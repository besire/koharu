use std::sync::{Arc, Mutex};

use super::{StageInput, StageProcessor, finish, generation};
use crate::{ModelCell, OcrModel, scope::geometry_extents};
use anyhow::{Context as _, Result, anyhow, bail};
use async_trait::async_trait;
use image::{DynamicImage, Rgb, RgbImage, imageops};
use koharu_ml::{
    baberu_ocr::BaberuOcr,
    hayai_ocr::HayaiOcr,
    manga_ocr::MangaOcr,
    paddle_ocr_vl::PaddleOCRVLTask,
    paddle_ocr_vl_quantized::PaddleOCRVLQuantized,
    pp_ocr_v6::{
        det::{PPOCRV6MediumDet, TextDetection},
        rec::PPOCRV6MediumRec,
    },
};
use koharu_scene::{
    At, Authored, EntityId, Geometry, LanguageTag, OcrAnalysis, Origin, Presents, RecognizedFrom,
    Region, RegionSpec, SourceText, TextDirection, TextLayout, TextRegion,
};
use koharu_translator::Language;

const PRODUCER: &str = "dev.koharu.pipeline.ocr";

pub(super) struct Processor {
    config: OcrModel,
    source_language: Option<Language>,
    device: koharu_ml::Device,
    model: ModelCell<Model>,
}

impl Processor {
    pub(super) fn new(
        config: OcrModel,
        source_language: Option<Language>,
        device: koharu_ml::Device,
    ) -> Self {
        Self {
            config,
            source_language,
            device,
            model: ModelCell::new(),
        }
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        match self.config {
            OcrModel::MangaOcr => "manga-ocr",
            OcrModel::BaberuOcr => "baberu-ocr",
            OcrModel::HayaiOcr => "hayai-ocr",
            OcrModel::PaddleOcrVl1_6 => "paddleocr-vl-1.6",
            OcrModel::PpOcrV6 => "pp-ocr-v6",
        }
    }

    fn unload(&self) -> bool {
        self.model.unload()
    }

    async fn load(&self) -> Result<()> {
        self.model
            .ensure(|| Model::load(self.device.clone(), &self.config))
            .await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        self.model
            .lock()
            .await
            .as_ref()
            .ok_or_else(|| anyhow!("OCR model is not loaded"))?
            .run(input, &self.config, self.source_language)
            .await
    }
}

enum Model {
    Manga(Arc<Mutex<MangaOcr>>),
    Baberu(Arc<Mutex<BaberuOcr>>),
    Hayai(Arc<Mutex<HayaiOcr>>),
    Paddle(Arc<Mutex<PaddleOCRVLQuantized>>),
    PpOcrV6 {
        det: Arc<Mutex<PPOCRV6MediumDet>>,
        rec: Arc<Mutex<PPOCRV6MediumRec>>,
    },
}

impl Model {
    async fn load(device: koharu_ml::Device, config: &OcrModel) -> Result<Self> {
        match config {
            OcrModel::MangaOcr => Ok(Self::Manga(Arc::new(Mutex::new(
                MangaOcr::load(device).await?,
            )))),
            OcrModel::BaberuOcr => Ok(Self::Baberu(Arc::new(Mutex::new(
                BaberuOcr::load(device).await?,
            )))),
            OcrModel::HayaiOcr => Ok(Self::Hayai(Arc::new(Mutex::new(
                HayaiOcr::load(device).await?,
            )))),
            OcrModel::PaddleOcrVl1_6 => Ok(Self::Paddle(Arc::new(Mutex::new(
                PaddleOCRVLQuantized::load(device).await?,
            )))),
            OcrModel::PpOcrV6 => Ok(Self::PpOcrV6 {
                det: Arc::new(Mutex::new(PPOCRV6MediumDet::load(device.clone()).await?)),
                rec: Arc::new(Mutex::new(PPOCRV6MediumRec::load(device).await?)),
            }),
        }
    }

    async fn run(
        &self,
        input: StageInput,
        ocr_model: &OcrModel,
        source_language: Option<Language>,
    ) -> Result<koharu_scene::Patch> {
        let model_name = match self {
            Self::Manga(_) => "manga-ocr",
            Self::Baberu(_) => "baberu-ocr",
            Self::Hayai(_) => "hayai-ocr",
            Self::Paddle(_) => "paddleocr-vl-1.6",
            Self::PpOcrV6 { .. } => "pp-ocr-v6",
        };
        let page = input.page;
        let mut targets = Vec::new();
        let source = input
            .images
            .get(&input.scene, page, "source")
            .await?
            .ok_or_else(|| anyhow!("page {page} has no source image"))?;
        for entity in input.scene.descendants(page)? {
            let region = entity.id();
            if !input.contains_entity(region)? {
                continue;
            }
            let is_text_region = input
                .scene
                .component::<Region>(region)?
                .is_some_and(|value| value.kind == TextRegion::kind());
            if !is_text_region {
                continue;
            }
            let geometry = input
                .scene
                .component::<Geometry>(region)?
                .ok_or_else(|| anyhow!("text region {region} has no geometry"))?;
            let crop = crop(&source, &geometry)
                .with_context(|| format!("text region {region} is outside its source image"))?;
            for relation in input.scene.relations_to_as::<RecognizedFrom>(region) {
                let content = relation.value().source;
                let previous = input.scene.component::<SourceText>(content)?;
                if previous.as_ref().is_some_and(|value| {
                    matches!(value.text.origin, Origin::User)
                        && (!input.has_entity_scope() || !value.text.value.is_empty())
                }) {
                    continue;
                }
                targets.push(OcrTarget {
                    content,
                    region: Some(region),
                    geometry: geometry.clone(),
                    previous,
                    image: crop.clone(),
                });
            }
        }

        if input.has_entity_scope() {
            for entity in input.scene.descendants(page)? {
                let layer = entity.id();
                if !input.contains_entity(layer)?
                    || input.scene.component::<TextLayout>(layer)?.is_none()
                    || input.scene.component::<Geometry>(layer)?.is_none()
                {
                    continue;
                }
                let Some(content) = input
                    .scene
                    .relation_from::<Presents>(layer)?
                    .map(|relation| relation.value().target)
                else {
                    continue;
                };
                let existing_region = input
                    .scene
                    .relation_from::<RecognizedFrom>(content)?
                    .map(|relation| relation.value().target);
                let geometry = input
                    .scene
                    .component::<Geometry>(layer)?
                    .ok_or_else(|| anyhow!("custom text layer {layer} has no geometry"))?;
                let crop = crop(&source, &geometry).with_context(|| {
                    format!("custom text layer {layer} is outside its source image")
                })?;
                let previous = input.scene.component::<SourceText>(content)?;
                if previous.as_ref().is_some_and(|value| {
                    matches!(value.text.origin, Origin::User) && !value.text.value.is_empty()
                }) {
                    continue;
                }
                targets.push(OcrTarget {
                    content,
                    region: existing_region,
                    geometry,
                    previous,
                    image: crop,
                });
            }
        }

        let results = match self {
            Self::Manga(model) => {
                infer_text(model.clone(), targets, |model, image| {
                    model.inference(image)
                })
                .await?
            }
            Self::Baberu(model) => {
                infer_text(model.clone(), targets, |model, image| {
                    Ok(normalize_ellipsis(&model.inference(image)?))
                })
                .await?
            }
            Self::Hayai(model) => {
                infer_text(model.clone(), targets, |model, image| {
                    model.inference(image)
                })
                .await?
            }
            Self::Paddle(model) => {
                infer_text(model.clone(), targets, |model, image| {
                    Ok(model.inference(image, PaddleOCRVLTask::Ocr)?.text)
                })
                .await?
            }
            Self::PpOcrV6 { det, rec } => {
                infer_pp_ocr_v6_targets(det.clone(), rec.clone(), targets).await?
            }
        };

        let generation = generation(PRODUCER, model_name)?;
        let mut edit = input.scene.edit_as(generation.clone());
        edit.observe_assets(page)?;
        for result in &results {
            if let Some(region) = result.region {
                edit.observe::<Region>(region)?;
                edit.observe::<Geometry>(region)?;
            }
            edit.observe::<SourceText>(result.content)?;
        }
        for result in results {
            let region = match result.region {
                Some(region) => region,
                None => {
                    let region = edit.add_entity(page, At::End)?;
                    edit.set(region, &result.geometry)?;
                    edit.set(
                        region,
                        &Region {
                            origin: Origin::Generated(generation.clone()),
                            kind: TextRegion::kind(),
                            label: Some("custom-text-frame".to_owned()),
                        },
                    )?;
                    edit.relate::<RecognizedFrom>(result.content, region)?;
                    region
                }
            };
            let language = resolve_ocr_language(
                source_language,
                ocr_model,
                result.previous.as_ref(),
                &result.text,
            );
            let source = SourceText {
                text: Authored::generated(result.text, generation.clone()),
                language,
            };
            if input.has_entity_scope() {
                edit.set_generated_source_text_if_empty(result.content, &source)?;
            } else {
                edit.set(result.content, &source)?;
            }
            let (min_x, min_y, max_x, max_y) = geometry_extents(&result.geometry)
                .ok_or_else(|| anyhow!("text region {region} has empty geometry"))?;
            edit.set(
                region,
                &OcrAnalysis {
                    origin: Origin::Generated(generation.clone()),
                    direction: if max_y - min_y >= (max_x - min_x) * 1.15 {
                        TextDirection::Vertical
                    } else {
                        TextDirection::Horizontal
                    },
                    confidence: None,
                    line_boundaries: Vec::new(),
                },
            )?;
        }
        finish(edit)
    }
}

struct OcrTarget {
    content: EntityId,
    region: Option<EntityId>,
    geometry: Geometry,
    previous: Option<SourceText>,
    image: DynamicImage,
}

struct OcrResult {
    content: EntityId,
    region: Option<EntityId>,
    geometry: Geometry,
    previous: Option<SourceText>,
    text: String,
}

async fn infer_text<M: Send + 'static>(
    model: Arc<Mutex<M>>,
    targets: Vec<OcrTarget>,
    inference: impl Fn(&M, &DynamicImage) -> Result<String> + Send + Sync + 'static,
) -> Result<Vec<OcrResult>> {
    tokio_rayon::spawn(move || {
        let model = model
            .lock()
            .map_err(|_| anyhow!("OCR model lock is poisoned"))?;
        targets
            .into_iter()
            .map(|target| {
                Ok(OcrResult {
                    content: target.content,
                    region: target.region,
                    geometry: target.geometry,
                    previous: target.previous,
                    text: normalize_ocr_text(inference(&model, &target.image)?),
                })
            })
            .collect()
    })
    .await
}

async fn infer_pp_ocr_v6_targets(
    det: Arc<Mutex<PPOCRV6MediumDet>>,
    rec: Arc<Mutex<PPOCRV6MediumRec>>,
    targets: Vec<OcrTarget>,
) -> Result<Vec<OcrResult>> {
    tokio_rayon::spawn(move || {
        let det = det
            .lock()
            .map_err(|_| anyhow!("PP-OCRv6 detection lock is poisoned"))?;
        let rec = rec
            .lock()
            .map_err(|_| anyhow!("PP-OCRv6 recognition lock is poisoned"))?;
        targets
            .into_iter()
            .map(|target| {
                Ok(OcrResult {
                    content: target.content,
                    region: target.region,
                    geometry: target.geometry,
                    previous: target.previous,
                    text: normalize_ocr_text(infer_pp_ocr_v6(&det, &rec, &target.image)?),
                })
            })
            .collect()
    })
    .await
}

fn infer_pp_ocr_v6(
    det: &PPOCRV6MediumDet,
    rec: &PPOCRV6MediumRec,
    image: &DynamicImage,
) -> Result<String> {
    const PAD: u32 = 12;
    let rgb = image.to_rgb8();
    let mut padded_rgb = RgbImage::from_pixel(
        rgb.width().saturating_add(PAD * 2),
        rgb.height().saturating_add(PAD * 2),
        Rgb([255, 255, 255]),
    );
    imageops::replace(&mut padded_rgb, &rgb, i64::from(PAD), i64::from(PAD));
    let padded = DynamicImage::ImageRgb8(padded_rgb);

    let detections = det.inference(&padded)?.detections;
    let mut line_boxes = extract_pp_ocr_boxes(&detections, padded.width(), padded.height());
    if line_boxes.is_empty() {
        let oriented = orient_line_crop(image);
        return Ok(rec.inference(&oriented)?.text);
    }

    sort_pp_ocr_boxes(&mut line_boxes);
    let mut lines = Vec::with_capacity(line_boxes.len());
    for [x, y, width, height] in line_boxes {
        let crop = padded.crop_imm(x, y, width, height);
        let oriented = orient_line_crop(&crop);
        let recognized = rec.inference(&oriented)?.text;
        let trimmed = recognized.trim();
        if !trimmed.is_empty() {
            lines.push(trimmed.to_owned());
        }
    }
    Ok(lines.join("\n"))
}

fn extract_pp_ocr_boxes(detections: &[TextDetection], width: u32, height: u32) -> Vec<[u32; 4]> {
    detections
        .iter()
        .filter(|detection| detection.score >= 0.3)
        .filter_map(|detection| {
            let mut min_x = f32::INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            for [x, y] in detection.polygon {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
            let left = (min_x.floor() - 2.0).max(0.0) as u32;
            let top = (min_y.floor() - 2.0).max(0.0) as u32;
            let right = (max_x.ceil() + 2.0).max(0.0).min(width as f32) as u32;
            let bottom = (max_y.ceil() + 2.0).max(0.0).min(height as f32) as u32;
            (right > left && bottom > top).then_some([left, top, right - left, bottom - top])
        })
        .collect()
}

fn sort_pp_ocr_boxes(boxes: &mut [[u32; 4]]) {
    let vertical_count = boxes
        .iter()
        .filter(|&&[_, _, width, height]| height as f32 > width as f32 * 1.4)
        .count();
    let vertical_majority = vertical_count * 2 > boxes.len();
    if vertical_majority {
        boxes.sort_by(|left, right| {
            let left_cx = left[0] as f32 + left[2] as f32 * 0.5;
            let right_cx = right[0] as f32 + right[2] as f32 * 0.5;
            right_cx
                .total_cmp(&left_cx)
                .then_with(|| left[1].cmp(&right[1]))
        });
    } else {
        boxes.sort_by(|left, right| {
            let left_cy = left[1] as f32 + left[3] as f32 * 0.5;
            let right_cy = right[1] as f32 + right[3] as f32 * 0.5;
            left_cy
                .total_cmp(&right_cy)
                .then_with(|| left[0].cmp(&right[0]))
        });
    }
}

fn orient_line_crop(image: &DynamicImage) -> DynamicImage {
    if image.height() as f32 > image.width() as f32 * 1.4 {
        DynamicImage::ImageRgb8(imageops::rotate270(&image.to_rgb8()))
    } else {
        image.clone()
    }
}

// Manga OCR can emit replacement-box glyphs for an isolated Japanese ellipsis.
// Normalize only an all-placeholder sequence so ordinary OCR output is preserved.
fn normalize_ocr_text(text: String) -> String {
    let visible = text
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<Vec<_>>();
    if visible.len() >= 2
        && visible
            .iter()
            .all(|character| matches!(character, '☐' | '□' | '▢' | '▣' | '�'))
    {
        "…".to_owned()
    } else {
        text
    }
}

/// Baberu reads a manga ellipsis as a full-width colon (「あ：あなた？」) or as
/// a run of full-width periods, and translators then render "A: Cariño" or
/// "Favor: ve a comprarlos". Only Baberu's output goes through this: other
/// models, PaddleOCR-VL among them, read Chinese, where "：" and "．" are real
/// punctuation. Between two digits (12：00, 3．5) they are kept as well.
fn normalize_ellipsis(text: &str) -> String {
    let characters = text.chars().collect::<Vec<_>>();
    let is_digit = |index: Option<usize>| {
        index
            .and_then(|index| characters.get(index))
            .is_some_and(|character| character.is_ascii_digit() || matches!(character, '０'..='９'))
    };
    let mut normalized = String::with_capacity(text.len());
    let mut start = 0;
    while start < characters.len() {
        if !matches!(characters[start], '：' | '．') {
            normalized.push(characters[start]);
            start += 1;
            continue;
        }
        let end = characters[start..]
            .iter()
            .position(|character| !matches!(character, '：' | '．'))
            .map_or(characters.len(), |length| start + length);
        if is_digit(start.checked_sub(1)) && is_digit(Some(end)) {
            normalized.extend(&characters[start..end]);
        } else {
            normalized.push('…');
        }
        start = end;
    }
    normalized
}

fn crop(source: &DynamicImage, geometry: &Geometry) -> Result<DynamicImage> {
    let (min_x, min_y, max_x, max_y) =
        geometry_extents(geometry).ok_or_else(|| anyhow!("geometry is empty"))?;
    let x = min_x.floor().max(0.0) as u32;
    let y = min_y.floor().max(0.0) as u32;
    let right = max_x.ceil().max(0.0).min(f64::from(source.width())) as u32;
    let bottom = max_y.ceil().max(0.0).min(f64::from(source.height())) as u32;
    if right <= x || bottom <= y {
        bail!("geometry does not overlap the image");
    }
    Ok(source.crop_imm(x, y, right - x, bottom - y))
}

fn resolve_ocr_language(
    configured: Option<Language>,
    model: &OcrModel,
    previous: Option<&SourceText>,
    text: &str,
) -> Option<LanguageTag> {
    if let Some(previous) = previous
        && matches!(previous.text.origin, Origin::User)
        && previous.language.is_some()
    {
        return previous.language.clone();
    }
    if let Some(language) = configured {
        return LanguageTag::new(language.tag()).ok();
    }
    match model {
        OcrModel::MangaOcr | OcrModel::BaberuOcr => LanguageTag::new("ja-JP").ok(),
        OcrModel::PaddleOcrVl1_6 | OcrModel::HayaiOcr | OcrModel::PpOcrV6 => {
            infer_script_language(text)
        }
    }
}

fn infer_script_language(text: &str) -> Option<LanguageTag> {
    let mut has_kana = false;
    let mut has_hangul = false;
    for ch in text.chars() {
        if matches!(ch, '\u{3041}'..='\u{3096}' | '\u{30A1}'..='\u{30FA}' | '\u{FF66}'..='\u{FF9F}')
        {
            has_kana = true;
        } else if matches!(
            ch,
            '\u{AC00}'..='\u{D7A3}' | '\u{1100}'..='\u{11FF}' | '\u{3131}'..='\u{318E}'
        ) {
            has_hangul = true;
        }
    }
    match (has_kana, has_hangul) {
        (true, false) => LanguageTag::new("ja-JP").ok(),
        (false, true) => LanguageTag::new("ko-KR").ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use koharu_scene::{Authored, LanguageTag, SourceText};
    use koharu_translator::Language;

    use crate::OcrModel;

    use super::{normalize_ellipsis, normalize_ocr_text, resolve_ocr_language};

    #[test]
    fn repeated_placeholder_glyphs_are_an_ellipsis() {
        assert_eq!(normalize_ocr_text("☐ ☐ ☐".to_owned()), "…");
        assert_eq!(normalize_ocr_text("□\n□".to_owned()), "…");
    }

    #[test]
    fn ordinary_text_and_single_boxes_are_unchanged() {
        assert_eq!(normalize_ocr_text("待って…".to_owned()), "待って…");
        assert_eq!(normalize_ocr_text("☐".to_owned()), "☐");
    }

    #[test]
    fn baberu_full_width_colons_and_periods_are_an_ellipsis() {
        assert_eq!(normalize_ellipsis("あ：あなた？"), "あ…あなた？");
        assert_eq!(normalize_ellipsis("そんな．．．"), "そんな…");
        assert_eq!(normalize_ellipsis("：：待って"), "…待って");
    }

    #[test]
    fn full_width_punctuation_between_digits_is_kept() {
        assert_eq!(normalize_ellipsis("12：00"), "12：00");
        assert_eq!(normalize_ellipsis("３．５倍"), "３．５倍");
        assert_eq!(normalize_ellipsis("3．5倍"), "3．5倍");
        // Only the colon between digits is protected; the one after a word is not.
        assert_eq!(normalize_ellipsis("时间：12：00"), "时间…12：00");
        assert_eq!(normalize_ellipsis("12："), "12…");
    }

    #[test]
    fn other_models_keep_full_width_punctuation() {
        // Only Baberu's output is passed through normalize_ellipsis; every
        // model, PaddleOCR-VL included, goes through normalize_ocr_text.
        for text in ["老师：你好", "时间：12：00", "3．5倍"] {
            assert_eq!(normalize_ocr_text(text.to_owned()), text);
        }
        assert_eq!(
            normalize_ocr_text("あ：あなた？".to_owned()),
            "あ：あなた？"
        );
    }

    #[test]
    fn pp_ocr_line_sorting_and_vertical_rotation_follow_reading_order() {
        let mut vertical = [[10, 20, 24, 120], [80, 20, 24, 120], [45, 20, 24, 120]];
        super::sort_pp_ocr_boxes(&mut vertical);
        assert_eq!(vertical.map(|item| item[0]), [80, 45, 10]);

        let tall = image::DynamicImage::new_rgb8(20, 80);
        let oriented = super::orient_line_crop(&tall);
        assert_eq!((oriented.width(), oriented.height()), (80, 20));
    }

    #[test]
    fn ocr_source_language_honors_explicit_config_and_avoids_false_japanese_tags() {
        assert_eq!(
            resolve_ocr_language(
                Some(Language::Korean),
                &OcrModel::PaddleOcrVl1_6,
                None,
                "Hello"
            ),
            Some(LanguageTag::new("ko-KR").unwrap())
        );
        assert_eq!(
            resolve_ocr_language(None, &OcrModel::BaberuOcr, None, "了解"),
            Some(LanguageTag::new("ja-JP").unwrap())
        );
        assert_eq!(
            resolve_ocr_language(None, &OcrModel::PaddleOcrVl1_6, None, "しばらくは大人しく"),
            Some(LanguageTag::new("ja-JP").unwrap())
        );
        assert_eq!(
            resolve_ocr_language(None, &OcrModel::HayaiOcr, None, "안녕하세요"),
            Some(LanguageTag::new("ko-KR").unwrap())
        );
        assert_eq!(
            resolve_ocr_language(None, &OcrModel::PaddleOcrVl1_6, None, "What happened?"),
            None
        );
        assert_eq!(
            resolve_ocr_language(None, &OcrModel::PpOcrV6, None, "今天天气真好"),
            None
        );

        let user_tagged = SourceText {
            text: Authored::user(String::new()),
            language: Some(LanguageTag::new("fr-FR").unwrap()),
        };
        assert_eq!(
            resolve_ocr_language(
                Some(Language::Japanese),
                &OcrModel::PaddleOcrVl1_6,
                Some(&user_tagged),
                "Bonjour"
            ),
            Some(LanguageTag::new("fr-FR").unwrap())
        );
    }
}
