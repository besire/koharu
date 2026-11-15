use image::DynamicImage;
use koharu_ml::{baberu_ocr::BaberuOcr, hayai_ocr::HayaiOcr, manga_ocr_ar::MangaOcrAr};
use koharu_runtime::Device;
use serde::{Deserialize, Serialize};
use strum::{Display, VariantArray};

/// The recognizers, by id in the section and by name.
#[derive(
    Clone, Copy, Debug, Default, Display, Eq, Hash, PartialEq, Serialize, Deserialize, VariantArray,
)]
#[serde(rename_all = "kebab-case")]
#[derive(specta::Type)]
pub enum Recognizer {
    #[default]
    #[strum(to_string = "Manga OCR AR")]
    MangaOcrAr,
    #[strum(to_string = "Baberu OCR")]
    BaberuOcr,
    #[strum(to_string = "Hayai OCR")]
    HayaiOcr,
}

/// A recognizer, loaded.
pub(super) type Model = Box<dyn Recognize + Send>;

pub(super) trait Recognize {
    /// Each image's text, in order.
    fn inference(&self, images: &[DynamicImage]) -> anyhow::Result<Vec<String>>;

    /// The language it reads, as a BCP 47 tag: `und` for one that reads
    /// several and doesn't say which.
    fn language(&self) -> &'static str;
}

impl Recognize for MangaOcrAr {
    fn inference(&self, images: &[DynamicImage]) -> anyhow::Result<Vec<String>> {
        MangaOcrAr::inference(self, images)
    }

    fn language(&self) -> &'static str {
        "ja"
    }
}

impl Recognize for BaberuOcr {
    fn inference(&self, images: &[DynamicImage]) -> anyhow::Result<Vec<String>> {
        BaberuOcr::inference(self, images).map(|texts| {
            texts
                .into_iter()
                .map(|text| normalize_ellipsis(&text))
                .collect()
        })
    }

    fn language(&self) -> &'static str {
        "ja"
    }
}

/// Baberu reads a manga ellipsis as a full-width colon (`あ：あなた？`) or as a
/// run of full-width periods, which translators then misread as colons. Only
/// Baberu's output goes through this; between two digits (`12：00`, `3．5`)
/// the punctuation is kept.
fn normalize_ellipsis(text: &str) -> String {
    let characters: Vec<char> = text.chars().collect();
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

impl Recognize for HayaiOcr {
    fn inference(&self, images: &[DynamicImage]) -> anyhow::Result<Vec<String>> {
        HayaiOcr::inference(self, images)
    }

    fn language(&self) -> &'static str {
        "und"
    }
}

pub(super) async fn load(model: Recognizer, device: &'static Device) -> anyhow::Result<Model> {
    Ok(match model {
        Recognizer::MangaOcrAr => Box::new(MangaOcrAr::load(device).await?),
        Recognizer::BaberuOcr => Box::new(BaberuOcr::load(device).await?),
        Recognizer::HayaiOcr => Box::new(HayaiOcr::load(device).await?),
    })
}

#[cfg(test)]
mod tests {
    use super::normalize_ellipsis;

    #[test]
    fn baberu_full_width_colons_and_periods_normalize_to_ellipsis() {
        assert_eq!(normalize_ellipsis("あ：あなた？"), "あ…あなた？");
        assert_eq!(normalize_ellipsis("そんな．．．"), "そんな…");
        assert_eq!(normalize_ellipsis("：：待って"), "…待って");
        assert_eq!(normalize_ellipsis("12：00"), "12：00");
        assert_eq!(normalize_ellipsis("３．５倍"), "３．５倍");
        assert_eq!(normalize_ellipsis("时间：12：00"), "时间…12：00");
        assert_eq!(normalize_ellipsis("12："), "12…");
    }
}
