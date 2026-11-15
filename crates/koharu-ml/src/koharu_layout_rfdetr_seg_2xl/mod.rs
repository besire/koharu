//! High-resolution manga layout instance segmentation with RF-DETR Seg 2XL.
//!
//! Checkpoint and strict Python loader:
//! https://huggingface.co/mayocream/koharu-layout-rfdetr-seg-2xl-1152/tree/aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2
//! RF-DETR upstream implementation:
//! https://github.com/roboflow/rf-detr/tree/4ab7c18729de9d02ffd0495795d0831b5630f01b

mod config;
mod model;
mod processor;

use anyhow::{Context, Result};
use image::DynamicImage;
use koharu_torch::Device;

use crate::backend::TryIntoDevice;

pub use self::{
    config::{KoharuLayoutRFDetrSeg2XLConfig, KoharuLayoutThresholds},
    processor::{
        KoharuLayoutDetection, KoharuLayoutDetections, KoharuLayoutMask,
        KoharuLayoutRFDetrImageProcessor,
    },
};

use self::model::Model;

crate::model_repository!("mayocream/koharu-layout-rfdetr-seg-2xl-1152" @ "aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2" {
    CONFIG = "inference_config.json",
    WEIGHTS = "model.safetensors",
});

#[derive(Debug)]
pub struct KoharuLayoutRFDetrSeg2XL {
    device: Device,
    model: Model,
    processor: KoharuLayoutRFDetrImageProcessor,
}

impl KoharuLayoutRFDetrSeg2XL {
    pub async fn load(device: crate::Device) -> Result<Self> {
        let device: Device = device.try_into_device()?;
        let config_path = CONFIG
            .resolve()
            .await
            .context("failed to resolve KoharuLayout RF-DETR inference config")?;
        let weights_path = WEIGHTS
            .resolve()
            .await
            .context("failed to resolve KoharuLayout RF-DETR weights")?;
        let config = KoharuLayoutRFDetrSeg2XLConfig::from_file(&config_path)?;
        let processor = KoharuLayoutRFDetrImageProcessor::new(&config)?;
        let mut model = Model::new(device);
        model
            .load(&weights_path)
            .with_context(|| format!("failed to load {}", weights_path.display()))?;
        Ok(Self {
            device,
            model,
            processor,
        })
    }

    pub fn inference(&self, image: &DynamicImage) -> Result<KoharuLayoutDetections> {
        self.inference_with_thresholds(image, self.processor.recommended_thresholds())
    }

    pub fn inference_with_thresholds(
        &self,
        image: &DynamicImage,
        thresholds: KoharuLayoutThresholds,
    ) -> Result<KoharuLayoutDetections> {
        let windows = vertical_slice_windows(image.width(), image.height());
        if windows.len() <= 1 {
            return self.inference_single(image, thresholds);
        }

        let seams = slice_overlap_bands(&windows);
        let mut all_detections = Vec::new();
        for (start_y, end_y) in windows {
            let tile = image.crop_imm(0, start_y, image.width(), end_y - start_y);
            let mut tile_result = self.inference_single(&tile, thresholds)?;
            for detection in &mut tile_result.detections {
                offset_detection(detection, start_y);
            }
            all_detections.extend(tile_result.detections);
        }

        let detections = merge_sliced_detections(all_detections, &seams);
        Ok(KoharuLayoutDetections {
            image_width: image.width(),
            image_height: image.height(),
            detections,
        })
    }

    fn inference_single(
        &self,
        image: &DynamicImage,
        thresholds: KoharuLayoutThresholds,
    ) -> Result<KoharuLayoutDetections> {
        koharu_torch::no_grad(|| {
            let pixel_values = self.processor.preprocess(image, self.device)?;
            let output = self.model.forward(&pixel_values);
            self.processor
                .postprocess(&output, image.width(), image.height(), thresholds)
        })
    }

    pub fn recommended_thresholds(&self) -> KoharuLayoutThresholds {
        self.processor.recommended_thresholds()
    }
}

const SLICE_ASPECT_THRESHOLD: f32 = 2.4;
const MIN_SLICED_IMAGE_HEIGHT: u32 = 1728;
const MIN_SLICE_HEIGHT: u32 = 1152;
const MIN_SLICE_OVERLAP: u32 = 160;
const DUPLICATE_IOU_THRESHOLD: f32 = 0.45;
const CONTAINMENT_MERGE_THRESHOLD: f32 = 0.80;
const SEAM_X_OVERLAP_THRESHOLD: f32 = 0.65;
const SEAM_VERTICAL_GAP_PX: f32 = 8.0;

fn vertical_slice_windows(width: u32, height: u32) -> Vec<(u32, u32)> {
    if width == 0
        || height <= MIN_SLICED_IMAGE_HEIGHT
        || (height as f32 / width as f32) <= SLICE_ASPECT_THRESHOLD
    {
        return vec![(0, height)];
    }

    let target_height = (width as f32 * 1.5).round() as u32;
    let slice_height = target_height.clamp(MIN_SLICE_HEIGHT, height);
    if slice_height >= height {
        return vec![(0, height)];
    }

    let overlap = (slice_height / 4).clamp(MIN_SLICE_OVERLAP, slice_height / 2);
    let stride = slice_height.saturating_sub(overlap).max(1);

    let mut windows = Vec::new();
    let mut start_y = 0;
    loop {
        if start_y + slice_height >= height {
            let final_window = (height - slice_height, height);
            if windows.last().copied() != Some(final_window) {
                windows.push(final_window);
            }
            break;
        }
        windows.push((start_y, start_y + slice_height));
        start_y += stride;
    }
    windows
}

fn slice_overlap_bands(windows: &[(u32, u32)]) -> Vec<(f32, f32)> {
    windows
        .windows(2)
        .filter_map(|pair| {
            let top = pair[1].0 as f32;
            let bottom = pair[0].1 as f32;
            (bottom > top).then_some((top, bottom))
        })
        .collect()
}

fn offset_detection(detection: &mut KoharuLayoutDetection, start_y: u32) {
    let dy = start_y as f32;
    detection.bbox[1] += dy;
    detection.bbox[3] += dy;
    if detection.mask.width > 0 && detection.mask.height > 0 {
        detection.mask.y = detection.mask.y.saturating_add(start_y);
    }
}

fn merge_sliced_detections(
    mut detections: Vec<KoharuLayoutDetection>,
    seams: &[(f32, f32)],
) -> Vec<KoharuLayoutDetection> {
    detections.sort_by(|left, right| right.score.total_cmp(&left.score));
    let mut merged: Vec<KoharuLayoutDetection> = Vec::with_capacity(detections.len());

    'candidates: for candidate in detections {
        for existing in &mut merged {
            if existing.label_id == candidate.label_id
                && should_merge_across_slices(existing.bbox, candidate.bbox, seams)
            {
                existing.score = existing.score.max(candidate.score);
                existing.bbox = [
                    existing.bbox[0].min(candidate.bbox[0]),
                    existing.bbox[1].min(candidate.bbox[1]),
                    existing.bbox[2].max(candidate.bbox[2]),
                    existing.bbox[3].max(candidate.bbox[3]),
                ];
                let (mask, area) = union_layout_masks(&existing.mask, &candidate.mask);
                existing.mask = mask;
                existing.area = area;
                continue 'candidates;
            }
        }
        merged.push(candidate);
    }

    merged
}

fn should_merge_across_slices(a: [f32; 4], b: [f32; 4], seams: &[(f32, f32)]) -> bool {
    let area_a = bbox_area(a);
    let area_b = bbox_area(b);
    if area_a <= 0.0 || area_b <= 0.0 {
        return false;
    }
    let inter = bbox_intersection(a, b);
    if inter > 0.0 {
        let union = area_a + area_b - inter;
        if union > 0.0 && inter / union >= DUPLICATE_IOU_THRESHOLD {
            return true;
        }
        if inter / area_a.min(area_b) >= CONTAINMENT_MERGE_THRESHOLD {
            return true;
        }
    }

    let width_a = (a[2] - a[0]).max(1.0);
    let width_b = (b[2] - b[0]).max(1.0);
    let x_overlap = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    if x_overlap / width_a.min(width_b) < SEAM_X_OVERLAP_THRESHOLD {
        return false;
    }

    let y_gap = (a[1].max(b[1]) - a[3].min(b[3])).max(0.0);
    if y_gap > SEAM_VERTICAL_GAP_PX {
        return false;
    }

    seams.iter().any(|&(seam_top, seam_bottom)| {
        let top = seam_top - SEAM_VERTICAL_GAP_PX;
        let bottom = seam_bottom + SEAM_VERTICAL_GAP_PX;
        a[3] >= top && a[1] <= bottom && b[3] >= top && b[1] <= bottom
    })
}

fn bbox_area(bbox: [f32; 4]) -> f32 {
    (bbox[2] - bbox[0]).max(0.0) * (bbox[3] - bbox[1]).max(0.0)
}

fn bbox_intersection(a: [f32; 4], b: [f32; 4]) -> f32 {
    let width = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let height = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    width * height
}

fn union_layout_masks(a: &KoharuLayoutMask, b: &KoharuLayoutMask) -> (KoharuLayoutMask, u32) {
    let a_empty = a.width == 0 || a.height == 0 || a.pixels.is_empty();
    let b_empty = b.width == 0 || b.height == 0 || b.pixels.is_empty();
    if a_empty && b_empty {
        return (
            KoharuLayoutMask {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
                pixels: Vec::new(),
            },
            0,
        );
    }
    if a_empty {
        let area = b.pixels.iter().filter(|&&pixel| pixel != 0).count() as u32;
        return (b.clone(), area);
    }
    if b_empty {
        let area = a.pixels.iter().filter(|&&pixel| pixel != 0).count() as u32;
        return (a.clone(), area);
    }

    let a_right = a.x.saturating_add(a.width);
    let b_right = b.x.saturating_add(b.width);
    let a_bottom = a.y.saturating_add(a.height);
    let b_bottom = b.y.saturating_add(b.height);
    let x = a.x.min(b.x);
    let y = a.y.min(b.y);
    let right = a_right.max(b_right);
    let bottom = a_bottom.max(b_bottom);
    let width = right.saturating_sub(x);
    let height = bottom.saturating_sub(y);
    let mut pixels = vec![0_u8; width as usize * height as usize];

    for mask in [a, b] {
        let offset_x = (mask.x - x) as usize;
        let offset_y = (mask.y - y) as usize;
        for row in 0..mask.height as usize {
            let src_start = row * mask.width as usize;
            let dst_start = (offset_y + row) * width as usize + offset_x;
            for col in 0..mask.width as usize {
                if mask.pixels[src_start + col] != 0 {
                    pixels[dst_start + col] = 255;
                }
            }
        }
    }

    let area = pixels.iter().filter(|&&pixel| pixel != 0).count() as u32;
    (
        KoharuLayoutMask {
            x,
            y,
            width,
            height,
            pixels,
        },
        area,
    )
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use anyhow::Result;

    use super::{
        KoharuLayoutDetection, KoharuLayoutMask, KoharuLayoutRFDetrSeg2XL, merge_sliced_detections,
        slice_overlap_bands, vertical_slice_windows,
    };

    #[test]
    fn normal_pages_stay_single_pass_while_webtoons_slice_uniformly() {
        assert_eq!(vertical_slice_windows(800, 1200), vec![(0, 1200)]);

        let windows = vertical_slice_windows(800, 5000);
        assert!(windows.len() >= 5);
        assert_eq!(windows.first().copied(), Some((0, 1200)));
        assert_eq!(windows.last().copied(), Some((3800, 5000)));
        for pair in windows.windows(2) {
            assert!(pair[1].0 < pair[0].1);
            assert_eq!(pair[0].1 - pair[0].0, 1200);
        }
    }

    #[test]
    fn seam_merger_unions_split_vertical_bubbles_and_masks() {
        let windows = vertical_slice_windows(800, 5000);
        let seams = slice_overlap_bands(&windows);
        let top_half = KoharuLayoutDetection {
            label_id: 2,
            label: "bubble".to_owned(),
            score: 0.91,
            bbox: [200.0, 820.0, 360.0, 1198.0],
            area: 4,
            mask: KoharuLayoutMask {
                x: 200,
                y: 1196,
                width: 2,
                height: 2,
                pixels: vec![255; 4],
            },
        };
        let bottom_half = KoharuLayoutDetection {
            label_id: 2,
            label: "bubble".to_owned(),
            score: 0.95,
            bbox: [202.0, 905.0, 358.0, 1340.0],
            area: 4,
            mask: KoharuLayoutMask {
                x: 200,
                y: 1198,
                width: 2,
                height: 2,
                pixels: vec![255; 4],
            },
        };

        let merged = merge_sliced_detections(vec![top_half, bottom_half], &seams);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].bbox, [200.0, 820.0, 360.0, 1340.0]);
        assert_eq!(merged[0].area, 8);
        assert!(merged[0].mask.contains(200, 1196));
        assert!(merged[0].mask.contains(201, 1199));
    }

    #[tokio::test]
    #[ignore = "downloads the checkpoint and requires CUDA"]
    async fn checkpoint_matches_rfdetr_upstream_structured_output() -> Result<()> {
        crate::init().await?;
        let image = image::open(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("benches/fixtures/object_detection/1.jpg"),
        )?;
        let model = KoharuLayoutRFDetrSeg2XL::load(crate::Device::cuda(0)).await?;
        let result = model.inference(&image)?;

        // RF-DETR 4ab7c18, CUDA BF16, shape=(1152, 1152), antialias disabled.
        // CUDA kernels vary across LibTorch releases, so compare structured
        // geometry and mask area in addition to bounded confidence differences.
        let best = &result.detections[0];
        assert_eq!(best.label, "bubble");
        assert!((best.score - 0.968_856_2).abs() < 0.01);
        for (actual, expected) in best
            .bbox
            .into_iter()
            .zip([566.220_7, 550.019_5, 691.044_9, 724.043])
        {
            assert!((actual - expected).abs() < 6.0);
        }
        assert!(best.area.abs_diff(18_882) < 100);

        let lower_panel = result
            .detections
            .iter()
            .find(|detection| detection.label == "panel" && detection.bbox[1] > 700.0)
            .expect("lower page panel");
        let middle_panel = result
            .detections
            .iter()
            .find(|detection| {
                detection.label == "panel" && detection.bbox[1] > 500.0 && detection.bbox[1] < 700.0
            })
            .expect("middle page panel");
        for (actual, expected) in lower_panel
            .bbox
            .into_iter()
            .zip([69.179_69, 799.453_1, 700.820_3, 1006.171_9])
        {
            assert!((actual - expected).abs() < 7.0);
        }
        for (actual, expected) in middle_panel
            .bbox
            .into_iter()
            .zip([70.683_59, 550.546_9, 699.316_4, 782.578_1])
        {
            assert!((actual - expected).abs() < 3.0);
        }
        assert!(lower_panel.area.abs_diff(136_734) < 1_000);
        assert!(middle_panel.area.abs_diff(142_179) < 2_000);
        Ok(())
    }
}
