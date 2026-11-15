//! Overlapping vertical windows for long webtoon strips, stitched back into
//! page coordinates before layout suppression.

use super::{Detection, suppress};

const SLICE_ASPECT_THRESHOLD: f32 = 2.4;
const MIN_SLICED_IMAGE_HEIGHT: u32 = 1728;
const MIN_SLICE_HEIGHT: u32 = 1152;
const MIN_SLICE_OVERLAP: u32 = 160;
const DUPLICATE_IOU_THRESHOLD: f32 = 0.45;
const CONTAINMENT_MERGE_THRESHOLD: f32 = 0.80;
const SEAM_X_OVERLAP_THRESHOLD: f32 = 0.65;
const SEAM_VERTICAL_GAP_PX: f32 = 8.0;

pub(super) fn windows(width: u32, height: u32) -> Vec<(u32, u32)> {
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

pub(super) fn seams(windows: &[(u32, u32)]) -> Vec<(f32, f32)> {
    windows
        .windows(2)
        .filter_map(|pair| {
            let top = pair[1].0 as f32;
            let bottom = pair[0].1 as f32;
            (bottom > top).then_some((top, bottom))
        })
        .collect()
}

pub(super) fn offset(detection: &mut Detection, start_y: u32) {
    let dy = start_y as f32;
    detection.bbox[1] += dy;
    detection.bbox[3] += dy;
    if detection.mask.width > 0 && detection.mask.height > 0 {
        detection.mask.y = detection.mask.y.saturating_add(start_y);
    }
}

pub(super) fn stitch(mut detections: Vec<Detection>, seams: &[(f32, f32)]) -> Vec<Detection> {
    detections.sort_by(|left, right| right.score.total_cmp(&left.score));
    let mut merged: Vec<Detection> = Vec::with_capacity(detections.len());

    'candidates: for candidate in detections {
        for existing in &mut merged {
            if existing.label == candidate.label
                && across_seam(existing.bbox, candidate.bbox, seams)
            {
                existing.score = existing.score.max(candidate.score);
                existing.bbox = [
                    existing.bbox[0].min(candidate.bbox[0]),
                    existing.bbox[1].min(candidate.bbox[1]),
                    existing.bbox[2].max(candidate.bbox[2]),
                    existing.bbox[3].max(candidate.bbox[3]),
                ];
                existing.mask = suppress::union(&existing.mask, &candidate.mask);
                continue 'candidates;
            }
        }
        merged.push(candidate);
    }

    merged
}

fn across_seam(a: [f32; 4], b: [f32; 4], seams: &[(f32, f32)]) -> bool {
    let area_a = area(a);
    let area_b = area(b);
    if area_a <= 0.0 || area_b <= 0.0 {
        return false;
    }
    let inter = intersection(a, b);
    if inter > 0.0 {
        let whole = area_a + area_b - inter;
        if whole > 0.0 && inter / whole >= DUPLICATE_IOU_THRESHOLD {
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

fn area(bbox: [f32; 4]) -> f32 {
    (bbox[2] - bbox[0]).max(0.0) * (bbox[3] - bbox[1]).max(0.0)
}

fn intersection(a: [f32; 4], b: [f32; 4]) -> f32 {
    (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::{
        super::{Detection, Mask},
        seams, stitch, windows,
    };

    #[test]
    fn normal_pages_stay_single_pass_while_webtoons_slice_uniformly() {
        assert_eq!(windows(800, 1200), vec![(0, 1200)]);

        let slices = windows(800, 5000);
        assert!(slices.len() >= 5);
        assert_eq!(slices.first().copied(), Some((0, 1200)));
        assert_eq!(slices.last().copied(), Some((3800, 5000)));
        for pair in slices.windows(2) {
            assert!(pair[1].0 < pair[0].1);
            assert_eq!(pair[0].1 - pair[0].0, 1200);
        }
    }

    #[test]
    fn seam_merger_unions_split_vertical_bubbles_and_masks() {
        let slices = windows(800, 5000);
        let bands = seams(&slices);
        let top_half = Detection {
            label: "bubble".to_owned(),
            score: 0.91,
            bbox: [200.0, 820.0, 360.0, 1198.0],
            mask: Mask {
                x: 200,
                y: 1196,
                width: 2,
                height: 2,
                pixels: vec![255; 4],
            },
        };
        let bottom_half = Detection {
            label: "bubble".to_owned(),
            score: 0.95,
            bbox: [202.0, 905.0, 358.0, 1340.0],
            mask: Mask {
                x: 200,
                y: 1198,
                width: 2,
                height: 2,
                pixels: vec![255; 4],
            },
        };

        let merged = stitch(vec![top_half, bottom_half], &bands);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].bbox, [200.0, 820.0, 360.0, 1340.0]);
        assert!(merged[0].mask.contains(200, 1196));
        assert!(merged[0].mask.contains(201, 1199));
    }
}
