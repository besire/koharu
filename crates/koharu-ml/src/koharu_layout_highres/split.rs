//! Splits unhosted multi-speaker free-text detections along cross-line gaps or
//! distinct text paints before reading-order sorting.

use image::RgbImage;
use rayon::prelude::*;

use super::{Detection, Mask, measure};

const IN_BUBBLE: f32 = 0.5;
const MIN_SPLIT_PIXELS: usize = 48;
const MIN_SPLIT_EXTENT: u32 = 6;
const BBOX_MARGIN: f32 = 2.0;
const DISTINCT_INK_DISTANCE_SQUARED: u32 = 52 * 52;
const DISTINCT_OUTLINE_DISTANCE_SQUARED: u32 = 64 * 64;

#[derive(Clone, Copy)]
struct Pixel {
    x: u32,
    y: u32,
    value: u8,
}

#[derive(Clone, Copy)]
struct Band {
    start: usize,
    end: usize,
    peak: usize,
    center: f32,
}

#[derive(Clone, Copy)]
struct Boundary {
    empty_gap: usize,
    cut: f64,
    pitch: f32,
}

pub(super) fn split(detections: &mut Vec<Detection>, page: &RgbImage) {
    if detections.iter().all(|detection| detection.label != "text") {
        return;
    }
    let bubbles: Vec<&Mask> = detections
        .iter()
        .filter(|detection| detection.label == "bubble")
        .map(|detection| &detection.mask)
        .collect();
    let splits: Vec<Option<Vec<Detection>>> = detections
        .par_iter()
        .map(|detection| split_one(detection, &bubbles, page))
        .collect();
    if splits.iter().all(Option::is_none) {
        return;
    }
    let mut expanded = Vec::with_capacity(detections.len());
    for (original, split) in std::mem::take(detections).into_iter().zip(splits) {
        match split {
            Some(parts) => expanded.extend(parts),
            None => expanded.push(original),
        }
    }
    *detections = expanded;
}

fn split_one(detection: &Detection, bubbles: &[&Mask], page: &RgbImage) -> Option<Vec<Detection>> {
    if detection.label != "text"
        || detection.mask.width == 0
        || detection.mask.height == 0
        || detection.mask.pixels.is_empty()
    {
        return None;
    }
    let foreground = collect_pixels(&detection.mask, page.width(), page.height());
    if foreground.len() < MIN_SPLIT_PIXELS * 2 {
        return None;
    }
    if bubbles.iter().any(|bubble| {
        let inside = foreground
            .iter()
            .filter(|pixel| bubble.contains(pixel.x, pixel.y))
            .count();
        inside as f32 / foreground.len() as f32 >= IN_BUBBLE
    }) {
        return None;
    }

    let (vertical, tilt) = measure::typography_of(&detection.mask, page).map_or_else(
        || {
            let [x0, y0, x1, y1] = detection.bbox;
            (y1 - y0 >= x1 - x0, 0.0)
        },
        |found| (found.vertical, found.tilt),
    );
    let (sin, cos) = f64::from(tilt).to_radians().sin_cos();
    let (cross_x, cross_y) = if vertical { (cos, sin) } else { (-sin, cos) };
    let (inline_x, inline_y) = if vertical { (-sin, cos) } else { (cos, sin) };

    let (cross_groups, line_width) = split_cross(page, &foreground, cross_x, cross_y);
    let mut groups = Vec::with_capacity(cross_groups.len());
    for group in cross_groups {
        groups.extend(split_inline(group, inline_x, inline_y, line_width));
    }
    let groups = coalesce_small(groups);
    if groups.len() <= 1 {
        return None;
    }

    let parts: Vec<Detection> = groups
        .into_iter()
        .filter_map(|group| sub_detection(detection, &group, page.width(), page.height()))
        .collect();
    (parts.len() > 1).then_some(parts)
}

fn collect_pixels(mask: &Mask, page_width: u32, page_height: u32) -> Vec<Pixel> {
    let mut out = Vec::new();
    for local_y in 0..mask.height {
        let Some(y) = mask.y.checked_add(local_y).filter(|&y| y < page_height) else {
            continue;
        };
        let row = local_y as usize * mask.width as usize;
        for local_x in 0..mask.width {
            let Some(&value) = mask.pixels.get(row + local_x as usize) else {
                continue;
            };
            if value == 0 {
                continue;
            }
            let Some(x) = mask.x.checked_add(local_x).filter(|&x| x < page_width) else {
                continue;
            };
            out.push(Pixel { x, y, value });
        }
    }
    out
}

fn project_profile(foreground: &[Pixel], axis_x: f64, axis_y: f64) -> Option<(f64, Vec<u32>)> {
    let first = foreground.first()?;
    let initial = (f64::from(first.x) + 0.5) * axis_x + (f64::from(first.y) + 0.5) * axis_y;
    let (min_coord, max_coord) =
        foreground[1..]
            .iter()
            .fold((initial, initial), |(low, high), pixel| {
                let coord =
                    (f64::from(pixel.x) + 0.5) * axis_x + (f64::from(pixel.y) + 0.5) * axis_y;
                (low.min(coord), high.max(coord))
            });
    let origin = min_coord.floor();
    let len = (max_coord.ceil() - origin).max(0.0) as usize + 1;
    let mut profile = vec![0u32; len];
    for pixel in foreground {
        let coord = (f64::from(pixel.x) + 0.5) * axis_x + (f64::from(pixel.y) + 0.5) * axis_y;
        let bin = ((coord - origin).floor().max(0.0) as usize).min(len - 1);
        profile[bin] += 1;
    }
    Some((origin, profile))
}

fn split_cross(
    page: &RgbImage,
    foreground: &[Pixel],
    cross_x: f64,
    cross_y: f64,
) -> (Vec<Vec<Pixel>>, f32) {
    let Some((origin, profile)) = project_profile(foreground, cross_x, cross_y) else {
        return (vec![foreground.to_vec()], 16.0);
    };
    if profile.len() < 16 {
        return (vec![foreground.to_vec()], profile.len().max(12) as f32);
    }
    let max_count = profile.iter().copied().max().unwrap_or(0);
    if max_count == 0 {
        return (vec![foreground.to_vec()], 16.0);
    }
    let empty_ceiling = max_count / 50;
    let bands = detect_bands(&profile, empty_ceiling);
    let line_width = if bands.is_empty() {
        16.0
    } else {
        median(
            &bands
                .iter()
                .map(|band| (band.end - band.start) as f32)
                .collect::<Vec<_>>(),
        )
        .max(6.0)
    };
    if bands.len() < 2 {
        return (vec![foreground.to_vec()], line_width);
    }

    let boundaries: Vec<Boundary> = (0..bands.len() - 1)
        .map(|index| {
            measure_boundary(
                &profile,
                origin,
                empty_ceiling,
                bands[index],
                bands[index + 1],
            )
        })
        .collect();

    let mut cuts = Vec::new();
    for (index, boundary) in boundaries.iter().copied().enumerate() {
        if boundary.empty_gap < 6 {
            continue;
        }
        let other_gaps: Vec<f32> = boundaries
            .iter()
            .enumerate()
            .filter_map(|(other, candidate)| (other != index).then_some(candidate.empty_gap as f32))
            .collect();
        let other_pitches: Vec<f32> = boundaries
            .iter()
            .enumerate()
            .filter_map(|(other, candidate)| (other != index).then_some(candidate.pitch))
            .collect();
        let baseline_gap = median(&other_gaps);
        let baseline_pitch = median(&other_pitches);

        let geometric = if bands.len() >= 3 {
            boundary.empty_gap >= 8
                && (boundary.empty_gap as f32) >= line_width * 0.75
                && (boundary.empty_gap as f32) >= baseline_gap * 2.0 + 6.0
                && boundary.pitch >= baseline_pitch * 1.45
        } else {
            boundary.empty_gap >= 14 && (boundary.empty_gap as f32) >= line_width * 1.25
        };
        if geometric {
            cuts.push(boundary.cut);
            continue;
        }

        let paint_candidate = (boundary.empty_gap as f32) >= (line_width.min(32.0) * 0.25).max(6.0)
            && (bands.len() == 2 || (boundary.empty_gap as f32) >= baseline_gap * 1.4 + 3.0);
        if paint_candidate && distinct_paint(page, foreground, cross_x, cross_y, &boundaries, index)
        {
            cuts.push(boundary.cut);
        }
    }

    if cuts.is_empty() {
        (vec![foreground.to_vec()], line_width)
    } else {
        (partition(foreground, cross_x, cross_y, &cuts), line_width)
    }
}

fn split_inline(
    foreground: Vec<Pixel>,
    inline_x: f64,
    inline_y: f64,
    line_width: f32,
) -> Vec<Vec<Pixel>> {
    if foreground.len() < MIN_SPLIT_PIXELS * 2 {
        return vec![foreground];
    }
    let Some((origin, profile)) = project_profile(&foreground, inline_x, inline_y) else {
        return vec![foreground];
    };
    let max_count = profile.iter().copied().max().unwrap_or(0);
    if max_count == 0 {
        return vec![foreground];
    }
    let empty_ceiling = max_count / 50;
    let Some(first_active) = profile.iter().position(|&count| count > empty_ceiling) else {
        return vec![foreground];
    };
    let Some(last_active) = profile.iter().rposition(|&count| count > empty_ceiling) else {
        return vec![foreground];
    };
    let min_gap = ((line_width.max(12.0) * 2.4).ceil() as usize).max(28);

    let mut cuts = Vec::new();
    let mut index = first_active;
    while index <= last_active {
        if profile[index] > empty_ceiling {
            index += 1;
            continue;
        }
        let start = index;
        while index <= last_active && profile[index] <= empty_ceiling {
            index += 1;
        }
        if start > first_active && index <= last_active && index - start >= min_gap {
            cuts.push(origin + (start + index) as f64 * 0.5);
        }
    }

    if cuts.is_empty() {
        vec![foreground]
    } else {
        partition(&foreground, inline_x, inline_y, &cuts)
    }
}

fn detect_bands(profile: &[u32], empty_ceiling: u32) -> Vec<Band> {
    let smooth: Vec<u32> = (0..profile.len())
        .map(|index| {
            let at = |offset: isize| {
                profile[(index as isize + offset).clamp(0, profile.len() as isize - 1) as usize]
            };
            at(-2) + at(-1) * 2 + at(0) * 3 + at(1) * 2 + at(2)
        })
        .collect();

    let mut raw_spans = Vec::new();
    let mut cursor = 0;
    while cursor < profile.len() {
        if profile[cursor] <= empty_ceiling {
            cursor += 1;
            continue;
        }
        let start = cursor;
        while cursor < profile.len() && profile[cursor] > empty_ceiling {
            cursor += 1;
        }
        subdivide_bands(profile, &smooth, start, cursor, &mut raw_spans);
    }

    raw_spans
        .into_iter()
        .filter_map(|(start, end)| {
            if end - start < 2 {
                return None;
            }
            let total: u64 = profile[start..end]
                .iter()
                .map(|&count| u64::from(count))
                .sum();
            if total < 12 {
                return None;
            }
            let weighted: f64 = profile[start..end]
                .iter()
                .enumerate()
                .map(|(offset, &count)| {
                    (start + offset) as f64 * f64::from(count) + 0.5 * f64::from(count)
                })
                .sum();
            let peak = (start..end).max_by_key(|&bin| smooth[bin]).unwrap_or(start);
            Some(Band {
                start,
                end,
                peak,
                center: (weighted / total as f64) as f32,
            })
        })
        .collect()
}

fn subdivide_bands(
    profile: &[u32],
    smooth: &[u32],
    start: usize,
    end: usize,
    out: &mut Vec<(usize, usize)>,
) {
    if end - start < 24 {
        out.push((start, end));
        return;
    }
    let mut best_valley: Option<(usize, f64)> = None;
    for candidate in (start + 10)..=(end - 10) {
        let left_peak = smooth[start..candidate].iter().copied().max().unwrap_or(0);
        let right_peak = smooth[(candidate + 1)..end]
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let reference = left_peak.min(right_peak);
        let raw_reference = profile[start..candidate]
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .min(
                profile[(candidate + 1)..end]
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0),
            );
        if reference == 0 || raw_reference == 0 {
            continue;
        }
        let ratio = f64::from(smooth[candidate]) / f64::from(reference);
        let raw_ratio = f64::from(profile[candidate]) / f64::from(raw_reference);
        if ratio <= 0.48 && raw_ratio <= 0.30 && best_valley.is_none_or(|(_, best)| ratio < best) {
            best_valley = Some((candidate, ratio));
        }
    }
    if let Some((valley, _)) = best_valley {
        subdivide_bands(profile, smooth, start, valley, out);
        subdivide_bands(profile, smooth, valley, end, out);
    } else {
        out.push((start, end));
    }
}

fn measure_boundary(
    profile: &[u32],
    origin: f64,
    empty_ceiling: u32,
    left: Band,
    right: Band,
) -> Boundary {
    let search_start = (left.peak + 1).min(profile.len());
    let search_end = right.peak.min(profile.len());
    let mut best_gap = (left.end.min(profile.len()), left.end.min(profile.len()));
    let mut index = search_start;
    while index < search_end {
        if profile[index] > empty_ceiling {
            index += 1;
            continue;
        }
        let start = index;
        while index < search_end && profile[index] <= empty_ceiling {
            index += 1;
        }
        if index - start > best_gap.1 - best_gap.0 {
            best_gap = (start, index);
        }
    }
    Boundary {
        empty_gap: best_gap.1 - best_gap.0,
        cut: origin + (best_gap.0 + best_gap.1) as f64 * 0.5,
        pitch: right.center - left.center,
    }
}

fn distinct_paint(
    page: &RgbImage,
    foreground: &[Pixel],
    cross_x: f64,
    cross_y: f64,
    boundaries: &[Boundary],
    index: usize,
) -> bool {
    let cut = boundaries[index].cut;
    let min_gap = ((boundaries[index].empty_gap as f32 * 0.75).floor() as usize).max(6);
    let left_bound = boundaries[..index]
        .iter()
        .rfind(|boundary| boundary.empty_gap >= min_gap)
        .map_or(f64::NEG_INFINITY, |boundary| boundary.cut);
    let right_bound = boundaries[index + 1..]
        .iter()
        .find(|boundary| boundary.empty_gap >= min_gap)
        .map_or(f64::INFINITY, |boundary| boundary.cut);

    let mut left_pixels = Vec::new();
    let mut right_pixels = Vec::new();
    for &pixel in foreground {
        let coord = (f64::from(pixel.x) + 0.5) * cross_x + (f64::from(pixel.y) + 0.5) * cross_y;
        if coord >= left_bound && coord < cut {
            left_pixels.push(pixel);
        } else if coord >= cut && coord < right_bound {
            right_pixels.push(pixel);
        }
    }
    if left_pixels.len() < MIN_SPLIT_PIXELS || right_pixels.len() < MIN_SPLIT_PIXELS {
        return false;
    }
    let (Some((_, left_mask)), Some((_, right_mask))) = (
        mask_and_bbox(&left_pixels, page.width(), page.height()),
        mask_and_bbox(&right_pixels, page.width(), page.height()),
    ) else {
        return false;
    };
    let (Some(left), Some(right)) = (
        measure::typography_of(&left_mask, page),
        measure::typography_of(&right_mask, page),
    ) else {
        return false;
    };
    let ink_differs = left
        .ink
        .zip(right.ink)
        .is_some_and(|(a, b)| color_distance_squared(a, b) >= DISTINCT_INK_DISTANCE_SQUARED);
    let outline_differs = match (left.outline, right.outline) {
        (Some(a), Some(b)) => {
            color_distance_squared(a.color, b.color) >= DISTINCT_OUTLINE_DISTANCE_SQUARED
        }
        (Some(outline), None) => {
            outline.width >= 2.5
                && right.ink.is_some_and(|ink| {
                    color_distance_squared(outline.color, ink) >= DISTINCT_INK_DISTANCE_SQUARED
                })
        }
        (None, Some(outline)) => {
            outline.width >= 2.5
                && left.ink.is_some_and(|ink| {
                    color_distance_squared(outline.color, ink) >= DISTINCT_INK_DISTANCE_SQUARED
                })
        }
        (None, None) => false,
    };
    ink_differs || outline_differs
}

fn color_distance_squared(a: [u8; 3], b: [u8; 3]) -> u32 {
    (0..3)
        .map(|channel| {
            let diff = u32::from(a[channel].abs_diff(b[channel]));
            diff * diff
        })
        .sum()
}

fn partition(foreground: &[Pixel], axis_x: f64, axis_y: f64, cuts: &[f64]) -> Vec<Vec<Pixel>> {
    let mut groups = vec![Vec::new(); cuts.len() + 1];
    for &pixel in foreground {
        let coord = (f64::from(pixel.x) + 0.5) * axis_x + (f64::from(pixel.y) + 0.5) * axis_y;
        let bucket = cuts.partition_point(|&cut| coord >= cut);
        groups[bucket].push(pixel);
    }
    groups.retain(|group| !group.is_empty());
    groups
}

fn coalesce_small(mut groups: Vec<Vec<Pixel>>) -> Vec<Vec<Pixel>> {
    let valid = |group: &[Pixel]| {
        if group.len() < MIN_SPLIT_PIXELS {
            return false;
        }
        let Some((min_x, min_y, max_x, max_y)) = group_bounds(group) else {
            return false;
        };
        max_x - min_x >= MIN_SPLIT_EXTENT && max_y - min_y >= MIN_SPLIT_EXTENT
    };
    while groups.len() > 1 {
        let Some(small_index) = groups.iter().position(|group| !valid(group)) else {
            break;
        };
        let small = groups.remove(small_index);
        let small_center = group_center(&small);
        let target = groups
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                squared_distance(small_center, group_center(left))
                    .total_cmp(&squared_distance(small_center, group_center(right)))
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
        groups[target].extend(small);
    }
    groups
}

fn group_bounds(group: &[Pixel]) -> Option<(u32, u32, u32, u32)> {
    let first = group.first()?;
    Some(group[1..].iter().fold(
        (first.x, first.y, first.x + 1, first.y + 1),
        |(min_x, min_y, max_x, max_y), pixel| {
            (
                min_x.min(pixel.x),
                min_y.min(pixel.y),
                max_x.max(pixel.x + 1),
                max_y.max(pixel.y + 1),
            )
        },
    ))
}

fn group_center(group: &[Pixel]) -> (f64, f64) {
    if group.is_empty() {
        return (0.0, 0.0);
    }
    let (sum_x, sum_y) = group.iter().fold((0.0, 0.0), |(sx, sy), pixel| {
        (sx + f64::from(pixel.x) + 0.5, sy + f64::from(pixel.y) + 0.5)
    });
    let inv = 1.0 / group.len() as f64;
    (sum_x * inv, sum_y * inv)
}

fn squared_distance(left: (f64, f64), right: (f64, f64)) -> f64 {
    let dx = left.0 - right.0;
    let dy = left.1 - right.1;
    dx * dx + dy * dy
}

fn mask_and_bbox(group: &[Pixel], page_width: u32, page_height: u32) -> Option<([f32; 4], Mask)> {
    let (min_x, min_y, max_x, max_y) = group_bounds(group)?;
    let width = max_x - min_x;
    let height = max_y - min_y;
    let mut pixels = vec![0u8; (width * height) as usize];
    for pixel in group {
        let local_x = pixel.x - min_x;
        let local_y = pixel.y - min_y;
        pixels[(local_y * width + local_x) as usize] = pixel.value;
    }
    let bbox = [
        (min_x as f32 - BBOX_MARGIN).max(0.0),
        (min_y as f32 - BBOX_MARGIN).max(0.0),
        (max_x as f32 + BBOX_MARGIN).min(page_width as f32),
        (max_y as f32 + BBOX_MARGIN).min(page_height as f32),
    ];
    Some((
        bbox,
        Mask {
            x: min_x,
            y: min_y,
            width,
            height,
            pixels,
        },
    ))
}

fn sub_detection(
    parent: &Detection,
    group: &[Pixel],
    page_width: u32,
    page_height: u32,
) -> Option<Detection> {
    let (raw_bbox, mask) = mask_and_bbox(group, page_width, page_height)?;
    let [parent_left, parent_top, parent_right, parent_bottom] = parent.bbox;
    let bbox = [
        raw_bbox[0].max(parent_left),
        raw_bbox[1].max(parent_top),
        raw_bbox[2].min(parent_right),
        raw_bbox[3].min(parent_bottom),
    ];
    let bbox = if bbox[2] > bbox[0] && bbox[3] > bbox[1] {
        bbox
    } else {
        raw_bbox
    };
    Some(Detection {
        label: parent.label.clone(),
        score: parent.score,
        bbox,
        mask,
    })
}

fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::{
        super::{Detection, Mask, measure, order},
        split,
    };

    struct Column {
        left: u32,
        top: u32,
        bottom: u32,
        fill: [u8; 3],
        stroke: [u8; 3],
    }

    fn vertical_scene(
        width: u32,
        height: u32,
        background: [u8; 3],
        columns: &[Column],
    ) -> (RgbImage, Detection) {
        let mut page = RgbImage::from_pixel(width, height, Rgb(background));
        let mut full = vec![0u8; (width * height) as usize];
        let glyph_w = 14u32;
        let glyph_h = 14u32;
        let stroke_w = 4u32;

        for column in columns {
            let mut y = column.top;
            while y + glyph_h <= column.bottom {
                let s_left = column.left.saturating_sub(stroke_w);
                let s_top = y.saturating_sub(stroke_w);
                let s_right = (column.left + glyph_w + stroke_w).min(width);
                let s_bottom = (y + glyph_h + stroke_w).min(height);
                for py in s_top..s_bottom {
                    for px in s_left..s_right {
                        page.put_pixel(px, py, Rgb(column.stroke));
                    }
                }
                for py in y..(y + glyph_h) {
                    for px in column.left..(column.left + glyph_w) {
                        page.put_pixel(px, py, Rgb(column.fill));
                        full[(py * width + px) as usize] = u8::MAX;
                    }
                }
                y += glyph_h + 6;
            }
        }

        let min_x = columns
            .iter()
            .map(|col| col.left.saturating_sub(stroke_w))
            .min()
            .unwrap();
        let min_y = columns
            .iter()
            .map(|col| col.top.saturating_sub(stroke_w))
            .min()
            .unwrap();
        let max_x = columns
            .iter()
            .map(|col| (col.left + glyph_w + stroke_w).min(width))
            .max()
            .unwrap();
        let max_y = columns
            .iter()
            .map(|col| (col.bottom + stroke_w).min(height))
            .max()
            .unwrap();

        let mask_w = max_x - min_x;
        let mask_h = max_y - min_y;
        let mut pixels = vec![0u8; (mask_w * mask_h) as usize];
        for y in min_y..max_y {
            for x in min_x..max_x {
                pixels[((y - min_y) * mask_w + (x - min_x)) as usize] =
                    full[(y * width + x) as usize];
            }
        }

        let bbox = [min_x as f32, min_y as f32, max_x as f32, max_y as f32];
        let raw_mask = Mask {
            x: min_x,
            y: min_y,
            width: mask_w,
            height: mask_h,
            pixels,
        };
        let fitted = measure::glyph_mask(&raw_mask, bbox, &page);
        (
            page,
            Detection {
                label: "text".to_owned(),
                score: 0.95,
                bbox,
                mask: fitted,
            },
        )
    }

    #[test]
    fn multi_speaker_outlined_columns_split_and_preserve_distinct_ink_and_outline() {
        // Two speakers over mid-tone CG art: left speaker has white text with pink
        // outline; right speaker has black text with white outline.
        let columns = [
            Column {
                left: 26,
                top: 28,
                bottom: 164,
                fill: [255, 255, 255],
                stroke: [235, 95, 130],
            },
            Column {
                left: 52,
                top: 28,
                bottom: 164,
                fill: [255, 255, 255],
                stroke: [235, 95, 130],
            },
            Column {
                left: 88,
                top: 28,
                bottom: 248,
                fill: [15, 15, 15],
                stroke: [255, 255, 255],
            },
            Column {
                left: 114,
                top: 28,
                bottom: 248,
                fill: [15, 15, 15],
                stroke: [255, 255, 255],
            },
        ];
        let (page, merged) = vertical_scene(220, 290, [92, 64, 58], &columns);
        let mut detections = vec![merged];

        split(&mut detections, &page);
        order::sort(&mut detections);

        assert_eq!(detections.len(), 2);
        assert!(
            detections[0].bbox[0] > detections[1].bbox[2],
            "right speaker precedes left speaker in vertical reading order: {:?}",
            [detections[0].bbox, detections[1].bbox]
        );
        assert!(detections[1].bbox[3] <= 180.0);

        let right = measure::typography_of(&detections[0].mask, &page).expect("right typography");
        let left = measure::typography_of(&detections[1].mask, &page).expect("left typography");
        assert!(right.ink.unwrap().iter().all(|&c| c <= 32), "{right:?}");
        assert!(
            right.outline.unwrap().color.iter().all(|&c| c >= 230),
            "{right:?}"
        );
        assert!(left.ink.unwrap().iter().all(|&c| c >= 230), "{left:?}");
        assert!(left.outline.unwrap().color[0] > left.outline.unwrap().color[1] + 80);
    }

    #[test]
    fn outlier_gap_splits_unhosted_blocks_while_bubbles_and_uniform_paragraphs_stay_whole() {
        let two_blocks = [
            Column {
                left: 24,
                top: 24,
                bottom: 220,
                fill: [15, 15, 15],
                stroke: [255, 255, 255],
            },
            Column {
                left: 48,
                top: 24,
                bottom: 220,
                fill: [15, 15, 15],
                stroke: [255, 255, 255],
            },
            Column {
                left: 96,
                top: 24,
                bottom: 220,
                fill: [15, 15, 15],
                stroke: [255, 255, 255],
            },
            Column {
                left: 120,
                top: 24,
                bottom: 220,
                fill: [15, 15, 15],
                stroke: [255, 255, 255],
            },
        ];
        let (page, merged) = vertical_scene(200, 260, [92, 64, 58], &two_blocks);
        let mut free = vec![merged.clone()];
        split(&mut free, &page);
        assert_eq!(free.len(), 2);

        let bubble = Detection {
            label: "bubble".to_owned(),
            score: 0.98,
            bbox: [10.0, 10.0, 160.0, 250.0],
            mask: Mask {
                x: 10,
                y: 10,
                width: 150,
                height: 240,
                pixels: vec![u8::MAX; 150 * 240],
            },
        };
        let mut hosted = vec![bubble, merged];
        split(&mut hosted, &page);
        assert_eq!(hosted.iter().filter(|d| d.label == "text").count(), 1);

        let uniform: Vec<Column> = (0..4)
            .map(|index| Column {
                left: 24 + index * 28,
                top: 24,
                bottom: 220,
                fill: [15, 15, 15],
                stroke: [255, 255, 255],
            })
            .collect();
        let (uniform_page, uniform_det) = vertical_scene(200, 260, [92, 64, 58], &uniform);
        let mut single = vec![uniform_det];
        split(&mut single, &uniform_page);
        assert_eq!(single.len(), 1);
    }
}
