/// Sixel encoder: converts a tiny-skia Pixmap to a Sixel byte string
/// for inline display in Sixel-capable terminals (WezTerm, foot, iTerm2, etc.).
///
/// Algorithm:
/// 1. Collect unique colors from the Pixmap (our diagrams use ≤10 colors)
/// 2. Write DCS header
/// 3. Process in 6-row bands, emitting sixel data per color
/// 4. Write ST terminator

use std::collections::HashMap;
use tiny_skia::Pixmap;

/// Convert a Pixmap to a Sixel-encoded byte string.
pub fn pixmap_to_sixel(pixmap: &Pixmap) -> Vec<u8> {
    let width = pixmap.width() as usize;
    let height = pixmap.height() as usize;
    let pixels = pixmap.pixels();

    // Collect unique colors and build palette
    let mut color_map: HashMap<(u8, u8, u8), usize> = HashMap::new();
    let mut palette: Vec<(u8, u8, u8)> = Vec::new();

    for pixel in pixels {
        let r = pixel.red();
        let g = pixel.green();
        let b = pixel.blue();
        let key = (r, g, b);
        if !color_map.contains_key(&key) {
            let idx = palette.len();
            color_map.insert(key, idx);
            palette.push(key);
        }
    }

    let mut out = Vec::with_capacity(width * height);

    // DCS header: P7;1q  (P7 = 256 colors, 1 = transparent bg)
    out.extend_from_slice(b"\x1bPq");

    // Raster attributes: "1;1;width;height
    out.extend_from_slice(format!("\"1;1;{};{}", width, height).as_bytes());

    // Define palette colors
    for (idx, &(r, g, b)) in palette.iter().enumerate() {
        // Sixel color percentages (0-100)
        let rp = (r as u32 * 100 + 127) / 255;
        let gp = (g as u32 * 100 + 127) / 255;
        let bp = (b as u32 * 100 + 127) / 255;
        out.extend_from_slice(format!("#{};2;{};{};{}", idx, rp, gp, bp).as_bytes());
    }

    // Process in 6-row bands
    let num_bands = (height + 5) / 6;

    for band in 0..num_bands {
        let y_start = band * 6;

        // For each color used in this band, emit sixel data
        let mut colors_in_band: Vec<usize> = Vec::new();
        for (ci, _) in palette.iter().enumerate() {
            // Check if this color appears in the band
            let mut used = false;
            'check: for row in 0..6 {
                let y = y_start + row;
                if y >= height {
                    break;
                }
                for x in 0..width {
                    let pixel = pixels[y * width + x];
                    let key = (pixel.red(), pixel.green(), pixel.blue());
                    if color_map[&key] == ci {
                        used = true;
                        break 'check;
                    }
                }
            }
            if used {
                colors_in_band.push(ci);
            }
        }

        for (ci_idx, &ci) in colors_in_band.iter().enumerate() {
            // Select color
            out.extend_from_slice(format!("#{}", ci).as_bytes());

            // Build sixel data for this color across the band
            let mut col_data = Vec::with_capacity(width);
            for x in 0..width {
                let mut mask: u8 = 0;
                for row in 0..6 {
                    let y = y_start + row;
                    if y >= height {
                        break;
                    }
                    let pixel = pixels[y * width + x];
                    let key = (pixel.red(), pixel.green(), pixel.blue());
                    if color_map[&key] == ci {
                        mask |= 1 << row;
                    }
                }
                col_data.push(0x3F + mask);
            }

            // RLE-compress the sixel data
            rle_encode(&col_data, &mut out);

            if ci_idx + 1 < colors_in_band.len() {
                // Carriage return (same band, next color)
                out.push(b'$');
            }
        }

        if band + 1 < num_bands {
            // Newline (next band)
            out.push(b'-');
        }
    }

    // ST terminator
    out.extend_from_slice(b"\x1b\\");

    out
}

/// RLE-compress sixel data: runs of identical bytes become "!count{byte}"
fn rle_encode(data: &[u8], out: &mut Vec<u8>) {
    if data.is_empty() {
        return;
    }

    let mut i = 0;
    while i < data.len() {
        let ch = data[i];
        let mut count = 1;
        while i + count < data.len() && data[i + count] == ch {
            count += 1;
        }
        if count >= 4 {
            out.extend_from_slice(format!("!{}", count).as_bytes());
            out.push(ch);
        } else {
            for _ in 0..count {
                out.push(ch);
            }
        }
        i += count;
    }
}
