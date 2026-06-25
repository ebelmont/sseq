/// Bitmap rendering using tiny-skia.
///
/// Draws cell diagrams to a Pixmap: nodes as filled circles,
/// Sq1 as straight lines, Sq2 as cubic Beziers, Sq3+ as
/// rectangular orthogonal routes.

use std::collections::HashSet;
use tiny_skia::{
    Color, FillRule, LineCap, LineJoin, Paint, PathBuilder, Pixmap, Stroke, Transform,
};

use crate::layout::LayoutResult;
use crate::module::Module;

/// Rendering configuration.
pub struct RenderConfig {
    pub node_radius: f32,
    pub stroke_width: f32,
    pub bg_color: Color,
    pub node_color: Color,
    /// Border drawn behind each node for contrast on dark backgrounds.
    pub node_border_color: Option<Color>,
    pub corner_radius: f32,
    pub bezier_bulge_base: f32,
    pub bezier_bulge_per_span: f32,
    pub rail_h0: f32,
    pub rail_hstep: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        use crate::theme::mocha;
        Self {
            node_radius: 7.0,
            stroke_width: 2.2,
            bg_color: mocha::transparent(),
            node_color: mocha::text(),
            node_border_color: Some(mocha::base()),
            corner_radius: 8.0,
            bezier_bulge_base: 28.0,
            bezier_bulge_per_span: 16.0,
            rail_h0: 72.0,
            rail_hstep: 22.0,
        }
    }
}

/// Rail depth strategy for Sq3+ rectangular routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RailStrategy {
    FixedByOrder,
    EnclosureAware,
}

/// Color palette per operation degree (Catppuccin Mocha).
///
/// Uses accent colors that avoid green/red/yellow (reserved for status indicators).
pub fn op_color(sq: u32) -> Color {
    use crate::theme::mocha;
    match sq {
        1 => mocha::blue(),      // #89b4fa
        2 => mocha::mauve(),     // #cba6f7
        4 => mocha::peach(),     // #fab387
        8 => mocha::teal(),      // #94e2d5
        16 => mocha::sky(),      // #89dceb
        _ => mocha::pink(),      // #f5c2e7
    }
}

/// Render the complete cell diagram to a Pixmap.
pub fn render_diagram(
    module: &Module,
    layout: &LayoutResult,
    sides: &[i8],
    config: &RenderConfig,
    rail_strategy: RailStrategy,
) -> Pixmap {
    // For EnclosureAware: precompute extents of each arc
    let extents = if rail_strategy == RailStrategy::EnclosureAware {
        compute_enclosure_extents(module, layout, sides, config)
    } else {
        vec![]
    };

    // Pre-scan: compute extra width needed for rails extending beyond layout bounds
    let mut extra_left = 0.0f32;
    let mut extra_right = 0.0f32;
    let margin_pad = 40.0; // extra padding for labels next to rails

    for (ei, edge) in module.edges.iter().enumerate() {
        let sq = edge.op_degree;
        let side = sides[ei];
        if sq >= 3 {
            let depth = match rail_strategy {
                RailStrategy::FixedByOrder => {
                    config.rail_h0 + (sq as f32 - 3.0) * config.rail_hstep
                }
                RailStrategy::EnclosureAware => {
                    extents.get(ei).copied().unwrap_or(config.rail_h0)
                }
            };
            let src = &layout.positions[edge.from_idx];
            let dst = &layout.positions[edge.to_idx];
            let x_rail = if side > 0 {
                src.x.max(dst.x) + depth
            } else {
                src.x.min(dst.x) - depth
            };
            let needed_right = (x_rail + margin_pad) - layout.width;
            let needed_left = -(x_rail - margin_pad);
            if needed_right > extra_right {
                extra_right = needed_right;
            }
            if needed_left > extra_left {
                extra_left = needed_left;
            }
        } else if sq == 2 {
            let src = &layout.positions[edge.from_idx];
            let dst = &layout.positions[edge.to_idx];
            let span = ((dst.y - src.y).abs() / 80.0).max(1.0);
            let bulge = (config.bezier_bulge_base + config.bezier_bulge_per_span * span) * side as f32;
            let apex_x = src.x + bulge; // approximate
            let needed_right = (apex_x + margin_pad) - layout.width;
            let needed_left = -(apex_x - margin_pad);
            if needed_right > extra_right {
                extra_right = needed_right;
            }
            if needed_left > extra_left {
                extra_left = needed_left;
            }
        }
    }

    extra_left = extra_left.max(0.0);
    extra_right = extra_right.max(0.0);

    let total_width = layout.width + extra_left + extra_right;
    let x_offset = extra_left; // shift all positions right by this amount

    let w = total_width.ceil() as u32;
    let h = layout.height.ceil() as u32;
    let mut pixmap = Pixmap::new(w.max(1), h.max(1)).unwrap();

    // Layer 0: white background
    pixmap.fill(config.bg_color);

    // Layer 1: edges (draw before nodes so nodes sit on top)
    let mut seen_ops: HashSet<u32> = HashSet::new();
    let mut labels: Vec<LabelInfo> = Vec::new();

    for (ei, edge) in module.edges.iter().enumerate() {
        let sq = edge.op_degree;
        let side = sides[ei];
        let color = op_color(sq);
        let src = &layout.positions[edge.from_idx];
        let dst = &layout.positions[edge.to_idx];
        let sx = src.x + x_offset;
        let sy = src.y;
        let dx = dst.x + x_offset;
        let dy = dst.y;

        let is_first = !seen_ops.contains(&sq);

        if sq == 1 {
            // Straight line
            draw_line(&mut pixmap, sx, sy, dx, dy, color, config.stroke_width);
        } else if sq == 2 {
            // Cubic Bezier arc
            let span = ((dy - sy).abs() / 80.0).max(1.0); // span in degrees
            let label_pos = draw_bezier_arc(
                &mut pixmap, sx, sy, dx, dy, side, span, color, config,
            );
            if is_first {
                labels.push(LabelInfo { sq, color, x: label_pos.0, y: label_pos.1 });
            }
        } else {
            // Rectangular orthogonal route (Sq3+)
            let depth = match rail_strategy {
                RailStrategy::FixedByOrder => {
                    config.rail_h0 + (sq as f32 - 3.0) * config.rail_hstep
                }
                RailStrategy::EnclosureAware => {
                    extents.get(ei).copied().unwrap_or(config.rail_h0)
                }
            };
            let label_pos = draw_rectangular_route(
                &mut pixmap, sx, sy, dx, dy, side, depth, color, config,
            );
            if is_first {
                labels.push(LabelInfo { sq, color, x: label_pos.0, y: label_pos.1 });
            }
        }

        seen_ops.insert(sq);
    }

    // Layer 2: nodes (filled circles with optional border)
    for pos in &layout.positions {
        if let Some(border) = config.node_border_color {
            draw_filled_circle(&mut pixmap, pos.x + x_offset, pos.y, config.node_radius + 1.5, border);
        }
        draw_filled_circle(&mut pixmap, pos.x + x_offset, pos.y, config.node_radius, config.node_color);
    }

    // Layer 3: labels (first instance of each operation)
    let font = load_font();
    for label in &labels {
        let text = format_sq_label(label.sq);
        draw_text(&mut pixmap, &font, &text, label.x, label.y, label.color, 14.0);
    }

    pixmap
}

struct LabelInfo {
    sq: u32,
    color: Color,
    x: f32,
    y: f32,
}

/// Draw a straight line segment.
fn draw_line(pixmap: &mut Pixmap, x1: f32, y1: f32, x2: f32, y2: f32, color: Color, width: f32) {
    let mut pb = PathBuilder::new();
    pb.move_to(x1, y1);
    pb.line_to(x2, y2);
    if let Some(path) = pb.finish() {
        let mut paint = Paint::default();
        paint.set_color(color);
        paint.anti_alias = true;
        let stroke = Stroke {
            width,
            line_cap: LineCap::Round,
            line_join: LineJoin::Round,
            ..Stroke::default()
        };
        pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
    }
}

/// Draw a cubic Bezier arc for Sq2. Returns the label position (apex).
fn draw_bezier_arc(
    pixmap: &mut Pixmap,
    x_src: f32,
    y_src: f32,
    x_dst: f32,
    y_dst: f32,
    side: i8,
    span: f32,
    color: Color,
    config: &RenderConfig,
) -> (f32, f32) {
    let bulge = (config.bezier_bulge_base + config.bezier_bulge_per_span * span) * side as f32;
    let dy = y_dst - y_src;

    let cx1 = x_src + bulge;
    let cy1 = y_src + dy * 0.15;
    let cx2 = x_dst + bulge;
    let cy2 = y_dst - dy * 0.15;

    let mut pb = PathBuilder::new();
    pb.move_to(x_src, y_src);
    pb.cubic_to(cx1, cy1, cx2, cy2, x_dst, y_dst);
    if let Some(path) = pb.finish() {
        let mut paint = Paint::default();
        paint.set_color(color);
        paint.anti_alias = true;
        let stroke = Stroke {
            width: config.stroke_width,
            line_cap: LineCap::Round,
            line_join: LineJoin::Round,
            ..Stroke::default()
        };
        pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
    }

    // Label position: apex of the arc (approximate midpoint of bezier)
    let apex_x = 0.125 * x_src + 0.375 * cx1 + 0.375 * cx2 + 0.125 * x_dst;
    let apex_y = 0.125 * y_src + 0.375 * cy1 + 0.375 * cy2 + 0.125 * y_dst;
    (apex_x + side as f32 * 4.0, apex_y - 8.0)
}

/// Draw a rectangular orthogonal route for Sq3+.
/// Returns the label position (midpoint of vertical segment).
fn draw_rectangular_route(
    pixmap: &mut Pixmap,
    x_src: f32,
    y_src: f32,
    x_dst: f32,
    y_dst: f32,
    side: i8,
    depth: f32,
    color: Color,
    config: &RenderConfig,
) -> (f32, f32) {
    // Rail x placed outside both endpoints
    let x_rail = if side > 0 {
        x_src.max(x_dst) + depth
    } else {
        x_src.min(x_dst) - depth
    };

    let r = config.corner_radius.min((y_dst - y_src).abs() / 2.0).min((x_rail - x_src).abs() / 2.0).min((x_rail - x_dst).abs() / 2.0);
    let s = side as f32;

    let mut pb = PathBuilder::new();
    pb.move_to(x_src, y_src);

    if r > 0.5 {
        // Rounded corners
        // First horizontal segment: x_src -> x_rail (with corner)
        pb.line_to(x_rail - s * r, y_src);
        // Corner 1: turn from horizontal to vertical
        let vert_dir = if y_dst > y_src { 1.0 } else { -1.0 };
        pb.quad_to(x_rail, y_src, x_rail, y_src + vert_dir * r);
        // Vertical segment
        pb.line_to(x_rail, y_dst - vert_dir * r);
        // Corner 2: turn from vertical to horizontal
        pb.quad_to(x_rail, y_dst, x_rail - s * r, y_dst);
    } else {
        pb.line_to(x_rail, y_src);
        pb.line_to(x_rail, y_dst);
    }

    pb.line_to(x_dst, y_dst);

    if let Some(path) = pb.finish() {
        let mut paint = Paint::default();
        paint.set_color(color);
        paint.anti_alias = true;
        let stroke = Stroke {
            width: config.stroke_width,
            line_cap: LineCap::Round,
            line_join: LineJoin::Round,
            ..Stroke::default()
        };
        pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
    }

    // Label position: midpoint of the vertical segment
    let label_x = x_rail + side as f32 * 6.0;
    let label_y = (y_src + y_dst) / 2.0 - 8.0;
    (label_x, label_y)
}

/// Draw a filled circle (for nodes).
fn draw_filled_circle(pixmap: &mut Pixmap, cx: f32, cy: f32, radius: f32, color: Color) {
    let mut pb = PathBuilder::new();
    // Approximate circle with 4 cubic Beziers
    let k = 0.5522847498; // magic number for circle approximation
    let r = radius;
    pb.move_to(cx + r, cy);
    pb.cubic_to(cx + r, cy + r * k, cx + r * k, cy + r, cx, cy + r);
    pb.cubic_to(cx - r * k, cy + r, cx - r, cy + r * k, cx - r, cy);
    pb.cubic_to(cx - r, cy - r * k, cx - r * k, cy - r, cx, cy - r);
    pb.cubic_to(cx + r * k, cy - r, cx + r, cy - r * k, cx + r, cy);
    pb.close();

    if let Some(path) = pb.finish() {
        let mut paint = Paint::default();
        paint.set_color(color);
        paint.anti_alias = true;
        pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
    }
}

/// Load an embedded font for text rendering.
fn load_font() -> fontdue::Font {
    // Use fontdue's built-in font metrics for basic ASCII.
    // We embed a minimal subset — for Sq labels we only need S, q, and superscript digits.
    // Using the default settings with a built-in font.
    let font_data = include_bytes!("../fonts/DejaVuSansMono.ttf");
    fontdue::Font::from_bytes(
        font_data as &[u8],
        fontdue::FontSettings::default(),
    )
    .expect("failed to load embedded font")
}

/// Format operation label with superscript digits: "Sq2" -> "Sq²"
fn format_sq_label(sq: u32) -> String {
    let superscripts = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    let deg_str: String = sq
        .to_string()
        .chars()
        .map(|c| {
            let d = c.to_digit(10).unwrap() as usize;
            superscripts[d]
        })
        .collect();
    format!("Sq{deg_str}")
}

/// Draw text onto a pixmap using fontdue rasterization.
fn draw_text(
    pixmap: &mut Pixmap,
    font: &fontdue::Font,
    text: &str,
    x: f32,
    y: f32,
    color: Color,
    size: f32,
) {
    let r = (color.red() * 255.0) as u8;
    let g = (color.green() * 255.0) as u8;
    let b = (color.blue() * 255.0) as u8;

    let pw = pixmap.width() as i32;
    let ph = pixmap.height() as i32;

    let mut cursor_x = x;
    for ch in text.chars() {
        let (metrics, bitmap) = font.rasterize(ch, size);
        let gx = cursor_x as i32 + metrics.xmin;
        let gy = y as i32 - metrics.ymin - metrics.height as i32 + metrics.height as i32;

        // Stamp the glyph bitmap onto the pixmap
        let pixels = pixmap.pixels_mut();
        for row in 0..metrics.height {
            for col in 0..metrics.width {
                let coverage = bitmap[row * metrics.width + col];
                if coverage == 0 {
                    continue;
                }
                let px = gx + col as i32;
                let py = gy + row as i32;
                if px < 0 || py < 0 || px >= pw || py >= ph {
                    continue;
                }
                let idx = (py * pw + px) as usize;
                let alpha = coverage as f32 / 255.0;
                // Alpha-blend with existing pixel
                let existing = pixels[idx];
                let er = existing.red() as f32;
                let eg = existing.green() as f32;
                let eb = existing.blue() as f32;
                let nr = r as f32 * alpha + er * (1.0 - alpha);
                let ng = g as f32 * alpha + eg * (1.0 - alpha);
                let nb = b as f32 * alpha + eb * (1.0 - alpha);
                pixels[idx] = tiny_skia::PremultipliedColorU8::from_rgba(
                    nr as u8,
                    ng as u8,
                    nb as u8,
                    255,
                )
                .unwrap();
            }
        }

        cursor_x += metrics.advance_width;
    }
}

/// Compute enclosure-aware rail depths for EnclosureAware strategy.
/// Returns a Vec indexed by edge index with the depth for each Sq3+ edge.
fn compute_enclosure_extents(
    module: &Module,
    layout: &LayoutResult,
    sides: &[i8],
    config: &RenderConfig,
) -> Vec<f32> {
    let ne = module.edges.len();
    let mut depths = vec![0.0f32; ne];

    // Collect spanning edges (sq >= 3) that use rectangular routing
    let mut rect_edges: Vec<(usize, usize, usize, i8)> = Vec::new(); // (edge_idx, rank_lo, rank_hi, side)
    for (ei, edge) in module.edges.iter().enumerate() {
        if edge.op_degree >= 3 {
            let r_src = layout.positions[edge.from_idx].rank;
            let r_dst = layout.positions[edge.to_idx].rank;
            let (lo, hi) = if r_src < r_dst {
                (r_src, r_dst)
            } else {
                (r_dst, r_src)
            };
            rect_edges.push((ei, lo, hi, sides[ei]));
        }
    }

    // Sort by span (smallest first) for inside-out processing
    rect_edges.sort_by_key(|&(_, lo, hi, _)| hi - lo);

    // Track maximum extent on each side for each spine position
    // For simplicity with ≤30 cells, we track per-edge extents
    let margin = 20.0f32;

    for &(ei, lo, hi, side) in &rect_edges {
        let edge = &module.edges[ei];
        let x_src = layout.positions[edge.from_idx].x;
        let x_dst = layout.positions[edge.to_idx].x;

        // Find maximum extent of items enclosed by this route on the same side
        let mut max_extent = 0.0f32;

        // Check Sq2 arcs enclosed by this route
        for (ej, e2) in module.edges.iter().enumerate() {
            if e2.op_degree == 2 && sides[ej] == side {
                let r_src2 = layout.positions[e2.from_idx].rank;
                let r_dst2 = layout.positions[e2.to_idx].rank;
                let (lo2, hi2) = (r_src2.min(r_dst2), r_src2.max(r_dst2));
                if lo <= lo2 && hi2 <= hi {
                    // This Sq2 arc is enclosed
                    let span = ((layout.positions[e2.to_idx].y - layout.positions[e2.from_idx].y).abs() / 80.0).max(1.0);
                    let bulge = config.bezier_bulge_base + config.bezier_bulge_per_span * span;
                    max_extent = max_extent.max(bulge);
                }
            }
        }

        // Check other rectangular routes enclosed and already computed
        for &(ej, lo2, hi2, side2) in &rect_edges {
            if ej != ei && side2 == side && lo <= lo2 && hi2 <= hi && depths[ej] > 0.0 {
                max_extent = max_extent.max(depths[ej]);
            }
        }

        // Check node x-offsets enclosed
        for rank in lo..=hi {
            if let Some(&gen_idx) = layout.spine.get(rank) {
                let node_x = layout.positions[gen_idx].x;
                let ref_x = if side > 0 {
                    x_src.max(x_dst)
                } else {
                    x_src.min(x_dst)
                };
                let offset = if side > 0 {
                    (node_x - ref_x).max(0.0)
                } else {
                    (ref_x - node_x).max(0.0)
                };
                max_extent = max_extent.max(offset);
            }
        }

        depths[ei] = max_extent + margin;
    }

    depths
}
