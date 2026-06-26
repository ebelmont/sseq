//! Generates a random finite dimensional module over the Steenrod algebra and resolves it.
//!
//! Unlike `FPModule` (finitely presented modules), where Adem relations are automatically
//! satisfied by construction, `FDModule` requires explicitly specifying all Steenrod operations
//! and verifying Adem relations via `check_validity`. This example generates random generator
//! actions, uses `extend_actions` to compute derived operations, and checks that all Adem
//! relations hold. If any relation fails, it retries with a fresh random module.
//!
//! Run with: cargo run --example random_module

use std::collections::HashMap;
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;

use algebra::{
    Algebra, AdemAlgebra, GeneratedAlgebra, SteenrodAlgebra,
    module::{FDModule, Module, SteenrodModule},
};
use bivec::BiVec;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ext::chain_complex::{AugmentedChainComplex, ChainComplex, FreeChainComplex};
use fp::prime::ValidPrime;
use rand::Rng;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
    Frame, Terminal,
};
use ratatui_image::{picker::Picker, protocol::StatefulProtocol, StatefulImage};
use sseq::coordinates::Bidegree;

/// Generate a random valid `FDModule` over the Steenrod algebra with the given graded dimensions.
///
/// Repeatedly tries random sparse generator actions until Adem relations are satisfied.
fn generate_random_fd_module(
    rng: &mut impl Rng,
    algebra: &Arc<SteenrodAlgebra>,
    graded_dim: &BiVec<usize>,
    verbose: bool,
) -> FDModule<SteenrodAlgebra> {
    let max_attempts = 1000;
    let sparsity = 0.7; // probability that each action entry is zero

    for attempt in 0..max_attempts {
        let mut module =
            FDModule::new(Arc::clone(algebra), "random_fd".to_string(), graded_dim.clone());

        let min_deg = graded_dim.min_degree();
        let max_deg = graded_dim.len(); // one past the end

        let mut valid = true;

        // Process in the same order as from_json: input_deg from HIGH to LOW,
        // output_deg from LOW to HIGH
        'outer: for input_deg in (min_deg..max_deg).rev() {
            for output_deg in (input_deg + 1)..max_deg {
                let op_deg = output_deg - input_deg;
                let output_dim = module.dimension(output_deg);
                let input_dim = module.dimension(input_deg);

                if output_dim == 0 || input_dim == 0 {
                    continue;
                }

                // Set random actions for generator operations only
                for op_idx in algebra.generators(op_deg) {
                    for input_idx in 0..input_dim {
                        let mut output = vec![0u32; output_dim];
                        for entry in output.iter_mut() {
                            // With probability (1 - sparsity), set a random nonzero value
                            if rng.random_range(0..1000) >= (sparsity * 1000.0) as u32 {
                                *entry = 1; // p=2, so only nonzero value is 1
                            }
                        }
                        module.set_action(op_deg, op_idx, input_deg, input_idx, &output);
                    }
                }

                // Compute derived (non-generator) actions from the generators we just set
                module.extend_actions(input_deg, output_deg);

                // Check Adem relations for this (input_deg, output_deg) pair
                if module.check_validity(input_deg, output_deg).is_err() {
                    valid = false;
                    break 'outer;
                }
            }
        }

        if valid {
            if verbose && attempt > 0 {
                println!("Found valid module after {attempt} retries.");
            }
            return module;
        }
    }

    // Fallback: zero-action module (always valid)
    if verbose {
        println!("Falling back to zero-action module after {max_attempts} attempts.");
    }
    FDModule::new(
        Arc::clone(algebra),
        "random_fd".to_string(),
        graded_dim.clone(),
    )
}

/// Generate a connected `FDModule` by forcing a Sq^1 chain between consecutive occupied degrees.
///
/// Gaps between occupied degrees are filled with 1 cell each so that Sq^1 can chain through.
/// At least one Sq^1 action is forced from cell 0 in each degree to a random cell in the next.
fn generate_connected_fd_module(
    rng: &mut impl Rng,
    algebra: &Arc<SteenrodAlgebra>,
    graded_dim: &BiVec<usize>,
    verbose: bool,
) -> FDModule<SteenrodAlgebra> {
    let max_attempts = 1000;
    let sparsity = 0.7;

    // Patch graded dims: fill gaps between occupied degrees with 1 cell
    let min_deg = graded_dim.min_degree();
    let max_deg = graded_dim.len(); // one past the end
    let mut patched_vec: Vec<usize> = Vec::new();
    for d in min_deg..max_deg {
        patched_vec.push(graded_dim[d]);
    }

    // Find first and last occupied degrees (relative to min_deg)
    let first_occ = patched_vec.iter().position(|&d| d > 0);
    let last_occ = patched_vec.iter().rposition(|&d| d > 0);

    if let (Some(first), Some(last)) = (first_occ, last_occ) {
        for i in first..=last {
            if patched_vec[i] == 0 {
                patched_vec[i] = 1;
            }
        }
    }

    let patched_dim = BiVec::from_vec(min_deg, patched_vec);

    for attempt in 0..max_attempts {
        let mut module = FDModule::new(
            Arc::clone(algebra),
            "connected_fd".to_string(),
            patched_dim.clone(),
        );

        let p_min = patched_dim.min_degree();
        let p_max = patched_dim.len();

        let mut valid = true;

        // Process in the same order as generate_random_fd_module
        'outer: for input_deg in (p_min..p_max).rev() {
            for output_deg in (input_deg + 1)..p_max {
                let op_deg = output_deg - input_deg;
                let output_dim = module.dimension(output_deg);
                let input_dim = module.dimension(input_deg);

                if output_dim == 0 || input_dim == 0 {
                    continue;
                }

                for op_idx in algebra.generators(op_deg) {
                    for input_idx in 0..input_dim {
                        let mut output = vec![0u32; output_dim];

                        // Force Sq^1 chain: for op_deg == 1, input_idx == 0,
                        // ensure at least one nonzero output
                        if op_deg == 1 && input_idx == 0 {
                            let target = rng.random_range(0..output_dim);
                            output[target] = 1;
                        }

                        // Fill remaining entries randomly
                        for entry in output.iter_mut() {
                            if *entry == 0
                                && rng.random_range(0..1000) >= (sparsity * 1000.0) as u32
                            {
                                *entry = 1;
                            }
                        }
                        module.set_action(op_deg, op_idx, input_deg, input_idx, &output);
                    }
                }

                module.extend_actions(input_deg, output_deg);

                if module.check_validity(input_deg, output_deg).is_err() {
                    valid = false;
                    break 'outer;
                }
            }
        }

        if valid {
            if verbose && attempt > 0 {
                println!("Found valid connected module after {attempt} retries.");
            }
            return module;
        }
    }

    // Fallback: zero-action module
    if verbose {
        println!("Falling back to zero-action module after {max_attempts} attempts.");
    }
    FDModule::new(
        Arc::clone(algebra),
        "connected_fd".to_string(),
        patched_dim,
    )
}

/// Generate a cyclic quotient module starting from 1 cell in degree 0.
///
/// Graded dimensions are chosen randomly: always 1 cell in degree 0, then 0-2 cells per
/// subsequent degree until `target_cells` is reached. The resulting module is filtered
/// for connectivity.
fn generate_cyclic_quotient_module(
    rng: &mut impl Rng,
    algebra: &Arc<SteenrodAlgebra>,
    max_degree: i32,
    target_cells: usize,
    verbose: bool,
) -> FDModule<SteenrodAlgebra> {
    let max_outer_attempts = 500;
    let sparsity = 0.7;

    for outer in 0..max_outer_attempts {
        // Choose graded dimensions
        let mut dim_vec = vec![0usize; (max_degree + 1) as usize];
        dim_vec[0] = 1;
        let mut remaining = target_cells.saturating_sub(1);

        for d in 1..=(max_degree as usize) {
            if remaining == 0 {
                break;
            }
            let add = rng.random_range(0..=std::cmp::min(2, remaining));
            dim_vec[d] = add;
            remaining -= add;
        }

        // If we still have remaining cells, distribute them
        while remaining > 0 {
            let d = rng.random_range(1..=(max_degree as usize));
            dim_vec[d] += 1;
            remaining -= 1;
        }

        // Ensure at least 2 cells total
        let total: usize = dim_vec.iter().sum();
        if total < 2 {
            dim_vec[std::cmp::min(1, max_degree as usize)] += 1;
        }

        let graded_dim = BiVec::from_vec(0, dim_vec);

        // Try to generate a valid module with these dims
        let max_inner = 100;
        for _inner in 0..max_inner {
            let mut module = FDModule::new(
                Arc::clone(algebra),
                "cyclic_quotient".to_string(),
                graded_dim.clone(),
            );

            let min_deg = graded_dim.min_degree();
            let max_deg_end = graded_dim.len();

            let mut valid = true;

            'outer_loop: for input_deg in (min_deg..max_deg_end).rev() {
                for output_deg in (input_deg + 1)..max_deg_end {
                    let op_deg = output_deg - input_deg;
                    let output_dim = module.dimension(output_deg);
                    let input_dim = module.dimension(input_deg);

                    if output_dim == 0 || input_dim == 0 {
                        continue;
                    }

                    for op_idx in algebra.generators(op_deg) {
                        for input_idx in 0..input_dim {
                            let mut output = vec![0u32; output_dim];
                            for entry in output.iter_mut() {
                                if rng.random_range(0..1000) >= (sparsity * 1000.0) as u32 {
                                    *entry = 1;
                                }
                            }
                            module.set_action(op_deg, op_idx, input_deg, input_idx, &output);
                        }
                    }

                    module.extend_actions(input_deg, output_deg);

                    if module.check_validity(input_deg, output_deg).is_err() {
                        valid = false;
                        break 'outer_loop;
                    }
                }
            }

            if valid && module.is_connected() {
                if verbose && outer > 0 {
                    println!("Found valid cyclic quotient module after {outer} outer retries.");
                }
                return module;
            }
        }
    }

    // Fallback: single-cell module
    if verbose {
        println!(
            "Falling back to single-cell module after {max_outer_attempts} attempts."
        );
    }
    let dim_vec = vec![1usize];
    let graded_dim = BiVec::from_vec(0, dim_vec);
    FDModule::new(
        Arc::clone(algebra),
        "cyclic_quotient".to_string(),
        graded_dim,
    )
}

/// Return the ANSI color escape code for a given operation degree.
fn op_color(op_deg: u8) -> &'static str {
    match op_deg {
        0 => "\x1b[1m",   // bold (cells)
        1 => "\x1b[36m",  // cyan (Sq1)
        2 => "\x1b[33m",  // yellow (Sq2)
        4 => "\x1b[35m",  // magenta (Sq4)
        _ => "\x1b[32m",  // green (higher)
    }
}

/// Return the ANSI color for degree labels.
fn label_color() -> &'static str {
    "\x1b[2m" // dim
}

const RESET: &str = "\x1b[0m";

/// Braille-based canvas for high-resolution terminal graphics.
///
/// Each terminal character cell maps to a 2-wide x 4-tall grid of braille dots,
/// giving 8x the resolution of ASCII art. Dots are approximately square visually
/// because terminal characters are roughly 1:2 (width:height).
struct BrailleCanvas {
    char_width: usize,
    char_height: usize,
    dots: Vec<Vec<u8>>,   // [char_row][char_col] = braille dot bitmask
    colors: Vec<Vec<u8>>, // [char_row][char_col] = color category
}

impl BrailleCanvas {
    fn new(char_width: usize, char_height: usize) -> Self {
        Self {
            char_width,
            char_height,
            dots: vec![vec![0u8; char_width]; char_height],
            colors: vec![vec![255u8; char_width]; char_height],
        }
    }

    /// Set a single subpixel. Color uses last-writer-wins (draw order determines priority).
    fn set_pixel(&mut self, px: i32, py: i32, color: u8) {
        if px < 0 || py < 0 {
            return;
        }
        let (px, py) = (px as usize, py as usize);
        let (cx, cy) = (px / 2, py / 4);
        if cx >= self.char_width || cy >= self.char_height {
            return;
        }
        // Braille dot layout within a character cell:
        //   bit0(0,0)  bit3(1,0)
        //   bit1(0,1)  bit4(1,1)
        //   bit2(0,2)  bit5(1,2)
        //   bit6(0,3)  bit7(1,3)
        let bit: u8 = match (px % 2, py % 4) {
            (0, 0) => 0x01,
            (0, 1) => 0x02,
            (0, 2) => 0x04,
            (1, 0) => 0x08,
            (1, 1) => 0x10,
            (1, 2) => 0x20,
            (0, 3) => 0x40,
            (1, 3) => 0x80,
            _ => unreachable!(),
        };
        self.dots[cy][cx] |= bit;
        self.colors[cy][cx] = color;
    }

    /// Draw a quadratic Bezier curve: B(t) = (1-t)^2 * p0 + 2(1-t)t * ctrl + t^2 * p1.
    ///
    /// Thickness is achieved by drawing parallel curves offset perpendicular to the tangent:
    /// - 1 = single pixel line (Sq^1)
    /// - 2 = double line (Sq^2)
    /// - 3 = triple line (Sq^4+)
    fn draw_bezier(
        &mut self,
        p0: (f64, f64),
        ctrl: (f64, f64),
        p1: (f64, f64),
        color: u8,
        thickness: u8,
    ) {
        let dist = ((p1.0 - p0.0).powi(2) + (p1.1 - p0.1).powi(2)).sqrt();
        let steps = (dist * 2.5).max(60.0) as usize;

        let offsets: &[f64] = match thickness {
            1 => &[0.0],
            2 => &[-0.5, 0.5],
            _ => &[-1.0, 0.0, 1.0],
        };

        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let u = 1.0 - t;
            let px = u * u * p0.0 + 2.0 * u * t * ctrl.0 + t * t * p1.0;
            let py = u * u * p0.1 + 2.0 * u * t * ctrl.1 + t * t * p1.1;

            // Tangent vector for perpendicular offset direction
            let tx = 2.0 * (u * (ctrl.0 - p0.0) + t * (p1.0 - ctrl.0));
            let ty = 2.0 * (u * (ctrl.1 - p0.1) + t * (p1.1 - ctrl.1));
            let tlen = (tx * tx + ty * ty).sqrt().max(0.001);
            let perp = (-ty / tlen, tx / tlen);

            for &off in offsets {
                self.set_pixel(
                    (px + off * perp.0).round() as i32,
                    (py + off * perp.1).round() as i32,
                    color,
                );
            }
        }
    }

    /// Draw a filled circle at subpixel coordinates.
    fn draw_filled_circle(&mut self, cx: f64, cy: f64, r: f64, color: u8) {
        let ri = r.ceil() as i32 + 1;
        let r2 = r * r;
        for dy in -ri..=ri {
            for dx in -ri..=ri {
                if (dx * dx + dy * dy) as f64 <= r2 {
                    self.set_pixel(cx.round() as i32 + dx, cy.round() as i32 + dy, color);
                }
            }
        }
    }

    /// Render one character row as a string, optionally with ANSI colors.
    fn render_line(&self, row: usize, use_color: bool) -> String {
        let mut s = String::new();
        if use_color {
            let mut cur: Option<u8> = None;
            for col in 0..self.char_width {
                let bits = self.dots[row][col];
                if bits == 0 {
                    if cur.is_some() {
                        s.push_str(RESET);
                        cur = None;
                    }
                    s.push(' ');
                } else {
                    let c = self.colors[row][col];
                    if cur != Some(c) && c != 255 {
                        s.push_str(op_color(c));
                        cur = Some(c);
                    }
                    s.push(char::from_u32(0x2800 + bits as u32).unwrap());
                }
            }
            if cur.is_some() {
                s.push_str(RESET);
            }
        } else {
            for col in 0..self.char_width {
                let bits = self.dots[row][col];
                if bits == 0 {
                    s.push(' ');
                } else {
                    s.push(char::from_u32(0x2800 + bits as u32).unwrap());
                }
            }
        }
        s
    }
}

/// Print a cell diagram using braille graphics for smooth, non-overlapping curves.
///
/// Each Sq^n operation is drawn as a quadratic Bezier curve that bows in an alternating
/// direction (Sq^1 left, Sq^2 right, Sq^4 left, ...) so connections never overlap.
/// Thickness distinguishes operations: Sq^1 thin, Sq^2 medium, Sq^4+ thick.
/// When stdout is a terminal, uses ANSI colors for additional distinction.
fn print_cell_diagram(
    module: &FDModule<SteenrodAlgebra>,
    algebra: &Arc<SteenrodAlgebra>,
) {
    let min_deg = module.min_degree();
    let max_deg = match module.max_degree() {
        Some(d) => d,
        None => return,
    };

    let max_dim = (min_deg..=max_deg)
        .map(|d| module.dimension(d))
        .max()
        .unwrap_or(0);

    if max_dim == 0 {
        return;
    }

    let use_color = std::io::stdout().is_terminal();

    // Layout parameters in subpixels (braille dots).
    // Each character cell = 2 dots wide x 4 dots tall.
    let col_sp: usize = 14; // subpixels between cell centers horizontally (7 chars)
    let row_sp: usize = 20; // subpixels between degree rows vertically (5 chars)
    let pad: usize = 4; // padding around edges
    let cell_r: f64 = 2.0; // cell dot radius in subpixels

    // Canvas dimensions in subpixels
    let px_w = pad + max_dim.saturating_sub(1) * col_sp + pad;
    let px_h = pad + (max_deg - min_deg) as usize * row_sp + pad;

    // Convert to character dimensions (round up)
    let char_w = (px_w + 1) / 2 + 1;
    let char_h = (px_h + 3) / 4 + 1;

    let mut canvas = BrailleCanvas::new(char_w, char_h);

    // Cell positions in subpixels (higher degree = smaller y = top of canvas)
    let cell_x = |i: usize| -> f64 { (pad + i * col_sp) as f64 };
    let cell_y = |d: i32| -> f64 { (pad + (max_deg - d) as usize * row_sp) as f64 };

    // Collect connections
    struct Conn {
        src_deg: i32,
        src_idx: usize,
        tgt_deg: i32,
        tgt_idx: usize,
        op_deg: i32,
    }

    let mut connections: Vec<Conn> = Vec::new();
    for input_deg in min_deg..=max_deg {
        for output_deg in (input_deg + 1)..=max_deg {
            let op_deg = output_deg - input_deg;
            if module.dimension(output_deg) == 0 || module.dimension(input_deg) == 0 {
                continue;
            }
            for op_idx in algebra.generators(op_deg) {
                for input_idx in 0..module.dimension(input_deg) {
                    let action = module.action(op_deg, op_idx, input_deg, input_idx);
                    for output_idx in 0..module.dimension(output_deg) {
                        if action.entry(output_idx) != 0 {
                            connections.push(Conn {
                                src_deg: input_deg,
                                src_idx: input_idx,
                                tgt_deg: output_deg,
                                tgt_idx: output_idx,
                                op_deg,
                            });
                        }
                    }
                }
            }
        }
    }

    // Sort: longer ops first so shorter ones draw on top (last-writer-wins for color)
    connections.sort_by(|a, b| b.op_deg.cmp(&a.op_deg));

    // Group by (src, tgt) cell pair to spread overlapping connections apart
    let mut groups: HashMap<(i32, usize, i32, usize), Vec<usize>> = HashMap::new();
    for (i, c) in connections.iter().enumerate() {
        groups
            .entry((c.src_deg, c.src_idx, c.tgt_deg, c.tgt_idx))
            .or_default()
            .push(i);
    }

    let mut group_offsets = vec![0.0f64; connections.len()];
    for indices in groups.values() {
        let n = indices.len();
        if n > 1 {
            for (j, &idx) in indices.iter().enumerate() {
                group_offsets[idx] = (j as f64 - (n - 1) as f64 / 2.0) * 4.0;
            }
        }
    }

    // Draw connections as Bezier curves
    for (i, c) in connections.iter().enumerate() {
        let sx = cell_x(c.src_idx);
        let sy = cell_y(c.src_deg);
        let tx = cell_x(c.tgt_idx);
        let ty = cell_y(c.tgt_deg);

        // Direction and perpendicular vectors
        let dx = tx - sx;
        let dy = ty - sy;
        let len = (dx * dx + dy * dy).sqrt().max(1.0);
        let perp = (-dy / len, dx / len);

        // Base bow: alternates direction by operation degree to separate Sq^1, Sq^2, Sq^4
        let base_bow = match c.op_deg {
            1 => -4.0,
            2 => 5.0,
            4 => -6.0,
            8 => 7.0,
            16 => -8.0,
            _ => {
                if c.op_deg % 2 == 0 {
                    6.0
                } else {
                    -6.0
                }
            }
        };

        let total_offset = base_bow + group_offsets[i];

        // Control point at midpoint, offset perpendicular to the straight line
        let mid_x = (sx + tx) / 2.0;
        let mid_y = (sy + ty) / 2.0;
        let ctrl = (
            mid_x + total_offset * perp.0,
            mid_y + total_offset * perp.1,
        );

        // Thickness: Sq^1 thin, Sq^2 medium, Sq^4+ thick
        let thickness: u8 = match c.op_deg {
            1 => 1,
            2 => 2,
            _ => 3,
        };

        canvas.draw_bezier((sx, sy), ctrl, (tx, ty), c.op_deg as u8, thickness);
    }

    // Draw cells on top of curves
    for d in min_deg..=max_deg {
        for i in 0..module.dimension(d) {
            canvas.draw_filled_circle(cell_x(i), cell_y(d), cell_r, 0);
        }
    }

    // Print diagram with degree labels
    println!("\n  Cell Diagram");
    for row in 0..char_h {
        // Build degree label prefix
        let mut prefix = String::from("    \u{2502} ");
        for d in min_deg..=max_deg {
            let deg_char_row = cell_y(d) as usize / 4;
            if deg_char_row == row {
                if use_color {
                    prefix = format!("{}{d:>3}{} \u{2502} ", label_color(), RESET);
                } else {
                    prefix = format!("{d:>3} \u{2502} ");
                }
                break;
            }
        }

        let content = canvas.render_line(row, use_color);
        println!("{}{}", prefix, content.trim_end());
    }

    // Legend
    let mut ops: Vec<i32> = connections.iter().map(|c| c.op_deg).collect();
    ops.sort();
    ops.dedup();
    if !ops.is_empty() {
        let thickness_label = |d: i32| -> &'static str {
            match d {
                1 => "thin",
                2 => "medium",
                _ => "thick",
            }
        };
        if use_color {
            let legend: Vec<String> = ops
                .iter()
                .map(|d| {
                    format!(
                        "{}Sq{} ({}){}",
                        op_color(*d as u8),
                        d,
                        thickness_label(*d),
                        RESET,
                    )
                })
                .collect();
            println!("        ({})", legend.join(", "));
        } else {
            let legend: Vec<String> = ops
                .iter()
                .map(|d| format!("Sq{} ({})", d, thickness_label(*d)))
                .collect();
            println!("        ({})", legend.join(", "));
        }
    }
}

/// Convert an FDModule to sseq-format JSON for use with steenrod-art.
///
/// Uses flat indexing (x0, x1, ...) and records only generator actions.
/// Multi-target actions (Sq^k maps one basis element to a sum) produce one
/// action entry per nonzero coefficient, giving one edge per target in the diagram.
fn fdmodule_to_sseq_json(
    module: &FDModule<SteenrodAlgebra>,
    algebra: &Arc<SteenrodAlgebra>,
    name: &str,
) -> String {
    let min_deg = module.min_degree();
    let max_deg = module.max_degree().unwrap_or(min_deg);

    // Build flat index mapping: degree -> starting flat index
    let mut flat_idx = 0usize;
    let mut offsets: HashMap<i32, usize> = HashMap::new();
    let mut gen_entries: Vec<(String, i32)> = Vec::new();

    for d in min_deg..=max_deg {
        let dim = module.dimension(d);
        offsets.insert(d, flat_idx);
        for _ in 0..dim {
            gen_entries.push((format!("x{flat_idx}"), d));
            flat_idx += 1;
        }
    }

    // Build gens JSON
    let gens_json: String = gen_entries
        .iter()
        .map(|(n, d)| format!("\"{n}\": {d}"))
        .collect::<Vec<_>>()
        .join(", ");

    // Build actions: one entry per nonzero coefficient in each generator action
    let mut actions: Vec<String> = Vec::new();
    for input_deg in min_deg..=max_deg {
        for output_deg in (input_deg + 1)..=max_deg {
            let op_deg = output_deg - input_deg;
            let output_dim = module.dimension(output_deg);
            let input_dim = module.dimension(input_deg);
            if output_dim == 0 || input_dim == 0 {
                continue;
            }

            let input_off = offsets[&input_deg];
            let output_off = offsets[&output_deg];

            for op_idx in algebra.generators(op_deg) {
                for input_idx in 0..input_dim {
                    let action = module.action(op_deg, op_idx, input_deg, input_idx);
                    for output_idx in 0..output_dim {
                        if action.entry(output_idx) != 0 {
                            actions.push(format!(
                                "Sq{} x{} = x{}",
                                op_deg,
                                input_off + input_idx,
                                output_off + output_idx,
                            ));
                        }
                    }
                }
            }
        }
    }

    let actions_json: String = actions
        .iter()
        .map(|a| format!("        \"{a}\""))
        .collect::<Vec<_>>()
        .join(",\n");

    format!(
        "{{\n    \"p\": 2,\n    \"type\": \"finite dimensional module\",\n    \"name\": \"{name}\",\n    \"gens\": {{ {gens_json} }},\n    \"actions\": [\n{actions_json}\n    ]\n}}"
    )
}

/// Render a module's cell diagram inline as sixel graphics.
///
/// Returns `true` if the sixel was successfully written to stdout, `false` on error.
#[allow(dead_code)]
fn render_inline_sixel(
    module: &FDModule<SteenrodAlgebra>,
    algebra: &Arc<SteenrodAlgebra>,
    name: &str,
) -> bool {
    let json = fdmodule_to_sseq_json(module, algebra, name);
    let art_module = match steenrod_art::module::load_module_from_str(&json, name) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("steenrod-art: failed to load module: {e}");
            return false;
        }
    };

    let layout_config = steenrod_art::layout::LayoutConfig::default();
    let render_config = steenrod_art::render::RenderConfig::default();

    let layout_result = steenrod_art::layout::compute_layout(&art_module, &layout_config);
    let assignment = steenrod_art::crossing::assign_sides(&art_module, &layout_result);
    let pixmap = steenrod_art::render::render_diagram(
        &art_module,
        &layout_result,
        &assignment.sides,
        &render_config,
        steenrod_art::render::RailStrategy::FixedByOrder,
    );
    let sixel_data = steenrod_art::sixel::pixmap_to_sixel(&pixmap);

    let mut stdout = std::io::stdout();
    if stdout.write_all(&sixel_data).is_err() {
        return false;
    }
    let _ = stdout.flush();
    true
}

/// Write realizable modules to JSON files and optionally launch steenrod-art TUI.
#[allow(dead_code)]
fn write_and_view_modules(
    modules: &[(usize, FDModule<SteenrodAlgebra>)],
    algebra: &Arc<SteenrodAlgebra>,
    label: &str,
) -> anyhow::Result<Vec<PathBuf>> {
    let output_dir = std::env::temp_dir().join("steenrod-art-enumerate");
    std::fs::create_dir_all(&output_dir)?;

    let mut paths = Vec::new();
    for (num, (_orig_idx, m)) in modules.iter().enumerate() {
        let name = format!("{label}_{}", num + 1);
        let json = fdmodule_to_sseq_json(m, algebra, &name);
        let path = output_dir.join(format!("{name}.json"));
        std::fs::write(&path, &json)?;
        paths.push(path);
    }

    println!(
        "\nModule JSON files written to: {}",
        output_dir.display()
    );

    paths.iter().enumerate().for_each(|(i, p)| {
        println!("  {}: {}", i + 1, p.display());
    });

    Ok(paths)
}

/// Launch steenrod-art TUI with the given module JSON files.
#[allow(dead_code)]
fn launch_steenrod_art(paths: &[PathBuf]) -> anyhow::Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let steenrod_art_manifest = format!("{manifest_dir}/steenrod-art/Cargo.toml");

    if !std::path::Path::new(&steenrod_art_manifest).exists() {
        println!(
            "steenrod-art not found at {steenrod_art_manifest}. \
             Build it first: cd {manifest_dir}/steenrod-art && cargo build"
        );
        return Ok(());
    }

    println!("\nLaunching steenrod-art TUI...");

    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("run")
        .arg("--manifest-path")
        .arg(&steenrod_art_manifest)
        .arg("--");

    for p in paths {
        cmd.arg(p);
    }

    let status = cmd.status()?;
    if !status.success() {
        println!("steenrod-art exited with status: {status}");
    }

    Ok(())
}

/// Print nonzero generator actions for a module.
fn print_actions(module: &FDModule<SteenrodAlgebra>, algebra: &Arc<SteenrodAlgebra>) {
    let min_deg = module.min_degree();
    let max_deg = module.max_degree().unwrap();

    let mut any_nonzero = false;
    for input_deg in min_deg..=max_deg {
        for output_deg in (input_deg + 1)..=max_deg {
            let op_deg = output_deg - input_deg;
            let output_dim = module.dimension(output_deg);
            if output_dim == 0 {
                continue;
            }
            for op_idx in algebra.generators(op_deg) {
                for input_idx in 0..module.dimension(input_deg) {
                    let action = module.action(op_deg, op_idx, input_deg, input_idx);
                    if !action.is_zero() {
                        any_nonzero = true;
                        println!(
                            "  {} {} = {}",
                            algebra.generator_to_string(op_deg, op_idx),
                            module.basis_element_to_string(input_deg, input_idx),
                            module.element_to_string(output_deg, action.as_slice()),
                        );
                    }
                }
            }
        }
    }
    if !any_nonzero {
        println!("  (all zero)");
    }
}

// === Gallery TUI ===

/// Catppuccin Mocha palette for ratatui widgets.
mod mocha_tui {
    use ratatui::style::Color;
    pub const SURFACE0: Color = Color::Rgb(0x31, 0x32, 0x44);
    pub const SURFACE1: Color = Color::Rgb(0x45, 0x47, 0x5a);
    pub const OVERLAY0: Color = Color::Rgb(0x6c, 0x70, 0x86);
    pub const OVERLAY1: Color = Color::Rgb(0x7f, 0x84, 0x9c);
    pub const SUBTEXT1: Color = Color::Rgb(0xba, 0xc2, 0xde);
    pub const TEXT: Color = Color::Rgb(0xcd, 0xd6, 0xf4);
    pub const LAVENDER: Color = Color::Rgb(0xb4, 0xbe, 0xfe);
    pub const MAUVE: Color = Color::Rgb(0xcb, 0xa6, 0xf7);
    pub const GREEN: Color = Color::Rgb(0xa6, 0xe3, 0xa1);
    pub const RED: Color = Color::Rgb(0xf3, 0x8b, 0xa8);
    pub const YELLOW: Color = Color::Rgb(0xf9, 0xe2, 0xaf);
}

#[derive(Clone, Copy, PartialEq)]
enum ModuleStatus {
    Realizable,
    Obstructed,
    Unknown,
}

struct GalleryEntry {
    module: FDModule<SteenrodAlgebra>,
    status: ModuleStatus,
    actions_text: String,
}

#[derive(Clone, Copy, PartialEq)]
enum StatusFilter {
    All,
    Realizable,
    Obstructed,
    Unknown,
}

struct HomResult {
    hom_dim: usize,
    stem_minus_1: Vec<(i32, usize)>,
    all_lift: bool,
}

struct GalleryApp {
    entries: Vec<GalleryEntry>,
    selected: usize,
    list_state: ListState,
    filter: StatusFilter,
    filtered_indices: Vec<usize>,
    dims_label: String,
    picker: Picker,
    image_cache: HashMap<usize, StatefulProtocol>,
    algebra: Arc<SteenrodAlgebra>,
    resolved_cache: HashMap<usize, String>,
    mark_source: Option<usize>,
    hom_cache: HashMap<(usize, usize), HomResult>,
}

impl GalleryApp {
    fn new(
        entries: Vec<GalleryEntry>,
        dims_label: String,
        picker: Picker,
        algebra: Arc<SteenrodAlgebra>,
        hom_cache: HashMap<(usize, usize), HomResult>,
    ) -> Self {
        let filtered_indices: Vec<usize> = (0..entries.len()).collect();
        let mut list_state = ListState::default();
        if !filtered_indices.is_empty() {
            list_state.select(Some(0));
        }
        Self {
            entries,
            selected: 0,
            list_state,
            filter: StatusFilter::All,
            filtered_indices,
            dims_label,
            picker,
            image_cache: HashMap::new(),
            algebra,
            resolved_cache: HashMap::new(),
            mark_source: None,
            hom_cache,
        }
    }

    fn refilter(&mut self) {
        self.filtered_indices = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| match self.filter {
                StatusFilter::All => true,
                StatusFilter::Realizable => e.status == ModuleStatus::Realizable,
                StatusFilter::Obstructed => e.status == ModuleStatus::Obstructed,
                StatusFilter::Unknown => e.status == ModuleStatus::Unknown,
            })
            .map(|(i, _)| i)
            .collect();
        self.selected = 0;
        self.list_state.select(if self.filtered_indices.is_empty() {
            None
        } else {
            Some(0)
        });
    }

    fn selected_entry_index(&self) -> Option<usize> {
        self.filtered_indices.get(self.selected).copied()
    }

    fn next(&mut self) {
        if !self.filtered_indices.is_empty() {
            self.selected = (self.selected + 1) % self.filtered_indices.len();
            self.list_state.select(Some(self.selected));
        }
    }

    fn prev(&mut self) {
        if !self.filtered_indices.is_empty() {
            self.selected = if self.selected == 0 {
                self.filtered_indices.len() - 1
            } else {
                self.selected - 1
            };
            self.list_state.select(Some(self.selected));
        }
    }

    fn cycle_filter(&mut self) {
        self.filter = match self.filter {
            StatusFilter::All => StatusFilter::Realizable,
            StatusFilter::Realizable => StatusFilter::Obstructed,
            StatusFilter::Obstructed => StatusFilter::Unknown,
            StatusFilter::Unknown => StatusFilter::All,
        };
        self.refilter();
    }

    fn ensure_image(&mut self, idx: usize) {
        if self.image_cache.contains_key(&idx) {
            return;
        }
        let entry = &self.entries[idx];
        let name = format!("module_{}", idx + 1);
        if let Some(pixmap) = render_module_pixmap(&entry.module, &self.algebra, &name) {
            let dyn_img = pixmap_to_dynamic_image(&pixmap);
            let state = self.picker.new_resize_protocol(dyn_img);
            self.image_cache.insert(idx, state);
        }
    }
}

/// Format actions text for display (like print_actions but returns a String).
fn format_actions_text(
    module: &FDModule<SteenrodAlgebra>,
    algebra: &Arc<SteenrodAlgebra>,
) -> String {
    let min_deg = module.min_degree();
    let max_deg = match module.max_degree() {
        Some(d) => d,
        None => return "(empty module)".to_string(),
    };
    let mut lines = Vec::new();
    for input_deg in min_deg..=max_deg {
        for output_deg in (input_deg + 1)..=max_deg {
            let op_deg = output_deg - input_deg;
            let output_dim = module.dimension(output_deg);
            if output_dim == 0 {
                continue;
            }
            for op_idx in algebra.generators(op_deg) {
                for input_idx in 0..module.dimension(input_deg) {
                    let action = module.action(op_deg, op_idx, input_deg, input_idx);
                    if !action.is_zero() {
                        lines.push(format!(
                            "{} {} = {}",
                            algebra.generator_to_string(op_deg, op_idx),
                            module.basis_element_to_string(input_deg, input_idx),
                            module.element_to_string(output_deg, action.as_slice()),
                        ));
                    }
                }
            }
        }
    }
    if lines.is_empty() {
        "(all zero)".to_string()
    } else {
        lines.join("\n")
    }
}

/// Render an FDModule to a tiny-skia Pixmap using the steenrod-art pipeline.
fn render_module_pixmap(
    module: &FDModule<SteenrodAlgebra>,
    algebra: &Arc<SteenrodAlgebra>,
    name: &str,
) -> Option<tiny_skia::Pixmap> {
    let json = fdmodule_to_sseq_json(module, algebra, name);
    let art_module = steenrod_art::module::load_module_from_str(&json, name).ok()?;
    let layout_config = steenrod_art::layout::LayoutConfig::default();
    let render_config = steenrod_art::render::RenderConfig::default();
    let layout_result = steenrod_art::layout::compute_layout(&art_module, &layout_config);
    let assignment = steenrod_art::crossing::assign_sides(&art_module, &layout_result);
    Some(steenrod_art::render::render_diagram(
        &art_module,
        &layout_result,
        &assignment.sides,
        &render_config,
        steenrod_art::render::RailStrategy::FixedByOrder,
    ))
}

/// Convert a tiny-skia Pixmap (premultiplied alpha) to an image::DynamicImage
/// with straight (un-premultiplied) alpha for correct compositing.
fn pixmap_to_dynamic_image(pixmap: &tiny_skia::Pixmap) -> image::DynamicImage {
    let w = pixmap.width();
    let h = pixmap.height();
    let mut img = image::RgbaImage::new(w, h);
    for (i, px) in pixmap.pixels().iter().enumerate() {
        let a = px.alpha();
        let (r, g, b) = if a == 0 {
            (0, 0, 0)
        } else {
            let a16 = a as u16;
            (
                ((px.red() as u16 * 255 + a16 / 2) / a16) as u8,
                ((px.green() as u16 * 255 + a16 / 2) / a16) as u8,
                ((px.blue() as u16 * 255 + a16 / 2) / a16) as u8,
            )
        };
        let x = (i as u32) % w;
        let y = (i as u32) / w;
        img.put_pixel(x, y, image::Rgba([r, g, b, a]));
    }
    image::DynamicImage::ImageRgba8(img)
}

/// Render the gallery TUI.
fn gallery_ui(frame: &mut Frame, app: &mut GalleryApp) {
    use mocha_tui::*;

    let outer = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(5),
        Constraint::Length(1),
    ])
    .split(frame.area());

    // --- Header ---
    let (n_real, n_obst, n_unk) = app.entries.iter().fold((0, 0, 0), |(r, o, u), e| match e.status {
        ModuleStatus::Realizable => (r + 1, o, u),
        ModuleStatus::Obstructed => (r, o + 1, u),
        ModuleStatus::Unknown => (r, o, u + 1),
    });
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            format!(" enumerate dims [{}]", app.dims_label),
            Style::default().fg(MAUVE).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" · {} modules · ", app.entries.len()),
            Style::default().fg(OVERLAY1),
        ),
        Span::styled(format!("{n_real}"), Style::default().fg(GREEN)),
        Span::styled("✓ ", Style::default().fg(GREEN)),
        Span::styled(format!("{n_obst}"), Style::default().fg(RED)),
        Span::styled("✗ ", Style::default().fg(RED)),
        Span::styled(format!("{n_unk}"), Style::default().fg(YELLOW)),
        Span::styled("?", Style::default().fg(YELLOW)),
    ]))
    .block(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(SURFACE1)),
    );
    frame.render_widget(header, outer[0]);

    // --- Main area ---
    let main = Layout::horizontal([Constraint::Percentage(25), Constraint::Percentage(75)])
        .split(outer[1]);

    // --- List pane ---
    let items: Vec<ListItem> = app
        .filtered_indices
        .iter()
        .map(|&idx| {
            let entry = &app.entries[idx];
            let (sym, sc) = match entry.status {
                ModuleStatus::Realizable => ("✓", GREEN),
                ModuleStatus::Obstructed => ("✗", RED),
                ModuleStatus::Unknown => ("?", YELLOW),
            };
            let label = match entry.status {
                ModuleStatus::Realizable => "realizable",
                ModuleStatus::Obstructed => "obstructed",
                ModuleStatus::Unknown => "unknown",
            };
            ListItem::new(Line::from(vec![
                Span::styled(format!("{:02}  ", idx + 1), Style::default().fg(TEXT)),
                Span::styled(format!("{sym} "), Style::default().fg(sc)),
                Span::styled(label, Style::default().fg(sc)),
            ]))
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::RIGHT)
                .border_style(Style::default().fg(SURFACE1)),
        )
        .highlight_style(Style::default().bg(SURFACE0).fg(TEXT))
        .highlight_symbol("▸ ");
    frame.render_stateful_widget(list, main[0], &mut app.list_state);

    // --- Right panel (diagram + actions) ---
    let right = Layout::vertical([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(main[1]);

    // Diagram pane
    if let Some(idx) = app.selected_entry_index() {
        app.ensure_image(idx);
        if let Some(state) = app.image_cache.get_mut(&idx) {
            let widget = StatefulImage::new();
            frame.render_stateful_widget(widget, right[0], state);
        } else {
            let placeholder = Paragraph::new("rendering…")
                .style(Style::default().fg(OVERLAY0));
            frame.render_widget(placeholder, right[0]);
        }
    }

    // Actions pane
    if let Some(idx) = app.selected_entry_index() {
        let mut text = if let Some(chart) = app.resolved_cache.get(&idx) {
            format!("Ext chart:\n{chart}\n\nActions:\n{}", app.entries[idx].actions_text)
        } else {
            app.entries[idx].actions_text.clone()
        };

        // Show cached hom results involving this module
        for (&(src, tgt), result) in &app.hom_cache {
            if src == idx {
                text.push_str(&format!(
                    "\n\nMaps to #{}: Hom = F2^{}",
                    tgt + 1,
                    result.hom_dim,
                ));
                if result.all_lift {
                    text.push_str(&format!(
                        "\n  All {} maps lift to spectrum maps",
                        result.hom_dim,
                    ));
                } else {
                    let ss: Vec<String> = result.stem_minus_1.iter()
                        .map(|(s, d)| format!("s={s} (dim {d})"))
                        .collect();
                    text.push_str(&format!(
                        "\n  Lifting obstructions at {}; d_2 needed",
                        ss.join(", "),
                    ));
                }
            } else if tgt == idx {
                text.push_str(&format!(
                    "\n\nMaps from #{}: Hom = F2^{}",
                    src + 1,
                    result.hom_dim,
                ));
                if result.all_lift {
                    text.push_str(&format!(
                        "\n  All {} maps lift to spectrum maps",
                        result.hom_dim,
                    ));
                } else {
                    let ss: Vec<String> = result.stem_minus_1.iter()
                        .map(|(s, d)| format!("s={s} (dim {d})"))
                        .collect();
                    text.push_str(&format!(
                        "\n  Lifting obstructions at {}; d_2 needed",
                        ss.join(", "),
                    ));
                }
            }
        }

        let actions = Paragraph::new(text)
            .style(Style::default().fg(SUBTEXT1))
            .block(
                Block::default()
                    .borders(Borders::TOP)
                    .border_style(Style::default().fg(SURFACE1)),
            )
            .wrap(Wrap { trim: false });
        frame.render_widget(actions, right[1]);
    }

    // --- Footer ---
    let footer = if let Some(src_idx) = app.mark_source {
        Paragraph::new(Line::from(vec![
            Span::styled(
                format!(" source = #{} selected", src_idx + 1),
                Style::default().fg(MAUVE).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" — press ", Style::default().fg(OVERLAY1)),
            Span::styled("m", Style::default().fg(LAVENDER)),
            Span::styled(" on target, ", Style::default().fg(OVERLAY1)),
            Span::styled("Esc", Style::default().fg(LAVENDER)),
            Span::styled(" to cancel", Style::default().fg(OVERLAY1)),
        ]))
    } else {
        let filter_label = match app.filter {
            StatusFilter::All => "all",
            StatusFilter::Realizable => "✓",
            StatusFilter::Obstructed => "✗",
            StatusFilter::Unknown => "?",
        };
        Paragraph::new(Line::from(vec![
            Span::styled(" j", Style::default().fg(LAVENDER)),
            Span::styled("/", Style::default().fg(OVERLAY1)),
            Span::styled("k", Style::default().fg(LAVENDER)),
            Span::styled(" move  ", Style::default().fg(OVERLAY1)),
            Span::styled("f", Style::default().fg(LAVENDER)),
            Span::styled(format!(" filter:{filter_label}  "), Style::default().fg(OVERLAY1)),
            Span::styled("enter", Style::default().fg(LAVENDER)),
            Span::styled(" resolve  ", Style::default().fg(OVERLAY1)),
            Span::styled("m", Style::default().fg(LAVENDER)),
            Span::styled(" map  ", Style::default().fg(OVERLAY1)),
            Span::styled("o", Style::default().fg(LAVENDER)),
            Span::styled(" open  ", Style::default().fg(OVERLAY1)),
            Span::styled("y", Style::default().fg(LAVENDER)),
            Span::styled(" yank  ", Style::default().fg(OVERLAY1)),
            Span::styled("q", Style::default().fg(LAVENDER)),
            Span::styled(" quit", Style::default().fg(OVERLAY1)),
        ]))
    };
    frame.render_widget(footer, outer[2]);
}

/// Launch the interactive gallery TUI for browsing enumerated modules.
fn run_gallery(
    entries: Vec<GalleryEntry>,
    dims_label: String,
    algebra: Arc<SteenrodAlgebra>,
    hom_cache: HashMap<(usize, usize), HomResult>,
) -> anyhow::Result<()> {
    // Probe terminal graphics protocol before entering raw mode
    let picker = match Picker::from_query_stdio() {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "Could not detect graphics protocol ({e}). Falling back to text output."
            );
            for (num, entry) in entries.iter().enumerate() {
                let label = match entry.status {
                    ModuleStatus::Realizable => "realizable",
                    ModuleStatus::Obstructed => "obstructed",
                    ModuleStatus::Unknown => "unknown",
                };
                println!("\n=== Indecomposable #{} ({}) ===", num + 1, label);
                println!("{}", entry.actions_text);
            }
            return Ok(());
        }
    };

    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = GalleryApp::new(entries, dims_label, picker, algebra, hom_cache);

    let result = (|| -> anyhow::Result<()> {
        loop {
            terminal.draw(|f| gallery_ui(f, &mut app))?;

            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Esc => {
                        if app.mark_source.is_some() {
                            app.mark_source = None;
                        } else {
                            break;
                        }
                    }
                    KeyCode::Char('j') | KeyCode::Down => app.next(),
                    KeyCode::Char('k') | KeyCode::Up => app.prev(),
                    KeyCode::Char('g') => {
                        app.selected = 0;
                        app.list_state.select(Some(0));
                    }
                    KeyCode::Char('G') => {
                        if !app.filtered_indices.is_empty() {
                            app.selected = app.filtered_indices.len() - 1;
                            app.list_state.select(Some(app.selected));
                        }
                    }
                    KeyCode::Char('f') => app.cycle_filter(),
                    KeyCode::Char('o') => {
                        if let Some(idx) = app.selected_entry_index() {
                            let name = format!("module_{}", idx + 1);
                            if let Some(pixmap) =
                                render_module_pixmap(&app.entries[idx].module, &app.algebra, &name)
                            {
                                let path = std::env::temp_dir().join(format!("{name}.png"));
                                if pixmap.save_png(&path).is_ok() {
                                    let _ =
                                        std::process::Command::new("open").arg(&path).spawn();
                                }
                            }
                        }
                    }
                    KeyCode::Char('y') => {
                        if let Some(idx) = app.selected_entry_index() {
                            let name = format!("module_{}", idx + 1);
                            let json = fdmodule_to_sseq_json(
                                &app.entries[idx].module,
                                &app.algebra,
                                &name,
                            );
                            if let Ok(mut child) = std::process::Command::new("pbcopy")
                                .stdin(std::process::Stdio::piped())
                                .spawn()
                            {
                                if let Some(stdin) = child.stdin.as_mut() {
                                    let _ = stdin.write_all(json.as_bytes());
                                }
                                let _ = child.wait();
                            }
                        }
                    }
                    KeyCode::Enter | KeyCode::Char('r') => {
                        if let Some(idx) = app.selected_entry_index() {
                            if !app.resolved_cache.contains_key(&idx) {
                                let entry = &app.entries[idx];
                                let sm: SteenrodModule = Arc::new(entry.module.clone());
                                let sm = Arc::new(sm);
                                let cc: Arc<ext::CCC> =
                                    Arc::new(ext::chain_complex::FiniteChainComplex::ccdz(sm));
                                let res = ext::resolution::Resolution::new(Arc::clone(&cc));
                                let max = Bidegree::s_t(10, 30);
                                res.compute_through_bidegree(max);
                                app.resolved_cache
                                    .insert(idx, res.graded_dimension_string());
                            }
                        }
                    }
                    KeyCode::Char('m') => {
                        if let Some(target_idx) = app.selected_entry_index() {
                            if let Some(source_idx) = app.mark_source.take() {
                                // Second press: compute hom
                                if !app.hom_cache.contains_key(&(source_idx, target_idx)) {
                                    if let Some(result) = compute_hom(
                                        &app.entries[source_idx].module,
                                        &app.entries[target_idx].module,
                                        &app.algebra,
                                    ) {
                                        app.hom_cache.insert((source_idx, target_idx), result);
                                    }
                                }
                            } else {
                                // First press: mark source
                                app.mark_source = Some(target_idx);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    })();

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    result
}

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let p = ValidPrime::new(2);
    let algebra = Arc::new(SteenrodAlgebra::AdemAlgebra(AdemAlgebra::new(p, false)));

    let mode: String = query::with_default(
        "Generation mode (random/connected/cyclic/enumerate)",
        "random",
        |response: &str| {
            let r = response.trim().to_lowercase();
            if r == "random" || r == "connected" || r == "cyclic" || r == "enumerate" {
                Ok(r)
            } else {
                Err(format!(
                    "unrecognized mode '{response}'. Should be 'random', 'connected', 'cyclic', or 'enumerate'"
                ))
            }
        },
    );

    if mode == "enumerate" {
        return run_enumerate_mode(&algebra);
    }

    let mut rng = rand::rng();

    let num_cells: usize = query::with_default("Number of cells (total dimension)", "6", str::parse);
    let max_degree: i32 = query::with_default("Max degree", "4", str::parse);
    let require_indecomposable: bool = query::with_default(
        "Require indecomposable? (y/n)",
        "n",
        |response: &str| {
            if response.starts_with('y') || response.starts_with('n') {
                Ok(response.starts_with('y'))
            } else {
                Err(format!(
                    "unrecognized response '{response}'. Should be '(y)es' or '(n)o'"
                ))
            }
        },
    );

    println!("\n=== Random Finite Dimensional Steenrod Module (p = 2) ===\n");

    // For random and connected modes, distribute cells across degrees up front
    let graded_dim = if mode != "cyclic" {
        let num_degrees = (max_degree + 1) as usize;
        let mut dim_vec = vec![0usize; num_degrees];
        for _ in 0..num_cells {
            let deg = rng.random_range(0..num_degrees);
            dim_vec[deg] += 1;
        }
        let gd = BiVec::from_vec(0, dim_vec);

        println!("Graded dimensions:");
        for (deg, &dim) in gd.iter_enum() {
            if dim > 0 {
                println!("  degree {deg}: {dim}");
            }
        }
        Some(gd)
    } else {
        None
    };

    // Generate the initial module using the chosen mode
    let generate_module =
        |rng: &mut rand::rngs::ThreadRng, verbose: bool| -> FDModule<SteenrodAlgebra> {
            match mode.as_str() {
                "connected" => generate_connected_fd_module(
                    rng,
                    &algebra,
                    graded_dim.as_ref().unwrap(),
                    verbose,
                ),
                "cyclic" => {
                    generate_cyclic_quotient_module(rng, &algebra, max_degree, num_cells, verbose)
                }
                _ => generate_random_fd_module(
                    rng,
                    &algebra,
                    graded_dim.as_ref().unwrap(),
                    verbose,
                ),
            }
        };

    let max_retry = 200;
    let mut module = generate_module(&mut rng, true);

    // For cyclic mode, print graded dims after generation (since they're chosen internally)
    if mode == "cyclic" {
        println!("Graded dimensions:");
        let min_d = module.min_degree();
        let max_d = module.max_degree().unwrap_or(min_d);
        for d in min_d..=max_d {
            let dim = module.dimension(d);
            if dim > 0 {
                println!("  degree {d}: {dim}");
            }
        }
    }

    let (mut indecomp_result, mut dim_end) = module.check_indecomposable();

    if require_indecomposable && indecomp_result != Some(true) {
        let mut attempt = 1;
        while attempt < max_retry {
            module = generate_module(&mut rng, false);

            // Fast pre-check: skip disconnected modules
            if !module.is_connected() {
                attempt += 1;
                continue;
            }

            let result = module.check_indecomposable();
            indecomp_result = result.0;
            dim_end = result.1;
            if indecomp_result == Some(true) {
                println!("Found indecomposable module after {attempt} retries.");
                // Print dims for cyclic mode on successful retry
                if mode == "cyclic" {
                    println!("Graded dimensions:");
                    let min_d = module.min_degree();
                    let max_d = module.max_degree().unwrap_or(min_d);
                    for d in min_d..=max_d {
                        let dim = module.dimension(d);
                        if dim > 0 {
                            println!("  degree {d}: {dim}");
                        }
                    }
                }
                break;
            }
            attempt += 1;
        }
        if indecomp_result != Some(true) {
            println!("Could not find indecomposable module after {max_retry} attempts, proceeding with last attempt.");
        }
    }

    println!("\nNonzero generator actions:");
    print_actions(&module, &algebra);

    println!("\nAdem relations verified.");

    // Print indecomposability result
    match indecomp_result {
        Some(true) => println!("Module is indecomposable (dim End_A(M) = {dim_end})"),
        Some(false) => println!("Module is decomposable (dim End_A(M) = {dim_end})"),
        None => println!("dim End_A(M) = {dim_end} (too large for idempotent search)"),
    }

    print_cell_diagram(&module, &algebra);

    // Resolve and display Ext chart
    let t_max: i32 = query::with_default("Max t", "30", str::parse);
    let s_max: i32 = query::with_default("Max s", "10", str::parse);

    let module: SteenrodModule = Arc::new(module);
    let module = Arc::new(module);
    let cc: Arc<ext::CCC> = Arc::new(ext::chain_complex::FiniteChainComplex::ccdz(module));
    let res = ext::resolution::Resolution::new(Arc::clone(&cc));

    let max = Bidegree::s_t(s_max, t_max);
    res.compute_through_bidegree(max);

    println!("\nExt chart:");
    println!("{}", res.graded_dimension_string());

    Ok(())
}

/// Run the enumerate mode: systematically enumerate all valid FDModules with given graded dims.
fn run_enumerate_mode(algebra: &Arc<SteenrodAlgebra>) -> anyhow::Result<()> {
    let dims_str: String = query::raw(
        "Graded dimensions (comma-separated, degree 0 first)",
        |s: &str| Ok::<String, String>(s.to_string()),
    );
    let dims: Vec<usize> = dims_str
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .expect("Invalid graded dimensions: expected comma-separated non-negative integers");

    let graded_dim = BiVec::from_vec(0, dims.clone());

    println!("\n=== Systematic Enumeration of FDModules (p = 2) ===\n");
    println!("Graded dimensions: {:?}", dims);
    let total_cells: usize = dims.iter().sum();
    println!("Total cells: {}", total_cells);

    let modules = FDModule::enumerate(algebra, &graded_dim);

    // Filter to connected indecomposables
    let mut connected_count = 0usize;
    let mut indecomposables: Vec<FDModule<SteenrodAlgebra>> = Vec::new();

    for m in &modules {
        if m.is_connected() {
            connected_count += 1;
            let (result, _) = m.check_indecomposable();
            if result == Some(true) {
                indecomposables.push(m.clone());
            }
        }
    }

    println!("Total valid: {}, Connected: {}, Indecomposable: {}",
             modules.len(), connected_count, indecomposables.len());

    // Deduplicate isomorphism classes
    eprint!("Deduplicating isomorphism classes...\r");
    let rep_indices = FDModule::dedup_isomorphism_classes(&indecomposables);
    let before = indecomposables.len();
    let indecomposables: Vec<FDModule<SteenrodAlgebra>> =
        rep_indices.into_iter().map(|i| indecomposables[i].clone()).collect();
    eprintln!("Isomorphism classes: {} (from {} indecomposables)     ", indecomposables.len(), before);

    // Check realization and build gallery entries
    let mut gallery_entries: Vec<GalleryEntry> = Vec::new();
    for (num, m) in indecomposables.iter().enumerate() {
        eprint!("Checking realization: {}/{}...\r", num + 1, indecomposables.len());
        let status = match check_realizable(m, algebra) {
            Some(true) => ModuleStatus::Realizable,
            Some(false) => ModuleStatus::Obstructed,
            None => ModuleStatus::Unknown,
        };
        gallery_entries.push(GalleryEntry {
            module: m.clone(),
            status,
            actions_text: format_actions_text(m, algebra),
        });
    }

    let n_real = gallery_entries.iter().filter(|e| e.status == ModuleStatus::Realizable).count();
    let n_obst = gallery_entries.iter().filter(|e| e.status == ModuleStatus::Obstructed).count();
    let n_unk = gallery_entries.iter().filter(|e| e.status == ModuleStatus::Unknown).count();
    eprintln!("\rRealizable: {n_real}, Obstructed: {n_obst}, Unknown: {n_unk}     ");

    if gallery_entries.is_empty() {
        println!("No indecomposable modules found.");
        return Ok(());
    }

    // TODO: scan_all_pairs is available but disabled pending review of
    // map-lifting obstruction semantics (stem -1 results are misleading
    // when the source or target module itself fails realization).
    let hom_cache = HashMap::new();

    let dims_label = dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
    run_gallery(gallery_entries, dims_label, Arc::clone(algebra), hom_cache)
}

/// Check whether a module passes the realization obstruction test.
///
/// Returns `Some(true)` if all stem -2 groups vanish (realizable),
/// `Some(false)` if any are nonzero (obstructed/inconclusive),
/// `None` if the module is empty.
fn check_realizable(
    module: &FDModule<SteenrodAlgebra>,
    algebra: &Arc<SteenrodAlgebra>,
) -> Option<bool> {
    use ext::chain_complex::HomCochainComplex;

    let max_deg = module.max_degree()?;
    let min_deg = module.min_degree();
    let diam = max_deg - min_deg;

    let s_max = diam + 4;

    // Build the resolution
    let sm: SteenrodModule = Arc::new(module.clone());
    let sm = Arc::new(sm);
    let cc: Arc<ext::CCC> = Arc::new(ext::chain_complex::FiniteChainComplex::ccdz(sm));
    let resolution = ext::resolution::Resolution::new(Arc::clone(&cc));

    let hom_max = Bidegree::n_s(0, s_max);
    let res_max = hom_max + Bidegree::n_s(max_deg, 1);
    resolution.compute_through_stem(res_max);
    algebra.compute_basis(hom_max.t() + max_deg + 2);

    // Get the module back from the chain complex for HomCochainComplex
    let target_cc = resolution.target();
    let module_arc = target_cc.module(0);

    let hom_cc = HomCochainComplex::new(Arc::new(resolution), Arc::clone(&module_arc));
    hom_cc.compute_through_stem(hom_max);

    // Scan stem -2: Ext^{s, s-2}(M, M) for s = 3..=diam+2
    for s in 3..=diam + 2 {
        let b = Bidegree::n_s(-2, s);
        let dim = hom_cc.homology_dimension(b);
        if dim > 0 {
            return Some(false);
        }
    }

    Some(true)
}

/// Compute the space of A-module maps Hom_A(source, target) = Ext^{0,0}(N, M)
/// and check whether map-lifting obstructions exist on stem -1 of Ext(N, M).
fn compute_hom(
    source: &FDModule<SteenrodAlgebra>,
    target: &FDModule<SteenrodAlgebra>,
    algebra: &Arc<SteenrodAlgebra>,
) -> Option<HomResult> {
    use ext::chain_complex::HomCochainComplex;

    let src_max = source.max_degree()?;
    let src_min = source.min_degree();
    let tgt_max = target.max_degree()?;
    let tgt_min = target.min_degree();

    let diam = std::cmp::max(src_max - src_min, tgt_max - tgt_min);
    let s_max = diam + 4;

    // Build resolution of source
    let sm: SteenrodModule = Arc::new(source.clone());
    let sm = Arc::new(sm);
    let cc: Arc<ext::CCC> = Arc::new(ext::chain_complex::FiniteChainComplex::ccdz(sm));
    let resolution = ext::resolution::Resolution::new(Arc::clone(&cc));

    let hom_max = Bidegree::n_s(0, s_max);
    let res_max = hom_max + Bidegree::n_s(tgt_max, 1);
    resolution.compute_through_stem(res_max);
    algebra.compute_basis(hom_max.t() + tgt_max + 2);

    // Build HomCochainComplex with target module
    let target_sm: SteenrodModule = Arc::new(target.clone());
    let target_arc: Arc<SteenrodModule> = Arc::new(target_sm);
    let hom_cc = HomCochainComplex::new(Arc::new(resolution), target_arc);
    hom_cc.compute_through_stem(hom_max);

    // Ext^{0,0}(N, M) = Hom_A(N, M)
    let hom_dim = hom_cc.homology_dimension(Bidegree::n_s(0, 0));

    // Scan stem -1: Ext^{s, s-1}(N, M) for s = 2..=diam+2
    let mut stem_minus_1 = Vec::new();
    for s in 2..=diam + 2 {
        let b = Bidegree::n_s(-1, s);
        let dim = hom_cc.homology_dimension(b);
        if dim > 0 {
            stem_minus_1.push((s, dim));
        }
    }

    let all_lift = stem_minus_1.is_empty();

    Some(HomResult {
        hom_dim,
        stem_minus_1,
        all_lift,
    })
}

/// Scan all ordered pairs (i, j) of modules for Hom spaces and map-lifting obstructions.
///
/// Optimisation: all modules share the same graded dimensions, so for a fixed source
/// the resolution is built once and reused across all targets.
fn scan_all_pairs(
    modules: &[FDModule<SteenrodAlgebra>],
    algebra: &Arc<SteenrodAlgebra>,
) -> HashMap<(usize, usize), HomResult> {
    use ext::chain_complex::HomCochainComplex;

    let mut results = HashMap::new();
    if modules.is_empty() {
        return results;
    }

    // All modules share the same graded dimensions, so compute parameters once.
    let max_deg = match modules[0].max_degree() {
        Some(d) => d,
        None => return results,
    };
    let min_deg = modules[0].min_degree();
    let diam = max_deg - min_deg;
    let s_max = diam + 4;
    let hom_max = Bidegree::n_s(0, s_max);
    let res_max = hom_max + Bidegree::n_s(max_deg, 1);

    let n = modules.len();
    for i in 0..n {
        eprint!("Scanning maps from module {}/{}...\r", i + 1, n);

        // Build resolution of source module once, reuse for all targets.
        let sm: SteenrodModule = Arc::new(modules[i].clone());
        let sm = Arc::new(sm);
        let cc: Arc<ext::CCC> = Arc::new(ext::chain_complex::FiniteChainComplex::ccdz(sm));
        let resolution = ext::resolution::Resolution::new(Arc::clone(&cc));
        resolution.compute_through_stem(res_max);
        algebra.compute_basis(hom_max.t() + max_deg + 2);
        let resolution = Arc::new(resolution);

        for j in 0..n {
            let target_sm: SteenrodModule = Arc::new(modules[j].clone());
            let target_arc: Arc<SteenrodModule> = Arc::new(target_sm);
            let hom_cc = HomCochainComplex::new(Arc::clone(&resolution), target_arc);
            hom_cc.compute_through_stem(hom_max);

            let hom_dim = hom_cc.homology_dimension(Bidegree::n_s(0, 0));

            let mut stem_minus_1 = Vec::new();
            for s in 2..=diam + 2 {
                let b = Bidegree::n_s(-1, s);
                let dim = hom_cc.homology_dimension(b);
                if dim > 0 {
                    stem_minus_1.push((s, dim));
                }
            }

            let all_lift = stem_minus_1.is_empty();

            results.insert((i, j), HomResult {
                hom_dim,
                stem_minus_1,
                all_lift,
            });
        }
    }
    eprintln!("Map scan complete: {} pairs checked.              ", n * n);

    results
}
