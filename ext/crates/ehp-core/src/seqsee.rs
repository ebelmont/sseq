//! Convert SATPage + SATResult into SeqSee-compatible formats.
//!
//! Provides two output paths:
//! - **JSON**: `sat_page_to_seqsee_json()` for the interactive WebSocket viewer
//! - **CSV**: `write_ehp_csv()` for the SeqSee pipeline (jsonmaker.py → main.py → HTML)

use std::io::{self, Write as _};

use hashbrown::HashMap;
use serde::Serialize;
use serde_json::{json, Map, Value};

use crate::constraints::DiffVar;
use crate::gf2::{mat_get, vec_get};
use crate::map::MapKind;
use crate::page::SATPage;
use crate::result::SATResult;
use crate::tridegree::Tridegree;

// =============================================================================
// Differential overlay — sent via WebSocket for live updates
// =============================================================================

/// Overlay data for differentials, sent to the client via WebSocket.
/// The base chart (nodes) is rendered by SeqSee; this overlay adds
/// differential edges and marks uncertain nodes.
#[derive(Clone, Debug, Serialize)]
pub struct DiffOverlay {
    pub edges: Vec<OverlayEdge>,
    pub uncertain: Vec<String>,
}

/// A single differential edge in the overlay.
#[derive(Clone, Debug, Serialize)]
pub struct OverlayEdge {
    pub source: String,
    pub target: String,
    /// "determined" or "unknown"
    pub style: String,
    pub source_tri: [i32; 3],
    pub source_idx: usize,
    pub target_tri: [i32; 3],
    pub target_idx: usize,
}

/// Build a differential overlay from a solve result.
pub fn build_diff_overlay(page: &SATPage, result: &SATResult) -> DiffOverlay {
    let mut edges = Vec::new();
    let mut uncertain = Vec::new();

    // Collect uncertain node IDs.
    let mut tridegrees: Vec<Tridegree> = page
        .dimension
        .iter()
        .filter(|(_, &d)| d > 0)
        .map(|(&t, _)| t)
        .collect();
    tridegrees.sort();

    for &t in &tridegrees {
        if result.is_tridegree_uncertain(t) {
            let dim = page.dim_at(t);
            for idx in 0..dim {
                uncertain.push(node_id(t, idx));
            }
        }
    }

    // Build differential edges.
    let mut by_tri: HashMap<Tridegree, Vec<&DiffVar>> = HashMap::new();
    for v in &result.vars {
        by_tri.entry(v.tridegree()).or_default().push(v);
    }

    for (&tri, _) in &by_tri {
        let src_dim = page.dim_at(tri);
        let tgt_tri = tri.diff_target(page.r);
        let tgt_dim = page.dim_at(tgt_tri);

        if src_dim == 0 || tgt_dim == 0 {
            continue;
        }

        for col in 0..src_dim {
            for row in 0..tgt_dim {
                let dv = DiffVar::new(
                    tri.n.min(tri.s + 2),
                    tri.s,
                    tri.f,
                    row as u16,
                    col as u16,
                );
                let Some(&idx) = result.var_index.get(&dv) else {
                    continue;
                };

                let is_unknown = result.unknown.contains(&idx);
                let is_nonzero = !is_unknown && vec_get(&result.offset, idx);

                if !is_nonzero && !is_unknown {
                    continue;
                }

                let style = if is_unknown { "unknown" } else { "determined" }.to_string();

                edges.push(OverlayEdge {
                    source: node_id(tri, col),
                    target: node_id(tgt_tri, row),
                    style,
                    source_tri: [tri.n, tri.s, tri.f],
                    source_idx: col,
                    target_tri: [tgt_tri.n, tgt_tri.s, tgt_tri.f],
                    target_idx: row,
                });
            }
        }
    }

    DiffOverlay { edges, uncertain }
}

/// Compute differential edges for a single sphere, using gen_name-style node IDs
/// (matching the SVG node IDs produced by the SeqSee pipeline).
///
/// Returns `(source_id, target_id, is_determined)` triples.
pub fn compute_sphere_diff_edges(
    page: &SATPage,
    result: &SATResult,
    sphere_n: i32,
) -> Vec<(String, String, bool)> {
    let mut edges = Vec::new();

    for (&tri, &dim) in &page.dimension {
        if tri.n != sphere_n || dim == 0 {
            continue;
        }

        let tgt_tri = tri.diff_target(page.r);
        let tgt_dim = page.dim_at(tgt_tri);
        if tgt_dim == 0 {
            continue;
        }

        let n_var = tri.n.min(tri.s + 2);

        // Uncertainty-excluded degrees have no differential variables at all —
        // the algorithm has no opinion there. Show every potential target as
        // an unknown (dashed) differential, as the original's write_spheres
        // did, so "excluded" is never mistaken for "determined zero".
        let first_var = DiffVar::new(n_var, tri.s, tri.f, 0, 0);
        if !result.var_index.contains_key(&first_var)
            && (page.is_excluded(tri) || page.is_excluded(tgt_tri))
        {
            for col in 0..dim {
                for row in 0..tgt_dim {
                    edges.push((
                        gen_name(tri.n, tri.s, tri.f, col, dim),
                        gen_name(tgt_tri.n, tgt_tri.s, tgt_tri.f, row, tgt_dim),
                        false,
                    ));
                }
            }
            continue;
        }

        for col in 0..dim {
            for row in 0..tgt_dim {
                let dv = DiffVar::new(n_var, tri.s, tri.f, row as u16, col as u16);
                let Some(&idx) = result.var_index.get(&dv) else {
                    continue;
                };

                let is_unknown = result.unknown.contains(&idx);
                let is_nonzero = !is_unknown && vec_get(&result.offset, idx);

                if !is_nonzero && !is_unknown {
                    continue;
                }

                let src_name = gen_name(tri.n, tri.s, tri.f, col, dim);
                let tgt_name = gen_name(tgt_tri.n, tgt_tri.s, tgt_tri.f, row, tgt_dim);
                edges.push((src_name, tgt_name, !is_unknown));
            }
        }
    }
    edges
}

/// Generate a SeqSee node ID from tridegree and basis index.
pub fn node_id(t: Tridegree, idx: usize) -> String {
    format!("n{}_s{}_f{}_i{}", t.n, t.s, t.f, idx)
}

/// Convert a SATPage (with optional solve result) into SeqSee JSON.
///
/// The chart uses Adams grading: x = s + f (total degree), y = s (stem).
/// Multiple generators at the same bidegree get sequential `position` values
/// so SeqSee stacks them automatically.
pub fn sat_page_to_seqsee_json(
    page: &SATPage,
    result: Option<&SATResult>,
    title: &str,
) -> Value {
    let mut nodes = Map::new();
    let mut edges: Vec<Value> = Vec::new();

    // Collect tridegrees with nonzero dimension, sorted.
    let mut tridegrees: Vec<Tridegree> = page
        .dimension
        .iter()
        .filter(|(_, &d)| d > 0)
        .map(|(&t, _)| t)
        .collect();
    tridegrees.sort();

    // Build nodes.
    for &t in &tridegrees {
        let dim = page.dim_at(t);
        if dim == 0 {
            continue;
        }

        // Adams chart: x = s + f, y = s.
        let chart_x = t.s + t.f;
        let chart_y = t.s;

        for idx in 0..dim {
            let id = node_id(t, idx);
            let label = make_label(page, t, idx);

            let mut node = json!({
                "x": chart_x,
                "y": chart_y,
                "position": idx as i64,
                "label": label,
            });

            // Color uncertain nodes gray.
            if let Some(res) = result {
                if res.is_tridegree_uncertain(t) {
                    node.as_object_mut().unwrap().insert(
                        "attributes".into(),
                        json!([{"color": "#999999"}]),
                    );
                }
            }

            nodes.insert(id, node);
        }
    }

    // Build differential edges from solve result.
    if let Some(res) = result {
        // Group vars by source tridegree.
        let mut by_tri: HashMap<Tridegree, Vec<&DiffVar>> = HashMap::new();
        for v in &res.vars {
            by_tri.entry(v.tridegree()).or_default().push(v);
        }

        for (&tri, _vars) in &by_tri {
            let src_dim = page.dim_at(tri);
            let tgt_tri = tri.diff_target(page.r);
            let tgt_dim = page.dim_at(tgt_tri);

            if src_dim == 0 || tgt_dim == 0 {
                continue;
            }

            for col in 0..src_dim {
                for row in 0..tgt_dim {
                    let dv = DiffVar::new(
                        tri.n.min(tri.s + 2),
                        tri.s,
                        tri.f,
                        row as u16,
                        col as u16,
                    );
                    let Some(&idx) = res.var_index.get(&dv) else {
                        continue;
                    };

                    let is_unknown = res.unknown.contains(&idx);
                    let is_nonzero = !is_unknown && vec_get(&res.offset, idx);

                    if !is_nonzero && !is_unknown {
                        continue;
                    }

                    let mut attrs: Vec<Value> = Vec::new();
                    if is_unknown {
                        attrs.push(json!({"color": "#cc8800"}));
                        attrs.push(json!({"pattern": "dashed"}));
                    } else {
                        attrs.push(json!({"color": "#d00000"}));
                    }

                    edges.push(json!({
                        "source": node_id(tri, col),
                        "target": node_id(tgt_tri, row),
                        "attributes": attrs,
                    }));
                }
            }
        }
    }

    json!({
        "header": {
            "metadata": {
                "htmltitle": title,
            },
            "chart": {
                "scale": 60,
                "nodeSize": 0.04,
                "nodeSpacing": 0.02,
                "nodeSlope": null,
            },
        },
        "nodes": nodes,
        "edges": edges,
    })
}

/// Generate a label for a basis element.
fn make_label(page: &SATPage, t: Tridegree, idx: usize) -> String {
    let key = format!("{},{},{},{}", t.n, t.s, t.f, idx);
    if let Some(name) = page.names.get(&key) {
        return name.clone();
    }
    if page.dim_at(t) == 1 {
        format!("$({},{},{})$", t.n, t.s, t.f)
    } else {
        format!("$({},{},{})[{}]$", t.n, t.s, t.f, idx)
    }
}

// =============================================================================
// CSV export — SeqSee E2-format compatible
// =============================================================================

/// Format a generator name following the SeqSee `n_stem_filtration[_idx]` convention.
pub fn gen_name(n: i32, s: i32, f: i32, idx: usize, dim: usize) -> String {
    if dim == 1 {
        format!("S{}_{}_{}",  n, s, f)
    } else {
        format!("S{}_{}_{}_{}",  n, s, f, idx)
    }
}

/// Collect target generator names for h_i multiplication on a generator.
///
/// h_i lives at tridegree `(n + s, hi_stem, 1)` and the product lands at
/// `(n, s + hi_stem, f + 1)`.  The hi_stem values are 0, 1, 3, 7 for
/// h_0, h_1, h_2, h_3 respectively.
fn hi_target_names(
    page: &SATPage,
    src_t: Tridegree,
    col: usize,
    hi_stem: i32,
) -> Vec<String> {
    use crate::products::ProductKey;

    let hi_t = Tridegree::new(src_t.n + src_t.s, hi_stem, 1);
    let hi_dim = page.dim_at(hi_t);
    if hi_dim == 0 {
        return Vec::new();
    }
    let tgt_t = Tridegree::new(src_t.n, src_t.s + hi_stem, src_t.f + 1);
    let tgt_dim = page.dim_at(tgt_t);
    if tgt_dim == 0 {
        return Vec::new();
    }
    // h_i is typically the 0-th basis element at its tridegree
    let key = ProductKey::new(src_t, col as u16, hi_t, 0);
    let result = match page.products.get(&key) {
        Some(v) => v,
        None => return Vec::new(),
    };
    let mut names = Vec::new();
    for j in 0..tgt_dim {
        if j < result.len() && vec_get(&result, j) {
            names.push(gen_name(tgt_t.n, tgt_t.s, tgt_t.f, j, tgt_dim));
        }
    }
    names
}

/// Collect target generator names for a map (E, H, or P) applied to generator
/// `(src_t, col)`. Reads the map matrix row for `col` and returns names of
/// nonzero target entries.
pub fn map_target_names(page: &SATPage, kind: MapKind, src_t: Tridegree, col: usize) -> Vec<String> {
    let map_table = match page.maps.get(&kind) {
        Some(mt) => mt,
        None => return Vec::new(),
    };
    let mat = match map_table.matrix_at(src_t) {
        Some(m) => m,
        None => return Vec::new(),
    };
    let tgt_t = kind.target_degree(src_t);
    let tgt_dim = page.dim_at(tgt_t);
    if tgt_dim == 0 {
        return Vec::new();
    }
    // Convention: rows = source basis, columns = target coordinates.
    // mat_vec_mul uses v^T * M, but for individual basis vector we just read
    // row `col` directly.
    let mut names = Vec::new();
    for j in 0..tgt_dim {
        if col < mat.rows() && j < mat.columns() && mat_get(mat, col, j) {
            names.push(gen_name(tgt_t.n, tgt_t.s, tgt_t.f, j, tgt_dim));
        }
    }
    names
}

/// Collect both determined-nonzero and unknown differential target names for
/// a generator in a single pass over the target basis (fuses what used to be
/// three separate scans: `diff_target_names`, `null_diff_target_names`, and
/// a redundant `has_nonzero_diff` — the first return value's emptiness is
/// exactly what `has_nonzero_diff` was recomputing).
fn diff_and_null_target_names(
    page: &SATPage,
    result: &SATResult,
    t: Tridegree,
    col: usize,
) -> (Vec<String>, Vec<String>) {
    let tgt_t = t.diff_target(page.r);
    let tgt_dim = page.dim_at(tgt_t);
    if tgt_dim == 0 {
        return (Vec::new(), Vec::new());
    }
    let n_var = t.n.min(t.s + 2);
    let mut targets = Vec::new();
    let mut nulls = Vec::new();
    for row in 0..tgt_dim {
        let dv = DiffVar::new(n_var, t.s, t.f, row as u16, col as u16);
        if let Some(&idx) = result.var_index.get(&dv) {
            if result.unknown.contains(&idx) {
                nulls.push(gen_name(tgt_t.n, tgt_t.s, tgt_t.f, row, tgt_dim));
            } else if vec_get(&result.offset, idx) {
                targets.push(gen_name(tgt_t.n, tgt_t.s, tgt_t.f, row, tgt_dim));
            }
        }
    }
    (targets, nulls)
}

/// Write EHP spectral sequence data as SeqSee E2-format CSV.
///
/// Columns match `SeqseeChart::write_e2_csv()`:
/// `name, n, stem, Adams filtration, shift, h0target, h1target, h2target, h3target,
///  E, H, P, C2, drinfo, drtarget, nulldif, XX`
///
/// Map columns (E, H, P) are populated from the page's map tables.
/// Differential columns are populated from the solve result when available.
pub fn write_ehp_csv<W: io::Write>(
    page: &SATPage,
    result: Option<&SATResult>,
    writer: &mut W,
) -> io::Result<()> {
    // Every call site passes a raw File; without buffering each writeln!
    // below is its own write(2) syscall.
    let mut writer = io::BufWriter::new(writer);
    let writer = &mut writer;
    writeln!(
        writer,
        "name,n,stem,Adams filtration,shift,\
         h0target,h1target,h2target,h3target,\
         E,H,P,C2,drinfo,drtarget,nulldif,XX"
    )?;

    let mut tridegrees: Vec<Tridegree> = page
        .dimension
        .iter()
        .filter(|(_, &d)| d > 0)
        .map(|(&t, _)| t)
        .collect();
    tridegrees.sort();

    for &t in &tridegrees {
        let dim = page.dim_at(t);
        if dim == 0 {
            continue;
        }

        for idx in 0..dim {
            let name = gen_name(t.n, t.s, t.f, idx, dim);

            // h_i extension targets
            let h0_targets = hi_target_names(page, t, idx, 0);
            let h1_targets = hi_target_names(page, t, idx, 1);
            let h2_targets = hi_target_names(page, t, idx, 3);
            let h3_targets = hi_target_names(page, t, idx, 7);

            // Map targets
            let e_targets = map_target_names(page, MapKind::E, t, idx);
            let map_h_targets = map_target_names(page, MapKind::H, t, idx);
            let p_targets = map_target_names(page, MapKind::P, t, idx);

            // Differential info
            let (dr_info, dr_targets, null_targets) = if let Some(res) = result {
                let tgt_t = t.diff_target(page.r);
                let n_var = t.n.min(t.s + 2);
                let no_vars = !res
                    .var_index
                    .contains_key(&DiffVar::new(n_var, t.s, t.f, 0, 0));
                if no_vars && (page.is_excluded(t) || page.is_excluded(tgt_t)) {
                    // Uncertainty-excluded: dash to every potential target
                    // (mirrors the original write_spheres) so exclusion is
                    // never mistaken for a determined-zero differential.
                    let tgt_dim = page.dim_at(tgt_t);
                    let nulls = (0..tgt_dim)
                        .map(|j| gen_name(tgt_t.n, tgt_t.s, tgt_t.f, j, tgt_dim))
                        .collect();
                    (None, Vec::new(), nulls)
                } else {
                    let (targets, nulls) = diff_and_null_target_names(page, res, t, idx);
                    if !targets.is_empty() {
                        (Some(page.r), targets, nulls)
                    } else if !nulls.is_empty() {
                        (None, Vec::new(), nulls)
                    } else {
                        (None, Vec::new(), Vec::new())
                    }
                }
            } else {
                (None, Vec::new(), Vec::new())
            };

            writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{},{},{},{},,{},{},{},XX",
                name,
                t.n,
                t.s,
                t.f,
                0, // shift
                h0_targets.join(";"),
                h1_targets.join(";"),
                h2_targets.join(";"),
                h3_targets.join(";"),
                e_targets.join(";"),
                map_h_targets.join(";"),
                p_targets.join(";"),
                dr_info.map(|d| d.to_string()).unwrap_or_default(),
                dr_targets.join(";"),
                null_targets.join(";"),
            )?;
        }
    }

    writer.flush()
}
