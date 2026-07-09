//! Export spectral sequence data to SeqSee CSV format.
//!
//! [SeqSee](https://github.com/JoeyBF/SeqSee) is a spectral sequence visualization tool.
//! This module provides functionality to export [`Sseq`] data to CSV files
//! compatible with SeqSee's input format.
//!
//! Two CSV formats are supported:
//! - **E2 format** (17 columns): used for E2-page data (sphere-indexed charts)
//! - **E3+ format** (24 columns): used for E3 and higher pages (ext-computed data)

use std::collections::HashMap;
use std::io;

use fp::matrix::Subquotient;

use crate::coordinates::Bidegree;
use crate::{Adams, Product, Sseq, SseqProfile};

/// A single row in the SeqSee CSV file, representing one generator
/// of the spectral sequence at a given bidegree.
#[derive(Debug, Clone)]
pub struct SeqseeRow {
    /// Generator identifier (e.g., `"2_0_0"` or `"15_15_2_1"`)
    pub name: String,
    /// Space dimension or identifier (the `n` column)
    pub n: i32,
    /// Stem degree (horizontal axis, `t - s` in Adams conventions)
    pub stem: i32,
    /// Adams filtration (vertical axis, `s` in Adams conventions)
    pub filtration: i32,
    /// Degree shift (usually 0)
    pub shift: i32,
    /// Product targets: `h_targets[i]` lists the names of generators hit by `h_i`
    pub h_targets: [Vec<String>; 4],
    /// Target under the suspension map E (empty for single-space export)
    pub e_target: String,
    /// Differential page number (e.g., 2 for d_2)
    pub dr_info: Option<i32>,
    /// Names of generators hit by the differential
    pub dr_targets: Vec<String>,
}

/// A collection of spectral sequence generators and their relationships,
/// ready for export to SeqSee CSV format.
#[derive(Debug)]
pub struct SeqseeChart {
    /// All generator rows.
    pub rows: Vec<SeqseeRow>,
    /// Lookup from `(stem, filtration, page_index)` to index in `rows`.
    name_lookup: HashMap<(i32, i32, usize), usize>,
}

impl Default for SeqseeChart {
    fn default() -> Self {
        Self::new()
    }
}

impl SeqseeChart {
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            name_lookup: HashMap::new(),
        }
    }

    /// Get the name of a generator at the given position, if it exists.
    pub fn generator_name(&self, stem: i32, filtration: i32, idx: usize) -> Option<&str> {
        self.name_lookup
            .get(&(stem, filtration, idx))
            .map(|&row_idx| self.rows[row_idx].name.as_str())
    }

    /// Build a [`SeqseeChart`] from spectral sequence data.
    ///
    /// This extracts generators, products, and differentials from the given
    /// [`Sseq`] at the specified page and produces a chart ready for CSV export.
    ///
    /// # Arguments
    ///
    /// * `sseq` - The spectral sequence (E2 page with optional d_2, or higher)
    /// * `page` - Which E-page to export (2 for E2, 3 for E3, etc.)
    /// * `products` - Filtration-one products (typically h_0 through h_3)
    /// * `differentials` - Whether to include d_r differential annotations
    /// * `n_value` - Value for the `n` column (space dimension or module id)
    pub fn from_sseq(
        sseq: &Sseq<2, Adams>,
        page: i32,
        products: &[(String, Product<2>)],
        differentials: bool,
        n_value: i32,
    ) -> Self {
        let mut chart = Self::new();

        // Step 1: Create generator rows for every bidegree with nonzero
        // dimension on this page.
        for b in sseq.iter_degrees() {
            let bd = sseq.page_data(b).get_max(page);
            if bd.is_empty() {
                continue;
            }

            let stem = b.x();
            let filtration = b.y();
            let dim = bd.dimension();

            for idx in 0..dim {
                let name = if dim == 1 {
                    format!("{n_value}_{stem}_{filtration}")
                } else {
                    format!("{n_value}_{stem}_{filtration}_{idx}")
                };

                let row_idx = chart.rows.len();
                chart.name_lookup.insert((stem, filtration, idx), row_idx);
                chart.rows.push(SeqseeRow {
                    name,
                    n: n_value,
                    stem,
                    filtration,
                    shift: 0,
                    h_targets: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
                    e_target: String::new(),
                    dr_info: None,
                    dr_targets: Vec::new(),
                });
            }
        }

        // Step 2: Populate product targets (h_0 through h_3).
        for (i, (_name, prod)) in products.iter().enumerate() {
            if i >= 4 {
                break;
            }

            for b in sseq.iter_degrees() {
                let target_data = sseq.page_data(b).get_max(page);
                if target_data.is_empty() {
                    continue;
                }

                let source_b = b - prod.b;
                if !sseq.defined(source_b) {
                    continue;
                }

                let source_data = sseq.page_data(source_b).get_max(page);
                if source_data.is_empty() {
                    continue;
                }

                if let Some(matrix) = prod.matrices.get(source_b) {
                    let reduced = Subquotient::reduce_matrix(matrix, source_data, target_data);

                    let source_stem = source_b.x();
                    let source_filt = source_b.y();
                    let target_stem = b.x();
                    let target_filt = b.y();

                    let target_dim = target_data.dimension();

                    for (k, row) in reduced.iter().enumerate() {
                        let Some(&row_idx) =
                            chart.name_lookup.get(&(source_stem, source_filt, k))
                        else {
                            continue;
                        };
                        for (l, &v) in row.iter().enumerate() {
                            if v != 0 {
                                let target_name = format_gen_name(
                                    n_value, target_stem, target_filt, l, target_dim,
                                );
                                chart.rows[row_idx].h_targets[i].push(target_name);
                            }
                        }
                    }
                }
            }
        }

        // Step 3: Annotate differentials.
        if differentials {
            for b in sseq.iter_degrees() {
                let bd = sseq.page_data(b).get_max(page);
                if bd.is_empty() {
                    continue;
                }

                let target_b = Adams::profile(page, b);
                if target_b.x() < 0 || !sseq.defined(target_b) {
                    continue;
                }

                let d = sseq.differentials(b);
                if d.len() <= page {
                    continue;
                }
                let d = &d[page];
                let target_data = sseq.page_data(target_b).get_max(page);
                let target_dim = target_data.dimension();

                let stem = b.x();
                let filtration = b.y();
                let target_stem = target_b.x();
                let target_filt = target_b.y();

                for (mut source_vec, mut target_vec) in d.get_source_target_pairs() {
                    let source_reduced = bd.reduce(source_vec.as_slice_mut());
                    let target_reduced = target_data.reduce(target_vec.as_slice_mut());

                    for (i, &sv) in source_reduced.iter().enumerate() {
                        if sv == 0 {
                            continue;
                        }
                        let Some(&row_idx) = chart.name_lookup.get(&(stem, filtration, i)) else {
                            continue;
                        };

                        for (j, &tv) in target_reduced.iter().enumerate() {
                            if tv == 0 {
                                continue;
                            }
                            let tn = format_gen_name(
                                n_value, target_stem, target_filt, j, target_dim,
                            );
                            if !chart.rows[row_idx].dr_targets.contains(&tn) {
                                chart.rows[row_idx].dr_targets.push(tn);
                            }
                        }

                        if !chart.rows[row_idx].dr_targets.is_empty() {
                            chart.rows[row_idx].dr_info = Some(page);
                        }
                    }
                }
            }
        }

        chart
    }

    /// Build a [`SeqseeChart`] for a family of spheres S^n.
    ///
    /// This creates one row per sphere dimension `n` (from `min_n` to `max_n`)
    /// for each generator in the spectral sequence, with suspension maps (E targets)
    /// linking S^n to S^{n+1}.
    pub fn from_sseq_spheres(
        sseq: &Sseq<2, Adams>,
        page: i32,
        products: &[(String, Product<2>)],
        differentials: bool,
        min_n: i32,
        max_n: i32,
    ) -> Self {
        let mut chart = Self::new();

        // Collect all bidegrees and their dimensions first.
        let bidegrees: Vec<(Bidegree, usize)> = sseq
            .iter_degrees()
            .filter_map(|b| {
                let bd = sseq.page_data(b).get_max(page);
                if bd.is_empty() {
                    None
                } else {
                    Some((b, bd.dimension()))
                }
            })
            .collect();

        // Step 1: Create rows for each (n, stem, filt) triple.
        for n in min_n..=max_n {
            for &(b, dim) in &bidegrees {
                let stem = b.x();
                let filtration = b.y();

                for idx in 0..dim {
                    let name = if dim == 1 {
                        format!("{n}_{stem}_{filtration}")
                    } else {
                        format!("{n}_{stem}_{filtration}_{idx}")
                    };

                    // E target: same position at n+1
                    let e_target = if n < max_n {
                        if dim == 1 {
                            format!("{}_{stem}_{filtration}", n + 1)
                        } else {
                            format!("{}_{stem}_{filtration}_{idx}", n + 1)
                        }
                    } else {
                        String::new()
                    };

                    let row_idx = chart.rows.len();
                    chart.name_lookup.insert((stem, filtration, idx), row_idx);
                    chart.rows.push(SeqseeRow {
                        name,
                        n,
                        stem,
                        filtration,
                        shift: 0,
                        h_targets: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
                        e_target,
                        dr_info: None,
                        dr_targets: Vec::new(),
                    });
                }
            }
        }

        // For sphere families, products and differentials are shared across all n.
        // We build them once for a reference n and then replicate the names.
        // For simplicity, populate products for each n using the same sseq data.
        for n in min_n..=max_n {
            for (i, (_name, prod)) in products.iter().enumerate() {
                if i >= 4 {
                    break;
                }

                for &(b, _) in &bidegrees {
                    let target_data = sseq.page_data(b).get_max(page);
                    let source_b = b - prod.b;
                    if !sseq.defined(source_b) {
                        continue;
                    }
                    let source_data = sseq.page_data(source_b).get_max(page);
                    if source_data.is_empty() {
                        continue;
                    }

                    if let Some(matrix) = prod.matrices.get(source_b) {
                        let reduced =
                            Subquotient::reduce_matrix(matrix, source_data, target_data);
                        let source_stem = source_b.x();
                        let source_filt = source_b.y();
                        let target_stem = b.x();
                        let target_filt = b.y();
                        let target_dim = target_data.dimension();

                        for (k, row) in reduced.iter().enumerate() {
                            // Find this generator's row for this n.
                            let source_name = format_gen_name(
                                n,
                                source_stem,
                                source_filt,
                                k,
                                source_data.dimension(),
                            );
                            let Some(row_idx) = chart
                                .rows
                                .iter()
                                .position(|r| r.name == source_name && r.n == n)
                            else {
                                continue;
                            };

                            for (l, &v) in row.iter().enumerate() {
                                if v != 0 {
                                    let target_name = format_gen_name(
                                        n,
                                        target_stem,
                                        target_filt,
                                        l,
                                        target_dim,
                                    );
                                    chart.rows[row_idx].h_targets[i]
                                        .push(target_name);
                                }
                            }
                        }
                    }
                }
            }

            if differentials {
                for &(b, _) in &bidegrees {
                    let bd = sseq.page_data(b).get_max(page);
                    let target_b = Adams::profile(page, b);
                    if target_b.x() < 0 || !sseq.defined(target_b) {
                        continue;
                    }
                    let d = sseq.differentials(b);
                    if d.len() <= page {
                        continue;
                    }
                    let d = &d[page];
                    let target_data = sseq.page_data(target_b).get_max(page);
                    let target_dim = target_data.dimension();

                    let stem = b.x();
                    let filtration = b.y();
                    let target_stem = target_b.x();
                    let target_filt = target_b.y();

                    for (mut source_vec, mut target_vec) in d.get_source_target_pairs() {
                        let source_reduced = bd.reduce(source_vec.as_slice_mut());
                        let target_reduced = target_data.reduce(target_vec.as_slice_mut());

                        for (si, &sv) in source_reduced.iter().enumerate() {
                            if sv == 0 {
                                continue;
                            }
                            let source_name =
                                format_gen_name(n, stem, filtration, si, bd.dimension());
                            let Some(row_idx) = chart
                                .rows
                                .iter()
                                .position(|r| r.name == source_name && r.n == n)
                            else {
                                continue;
                            };

                            for (tj, &tv) in target_reduced.iter().enumerate() {
                                if tv == 0 {
                                    continue;
                                }
                                let tn = format_gen_name(
                                    n,
                                    target_stem,
                                    target_filt,
                                    tj,
                                    target_dim,
                                );
                                if !chart.rows[row_idx].dr_targets.contains(&tn) {
                                    chart.rows[row_idx].dr_targets.push(tn);
                                }
                            }

                            if !chart.rows[row_idx].dr_targets.is_empty() {
                                chart.rows[row_idx].dr_info = Some(page);
                            }
                        }
                    }
                }
            }
        }

        chart
    }

    /// Write the chart in E2-format CSV (17 columns).
    ///
    /// Columns: `name, n, stem, Adams filtration, shift, h0target, h1target,
    /// h2target, h3target, E, H, P, C2, drinfo, drtarget, nulldif, XX`
    pub fn write_e2_csv<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writeln!(
            writer,
            "name,n,stem,Adams filtration,shift,\
             h0target,h1target,h2target,h3target,\
             E,H,P,C2,drinfo,drtarget,nulldif,XX"
        )?;

        for row in &self.rows {
            // 17 fields: name,n,stem,filt,shift,
            //   h0target,h1target,h2target,h3target,
            //   E,H,P,C2,drinfo,drtarget,nulldif,XX
            writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{},{},,,,{},{},,XX",
                row.name,
                row.n,
                row.stem,
                row.filtration,
                row.shift,
                row.h_targets[0].join(";"),
                row.h_targets[1].join(";"),
                row.h_targets[2].join(";"),
                row.h_targets[3].join(";"),
                row.e_target,
                row.dr_info.map(|d| d.to_string()).unwrap_or_default(),
                row.dr_targets.join(";"),
            )?;
        }

        Ok(())
    }

    /// Write the chart in E3+-format CSV (24 columns).
    ///
    /// Columns: `name, n, stem, Adams filtration, shift, tautorsion,
    /// h0info, h0target, h1info, h1target, h2info, h2target, h3info, h3target,
    /// E, H, P, C2, label, angle, drinfo, drtarget, nulldif, XX`
    pub fn write_e3_csv<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writeln!(
            writer,
            "name,n,stem,Adams filtration,shift,tautorsion,\
             h0info,h0target,h1info,h1target,h2info,h2target,h3info,h3target,\
             E,H,P,C2,label,angle,drinfo,drtarget,nulldif,XX"
        )?;

        for row in &self.rows {
            // 24 fields: name,n,stem,filt,shift,tautorsion,
            //   h0info,h0target,h1info,h1target,h2info,h2target,h3info,h3target,
            //   E,H,P,C2,label,angle,drinfo,drtarget,nulldif,XX
            writeln!(
                writer,
                "{},{},{},{},{},0,,{},,{},,{},,{},{},,,,{},,{},{},,XX",
                row.name,
                row.n,
                row.stem,
                row.filtration,
                row.shift,
                row.h_targets[0].join(";"),
                row.h_targets[1].join(";"),
                row.h_targets[2].join(";"),
                row.h_targets[3].join(";"),
                row.e_target,
                row.name,
                row.dr_info.map(|d| d.to_string()).unwrap_or_default(),
                row.dr_targets.join(";"),
            )?;
        }

        Ok(())
    }
}

/// Format a generator name following the SeqSee convention.
fn format_gen_name(n: i32, stem: i32, filtration: i32, idx: usize, dim: usize) -> String {
    if dim == 1 {
        format!("{n}_{stem}_{filtration}")
    } else {
        format!("{n}_{stem}_{filtration}_{idx}")
    }
}
