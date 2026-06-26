#[cfg(feature = "nassau")]
compile_error!("This example does not support the nassau feature");

use std::sync::Arc;

use algebra::{
    module::{FDModule, Module},
    Algebra,
};
use ext::chain_complex::{
    AugmentedChainComplex, BoundedChainComplex, ChainComplex, FreeChainComplex, HomCochainComplex,
};
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    eprintln!("This script checks for realization obstructions in Ext(M, M).");
    eprintln!("It scans the stem -2 line for nonzero groups.");
    eprintln!();

    let resolution = ext::utils::query_module_only("Module", None, false)?;
    let target_cc = resolution.target();

    if target_cc.max_s() > 1 {
        anyhow::bail!("Input must be a module, not a chain complex (max_s must be 1)");
    }

    let module = target_cc.module(0);

    let max_mod_deg = module
        .max_degree()
        .ok_or_else(|| anyhow::anyhow!("Module must be bounded (finite dimensional)"))?;
    let min_mod_deg = module.min_degree();
    let diam = max_mod_deg - min_mod_deg;

    eprintln!("Module degree range: [{min_mod_deg}, {max_mod_deg}], diameter = {diam}");

    // We need to check Ext^{s, s-2}(M, M) for s in [3, diam+2].
    // We also scan stem -1 for informational purposes.
    // s_max covers both: max of (diam + 2) for stem -2, and (diam + 2) for stem -1.
    let s_max = diam + 4;

    // Resolution range: HomCochainComplex needs resolution through
    // stem = hom_max.n() + max_mod_deg, filtration = hom_max.s() + 1
    // We need stem 0 for the self-test (Ext^{0,0}), stem -1 and stem -2 for scans.
    let hom_max = Bidegree::n_s(0, s_max);
    let res_max = hom_max + Bidegree::n_s(max_mod_deg, 1);
    resolution.compute_through_stem(res_max);
    resolution
        .algebra()
        .compute_basis(hom_max.t() + max_mod_deg + 2);

    let resolution = Arc::new(resolution);

    let hom_cc = HomCochainComplex::new(Arc::clone(&resolution), Arc::clone(&module));
    hom_cc.compute_through_stem(hom_max);

    // Self-test: Ext^{0,0}(M, M) should be >= 1 (contains the identity)
    let ext_0_0 = hom_cc.homology_dimension(Bidegree::n_s(0, 0));
    assert!(
        ext_0_0 >= 1,
        "Self-test failed: Ext^{{0,0}}(M, M) = {ext_0_0}, expected >= 1"
    );
    eprintln!("Self-test: dim Ext^{{0,0}}(M, M) = {ext_0_0} (OK)");

    // Cross-check: Ext(M, F2) via HomCochainComplex must match resolution.number_of_gens_in_bidegree
    cross_check_num_gens(&resolution, hom_max)?;

    eprintln!();

    // Scan stem -2 line: Ext^{s, s-2}(M, M) for s = 3..=diam+2
    let mut stem_minus_2: Vec<(i32, i32, usize)> = Vec::new();
    for s in 0..=s_max {
        let b = Bidegree::n_s(-2, s);
        let dim = if s >= 3 {
            hom_cc.homology_dimension(b)
        } else {
            0
        };
        stem_minus_2.push((s, b.t(), dim));
    }

    let mut obstructions: Vec<(i32, i32, usize)> = Vec::new();
    for &(s, t, dim) in &stem_minus_2 {
        if s >= 3 && s <= diam + 2 && dim > 0 {
            obstructions.push((s, t, dim));
        }
    }

    // Scan stem -1 line
    let mut stem_minus_1: Vec<(i32, i32, usize)> = Vec::new();
    for s in 0..=s_max {
        let b = Bidegree::n_s(-1, s);
        let dim = if s >= 1 {
            hom_cc.homology_dimension(b)
        } else {
            0
        };
        stem_minus_1.push((s, b.t(), dim));
    }

    // Print ASCII chart
    print_ascii_chart(&stem_minus_2, &stem_minus_1, s_max, diam);

    // Verdict
    eprintln!();
    if obstructions.is_empty() {
        println!("REALIZABLE");
        eprintln!("Verdict: All stem -2 groups vanish for s in [3, {}]. No realization obstructions detected.", diam + 2);

        // Uniqueness analysis from stem -1
        let nonzero_minus_1: Vec<(i32, usize)> = stem_minus_1
            .iter()
            .filter(|&&(s, _, dim)| s >= 1 && dim > 0)
            .map(|&(s, _, dim)| (s, dim))
            .collect();

        if nonzero_minus_1.is_empty() {
            eprintln!("Uniqueness: Stem -1 is all zero => unique realization (up to 2-completion).");
        } else {
            eprintln!("Uniqueness: Realization is not unique. Nonzero Ext^{{s, s-1}}(M, M):");
            for (s, dim) in &nonzero_minus_1 {
                eprintln!("  s = {s}: dim = {dim}");
            }
        }
    } else {
        println!("INCONCLUSIVE");
        eprintln!(
            "Verdict: Nonzero groups detected on stem -2 line at {} bidegree(s):",
            obstructions.len()
        );
        for (s, t, dim) in &obstructions {
            eprintln!("  (s, t) = ({s}, {t}), dim = {dim}");
        }
        eprintln!("Realization is obstructed or requires further analysis.");
        eprintln!("(Obstruction class computation not yet implemented — group nonzero, class may still vanish.)");
    }

    Ok(())
}

/// Cross-check: build Ext(M, F2) via HomCochainComplex and compare against
/// resolution.number_of_gens_in_bidegree for all computed bidegrees.
fn cross_check_num_gens(
    resolution: &Arc<ext::utils::QueryModuleResolution>,
    hom_max: Bidegree,
) -> anyhow::Result<()> {
    eprintln!("Cross-check: comparing Ext(M, F2) via HomCochainComplex against num_gens...");

    let algebra = resolution.algebra();

    // Build F2 as an FDModule with the same algebra
    let f2 = Arc::new(FDModule::new(
        algebra,
        String::from("F2"),
        bivec::BiVec::from_vec(0, vec![1]),
    ));

    let f2_hom_cc = HomCochainComplex::new(Arc::clone(resolution), f2);
    f2_hom_cc.compute_through_stem(hom_max);

    let mut checked = 0;
    for b in resolution.iter_stem() {
        if b.s() > hom_max.s() || b.n() > hom_max.n() {
            continue;
        }
        let num_gens = resolution.number_of_gens_in_bidegree(b);
        let ext_dim = f2_hom_cc.homology_dimension(b);
        if num_gens != ext_dim {
            anyhow::bail!(
                "Cross-check FAILED at ({}, {}): num_gens = {}, Ext(M, F2) = {}",
                b.s(),
                b.t(),
                num_gens,
                ext_dim
            );
        }
        checked += 1;
    }
    eprintln!("Cross-check passed ({checked} bidegrees verified).");
    Ok(())
}

/// Print an ASCII chart of the stem -2 and stem -1 lines.
fn print_ascii_chart(
    stem_minus_2: &[(i32, i32, usize)],
    stem_minus_1: &[(i32, i32, usize)],
    s_max: i32,
    diam: i32,
) {
    eprintln!();
    eprintln!("=== Ext(M, M) chart (stem -2 and stem -1) ===");
    eprintln!();

    // Header: s values
    let s_range: Vec<i32> = (0..=s_max).collect();
    eprint!("         s = ");
    for s in &s_range {
        eprint!("{:>3}", s);
    }
    eprintln!();
    eprint!("             ");
    for _ in &s_range {
        eprint!("---");
    }
    eprintln!();

    // Stem -2 row
    eprint!("  stem -2:   ");
    for (i, s) in s_range.iter().enumerate() {
        let dim = stem_minus_2[i].2;
        let marker = if *s >= 3 && *s <= diam + 2 {
            ext::utils::unicode_num(dim)
        } else {
            '.'
        };
        eprint!("  {marker}");
    }
    eprintln!();

    // Stem -1 row
    eprint!("  stem -1:   ");
    for (i, s) in s_range.iter().enumerate() {
        let dim = stem_minus_1[i].2;
        let marker = if *s >= 1 {
            ext::utils::unicode_num(dim)
        } else {
            '.'
        };
        eprint!("  {marker}");
    }
    eprintln!();

    // Legend
    eprintln!();
    eprintln!("  Legend: ' ' = 0, '·' = 1, ':' = 2, ... '*' = 10+, '.' = out of range");
}
