/// Check whether extending the resolution range changes EXISTING d2 values
/// or just reveals NEW d2 entries.
///
/// For RP10 and RP23, computes d2 at two different ranges and compares
/// the entries that exist in both.

use std::sync::Arc;

use algebra::{
    module::{
        homomorphism::{FreeModuleHomomorphism, ModuleHomomorphism},
        HomModule, Module,
    },
    AlgebraType, Algebra,
};
use ext::chain_complex::{
    AugmentedChainComplex, ChainComplex,
};
use ext::secondary::{SecondaryLift, SecondaryResolution};
use ext::utils::construct_standard;
use fp::vector::FpVector;
use sseq::coordinates::Bidegree;

fn main() {
    for &(n, stem_small, stem_large) in &[(10, 15, 20), (23, 33, 38)] {
        println!("=== RP{n}: comparing stem {stem_small} vs stem {stem_large} ===");
        compare_d2_values(n, stem_small, stem_large);
        println!();
    }
}

fn compare_d2_values(n: i32, stem_small: i32, stem_large: i32) {
    let d2_small = compute_all_d2(n, stem_small, stem_small / 2);
    let d2_large = compute_all_d2(n, stem_large, stem_large / 2);

    // Compare entries that exist in both
    let mut shared = 0;
    let mut changed = 0;
    let mut only_large = 0;

    for (bideg, idx, values) in &d2_large {
        match d2_small.iter().find(|(b, i, _)| b == bideg && i == idx) {
            Some((_, _, small_values)) => {
                shared += 1;
                if values != small_values {
                    changed += 1;
                    println!("  CHANGED at ({},{}), idx {}: {:?} -> {:?}",
                        bideg.s(), bideg.t(), idx, small_values, values);
                }
            }
            None => {
                only_large += 1;
                let nonzero = values.iter().any(|&v| v != 0);
                if nonzero {
                    println!("  NEW nonzero d2 at ({},{}), idx {}: {:?}",
                        bideg.s(), bideg.t(), idx, values);
                }
            }
        }
    }

    let only_small: usize = d2_small.iter()
        .filter(|(b, i, _)| !d2_large.iter().any(|(b2, i2, _)| b == b2 && i == i2))
        .count();

    println!("  Small range: {} entries, Large range: {} entries", d2_small.len(), d2_large.len());
    println!("  Shared: {shared}, Changed: {changed}, Only in large: {only_large}, Only in small: {only_small}");

    if changed == 0 {
        println!("  CONCLUSION: Existing d2 values are STABLE. Changes come from new entries.");
    } else {
        println!("  CONCLUSION: BUG - existing d2 values CHANGED when extending range.");
    }
}

/// Compute d2 entries for seed 0 at the given resolution range.
/// Returns Vec<(Bidegree, gen_idx, Vec<u32>)>.
fn compute_all_d2(n: i32, max_stem: i32, max_s: i32) -> Vec<(Bidegree, usize, Vec<u32>)> {
    let config = serde_json::json!({
        "p": 2,
        "type": "real projective space",
        "min": 1,
        "max": n
    });

    let resolution = Arc::new(
        construct_standard::<false, _, _>((config, AlgebraType::Milnor), None).unwrap()
    );
    resolution.compute_through_stem(Bidegree::n_s(max_stem, max_s));

    let target_cc = resolution.target();
    let module = target_cc.module(0);
    let max_nonzero = module.max_degree().unwrap();

    let source = resolution.module(2);
    let m0 = resolution.module(0);

    let hom = HomModule::new(Arc::clone(&source), Arc::clone(&module));
    resolution.algebra().compute_basis(2 * max_nonzero);
    hom.compute_basis(max_nonzero);

    let g = resolution.chain_map(0);
    let p = source.prime();
    let max_src_deg = source.max_computed_degree();
    let degree = 1;
    let dim = hom.dimension(degree);

    eprintln!("  RP{n} at ({max_stem},{max_s}): dim={dim}");

    // Seed 0: trivial lift (no hom corrections)
    let base = SecondaryResolution::new(Arc::clone(&resolution));
    base.initialize_homotopies();
    base.compute_composites();
    base.compute_intermediates();

    // Seed the homotopy with zeros (seed 0)
    {
        let hom_field = &base.homotopies()[2].homotopies;
        let max_seed_deg = std::cmp::min(base.max().t(2) - 1, source.max_computed_degree());
        for d in hom_field.min_degree()..=max_seed_deg {
            let ng = source.number_of_gens_in_degree(d);
            let target_dim = m0.dimension(d - hom_field.degree_shift());
            let mut rows = Vec::with_capacity(ng);
            for _ in 0..ng {
                rows.push(FpVector::new(p, target_dim));
            }
            hom_field.add_generators_from_rows(d, rows);
        }
    }

    let min_t = base.homotopies()[2].homotopies.min_degree();
    let s_range = base.homotopies().range();
    let min = Bidegree::s_t(s_range.start + 1, min_t);
    let max = base.max().restrict(s_range.end);
    sseq::coordinates::iter_s_t(&|b| base.compute_homotopy_step(b), min, max);

    // Extract d2 data
    let mut d2_data: Vec<(Bidegree, usize, Vec<u32>)> = Vec::new();
    for b in resolution.iter_stem() {
        if b.s() < 2 { continue; }
        if b.t() - 1 > resolution.module(b.s() - 2).max_computed_degree() { continue; }
        let homotopy = base.homotopy(b.s());
        let m = homotopy.homotopies.hom_k(b.t() - 1);
        for (i, entry) in m.into_iter().enumerate() {
            d2_data.push((b, i, entry));
        }
    }
    d2_data
}
