/// Diagnostic: for each RPn, find the stem at which d2_groups stabilizes.
/// Tests increasing resolution ranges until the d2 group count stops changing.

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
use fp::matrix::{Matrix, Subspace};
use fp::vector::FpVector;
use sseq::coordinates::{Bidegree, BidegreeGenerator};

fn main() {
    let max_rp = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(30);

    println!("{:>4}  {:>5}  {:>12}  {:>6}  {:>10}  {:>12}  {:>10}  {:>14}",
        "RPn", "dim", "constraints", "rank", "free_dim", "valid_seeds", "d2_groups", "stable_stem");
    println!("{}", "-".repeat(95));

    for n in 2..=max_rp {
        let result = find_stable_range(n);
        println!("{:>4}  {:>5}  {:>12}  {:>6}  {:>10}  {:>12}  {:>10}  {:>14}",
            format!("RP{n}"), result.dim, result.num_constraints, result.rank,
            result.free_dim, result.num_valid_seeds, result.num_d2_groups,
            result.stable_stem);
    }
}

struct StabilityResult {
    dim: usize,
    num_constraints: usize,
    rank: usize,
    free_dim: usize,
    num_valid_seeds: String,
    num_d2_groups: String,
    stable_stem: String,
}

fn find_stable_range(n: i32) -> StabilityResult {
    // Test at increasing stems until d2_groups stabilizes for 2 consecutive increases
    let stems: Vec<i32> = (0..8).map(|k| n + 5 + k * 5).collect();

    let mut prev_groups: Option<u128> = None;
    let mut stable_count = 0;
    let mut stable_stem_val: Option<i32> = None;
    let mut last_result: Option<AnalysisResult> = None;

    for &stem in &stems {
        let max_s = std::cmp::max(stem / 2, 4);
        eprint!("  RP{n} at ({stem},{max_s})...");
        let result = analyze(n, stem, max_s);
        eprintln!(" d2_groups={}", result.num_d2_groups);

        if let Some(prev) = prev_groups {
            if result.num_d2_groups == prev {
                stable_count += 1;
                if stable_count == 1 && stable_stem_val.is_none() {
                    // First stem where it matched = the stem where it stabilized
                    stable_stem_val = Some(stem - 5); // the previous stem was where it first hit this value
                }
                if stable_count >= 2 {
                    last_result = Some(result);
                    break;
                }
            } else {
                stable_count = 0;
                stable_stem_val = None;
            }
        }

        prev_groups = Some(result.num_d2_groups);
        last_result = Some(result);
    }

    let r = last_result.unwrap();

    let valid_str = if r.free_dim <= 40 {
        format!("{}", 1u128 << r.free_dim)
    } else {
        format!("2^{}", r.free_dim)
    };

    let groups_str = if r.num_d2_groups <= (1u128 << 40) {
        format!("{}", r.num_d2_groups)
    } else {
        format!("2^{}", r.num_d2_groups.trailing_zeros())
    };

    let stable_str = match stable_stem_val {
        Some(s) => format!("{s}"),
        None => "not stable".to_string(),
    };

    StabilityResult {
        dim: r.dim,
        num_constraints: r.num_constraints,
        rank: r.rank,
        free_dim: r.free_dim,
        num_valid_seeds: valid_str,
        num_d2_groups: groups_str,
        stable_stem: stable_str,
    }
}

struct AnalysisResult {
    dim: usize,
    num_constraints: usize,
    rank: usize,
    free_dim: usize,
    num_d2_groups: u128,
}

fn analyze(n: i32, max_stem: i32, max_s: i32) -> AnalysisResult {
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
    let max_nonzero = match module.max_degree() {
        Some(d) => d,
        None => return AnalysisResult { dim: 0, num_constraints: 0, rank: 0, free_dim: 0, num_d2_groups: 1 },
    };

    let source = resolution.module(2);
    let m0 = resolution.module(0);
    let m1 = resolution.module(1);

    let hom = HomModule::new(Arc::clone(&source), Arc::clone(&module));
    resolution.algebra().compute_basis(2 * max_nonzero);
    hom.compute_basis(max_nonzero);

    let g = resolution.chain_map(0);
    let p = source.prime();
    let max_src_deg = source.max_computed_degree();

    let degree = 1;
    let dim = hom.dimension(degree);

    if dim == 0 {
        return AnalysisResult { dim: 0, num_constraints: 0, rank: 0, free_dim: 0, num_d2_groups: 1 };
    }

    // Precompute lifts
    let basis_lifts: Vec<_> = (0..dim)
        .map(|j| {
            let f = FreeModuleHomomorphism::new(Arc::clone(&source), g.target(), degree);
            let gbe = hom.block_structures[degree].index_to_generator_basis_elt(j);
            for gd in f.min_degree()..=max_src_deg {
                let ng = source.number_of_gens_in_degree(gd);
                let target_dim = module.dimension(gd - degree);
                let mut rows = Vec::with_capacity(ng);
                for gi in 0..ng {
                    let mut row = FpVector::new(p, target_dim);
                    if gd == gbe.generator_degree && gi == gbe.generator_index {
                        row.set_entry(gbe.basis_index, 1);
                    }
                    rows.push(row);
                }
                f.add_generators_from_rows(gd, rows);
            }
            f.lift_through(&g).expect("basis element not in image of g")
        })
        .collect();

    // Compute composites and intermediates
    let base = SecondaryResolution::new(Arc::clone(&resolution));
    base.initialize_homotopies();
    base.compute_composites();
    base.compute_intermediates();

    let d3 = resolution.differential(3);
    let d1 = resolution.differential(1);

    // Collect constraints
    let mut constraint_coeffs: Vec<FpVector> = Vec::new();
    let mut constraint_rhs: Vec<u32> = Vec::new();

    if max_s < 3 || base.homotopies().range().end <= 3 {
        return AnalysisResult {
            dim, num_constraints: 0, rank: 0, free_dim: dim,
            num_d2_groups: 1, // can't distinguish without s=3
        };
    }

    let min_t = base.homotopies()[3].homotopies.min_degree();
    let max_t = base.max().t(3);

    for t in min_t..max_t {
        let num_gens_s3 = resolution.module(3).number_of_gens_in_degree(t);
        if num_gens_s3 == 0 {
            continue;
        }

        let target_deg = t - 1;
        let target_dim = m0.dimension(target_deg);
        if target_dim == 0 {
            continue;
        }

        let m1_dim = m1.dimension(target_deg);
        let mut image = Subspace::new(p, target_dim);
        for basis_idx in 0..m1_dim {
            let mut v = FpVector::new(p, target_dim);
            d1.apply_to_basis_element(v.as_slice_mut(), 1, target_deg, basis_idx);
            image.add_vector(v.as_slice());
        }

        if image.dimension() == target_dim {
            continue;
        }

        for idx in 0..num_gens_s3 {
            let bg = BidegreeGenerator::s_t(3, t, idx);
            let mut base_int = base.compute_intermediate(bg);

            let mut contribs: Vec<FpVector> = (0..dim)
                .map(|j| {
                    let mut contrib = FpVector::new(p, target_dim);
                    basis_lifts[j].apply(
                        contrib.as_slice_mut(),
                        1,
                        t,
                        d3.output(t, idx).as_slice(),
                    );
                    contrib
                })
                .collect();

            image.reduce(base_int.as_slice_mut());
            for contrib in &mut contribs {
                image.reduce(contrib.as_slice_mut());
            }

            for c in 0..target_dim {
                let rhs_bit = base_int.entry(c);
                let any_contrib = contribs.iter().any(|v| v.entry(c) != 0);

                if rhs_bit == 0 && !any_contrib {
                    continue;
                }

                if rhs_bit != 0 && !any_contrib {
                    return AnalysisResult {
                        dim, num_constraints: constraint_coeffs.len(),
                        rank: 0, free_dim: 0, num_d2_groups: 0,
                    };
                }

                let mut row = FpVector::new(p, dim);
                for j in 0..dim {
                    row.set_entry(j, contribs[j].entry(c));
                }
                constraint_coeffs.push(row);
                constraint_rhs.push(rhs_bit);
            }
        }
    }

    let num_constraints = constraint_coeffs.len();

    // Row reduce
    let mut matrix = Matrix::new(p, std::cmp::max(num_constraints, 1), dim + 1);
    for (i, (row, &rhs)) in constraint_coeffs.iter().zip(&constraint_rhs).enumerate() {
        for j in 0..dim {
            matrix.row_mut(i).set_entry(j, row.entry(j));
        }
        matrix.row_mut(i).set_entry(dim, rhs);
    }
    matrix.initialize_pivots();
    matrix.row_reduce();

    if num_constraints > 0 && matrix.pivots()[dim] >= 0 {
        return AnalysisResult {
            dim, num_constraints, rank: 0, free_dim: 0, num_d2_groups: 0,
        };
    }

    let mut pivot_cols: Vec<usize> = Vec::new();
    for col in 0..dim {
        if matrix.pivots()[col] >= 0 {
            pivot_cols.push(col);
        }
    }

    let rank = pivot_cols.len();
    let free_dim = dim - rank;

    // d2 uniqueness check
    let free_vars: Vec<usize> = (0..dim).filter(|j| matrix.pivots()[*j] < 0).collect();

    let compute_d2 = |seed: u128| -> Vec<Vec<u32>> {
        let h_i = build_lift(seed, dim, &hom, &source, &module, &g, p, max_src_deg, degree);

        let lift = SecondaryResolution::new(Arc::clone(&resolution));
        lift.initialize_homotopies();
        lift.copy_composites_from(&base);
        lift.copy_intermediates_from(&base);

        {
            let hom_field = &lift.homotopies()[2].homotopies;
            let max_seed_deg = std::cmp::min(lift.max().t(2) - 1, source.max_computed_degree());
            for d in hom_field.min_degree()..=max_seed_deg {
                let ng = source.number_of_gens_in_degree(d);
                let target_dim = m0.dimension(d - hom_field.degree_shift());
                let mut rows = Vec::with_capacity(ng);
                for gi in 0..ng {
                    let mut row = FpVector::new(p, target_dim);
                    if d >= h_i.min_degree() && d < h_i.next_degree() {
                        let h_out = h_i.output(d, gi);
                        if !h_out.is_zero() {
                            row.add(h_out, 1);
                        }
                    }
                    rows.push(row);
                }
                hom_field.add_generators_from_rows(d, rows);
            }
        }

        let min_t = lift.homotopies()[2].homotopies.min_degree();
        let s_range = lift.homotopies().range();
        let min = Bidegree::s_t(s_range.start + 1, min_t);
        let max = lift.max().restrict(s_range.end);
        sseq::coordinates::iter_s_t(&|b| lift.compute_homotopy_step(b), min, max);

        let mut d2_data: Vec<Vec<u32>> = Vec::new();
        for b in resolution.iter_stem() {
            if b.s() < 2 { continue; }
            if b.t() - 1 > resolution.module(b.s() - 2).max_computed_degree() { continue; }
            let homotopy = lift.homotopy(b.s());
            let m = homotopy.homotopies.hom_k(b.t() - 1);
            for (_i, entry) in m.into_iter().enumerate() {
                d2_data.push(entry);
            }
        }
        d2_data
    };

    // Particular solution
    let mut particular_seed: u128 = 0;
    for &col in &pivot_cols {
        let row_idx = matrix.pivots()[col] as usize;
        if matrix.row(row_idx).entry(dim) != 0 {
            particular_seed |= 1u128 << col;
        }
    }

    let direction_seeds: Vec<u128> = free_vars
        .iter()
        .map(|&f_k| {
            let mut v = vec![0u32; dim];
            v[f_k] = 1;
            for &col in pivot_cols.iter().rev() {
                let row_idx = matrix.pivots()[col] as usize;
                let mut val = 0u32;
                for j in (col + 1)..dim {
                    val ^= matrix.row(row_idx).entry(j) * v[j];
                }
                v[col] = val;
            }
            let mut seed = particular_seed;
            for j in 0..dim {
                if v[j] != 0 { seed ^= 1u128 << j; }
            }
            seed
        })
        .collect();

    let d2_base = compute_d2(particular_seed);

    let mut differing_deltas: Vec<Vec<Vec<u32>>> = Vec::new();
    for &dir_seed in &direction_seeds {
        let d2_dir = compute_d2(dir_seed);
        let delta: Vec<Vec<u32>> = d2_dir.iter().zip(d2_base.iter())
            .map(|(row_dir, row_base)| {
                row_dir.iter().zip(row_base.iter()).map(|(&a, &b)| a ^ b).collect()
            })
            .collect();
        if delta.iter().any(|row| row.iter().any(|&x| x != 0)) {
            differing_deltas.push(delta);
        }
    }

    let delta_flat_len: usize = if differing_deltas.is_empty() { 0 }
        else { differing_deltas[0].iter().map(|row| row.len()).sum() };

    let mut independent_rank = 0usize;
    if delta_flat_len > 0 {
        let mut basis = Subspace::new(p, delta_flat_len);
        for delta in &differing_deltas {
            let mut v = FpVector::new(p, delta_flat_len);
            let mut col = 0;
            for row in delta {
                for &val in row { v.set_entry(col, val); col += 1; }
            }
            let mut v_copy = v.clone();
            basis.reduce(v_copy.as_slice_mut());
            if !v_copy.is_zero() {
                basis.add_vector(v.as_slice());
                independent_rank += 1;
            }
        }
    }

    let num_d2_groups = 1u128 << independent_rank;

    AnalysisResult { dim, num_constraints, rank, free_dim, num_d2_groups }
}

fn build_lift<M: Module>(
    seed: u128,
    dim: usize,
    hom: &HomModule<M>,
    source: &Arc<algebra::module::MuFreeModule<false, M::Algebra>>,
    module: &M,
    g: &FreeModuleHomomorphism<M>,
    p: fp::prime::ValidPrime,
    max_src_deg: i32,
    degree: i32,
) -> FreeModuleHomomorphism<algebra::module::MuFreeModule<false, M::Algebra>>
where
    M::Algebra: algebra::MuAlgebra<false>,
{
    let f = FreeModuleHomomorphism::new(Arc::clone(source), g.target(), degree);
    for gd in f.min_degree()..=max_src_deg {
        let ng = source.number_of_gens_in_degree(gd);
        let target_dim = module.dimension(gd - degree);
        let mut rows = Vec::with_capacity(ng);
        for gi in 0..ng {
            let mut row = FpVector::new(p, target_dim);
            for idx in 0..dim {
                if (seed >> idx) & 1 != 0 {
                    let gbe = hom.block_structures[degree].index_to_generator_basis_elt(idx);
                    if gd == gbe.generator_degree && gi == gbe.generator_index {
                        row.add_basis_element(gbe.basis_index, 1);
                    }
                }
            }
            rows.push(row);
        }
        f.add_generators_from_rows(gd, rows);
    }
    f.lift_through(g).expect("seed element not in image of g")
}
