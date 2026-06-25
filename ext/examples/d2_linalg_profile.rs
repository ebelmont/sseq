use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use algebra::{
    module::{
        homomorphism::{FreeModuleHomomorphism, ModuleHomomorphism},
        HomModule, Module, SteenrodModule,
    },
    Algebra,
};
use ext::chain_complex::{
    AugmentedChainComplex, BoundedChainComplex, ChainComplex,
};
use ext::secondary::{SecondaryLift, SecondaryResolution};
use ext::utils::query_module;
use fp::matrix::{Matrix, Subspace};
use fp::vector::FpVector;
use sseq::coordinates::{Bidegree, BidegreeGenerator};

fn main() -> anyhow::Result<()> {
    let resolution = Arc::new(query_module(Some(algebra::AlgebraType::Milnor), true)?);
    let target_cc = resolution.target();

    if target_cc.max_s() > 1 {
        anyhow::bail!("Cannot compute secondary space for non-module");
    }

    let module = target_cc.module(0);

    let max_nonzero = module
        .max_degree()
        .ok_or_else(|| anyhow::anyhow!("Expected bounded module"))?;

    let source = resolution.module(2); // M_2(X)
    let m0 = resolution.module(0); // M_0(X)
    let m1 = resolution.module(1); // M_1(X)

    let hom = HomModule::new(Arc::clone(&source), Arc::clone(&module));
    resolution.algebra().compute_basis(2 * max_nonzero);
    hom.compute_basis(max_nonzero);

    let g = resolution.chain_map(0);
    let p = source.prime();
    let max_src_deg = source.max_computed_degree();

    let degree = 1;
    let dim = hom.dimension(degree);

    if dim <= 127 {
        eprintln!(
            "Hom^1 dimension: {dim} (2^{dim} = {} candidate seeds)",
            1u128 << dim
        );
    } else {
        eprintln!("Hom^1 dimension: {dim} (2^{dim} candidate seeds)");
    }

    // Precompute lifts for each basis element (only dim of them, not 2^dim)
    let t_basis = Instant::now();
    let basis_lifts: Vec<_> = (0..dim)
        .map(|j| {
            let f = FreeModuleHomomorphism::new(Arc::clone(&source), g.target(), degree);
            let gbe = hom.block_structures[degree].index_to_generator_basis_elt(j);
            for gd in f.min_degree()..=max_src_deg {
                let n = source.number_of_gens_in_degree(gd);
                let target_dim = module.dimension(gd - degree);
                let mut rows = Vec::with_capacity(n);
                for gi in 0..n {
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

    eprintln!("[PROFILE] basis_lifts precompute: {:?}", t_basis.elapsed());

    // Generator-mapping constraints: require h(gen at src_deg in M_2) = gen at tgt_deg in M_0.
    // To disable these constraints, change to vec![].
    let gen_constraints: Vec<(i32, i32)> = vec![];

    // Compute composites and intermediates once (both are seed-independent)
    let t_composites = Instant::now();
    let base = SecondaryResolution::new(Arc::clone(&resolution));
    base.initialize_homotopies();
    base.compute_composites();
    eprintln!("[PROFILE] base composites: {:?}", t_composites.elapsed());
    let t_intermediates_base = Instant::now();
    base.compute_intermediates();
    eprintln!("[PROFILE] base intermediates: {:?}", t_intermediates_base.elapsed());

    let d3 = resolution.differential(3);
    let d1 = resolution.differential(1);

    // Collect constraint equations over F_2
    // Each constraint: sum_j c_j * a_j = rhs (where a_j are the seed parameters)
    let mut constraint_coeffs: Vec<FpVector> = Vec::new();
    let mut constraint_rhs: Vec<u32> = Vec::new();

    // Add generator-mapping constraints
    for &(src_deg, tgt_deg) in &gen_constraints {
        if source.number_of_gens_in_degree(src_deg) == 0 {
            eprintln!("Warning: no generators in M_2 at degree {src_deg}, skipping constraint ({src_deg}, {tgt_deg})");
            continue;
        }
        if m0.number_of_gens_in_degree(tgt_deg) == 0 {
            eprintln!("Warning: no generators in M_0 at degree {tgt_deg}, skipping constraint ({src_deg}, {tgt_deg})");
            continue;
        }

        let gen_offset = m0.generator_offset(tgt_deg, tgt_deg, 0);
        let tgt_dim = m0.dimension(tgt_deg);

        for k in 0..tgt_dim {
            let rhs_bit = if k == gen_offset { 1u32 } else { 0u32 };
            let any_coeff = (0..dim).any(|j| basis_lifts[j].output(src_deg, 0).entry(k) != 0);

            if rhs_bit == 0 && !any_coeff {
                continue;
            }

            if rhs_bit != 0 && !any_coeff {
                eprintln!("No valid seeds: generator constraint unsatisfiable at ({src_deg}, {tgt_deg}), position {k}");
                return Ok(());
            }

            let mut row = FpVector::new(p, dim);
            for j in 0..dim {
                row.set_entry(j, basis_lifts[j].output(src_deg, 0).entry(k));
            }
            constraint_coeffs.push(row);
            constraint_rhs.push(rhs_bit);
        }
        eprintln!(
            "Added generator constraint: h(gen at {src_deg} in M_2) = gen at {tgt_deg} in M_0"
        );
    }

    // Iterate over all degrees t at s=3
    let t_constraints = Instant::now();
    let min_t = base.homotopies()[3].homotopies.min_degree();
    let max_t = base.max().t(3);
    eprintln!("s=3 range: t in [{min_t}, {max_t})");

    for t in min_t..max_t {
        let num_gens_s3 = resolution.module(3).number_of_gens_in_degree(t);
        if num_gens_s3 == 0 {
            continue;
        }

        let target_deg = t - 1; // intermediate lives in module(0) at degree t-1
        let target_dim = m0.dimension(target_deg);
        if target_dim == 0 {
            continue;
        }

        // Build image subspace of d_1 at target_deg.
        let m1_dim = m1.dimension(target_deg);
        let mut image = Subspace::new(p, target_dim);
        for basis_idx in 0..m1_dim {
            let mut v = FpVector::new(p, target_dim);
            d1.apply_to_basis_element(v.as_slice_mut(), 1, target_deg, basis_idx);
            image.add_vector(v.as_slice());
        }

        // If image is the entire space, all intermediates are liftable — no constraint
        if image.dimension() == target_dim {
            continue;
        }

        // For each generator at (s=3, t):
        for idx in 0..num_gens_s3 {
            // Compute base intermediate (from composites only)
            let bg = BidegreeGenerator::s_t(3, t, idx);
            let mut base_int = base.compute_intermediate(bg);

            // Compute contribution from each seed basis element
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

            // Reduce all vectors modulo the image of d_1
            image.reduce(base_int.as_slice_mut());
            for contrib in &mut contribs {
                image.reduce(contrib.as_slice_mut());
            }

            // Extract constraints from non-zero positions
            for c in 0..target_dim {
                let rhs_bit = base_int.entry(c);
                let any_contrib = contribs.iter().any(|v| v.entry(c) != 0);

                if rhs_bit == 0 && !any_contrib {
                    continue; // trivially satisfied
                }

                if rhs_bit != 0 && !any_contrib {
                    // Unsatisfiable: no seed can fix this
                    eprintln!(
                        "No valid seeds: unsatisfiable constraint at (s=3, t={t}, idx={idx})"
                    );
                    return Ok(());
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

    eprintln!("[PROFILE] constraint building: {:?}", t_constraints.elapsed());

    let num_constraints = constraint_coeffs.len();
    eprintln!("Total constraints: {num_constraints} equations in {dim} unknowns");

    if num_constraints == 0 {
        eprintln!(
            "No constraints — all {} seeds are valid at s=3",
            1u128 << dim
        );
    }

    // Build augmented matrix [C | rhs] and row reduce
    let mut matrix = Matrix::new(p, std::cmp::max(num_constraints, 1), dim + 1);
    for (i, (row, &rhs)) in constraint_coeffs.iter().zip(&constraint_rhs).enumerate() {
        for j in 0..dim {
            matrix.row_mut(i).set_entry(j, row.entry(j));
        }
        matrix.row_mut(i).set_entry(dim, rhs);
    }
    matrix.initialize_pivots();
    matrix.row_reduce();

    // Check consistency: if there's a pivot in the rhs column, system is inconsistent
    if num_constraints > 0 && matrix.pivots()[dim] >= 0 {
        eprintln!("No valid seeds: inconsistent constraint system");
        return Ok(());
    }

    // Read off solution space
    let mut pivot_cols: Vec<usize> = Vec::new();
    for col in 0..dim {
        if matrix.pivots()[col] >= 0 {
            pivot_cols.push(col);
        }
    }

    let rank = pivot_cols.len();
    let free_dim = dim - rank;
    let num_solutions = 1u128 << free_dim;
    eprintln!("Rank: {rank}, Free dimensions: {free_dim}, Valid seeds at s=3: {num_solutions}");

    // Find the free variables
    let free_vars: Vec<usize> = (0..dim).filter(|j| matrix.pivots()[*j] < 0).collect();

    // --- d_2 uniqueness check via linear algebra ---

    // Helper: compute d_2 for a given seed value
    let compute_d2 = |seed: u128| -> Vec<(Bidegree, usize, Vec<u32>)> {
        let t0 = Instant::now();
        let h_i = build_lift(
            seed,
            dim,
            &hom,
            &source,
            &module,
            &g,
            p,
            max_src_deg,
            degree,
        );
        let t_build_lift = t0.elapsed();

        let t1 = Instant::now();
        let lift = SecondaryResolution::new(Arc::clone(&resolution));
        lift.initialize_homotopies();
        lift.copy_composites_from(&base);
        lift.copy_intermediates_from(&base);
        let t_composites = t1.elapsed();

        // Seed homotopy at s=2
        let t2 = Instant::now();
        {
            let hom_field = &lift.homotopies()[2].homotopies;
            let max_seed_deg = std::cmp::min(lift.max().t(2) - 1, source.max_computed_degree());
            for d in hom_field.min_degree()..=max_seed_deg {
                let n = source.number_of_gens_in_degree(d);
                let target_dim = m0.dimension(d - hom_field.degree_shift());
                let mut rows = Vec::with_capacity(n);
                for gi in 0..n {
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
        let t_seed = t2.elapsed();

        let t3 = Instant::now();
        // intermediates already copied above
        let t_intermediates = t3.elapsed();

        let t4 = Instant::now();
        let min_t = lift.homotopies()[2].homotopies.min_degree();
        let s_range = lift.homotopies().range();
        let min = Bidegree::s_t(s_range.start + 1, min_t);
        let max = lift.max().restrict(s_range.end);
        sseq::coordinates::iter_s_t(&|b| lift.compute_homotopy_step(b), min, max);
        let t_homotopy_steps = t4.elapsed();

        let t5 = Instant::now();
        // Extract d_2 data
        // hom_k(t) returns a target_dim × source_dim matrix where:
        //   target_dim = M_{s-2} gens at degree t (rows = d_2 source generators)
        //   source_dim = M_s gens at degree t+1 (cols = d_2 target generators)
        // We store (bidegree, generator_index, entry) tuples to avoid index drift.
        let mut d2_data: Vec<(Bidegree, usize, Vec<u32>)> = Vec::new();
        for b in resolution.iter_stem() {
            if b.s() < 2 {
                continue;
            }
            if b.t() - 1 > resolution.module(b.s() - 2).max_computed_degree() {
                continue;
            }
            let homotopy = lift.homotopy(b.s());
            let m = homotopy.homotopies.hom_k(b.t() - 1);
            for (i, entry) in m.into_iter().enumerate() {
                d2_data.push((b, i, entry));
            }
        }
        let t_extract = t5.elapsed();

        eprintln!(
            "[PROFILE] compute_d2(seed={seed}): build_lift={t_build_lift:?}, composites={t_composites:?}, seed={t_seed:?}, intermediates={t_intermediates:?}, homotopy_steps={t_homotopy_steps:?}, extract={t_extract:?}, total={:?}",
            t0.elapsed()
        );
        d2_data
    };

    // Compute particular solution seed (free vars = 0)
    let mut particular_seed: u128 = 0;
    for &col in &pivot_cols {
        let row = matrix.pivots()[col] as usize;
        if matrix.row(row).entry(dim) != 0 {
            particular_seed |= 1u128 << col;
        }
    }

    let particular_name = seed_name(particular_seed, dim, &hom, degree);
    eprintln!("Particular solution seed: {particular_seed} ({particular_name})");

    // Build seeds for each free direction (perturbed from particular solution)
    // Each direction_seed[k] = particular_seed XOR (the bits corresponding to free direction k)
    let direction_seeds: Vec<u128> = free_vars
        .iter()
        .enumerate()
        .map(|(_k, &f_k)| {
            // Build direction vector: set free var f_k = 1, others = 0, back-substitute pivots
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
            // XOR with particular solution
            let mut seed = particular_seed;
            for j in 0..dim {
                if v[j] != 0 {
                    seed ^= 1u128 << j;
                }
            }
            seed
        })
        .collect();

    eprintln!(
        "\nComputing d_2 for {} seeds (1 base + {} free directions)...",
        free_dim + 1,
        free_dim
    );

    // Compute d_2 for particular solution (base)
    let d2_base = compute_d2(particular_seed);
    let d2_len = d2_base.len();
    eprintln!("d_2 vector length: {d2_len} entries");

    // Check uniqueness: for each free direction, compute d_2 and compare to base
    // Store deltas (XOR with base) for directions that differ
    let mut differing_directions: Vec<usize> = Vec::new();
    let mut differing_deltas: Vec<Vec<Vec<u32>>> = Vec::new();
    for (k, &dir_seed) in direction_seeds.iter().enumerate() {
        let d2_dir = compute_d2(dir_seed);

        // Compute delta = entry-wise XOR of the d_2 coefficient vectors
        let delta: Vec<Vec<u32>> = d2_dir
            .iter()
            .zip(d2_base.iter())
            .map(|((_, _, row_dir), (_, _, row_base))| {
                row_dir.iter().zip(row_base.iter()).map(|(&a, &b)| a ^ b).collect()
            })
            .collect();

        let differs = delta.iter().any(|row| row.iter().any(|&x| x != 0));

        if differs {
            differing_directions.push(k);
            differing_deltas.push(delta);
            eprintln!(
                "d_2 differs along free direction {} (free var index {}, seed {})",
                k, free_vars[k], dir_seed
            );
        }
    }

    let num_differing = differing_directions.len();

    // Row-reduce deltas to find the rank of the delta space.
    // Linearly dependent deltas don't produce additional distinct d_2 groups.
    let delta_flat_len: usize = if differing_deltas.is_empty() {
        0
    } else {
        differing_deltas[0].iter().map(|row| row.len()).sum()
    };

    let mut independent: Vec<usize> = Vec::new(); // indices into differing_directions
    if delta_flat_len > 0 {
        let mut basis = Subspace::new(p, delta_flat_len);
        for (i, delta) in differing_deltas.iter().enumerate() {
            let mut v = FpVector::new(p, delta_flat_len);
            let mut col = 0;
            for row in delta {
                for &val in row {
                    v.set_entry(col, val);
                    col += 1;
                }
            }
            let mut v_copy = v.clone();
            basis.reduce(v_copy.as_slice_mut());
            if !v_copy.is_zero() {
                basis.add_vector(v.as_slice());
                independent.push(i);
            }
        }
    }

    let rank = independent.len();
    let num_d2_groups = 1u128 << rank;
    if rank == 0 {
        eprintln!(
            "\nAll {} valid seeds produce the same d_2 differentials.",
            num_solutions
        );
    } else {
        eprintln!(
            "\n{num_d2_groups} distinct d_2 groups ({num_differing} of {free_dim} free directions affect d_2, {rank} independent)."
        );
    }

    // Open output file: {module_name}_d2.txt
    std::fs::create_dir_all("output")?;
    let output_path = format!("output/{}_d2.txt", resolution.name());
    eprintln!("Writing output to {output_path}");
    let mut writer: Box<dyn Write> =
        Box::new(std::io::BufWriter::new(std::fs::File::create(&output_path)?));

    // Helper: format d_2 data as text
    let d2_shift = Bidegree::n_s(-1, 2);
    let format_d2 =
        |d2_data: &[(Bidegree, usize, Vec<u32>)], writer: &mut dyn Write| -> anyhow::Result<()> {
            for (b, i, entry) in d2_data {
                if entry.iter().any(|&x| x != 0) {
                    let source_gen = BidegreeGenerator::new(*b - d2_shift, *i);
                    writeln!(writer, "  d_2 x_{source_gen} = {entry:?}")?;
                }
            }
            Ok(())
        };

    // Enumerate all distinct d_2 groups
    writeln!(writer, "# d_2 uniqueness check")?;
    writeln!(writer, "# Hom^1 dimension: {dim}, rank: {rank}, free_dim: {free_dim}")?;
    writeln!(writer, "# {num_d2_groups} distinct d_2 groups ({num_differing} of {free_dim} free directions affect d_2, {rank} independent)")?;
    writeln!(writer, "# {num_solutions} total valid seeds at s=3")?;
    writeln!(writer)?;

    // Show which d_2 entries differ between groups
    if num_differing > 0 {
        writeln!(writer, "# d_2 entries that vary across groups:")?;
        for (idx, (b, i, base_entry)) in d2_base.iter().enumerate() {
            for (dir_idx, delta) in differing_deltas.iter().enumerate() {
                if idx < delta.len() && delta[idx].iter().any(|&x| x != 0) {
                    let source_gen = BidegreeGenerator::new(*b - d2_shift, *i);
                    let delta_entry = &delta[idx];
                    writeln!(
                        writer,
                        "#   d_2 x_{source_gen}: base={base_entry:?}, delta along dir {dir_idx}={delta_entry:?}"
                    )?;
                }
            }
        }
        writeln!(writer)?;
    }

    for group_bits in 0..num_d2_groups {
        // Build the seed for this group: start from particular, XOR in each active independent direction
        let mut group_seed = particular_seed;
        for (bit_idx, &indep_idx) in independent.iter().enumerate() {
            if (group_bits >> bit_idx) & 1 != 0 {
                let dir_k = differing_directions[indep_idx];
                group_seed ^= particular_seed ^ direction_seeds[dir_k];
            }
        }

        let group_name = seed_name(group_seed, dim, &hom, degree);
        writeln!(
            writer,
            "=== Group {} / {} (seed {group_seed}: {group_name}) ===",
            group_bits + 1,
            num_d2_groups
        )?;

        let d2_group = compute_d2(group_seed);
        let num_nonzero = d2_group.iter().filter(|(_, _, e)| e.iter().any(|&x| x != 0)).count();
        writeln!(writer, "# {num_nonzero} nonzero differentials")?;
        eprintln!("  Group {}: {num_nonzero} nonzero differentials", group_bits + 1);
        format_d2(&d2_group, &mut *writer)?;
        writeln!(writer)?;

        print_secondary_ops(&resolution, group_seed, dim, &hom, degree, &mut *writer);
    }

    writer.flush()?;
    eprintln!("Output written to {output_path}");

    Ok(())
}

fn seed_name<M: Module>(seed: u128, dim: usize, hom: &HomModule<M>, degree: i32) -> String {
    if seed == 0 {
        "0".to_string()
    } else {
        let parts: Vec<String> = (0..dim)
            .filter(|&idx| (seed >> idx) & 1 != 0)
            .map(|idx| hom.basis_element_to_string(degree, idx))
            .collect();
        parts.join(" + ")
    }
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
        let n = source.number_of_gens_in_degree(gd);
        let target_dim = module.dimension(gd - degree);
        let mut rows = Vec::with_capacity(n);
        for gi in 0..n {
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

fn print_secondary_ops(
    resolution: &ext::utils::QueryModuleResolution,
    seed: u128,
    dim: usize,
    hom: &HomModule<SteenrodModule>,
    degree: i32,
    writer: &mut dyn Write,
) {
    macro_rules! out {
        ($($arg:tt)*) => {{
            eprintln!($($arg)*);
            let _ = writeln!(writer, $($arg)*);
        }};
    }
    let target_cc = resolution.target();
    let module = target_cc.module(0);
    let source = resolution.module(2);
    let m0 = resolution.module(0);
    let m1 = resolution.module(1);
    let d0 = resolution.chain_map(0);
    let d1 = resolution.differential(1);
    let d2_diff = resolution.differential(2);
    let p = source.prime();
    let max_src_deg = source.max_computed_degree();

    let h_i = build_lift(seed, dim, hom, &source, &module, &d0, p, max_src_deg, degree);

    // Build M_0 generator → H*(X) name mapping via d_0
    let mut m0_gen_x_names: std::collections::HashMap<(i32, usize), String> =
        std::collections::HashMap::new();
    for gen_deg in m0.min_degree()..=m0.max_computed_degree() {
        for gen_idx in 0..m0.number_of_gens_in_degree(gen_deg) {
            let d0_out = d0.output(gen_deg, gen_idx);
            let x_str = module.element_to_string(gen_deg, d0_out.as_slice());
            m0_gen_x_names.insert((gen_deg, gen_idx), x_str);
        }
    }

    out!("\n--- Secondary operations for seed {seed} ---");
    let mut equation_count = 0usize;
    for d in source.min_degree()..=max_src_deg {
        let n = source.number_of_gens_in_degree(d);
        for gi in 0..n {
            let d2_out = d2_diff.output(d, gi);
            if d2_out.is_zero() {
                continue;
            }

            // Collect (outer_op · inner_op) terms grouped by M_0 generator
            let mut gen_terms: std::collections::BTreeMap<(i32, usize), Vec<String>> =
                std::collections::BTreeMap::new();

            for (idx, _coeff) in d2_out.as_slice().iter_nonzero() {
                let m1_opgen = m1.index_to_op_gen(d, idx);
                let outer = m1.algebra().basis_element_to_string(
                    m1_opgen.operation_degree,
                    m1_opgen.operation_index,
                );

                let d1_out =
                    d1.output(m1_opgen.generator_degree, m1_opgen.generator_index);

                for (d1_idx, _) in d1_out.as_slice().iter_nonzero() {
                    let m0_opgen =
                        m0.index_to_op_gen(m1_opgen.generator_degree, d1_idx);
                    let inner = m0.algebra().basis_element_to_string(
                        m0_opgen.operation_degree,
                        m0_opgen.operation_index,
                    );

                    let key = (m0_opgen.generator_degree, m0_opgen.generator_index);
                    let composite = match (outer.as_str(), inner.as_str()) {
                        ("1", "1") => "1".to_string(),
                        ("1", _) => inner.clone(),
                        (_, "1") => outer.clone(),
                        _ => format!("{outer}·{inner}"),
                    };
                    gen_terms.entry(key).or_default().push(composite);
                }
            }

            if gen_terms.is_empty() {
                continue;
            }

            // Compute d_0(seed(g)) for RHS
            let rhs = if d >= h_i.min_degree() && d < h_i.next_degree() {
                let h_out = h_i.output(d, gi);
                if h_out.is_zero() {
                    continue;
                }
                let m0_deg = d - degree;
                let x_dim = module.dimension(m0_deg);
                let mut d0_out = FpVector::new(p, x_dim);
                d0.apply(d0_out.as_slice_mut(), 1, m0_deg, h_out.as_slice());
                if d0_out.is_zero() {
                    continue;
                }
                module.element_to_string(m0_deg, d0_out.as_slice())
            } else {
                continue;
            };

            equation_count += 1;
            out!("\nEquation {equation_count}:\n");

            // Format each generator's terms with line wrapping
            let max_line = 72;
            let mut first_gen = true;
            for (&(gen_deg, gen_idx), composites) in &gen_terms {
                let x_name = &m0_gen_x_names[&(gen_deg, gen_idx)];
                let prefix = if first_gen { "  " } else { "+ " };
                first_gen = false;

                // Build the bracket contents with wrapping
                let mut bracket_lines: Vec<String> = Vec::new();
                let mut current_line = String::from("[ ");
                for (i, op) in composites.iter().enumerate() {
                    let addition = if i == 0 {
                        op.clone()
                    } else {
                        format!(" + {op}")
                    };
                    if current_line.len() + addition.len() > max_line {
                        bracket_lines.push(current_line);
                        current_line = format!("  + {op}");
                    } else {
                        current_line.push_str(&addition);
                    }
                }
                current_line.push_str(&format!(" ] · {x_name}"));
                bracket_lines.push(current_line);

                for (li, line) in bracket_lines.iter().enumerate() {
                    if li == 0 {
                        out!("{prefix}{line}");
                    } else {
                        out!("  {line}");
                    }
                }
            }
            out!("= {rhs}");
        }
    }
    out!("");
}
