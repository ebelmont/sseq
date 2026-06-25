//! Enumerates all A-module maps M → Σ^t N, lifts each to a chain map of resolutions, and computes
//! the secondary lift.
//!
//! # Usage
//! The program asks for two Steenrod modules M and N (which may be equal), along with a computation
//! range. It then:
//!
//! 1. Computes the space Hom_A(M, Σ^t N) for each degree shift t using the kernel of the pullback
//!    d₁*: Hom(C₀(M), N) → Hom(C₁(M), N).
//! 2. For each basis element of this kernel, lifts the corresponding module map to a chain map of
//!    resolutions.
//! 3. Computes the secondary resolution of both M and N.
//! 4. Lifts each chain map to the secondary level and outputs primary + secondary data.
//!
//! # Output
//! For each map f at degree shift t, the output consists of:
//! ```text
//! f_{t}_{idx} x_(s, n, i) = [primary values]
//! [f_{t}_{idx}] [basis_string] = [ext part] + λ [lambda part]
//! ```

use std::sync::Arc;

use algebra::module::{
    HomModule, Module,
    homomorphism::{FreeModuleHomomorphism, ModuleHomomorphism},
};
use ext::{
    chain_complex::{AugmentedChainComplex, ChainComplex, FreeChainComplex},
    resolution_homomorphism::ResolutionHomomorphism,
    secondary::*,
    utils,
};
use fp::{matrix::Matrix, vector::FpVector};
use itertools::Itertools;
use sseq::coordinates::{Bidegree, BidegreeElement, BidegreeGenerator};

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let source = Arc::new(utils::query_module_only(
        "Source module (M)",
        Some(algebra::AlgebraType::Milnor),
        true,
    )?);

    let source_name = source.name().to_owned();
    let target = query::with_default("Target module (N)", &source_name, |s| -> anyhow::Result<_> {
        if s == source_name {
            Ok(Arc::clone(&source))
        } else {
            let config: utils::Config = s.try_into()?;
            let mut target = utils::construct(config, None)?;
            target.set_name(s.to_owned());
            Ok(Arc::new(target))
        }
    });

    let same_module = Arc::ptr_eq(&source, &target);

    assert_eq!(source.prime(), target.prime());
    let p = source.prime();

    let max = Bidegree::n_s(
        query::with_default("Max n", "30", str::parse),
        query::with_default("Max s", "7", str::parse),
    );

    source.compute_through_stem(max);
    if !same_module {
        target.compute_through_stem(max);
    }

    // Get the underlying modules
    let source_module = source.target().module(0);
    let target_module = target.target().module(0);

    let target_max_deg = target_module
        .max_degree()
        .expect("secondary_map requires target module N to be bounded");

    // Set up Hom computation
    // source = Hom(C_0(M), N), target = Hom(C_1(M), N)
    let c0 = source.module(0);
    let c1 = source.module(1);
    let d1: Arc<FreeModuleHomomorphism<_>> = source.differential(1);

    let hom_source = Arc::new(HomModule::new(Arc::clone(&c0), Arc::clone(&target_module)));
    let hom_target = Arc::new(HomModule::new(Arc::clone(&c1), Arc::clone(&target_module)));

    let pullback =
        algebra::module::homomorphism::HomPullback::new(hom_source.clone(), hom_target, d1);

    // Compute HomModule basis through the range we need
    let hom_max_degree = c0.max_computed_degree() - target_module.min_degree();
    hom_source.compute_basis(hom_max_degree);
    pullback.compute_auxiliary_data_through_degree(hom_max_degree);

    // Compute secondary resolutions
    let source_lift = Arc::new(SecondaryResolution::new(Arc::clone(&source)));
    source_lift.extend_all();

    let target_lift = if same_module {
        Arc::clone(&source_lift)
    } else {
        let lift = SecondaryResolution::new(Arc::clone(&target));
        lift.extend_all();
        Arc::new(lift)
    };

    // Compute E3 page for source
    let source_sseq = Arc::new(source_lift.e3_page());

    fn get_page_data(sseq: &sseq::Sseq<2, sseq::Adams>, b: Bidegree) -> &fp::matrix::Subquotient {
        let d = sseq.page_data(b);
        &d[std::cmp::min(3, d.len() - 1)]
    }

    // Iterate over each degree t of the HomModule (= degree shift of the map)
    for shift_t in hom_source.min_degree()..=hom_max_degree {
        let kernel = match pullback.kernel(shift_t) {
            Some(k) => k,
            None => continue,
        };

        if kernel.dimension() == 0 {
            continue;
        }

        let block_structure = &hom_source.block_structures[shift_t];

        // For each basis element of the kernel
        for ker_idx in 0..kernel.dimension() {
            let ker_vec = kernel.basis().nth(ker_idx).unwrap();

            let name = format!("f_{shift_t}_{ker_idx}");

            // Decode the kernel vector into generator images for extend_step.
            // The kernel vector lives in Hom(C_0(M), N) at degree shift_t.
            // Its BlockStructure at degree shift_t has blocks indexed by (gen_deg, gen_idx) of
            // C_0(M), each of size dim(N, gen_deg - shift_t).
            //
            // For extend_step at s=0, we need to provide a Matrix for each generator degree d
            // of C_0(M):
            //   rows = number of generators of C_0(M) at degree d = number of generators of M at
            //          degree d
            //   columns = dim(N, d - shift_t) = dim of target module at output degree
            //
            // But extend_step expects the *augmentation* target images, i.e. a matrix whose rows
            // are images in the target module N. We call extend_step once per generator degree.

            let shift = Bidegree::s_t(0, shift_t);
            let hom = ResolutionHomomorphism::new(
                name.clone(),
                Arc::clone(&source),
                Arc::clone(&target),
                shift,
            );

            // Call extend_step for each generator degree of C_0(M) = the source module M's
            // resolution at s=0.
            // The generators of C_0(M) at degree d correspond to generators of M at degree d.
            let source_min = source_module.min_degree();
            let source_max = source_module
                .max_degree()
                .expect("secondary_map requires source module M to be bounded");

            for gen_deg in source_min..=source_max {
                let input = Bidegree::s_t(0, gen_deg);
                let output_t = gen_deg - shift_t;
                let num_gens = c0.number_of_gens_in_degree(gen_deg);
                let target_dim = if output_t >= target_module.min_degree()
                    && output_t <= target_max_deg
                {
                    target_module.dimension(output_t)
                } else {
                    0
                };

                if num_gens == 0 || target_dim == 0 {
                    hom.extend_step(input, None);
                    continue;
                }

                // Build the matrix of images in the augmentation target N
                let mut matrix = Matrix::new(p, num_gens, target_dim);

                for gen_idx in 0..num_gens {
                    let block = block_structure.generator_to_block(gen_deg, gen_idx);
                    for (j, k) in block.enumerate() {
                        matrix.row_mut(gen_idx).set_entry(j, ker_vec.entry(k));
                    }
                }

                hom.extend_step(input, Some(&matrix));
            }

            hom.extend_all();

            // Print primary chain map data
            for b in target.iter_stem() {
                let shifted_b = b + shift;
                if shifted_b.s() >= source.next_homological_degree()
                    || shifted_b.t() > source.module(shifted_b.s()).max_computed_degree()
                {
                    continue;
                }
                let matrix = hom.get_map(shifted_b.s()).hom_k(b.t());
                for (i, r) in matrix.iter().enumerate() {
                    let g = BidegreeGenerator::new(b, i);
                    println!("{name} x_{g} = {r:?}");
                }
            }

            // Secondary lift
            let hom = Arc::new(hom);
            let hom_lift = SecondaryResolutionHomomorphism::new(
                Arc::clone(&source_lift),
                Arc::clone(&target_lift),
                Arc::clone(&hom),
            );

            if let Some(s) = ext::utils::secondary_job() {
                hom_lift.compute_partial(s);
                continue;
            }

            hom_lift.extend_all();

            let hom_name = hom_lift.name();

            // Print secondary data
            for b in source.iter_nonzero_stem() {
                if !source.has_computed_bidegree(b + shift + LAMBDA_BIDEGREE) {
                    continue;
                }
                if !source.has_computed_bidegree(b + shift - Bidegree::s_t(1, 0)) {
                    continue;
                }

                let page_data = get_page_data(source_sseq.as_ref(), b);

                let target_num_gens = target.number_of_gens_in_bidegree(b + shift);
                let lambda_num_gens =
                    target.number_of_gens_in_bidegree(b + shift + LAMBDA_BIDEGREE);

                if target_num_gens == 0 && lambda_num_gens == 0 {
                    continue;
                }

                // Print products with non-surviving classes
                if target_num_gens > 0 {
                    let hom_k = hom.get_map((b + shift).s()).hom_k(b.t());
                    for i in page_data.complement_pivots() {
                        let g = BidegreeGenerator::new(b, i);
                        println!("{hom_name} λ x_{g} = λ {:?}", hom_k[i]);
                    }
                }

                // Print secondary products
                if page_data.subspace_dimension() == 0 {
                    continue;
                }

                let mut outputs = vec![
                    FpVector::new(p, target_num_gens + lambda_num_gens);
                    page_data.subspace_dimension()
                ];

                hom_lift.hom_k(
                    Some(&source_sseq),
                    b,
                    page_data.subspace_gens(),
                    outputs.iter_mut().map(FpVector::as_slice_mut),
                );
                for (g, output) in page_data.subspace_gens().zip_eq(outputs) {
                    println!(
                        "{hom_name} [{basis_string}] = {} + λ {}",
                        output.slice(0, target_num_gens),
                        output.slice(target_num_gens, target_num_gens + lambda_num_gens),
                        basis_string = BidegreeElement::new(b, g.to_owned()).to_basis_string(),
                    );
                }
            }
        }
    }
    Ok(())
}
