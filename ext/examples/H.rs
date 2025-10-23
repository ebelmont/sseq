//! Computes the Hopf map Ext^f(e_n, e_m) --> Ext^{f-1}(e_{2n-1}, e_{m-1}) where n is an input
//! variable
//!
//! Given an unstable Steenrod module $M$, compute the unstable Ext groups of $\Sigma^k M$ for all
//! $k$ up till the stable range. Each result is printed in the form
//! ```
//! n stem filt - matrix
//! ```
//! The entries are to be interpreted as follows:
//!  - `n` is as in the first line above
//!  - `stem` = m - n + f
//!  - `filt` = f
//!  - `matrix` is the *transpose* of the matrix representing the Hopf map on Ext.

use algebra::module::{steenrod_module, Module, SteenrodModule, SuspensionModule};
use algebra::AlgebraType;
use algebra::SteenrodAlgebra;
use ext::chain_complex::ChainComplex;
use ext::{
    chain_complex::{FiniteChainComplex, FreeChainComplex},
    resolution::UnstableResolution,
    resolution_homomorphism::UnstableResolutionHomomorphism,
};
use fp::vector::FpVector;
use serde_json::Value;
use sseq::coordinates::Bidegree;
use sseq::coordinates::BidegreeElement;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::sync::Arc;

/*
 * Let e_n := \Sigma^n F_2.
 * William Balderrama explains how H can be gotten from
 * Ext^{f+1}(e_n, e_{m+1}) = Ext^f(C_n, e_m) --> Ext^f(e_{2n-1}, e_m):
 * To find C_n, form \Omega(homological degree >= 1 of a minimal free resolution P of e_n)
 * where \Omega = algebraic loops (shift dimension down by 1 and apply unstable condition).
 * Then C_n = cokernel(\Omega P_1 <-- \Omega P_2) and the map on Ext comes from a map
 * e_{2n-1} --> C_n sending 1 --> x_n where the resolution differential sends x_n --> Sq^n \iota_n.
 *
 * The main work is to produce a map of resolutions from
 * (e_{2n-1} <-- ...) to (C_n <-- \Omega P_1 <--- ...).
 */

fn main() -> anyhow::Result<()> {
    let max_deg = query::raw("max degree", str::parse);
    let n = query::raw("n", str::parse);
    hopf(n, max_deg)
}

fn hopf(n: i32, max_deg : i32) -> anyhow::Result<()> {
    //let's think about the sphere
    let module = Arc::new(sphere()?);
    // this is the degree of the max stem for the target, so should be (max source stem) - n + 1, but also there is an
    // offset of 2n-1 (internal degree of the target sphere). The +1 is for < vs. <= in the loop.
    let max_n = max_deg + n+1;
    let max_s = max_deg;
    let max = Bidegree::n_s(max_n, max_s as u32);
    //let max = Bidegree::n_s(6, 3);
    //prepping out-file containing our data
    let file = File::create(format!("hopf{n}.txt")).expect("Failed to create log file");
    let mut writer = BufWriter::new(file);

    let min_degree = Bidegree::s_t(0, 2 * n - 1);

    //create a resolution for S^n
    let res_a: Arc<UnstableResolution<FiniteChainComplex<_>>> =
        Arc::new(UnstableResolution::new_with_save(
            Arc::new(FiniteChainComplex::ccdz(Arc::new(SuspensionModule::new(
                Arc::clone(&module),
                n,
            )))),
            None,
        )?);
    //create a resolution for S^2n-1
    let res_b: Arc<UnstableResolution<FiniteChainComplex<_>>> =
        Arc::new(UnstableResolution::new_with_save(
            Arc::new(FiniteChainComplex::ccdz(Arc::new(SuspensionModule::new(
                Arc::clone(&module),
                2 * n - 1,
            )))),
            None,
        )?);
    res_a.compute_through_stem(max);
    res_b.compute_through_stem(max);

    
    // Print res_a
    println!("=== Resolution 'res_a' ===");
    println!("Generators:");
    for b in res_a.iter_stem() {
        for i in 0..res_a.number_of_gens_in_bidegree(b) {
            let gen = sseq::coordinates::BidegreeGenerator::new(b, i);
            println!("({}, {}): x_{gen:#}", gen.n() - n, gen.s());
        }
    }
    println!("\nDifferentials:");
    for b in res_a.iter_stem() {
        for i in 0..res_a.number_of_gens_in_bidegree(b) {
            let gen = sseq::coordinates::BidegreeGenerator::new(b, i);
            let cocycle = res_a.cocycle_string(gen, true);
            println!("({}, {}): d x_{gen:#} = {cocycle}", b.n() - n, b.s());
        }
    }
    println!("=== End Resolution 'res_a' ===\n");

    //Prepare the vector representation of Sq^n x_{n, 0} in F(n)
    let dim = res_a.module(0).dimension(2 * n);
    let mut top: Vec<u32> = vec![0; dim];
    top[dim - 1] = 1;
    let top_degree = Bidegree::s_t(1, 2 * n);

    let dim2 = res_a.module(1).dimension(2 * n);

    let top_inv: Vec<u32> = vec![0; dim2];
    let target = FpVector::from_slice(module.prime(), top_inv.as_slice());
    let mut t = [target];
    let _ = res_a.apply_quasi_inverse(
        &mut t,
        top_degree,
        &[FpVector::from_slice(module.prime(), top.as_slice())],
    );
    println!("top_degree = {top_degree}, t = {t:?}");

    println!("=== Resolution 'res_a' after quasi-inverse ===");
    println!("Generators:");
    for b in res_a.iter_stem() {
        for i in 0..res_a.number_of_gens_in_bidegree(b) {
            let gen = sseq::coordinates::BidegreeGenerator::new(b, i);
            println!("({}, {}): x_{gen:#}", gen.n() - n, gen.s());
        }
    }
    println!("\nDifferentials:");
    for b in res_a.iter_stem() {
        for i in 0..res_a.number_of_gens_in_bidegree(b) {
            let gen = sseq::coordinates::BidegreeGenerator::new(b, i);
            let cocycle = res_a.cocycle_string(gen, true);
            println!("({}, {}): d x_{gen:#} = {cocycle}", b.n() - n, b.s());
        }
    }
    println!("=== End Resolution 'res_a' after quasi-inverse ===\n");

    // Apply algebraic loops to each module of the resolution, and also truncate by removing
    // homological degree zero.
    let new: Arc<UnstableResolution<FiniteChainComplex<_>>> =
        Arc::new(UnstableResolution::unaugmented_loops(&res_a, max_n));

    let suspension_shift = Bidegree::s_t(0, 0);
    let hom = UnstableResolutionHomomorphism::new(
        String::from("hopf"),
        Arc::clone(&res_b),
        Arc::clone(&new),
        suspension_shift,
    );

    // Print new
    println!("=== Resolution 'new' ===");
    println!("Generators:");
    for b in new.iter_stem() {
        for i in 0..new.number_of_gens_in_bidegree(b) {
            let gen = sseq::coordinates::BidegreeGenerator::new(b, i);
            println!("({}, {}): x_{gen:#}", gen.n() - n, gen.s());
        }
    }
    println!("\nDifferentials:");
    for b in new.iter_stem() {
        for i in 0..new.number_of_gens_in_bidegree(b) {
            let gen = sseq::coordinates::BidegreeGenerator::new(b, i);
            if gen.s() == 0 {
                continue;
            }
            let cocycle = new.cocycle_string(gen, true);
            println!("({}, {}): d x_{gen:#} = {cocycle}", b.n() - n, b.s());
        }
    }
    println!("=== End Resolution 'new' ===\n");

    println!("=== Resolution 'res_b' ===");
    println!("Generators:");
    for b in res_b.iter_stem() {
        for i in 0..res_b.number_of_gens_in_bidegree(b) {
            let gen = sseq::coordinates::BidegreeGenerator::new(b, i);
            println!("({}, {}): x_{gen:#}", gen.n() - (2*n-1), gen.s());
        }
    }
    println!("\nDifferentials:");
    for b in res_b.iter_stem() {
        for i in 0..res_b.number_of_gens_in_bidegree(b) {
            let gen = sseq::coordinates::BidegreeGenerator::new(b, i);
            if gen.s() == 0 {
                continue;
            }
            let cocycle = res_b.cocycle_string(gen, true);
            println!("({}, {}): d x_{gen:#} = {cocycle}", b.n() - (2*n-1), b.s());
        }
    }
    println!("=== End Resolution 'res_b' ===\n");


    hom.extend_step_raw(min_degree, Some(t.to_vec()));
    hom.extend_all();
//    let result_vector = hom.get_map(4).output(14, 0);

//    print(BidegreeElement::new(14, result_vector).to_string_module(&hom.target(), false));

    // Since we have a map from (res of e_{2n-1}) --> (res of C_n), "source" and "target" are
    // swapped from what they should be for H, and the matrix this code outputs is actually the
    // transpose of the matrix representing H.
    let vec = hom.get_map(4).return_output();
    let target_module = &hom.get_map(4).target;
    let source_module = &hom.get_map(4).source;
    println!("vec: {}", vec.len());
    for i in 0..vec.len() {
        println!("{}", vec[i]);
        println!("{}", target_module.basis_element_to_string(14, i));
        //println!("{}", target_module.element_to_string(14, vec[i].as_slice()));
    }

    println!("vec end");
    for stem in (2 * n - 1)..max.n() {
        for s in 0..=max.s() - 1 {
            let source = Bidegree::n_s(stem, s);
            let target = source - suspension_shift;
            let source_num_gens = res_b.number_of_gens_in_bidegree(source);
            let target_num_gens = new.number_of_gens_in_bidegree(target);
            println!("{n} {} {}", stem - (2*n-1) - 1 + n, s+1);
            let m = hom.get_map(target.s()).hom_k(target.t());
            let vec = hom.get_map(target.s()).return_output();
            /*if s != 0 && target.s() != 0 {
                for i in 0..vec.len() {
                    println!("{s}, {n}, {}", vec[i]);
                    //println!("{}", vec[i]);
                    let module = &hom.get_map(target.s()).target;
                    //println!("{}", BidegreeElement::new(target, vec[i].clone()).to_string_module(module, false));
                    //let opgen = module.index_to_op_gen(target.t(), i);
                    println!("{}", module.element_to_string(target.t(), vec[i].as_slice()));
                }
            }*/

            let m = format!(" - {m:?}");
            if source_num_gens != 0 && target_num_gens != 0 {
                // H: (stem, filt, sphere) --> (stem-sphere+1, filt-1, 2(sphere)-1)
                // The degrees (stem, s) are for the target of the H map, and we want to print
                // source degrees. Also, sseq's stem has an offset of 2n-1 from the actual stem.
                let realstem = stem - (2*n-1);
                writeln!(
                    writer,
                    "{n} {} {} {m}",
                    realstem - 1 + n,
                    s + 1
                )
                .expect("Failed to write to file");
                println!(
                    "{n} {} {} {m}",
                    realstem - 1 + n,
                    s + 1
                    );
            }
        }
    }

    Ok(())
}
fn sphere() -> anyhow::Result<SteenrodModule> {
    // Hard-coded JSON content
    let json_content = r#"
    {
        "p": 2,
        "type": "finite dimensional module",
        "gens": { "x0": 0 },
        "algebra": ["adem"],
        "actions": []
    }
    "#;

    // Parse the hard-coded JSON
    let json: Value = serde_json::from_str(json_content)
        .map_err(|e| anyhow::anyhow!("Failed to parse hard-coded JSON: {}", e))?;

    // Create the algebra
    let algebra = Arc::new(SteenrodAlgebra::from_json(
        &json,
        AlgebraType::Milnor,
        true,
    )?);

    // Create the module
    steenrod_module::from_json(algebra, &json)
}
