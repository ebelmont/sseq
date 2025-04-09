//! Computes the suspension map between different unstable Ext groups.
//!
//! Given an unstable Steenrod module $M$, compute the unstable Ext groups of $\Sigma^k M$ for all
//! $k$ up till the stable range. Each result is printed in the form
//! ```
//! n s k: num_gens - matrix
//! ```
//! The entries are to be interpreted as follows:
//!  - `n` is the stem, which is defined to be `t - s - min_degree`
//!  - `s` is the Adams filtration
//!  - `k` is the shift
//!  - `num_gens` is the number of generators in this Ext group
//!  - `matrix` is the matrix representing the suspension map from $\Sigma^k M$. This is omitted if
//!    the source or target of the suspension map is trivial, or if they have the same dimension
//!    and the matrix is the identity matrix.
//!
//! The output is best read after sorting with `sort -n -k 1 -k 2 -k 3`.

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
use sseq::coordinates::BidegreeGenerator;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::{path::PathBuf, sync::Arc};

fn main() -> anyhow::Result<()> {
    hopf(3)
}

fn hopf(n: i32) -> anyhow::Result<()> {
    //let's think about the sphere
    let module = Arc::new(sphere()?);
    let max_n = 40;
    let max_s = 25;
    let max = Bidegree::n_s(max_n, max_s);
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
    let shift = Bidegree::s_t(0, 0);
    res_a.compute_through_stem(max + shift);
    res_b.compute_through_stem(max);
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

    //augment the target chain complex since sseq doesn't allow maps which decrease Adams
    //filtration
    let new: Arc<UnstableResolution<FiniteChainComplex<_>>> =
        Arc::new(UnstableResolution::augmented_loops(&res_a, max_n));

    for b in new.iter_stem() {
        if b.s() == 0 {
            continue;
        }
        for i in 0..new.number_of_gens_in_bidegree(b) {
            let gen = BidegreeGenerator::new(b, i);
            let cocycle = new.cocycle_string(gen, true);
            println!("d x_{gen:#} = {cocycle}");
        }
    }
    let suspension_shift = Bidegree::s_t(0, 0);
    let hom = UnstableResolutionHomomorphism::new(
        String::from("hopf"),
        Arc::clone(&res_b),
        Arc::clone(&new),
        suspension_shift,
    );

    hom.extend_step_raw(min_degree, Some(t.to_vec()));
    hom.extend_all();

    for stem in (2 * n - 1)..max.n() {
        for s in 0..=max.s() - 1 {
            let source = Bidegree::n_s(stem, s);
            let target = source - suspension_shift;
            let source_num_gens = res_b.number_of_gens_in_bidegree(source);
            let target_num_gens = new.number_of_gens_in_bidegree(target);
            let m = hom.get_map(target.s()).hom_k(target.t());

            let m = format!(" - {m:?}");
            if source_num_gens != 0 || target_num_gens != 0 {
                writeln!(
                    writer,
                    "{stem} {s}: {source_num_gens} {target_num_gens} {m}",
                    stem = stem - (2 * n - 1)
                )
                .expect("Failed to write to file");
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
