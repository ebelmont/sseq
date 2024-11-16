//! Computes the product up to a sign
//!
//! This is optimized to compute a * _ for fixed a and all _ where a has small degree.

use std::sync::Arc;

use ext::{
    chain_complex::{ChainComplex, FreeChainComplex},
    resolution_homomorphism::ResolutionHomomorphism,
};
use fp::matrix::{AugmentedMatrix, Matrix};
use sseq::coordinates::{Bidegree, BidegreeElement, BidegreeGenerator};

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging();

    let resolution = Arc::new(ext::utils::query_module(None, true)?);
    let p = resolution.prime();

    let (is_unit, unit) = ext::utils::get_unit(Arc::clone(&resolution))?;

    eprintln!("\nComputing products a * _");
    eprintln!("\nEnter a:");

    let a = Bidegree::n_s(
        query::raw("n of Ext class a", str::parse),
        query::raw("s of Ext class a", str::parse::<std::num::NonZeroU32>).get(),
    );

    unit.compute_through_stem(a);

    let a_class = query::vector("Input Ext class a", unit.number_of_gens_in_bidegree(a));

    // The product shifts the bidegree by this amount
    let shift = a;

    if !is_unit {
        unit.compute_through_stem(shift);
    }

    if !resolution.has_computed_bidegree(shift + Bidegree::s_t(0, resolution.min_degree())) {
        eprintln!("No computable bidegrees");
        return Ok(());
    }

    for c in resolution.iter_stem() {
        println!("a = {}, c = {}", a, c);
        if !resolution.has_computed_bidegree(c + shift) {
            continue;
        }

        let num_gens = resolution.number_of_gens_in_bidegree(c);
        let product_num_gens = resolution.number_of_gens_in_bidegree(a + c);
        println!("c = {}, num_gens = {}, product_num_gens = {}", c, num_gens, product_num_gens);
        if num_gens == 0 {
            continue;
        }

        let mut product = AugmentedMatrix::<2>::new(p, num_gens, [product_num_gens, num_gens]);
        product.segment(1, 1).add_identity();

        let mut matrix = Matrix::new(p, num_gens, 1);
        for idx in 0..num_gens {
            let hom = Arc::new(ResolutionHomomorphism::new(
                String::new(),
                Arc::clone(&resolution),
                Arc::clone(&unit),
                c,
            ));

            matrix[idx].set_entry(0, 1);
            hom.extend_step(c, Some(&matrix));
            matrix[idx].set_entry(0, 0);

            hom.extend_through_stem(c + shift);

            //println!("a_class = {:?}", a_class);
            for (k, &v) in a_class.iter().enumerate() {
                //println!("k = {}, v = {}", k, v);
                if v != 0 {
                    let gen = BidegreeGenerator::new(a, k);
                    //println!("product[{}] = {}, gen = {}", idx, product[idx], gen);
                    hom.act(product[idx].slice_mut(0, product_num_gens), v, gen);
                }
            }
        }
        //println!("c = {}, product_num_gens = {}, product = {}", c, product_num_gens, product.inner);
        product.row_reduce();

        for (i,row) in product.iter().enumerate() {
            println!("row = {}", row);
            let row_trunc = row.slice(0, product_num_gens);
            if !row_trunc.iter().all(|x| x == 0) {
                println!(
                    "a * x_({}, {}, {}) = {}",
                    c.n(),
                    c.s(),
                    i,
                    row_trunc
                );
            }
        }
    }

    Ok(())
}
