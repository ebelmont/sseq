//! Computes the product S^{|a|+|c|+n} --c--> S^{|a|+n} --a--> S^n
//! for a single n and stem(a+c) <= a cutoff
//! 

use std::{path::PathBuf, sync::Arc};

use algebra::module::{Module, SuspensionModule};

use ext::{
    chain_complex::{FiniteChainComplex, FreeChainComplex},
    resolution_homomorphism::UnstableResolutionHomomorphism,
    resolution::UnstableResolution,
};
use fp::matrix::{AugmentedMatrix, Matrix};
use sseq::coordinates::{Bidegree, BidegreeGenerator};

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging();

    let module = Arc::new(ext::utils::query_unstable_module_only()?);
    let p = module.prime();

    let save_dir = {
        let base = query::optional("Module save directory", |x| {
            core::result::Result::<PathBuf, std::convert::Infallible>::Ok(PathBuf::from(x))
        });
        move |shift| {
            base.as_ref().cloned().map(|mut x| {
                x.push(format!("resolution{shift}"));
                x
            })
        }
    };



    eprintln!("\nComputing products --c--> --a--> for all c and all a in a range");

    let sphere1 = query::raw("sphere of Ext class a", str::parse::<std::num::NonZeroI32>).get();
    let ac_max = Bidegree::n_s(
        query::raw("Max stem", str::parse),
        query::raw("Max filtration", str::parse),
    );
    

    /*
     * The map of spheres S^{s3} --> S^{s2} --> S^{s1} where s2 = s1 + a.n and s3 = s2 + c.n
     * S^{|a| + |c| + s1} --> S^{|a| + s1} --> S^{s1} induces a map on
     * cohomology in the other direction. res1 is the resolution of Sigma^{s1} k and res2 is the
     * resolution of Sigma^{|a| + s1} k.
     * We compute the a map of resolutions
     * ... --> res1_{c.s + a.s} --> ... --> res1_{a.s} ----> ... --> Sigma^{s1} k
     * ... --> res2_{c.s}  -------> ... --> Sigma^{s2} k
     * ... --> Sigma^{s3} k
     * a is represented by a class in (s,t) = (a.s, s1 + a.t) in res1
     * c is represented by a class in (s,t) = (c.s, s2 + c.t) in res2
     * The first map of complexes has degree (s,t) = (a.s, s1 + a.t - s2) = (a.s, a.t - a.n).
     * The second map of complexes has degree (s,t) = (c.s, s2 + c.t - s3) = (c.s, c.t - c.n).
     * The product is res1_{c.s + a.s} --> res2_{c.s} --> Sigma^{s3} k.
     */

    let res1: Arc<UnstableResolution<FiniteChainComplex<_>>> =
        Arc::new(UnstableResolution::new_with_save(
            Arc::new(FiniteChainComplex::ccdz(Arc::new(SuspensionModule::new(
                Arc::clone(&module),
                sphere1,
            )))),
            save_dir(sphere1),
        )?);



    for a_n in 0..ac_max.n() {
        for a_s in 0..ac_max.s() {
            let a = Bidegree::n_s(a_n, a_s);
            let sphere2 = sphere1 + a.n();

            let res2: Arc<UnstableResolution<FiniteChainComplex<_>>> =
                Arc::new(UnstableResolution::new_with_save(
                    Arc::new(FiniteChainComplex::ccdz(Arc::new(SuspensionModule::new(
                        Arc::clone(&module),
                        sphere2
                    )))),
                    save_dir(sphere2),
                )?);

            for c_n in 0..(ac_max.n() - a_n) { // so |a+c| < ac_max
                for c_s in 0..(ac_max.s() - a_s) {
                   /* The stable range is a_n <= sphere1 - 2 and c_n <= sphere2 - 2.
                    * We are not interested in multiplying stable*stable, except where both
                    * are the first class in the stable range.
                    */
                    if a_n < sphere1 - 2 && c_n < sphere2 - 2 {
                        continue;
                    }
                    if (a_n == 0 && a_s == 0) || (c_n == 0 && c_s == 0) {
                        continue;
                    }
                    let c = Bidegree::n_s(c_n, c_s);

                    // c is a map res2_{c.s} --> Sigma^{sphere2}k and this is the source
                    // degree
                    res2.compute_through_stem(Bidegree::s_t(c.s(), sphere2 + c.t()));
                    let num_c_classes = res2.number_of_gens_in_bidegree(
                        Bidegree::s_t(c.s(), sphere2 + c.t())
                        );
                    if num_c_classes == 0 {
                        continue;
                    }

                    // ac is a map res1_{c.s + a.s} --> Sigma^{sphere3} k and this is the
                    // source degree
                    let res1_maxdeg = Bidegree::s_t(a.s() + c.s(), sphere2 + c.t() + a.t() - a.n());
                    res1.compute_through_stem(res1_maxdeg);
                    let mut c_class = vec!(0; num_c_classes);
                    for c_idx in 0..num_c_classes {
                        c_class[c_idx] = 1;

                        let num_gens_a = res1.number_of_gens_in_bidegree(Bidegree::s_t(a.s(), sphere1 + a.t()));
                        let product_num_gens = res1.number_of_gens_in_bidegree(res1_maxdeg);
                        if num_gens_a == 0 || product_num_gens == 0 {
                            continue;
                        }

                        let mut product = AugmentedMatrix::<2>::new(p, num_gens_a, [product_num_gens, num_gens_a]);
                        product.segment(1, 1).add_identity();
                        // This represents the map res1_{a.s} --> Sigma^{s2} k. It is used to build the rest of the map res1 --> res2.
                        let mut matrix = Matrix::new(p, num_gens_a, 1);
                        for idx in 0..num_gens_a {
                            let hom = Arc::new(UnstableResolutionHomomorphism::new(
                                String::new(),
                                Arc::clone(&res1),
                                Arc::clone(&res2),
                                // hom maps a class of this degree to a class in degree zero
                                Bidegree::s_t(a.s(), a.t() - a.n())
                            ));

                            matrix[idx].set_entry(0, 1); // set matrix to the standard basis vector e_idx.
                            hom.extend_step( // this is just mapping res1_{a.s} --> res2_0
                                Bidegree::s_t(a.s(), a.t() - a.n() + sphere2), // degree in res1_{a.s} to construct the map (this is the degree that maps to the generator of Sigma^{sphere2}k)
                                Some(&matrix)
                                );

                            matrix[idx].set_entry(0, 0); // reset this entry to help setting matrix to e_{idx+1} on the next iteration

                            hom.extend_through_stem(res1_maxdeg);

                            for (k, &v) in c_class.iter().enumerate() {
                                if v != 0 {
                                    let gen = BidegreeGenerator::new(Bidegree::s_t(c.s(), sphere2 + c.t()), k);
                                    hom.act(product[idx].slice_mut(0, product_num_gens), v, gen);
                                }
                            }
                        }
                        //product.row_reduce();

                        for (i,row) in product.iter().enumerate() {
                            let row_trunc = row.slice(0, product_num_gens);
                            if !row_trunc.iter().all(|x| x == 0) {
                                println!(
                                    "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                                    a.n(),
                                    a.s(),
                                    i,
                                    c.n(),
                                    c.s(),
                                    c_idx,
                                    row_trunc
                                );
                            }
                        }
                        c_class[c_idx] = 0; // reset c_class so we can set it to the next standard basis vector
                    }
                }
            }
        }
    }
    Ok(())
}
