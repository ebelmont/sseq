//! Computing products $E_2^{s_1,t_1}(S^{th_1}) \cdot E_2^(s_2,t_2)(S^{th_2}) \to E_2^{s_1+s_2, t_1+t_2}(S^{th_1})$
//! for given th1, th2 and all s1, s2, t1, t2.
//! The output is formatted as tsv, where the fields are:
//! th1, stem of elt1, filtration of elt1, basis index of elt1, th2, stem of elt2, filtration of elt2, basis index of elt2,
//! vector representing a linear combination of basis elements in the degree of elt1 . elt2.
//! The order is composition notation order. See the comment below for more on what this means.
//! 

use std::{path::PathBuf, sync::Arc};

use algebra::module::{Module, SuspensionModule};
use algebra::SteenrodAlgebra;

use ext::{
    chain_complex::{FiniteChainComplex, FreeChainComplex},
    resolution_homomorphism::UnstableResolutionHomomorphism,
    resolution::UnstableResolution,
};
use fp::matrix::{AugmentedMatrix, Matrix};
use fp::prime::ValidPrime;
use sseq::coordinates::{Bidegree, BidegreeGenerator};
use maybe_rayon::prelude::*;


#[derive(Clone)]
struct CompositeComputationData {
    p: ValidPrime,
    a: Bidegree,
    ac_max: Bidegree,
    resolution1: Arc<UnstableResolution<FiniteChainComplex<SuspensionModule<Box<dyn Module<Algebra = SteenrodAlgebra>>>>>>,
    resolution2: Arc<UnstableResolution<FiniteChainComplex<SuspensionModule<Box<dyn Module<Algebra = SteenrodAlgebra>>>>>>,
    sphere1: i32,
    sphere2: i32,
}


fn save_dir(name: &str, shift: i32) -> Option<PathBuf> {
    let base = Some(PathBuf::from(name));
        base.as_ref().cloned().map(|mut x| {
            x.push(format!("resolution{shift}"));
            x
        })
}


fn main() -> anyhow::Result<()> {

    tracing_subscriber::fmt::init();
    //ext::utils::init_logging();

    let module = Arc::new(ext::utils::query_unstable_module_only()?);
    let p = module.prime();

    let save_dir_name: String = query::raw("Save dir", str::parse);


    eprintln!("\nComputing products E_2^(s1,t1)(S^(th1)) . E_2^(s2,t2)(S^(th2))");
    eprintln!("for your choice of th1 and th2, and all (s1, t1, s2, t2) in a range.");
    eprintln!("If these are permanent cycles, the product is");
    eprintln!("S^(th1 + n1 - th2) x S^(th2 + n2) --elt2--> S^(th1 + n1 - th2) x S^(th2) --elt1--> S^(th1)");
    eprintln!("where n = t-s.\n\n");

    let sphere1 = query::raw("th1 (sphere for elt1) = ?", str::parse::<std::num::NonZeroI32>).get();
    let sphere2 = query::raw("th2 (sphere for elt2) = ?", str::parse::<std::num::NonZeroI32>).get();

    let ac_max = Bidegree::n_s(
        query::raw("Max stem", str::parse),
        query::raw("Max filtration", str::parse),
    );
    

    /*
     * Write e_n := Sigma^n F_2.
     * The composition product (a la Bousfield-Kan "Pairings..." section 10) in Adams
     * grading is
     * Ext^{th1+n1, f1}(e_{th1}, e_0) . Ext^{th2+n2, f2}(e_{th2}, e_0)
     * which (section 18) can be expressed as a Yoneda product
     * Ext^{m,f1}(e_{th1}, e_{th2}) . Ext^{0,f2}(e_{th2}, e_{th2+n2})
     * --> Ext^{m,f1+f2}(e_{th1}, e_{th2+n2}) where m = th1 - th2 + n1.
     * Write |a| = (n1, f1), |c| = (n2, f2) (Adams grading).
     * res1 is the resolution of e_{th1} and res2 is the resolution of e_{th2}.
     * We compute the a map of resolutions
     * ... --> res1_{c.s + a.s} --> ... --> res1_{a.s} ----> ... --> e_{th1}
     * ... --> res2_{c.s}  -------> ... --> e_{th2}
     * ... --> e_{th2+c.n}.
     * a is represented by a class in (s,t) = (a.s, th1 + a.t) in res1
     * c is represented by a class in (s,t) = (c.s, th2 + c.t) in res2
     * The first map of complexes has degree (s,t) = (a.s, th1 + a.t - th2) = (a.s, m + a.s).
     * The second map of complexes has degree (s,t) = (c.s, c.s).
     * The product is res1_{c.s + a.s} --> res2_{c.s} --> e_{th2+c.n}
     * and has s = c.s+a.s, t = (th2+c.n) + (m+a.s) + c.s = th1 + a.t + c.t.
     */


    let res1: Arc<UnstableResolution<FiniteChainComplex<_>>> =
        Arc::new(UnstableResolution::new_with_save(
            Arc::new(FiniteChainComplex::ccdz(Arc::new(SuspensionModule::new(
                Arc::clone(&module),
                sphere1,
            )))),
            save_dir(&save_dir_name, sphere1),
        )?);

    let res2: Arc<UnstableResolution<FiniteChainComplex<_>>> =
        Arc::new(UnstableResolution::new_with_save(
            Arc::new(FiniteChainComplex::ccdz(Arc::new(SuspensionModule::new(
                Arc::clone(&module),
                sphere2
            )))),
            save_dir(&save_dir_name, sphere2),
        )?);

    // ac is a map res1_{c.s + a.s} --> e_{th2+c.n} and the source degree is
    // (a.s + c.s, a.t + c.t + sphere1).
    // We compute this for the maximum values up front.
    let res1_maxdeg = Bidegree::s_t(ac_max.s(), sphere1 + ac_max.t());
    res1.compute_through_stem(res1_maxdeg);

    for a_n in 0..ac_max.n() {
        for a_s in 0..ac_max.s() {
            //TODO: Check if res1 is zero in degree (s,t) = (a.s, a.t + sphere1)
            let a = Bidegree::n_s(a_n, a_s);
            let cmax_s = ac_max.s() - a.s();
            let cmax_t = ac_max.t() - a.t();

            // c is a map res2_{c.s} --> Sigma^{sphere3}k where sphere3 = sphere2 + c.n
            // and the source degree is (c.s, c.t + sphere2).
            // We compute the max needed.
            res2.compute_through_stem(Bidegree::s_t(cmax_s, cmax_t + sphere2));

            let data = CompositeComputationData {
                p, 
                ac_max,
                a,
                resolution1: Arc::clone(&res1),
                resolution2: Arc::clone(&res2),
                sphere1,
                sphere2,
            };


            let cmax_n = ac_max.n() - a_n;
            (0..cmax_n)
                .maybe_par_bridge()
                .try_for_each(|c_n| {
                    compute_composites(c_n, data.clone())
                })?;
        }
    }
    Ok(())
}

fn compute_composites(
    c_n: i32,
    data: CompositeComputationData,
    ) -> anyhow::Result<()> {
    let a = data.a;
    let res1 = data.resolution1;
    let res2 = data.resolution2;
    let sphere1 = data.sphere1;
    let sphere2 = data.sphere2;
    let m = sphere1 - sphere2 + a.n();

    let cmax_s = data.ac_max.s() - a.s();


    for c_s in 0..cmax_s {
        let c = Bidegree::n_s(c_n, c_s);
       /* The stable range is a_n <= sphere1 - 2 and c_n <= sphere2 - 2.
        * We are not interested in multiplying stable*stable, except where both
        * are the first class in the stable range.
        */
        if a.n() < sphere1 - 2 && c_n < sphere2 - 2 {
            continue;
        }
        if (a.n() == 0 && a.s() == 0) || (c_n == 0 && c_s == 0) {
            continue;
        }
        let num_c_classes = res2.number_of_gens_in_bidegree(
            Bidegree::s_t(c.s(), sphere2 + c.t())
            );
        if num_c_classes == 0 {
            continue;
        }


        let mut c_class = vec!(0; num_c_classes);
        for c_idx in 0..num_c_classes {
            c_class[c_idx] = 1;

            let ac_deg = Bidegree::s_t(a.s() + c.s(), a.t() + c.t() + sphere1);
            let num_gens_a = res1.number_of_gens_in_bidegree(Bidegree::s_t(a.s(), a.t() + sphere1));
            let product_num_gens = res1.number_of_gens_in_bidegree(ac_deg);
            if num_gens_a == 0 || product_num_gens == 0 {
                continue;
            }

            let mut product = AugmentedMatrix::<2>::new(data.p, num_gens_a, [product_num_gens, num_gens_a]);
            product.segment(1, 1).add_identity();
            // This represents the map res1_{a.s} --> Sigma^{s2} k. It is used to build the rest of the map res1 --> res2.
            let mut matrix = Matrix::new(data.p, num_gens_a, 1);
            for idx in 0..num_gens_a {
                let hom = Arc::new(UnstableResolutionHomomorphism::new(
                    String::new(),
                    Arc::clone(&res1),
                    Arc::clone(&res2),
                    // hom maps a class of this degree to a class in degree zero
                    Bidegree::s_t(a.s(), (a.s() as i32) + m)
                ));

                matrix[idx].set_entry(0, 1); // set matrix to the standard basis vector e_idx.
                hom.extend_step( // this is just mapping res1_{a.s} --> res2_0
                    Bidegree::s_t(a.s(), sphere2 + (a.s() as i32) + m), // degree in res1_{a.s} to construct the map (this is the degree that maps to the generator of Sigma^{sphere2}k)
                    Some(&matrix)
                    );

                matrix[idx].set_entry(0, 0); // reset this entry to help setting matrix to the next standard basis vector e_{idx+1} on the next iteration

                hom.extend_through_stem(ac_deg);

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
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        sphere1,
                        a.n(),
                        a.s(),
                        i,
                        sphere2,
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
    Ok(())
}
