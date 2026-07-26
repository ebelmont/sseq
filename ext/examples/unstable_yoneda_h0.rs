//! This version only computes left h0-multiplication.
//! Computing the Yoneda product $E_2^{s_1}(\Sigma^{\theta_1}F_2, \Sigma^{\theta_2}F_2) \cdot E_2^{s_2}(\Sigma^{\theta_2}F_2, \Sigma^{\theta_3}F_2) \to E_2^{s_1+s_2}(\Sigma^{\theta_1}F_2, \Sigma^{\theta_3}F_2)$
//! for given $\theta_1$ and all $\theta_2, \theta_3, s_1, s_2$ in a range of degrees. Here $E_2^s$ means $Ext^{s,0}$ in the (s, t) grading.
//! The output is formatted as tsv, where the fields are:
//! theta1, theta2, theta3, s1, index1, s2, index2, product
//! where index is the index of the basis element, and the product is expressed as a vector (linear
//! combination of basis elements in the degree of the product).
//! See the comment below for more on what this means.
//! 

use std::{path::PathBuf, sync::Arc};

use algebra::module::{Module, SuspensionModule};
use algebra::SteenrodAlgebra;

use ext::{
    chain_complex::{ChainComplex, FiniteChainComplex, FreeChainComplex},
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
    res_max_deg: Bidegree,
    resolution1: Arc<UnstableResolution<FiniteChainComplex<SuspensionModule<Box<dyn Module<Algebra = SteenrodAlgebra>>>>>>,
    resolution2: Arc<UnstableResolution<FiniteChainComplex<SuspensionModule<Box<dyn Module<Algebra = SteenrodAlgebra>>>>>>,
    th1: i32,
    th2: i32,
    s1: i32,
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


    eprintln!("\nComputing the Yoneda product E_2^(s1,0)(H^* S^(th1), H^* S^(th2)) . E_2^(s2,0)(H^* S^(th2), H^* S^(th3))");
    eprintln!("for your choice of th1, and all (th2, th3, s1, s2) such that the corresponding");
    eprintln!("Ytilde product lands in internal degree <= prod_max_t for your choice of prod_max_t.");


    /*
     * Write e_n := Sigma^n F_2.
     * Resolve e_{th1} and e_{th2}.
     * We compute the a map of resolutions
     * ... --> res1_{s1 + s2} --> ... --> res1_{s1} ----> ... --> e_{th1}
     * ... --> res2_{s2}  -------> ... --> e_{th2}
     * ... --> e_{th3}.
     * elt1 is represented by a class in (s,t) = (s1, th2) in res1
     * elt2 is represented by a class in (s,t) = (s2, th3) in res2
     * The first map of complexes has degree (s,t) = (s1, 0).
     * The second map of complexes has degree (s,t) = (s2, 0).
     * The product is res1_{s1 + s2} --> res2_{s2} --> e_{th3}
     * and has (s,t) = (s1 + s2, th3).
     * The variable names in the code refer to a = elt1, c = elt2.
     *
     * The product that converges to (extended) composition in homotopy is the Ytilde product,
     * which is suspension composed with what we compute here. We want to restrict degrees here so
     * that the output of Ytilde is restricted to t <= (input degree).
     *
     * Ytilde: Ext^{s1}(e_{th1}, e_{n1+th1+s1}) x Ext^{s2}(e_{th2}, e_{n2+th2+s2})
     *    --suspend--> Ext^{s1}(e_{th1}, e_{n1+th1+s1}) x Ext^{s2}(e_{n1+th1+s1})
     *    --Yoneda--> Ext^{s1+s2}(e_{th1}, e_{n1+n2+th1+s1+s2})
     * where n = stem and s = filtration. So our th2 = n1+th1+s1 and our th3 = n1+n2+th1+s1+s2.
     *
     * t in the Ytilde product = stem+filtration = n1+n2+s1+s2 = th3 - th1.
     * With th1 fixed, this means we want to restrict:
     * th1 <= th3 <= prod_max_t + th1
     * th1 <= th2 <= th3
     *
     * To find the cutoff for filtration, we use the Adams vanishing line,
     * which says (with a little wiggle room) that s <= 1/2(t-s) + 5, or s <= (t+10)/3.
     * This means an upper bound for Adams filtration of the Ytilde product is
     * s1 + s2 <= (prod_max_t+10)/3, and Ytilde filtration = Y filtration.
     *
     */


    let prod_max_t : i32 = query::raw("Max internal degree (prod_max_t)", str::parse);
    let th1 = query::raw("th1", str::parse::<std::num::NonZeroI32>).get();

    /*
     * We need res1 up to degree (s,t) = (s1+s2, th3). Compute the maximum up front.
     */
    let max_s = (prod_max_t+10)/3+1;
    let res_max_deg = Bidegree::s_t(max_s as u32, prod_max_t + th1);
    
    let res1: Arc<UnstableResolution<FiniteChainComplex<_>>> =
        Arc::new(UnstableResolution::new_with_save(
            Arc::new(FiniteChainComplex::ccdz(Arc::new(SuspensionModule::new(
                Arc::clone(&module),
                th1,
            )))),
            save_dir(&save_dir_name, th1),
        )?);

    // ac is a map res1_{s1 + s2} --> e_{th3} and the source degree is
    // (s,t) = (s1 + s2, th3).
    // We compute this for the maximum values up front.
    //let res1_maxdeg = Bidegree::s_t(prod_max.s(), prod_max.t());
    res1.compute_through_bidegree(res_max_deg);

	let th2 = th1 + 1;
	let res2: Arc<UnstableResolution<FiniteChainComplex<_>>> =
		Arc::new(UnstableResolution::new_with_save(
			Arc::new(FiniteChainComplex::ccdz(Arc::new(SuspensionModule::new(
				Arc::clone(&module),
				th2,
			)))),
			save_dir(&save_dir_name, th2),
		)?);
	// We use res2 in degree (s,t) = (s2, th3). Compute this in the maximum degrees
	// needed.
	res2.compute_through_bidegree(res_max_deg);

	let s1 = 1;
	let data = CompositeComputationData {
		p, 
		res_max_deg,
		resolution1: Arc::clone(&res1),
		resolution2: Arc::clone(&res2),
		th1,
		th2,
		s1: s1 as i32,
	};


	let s2_max = res_max_deg.s() - s1;
	(0..s2_max+1)
		.maybe_par_bridge()
		.try_for_each(|s2| {
			compute_composites(s2 as i32, data.clone())
		})?;
    Ok(())
}

fn compute_composites(
    s2: i32,
    data: CompositeComputationData,
    ) -> anyhow::Result<()> {
    let res_max_deg = data.res_max_deg;
    let res1 = data.resolution1;
    let res2 = data.resolution2;
    let th1 = data.th1;
    let th2 = data.th2;
    let s1 = data.s1;


    // Want (stem of target) = th3-(s1+s2)-th1 <= prod_max.n()
    for th3 in th2..res_max_deg.t() {
       /* The stable range is stem <= sphere - 2. A class in Ext^{s1, 0}(e_{th1}, e_{th2})
        * converges to a class in [S^{th2-s1}, S^{th1}] with stem th2 - s1 - th1 and sphere S^{th1}.
        * The other class has stem = th3 - s2 - th2 and sphere S^{th2}.
        * We are not interested in multiplying stable*stable, except where both
        * are the first class in the stable range.
        */
        /*if th2 - s1 - th1 < th1 - 2 && th3 - s2 - th2 < th2 - 2 {
            continue;
        }
        if (th2 - s1 - th1 == 0 && s1 == 0) || (th3 - s2 - th2 == 0 && s2 == 0) {  // Don't multiply by degree (stem,filt) = (0,0)
            continue;
        }*/
        let num_c_classes = res2.number_of_gens_in_bidegree(
            Bidegree::s_t(s2 as u32, th3)
            );
        /*let num_a_classes = res1.number_of_gens_in_bidegree(
            Bidegree::s_t(s1 as u32, th2)
            );*/
        if num_c_classes == 0 {
            /*println!(
                "{}\t{}\t{}\t{}\t*\t{}\t*\t0",
                th1,
                th2, 
                th3, 
                s1,
                s2,
            );*/
            continue;
        }


        let mut c_class = vec!(0; num_c_classes);
        for c_idx in 0..num_c_classes {
            c_class[c_idx] = 1;

            let prod_deg = Bidegree::s_t((s1 + s2) as u32, th3);
            let num_gens_a = res1.number_of_gens_in_bidegree(Bidegree::s_t(s1 as u32, th2));
            let product_num_gens = res1.number_of_gens_in_bidegree(prod_deg);
            if num_gens_a == 0 {
                /*println!(
                    "{}\t{}\t{}\t{}\t*\t{}\t{}\t0",
                    th1,
                    th2, 
                    th3, 
                    s1,
                    s2,
                    c_idx
                );*/
                continue;
            }
            if product_num_gens == 0 {
                /*println!(
                    "{}\t{}\t{}\t{}\t*\t{}\t{}\t0",
                    th1,
                    th2, 
                    th3, 
                    s1,
                    s2,
                    c_idx
                );*/
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
                    Bidegree::s_t(s1 as u32, 0)
                ));

                matrix[idx].set_entry(0, 1); // set matrix to the standard basis vector e_idx.
                hom.extend_step( // this is just mapping res1_{s1} --> res2_0
                    Bidegree::s_t(s1 as u32, th2), // degree in res1_{s1} to construct the map (this is the degree that maps to the generator of e_{th2})
                    Some(&matrix)
                    );

                matrix[idx].set_entry(0, 0); // reset this entry to help setting matrix to the next standard basis vector e_{idx+1} on the next iteration

                hom.extend_through_stem(prod_deg);

                for (k, &v) in c_class.iter().enumerate() {
                    if v != 0 {
                        let gen = BidegreeGenerator::new(Bidegree::s_t(s2 as u32, th3), k);  // degree in res2_{s2} that maps to e_{th3}
                        hom.act(product[idx].slice_mut(0, product_num_gens), v, gen);
                    }
                }
            }

            for (i,row) in product.iter().enumerate() {
                let row_trunc = row.slice(0, product_num_gens);
                if !row_trunc.iter().all(|x| x == 0) {
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        th1,
                        th2, 
                        th3, 
                        s1,
                        i,
                        s2,
                        c_idx,
                        row_trunc
                    );
                }
                /*else {
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t0",
                        th1,
                        th2, 
                        th3, 
                        s1,
                        i,
                        s2,
                        c_idx,
                    );
                }*/
            }
            c_class[c_idx] = 0; // reset c_class so we can set it to the next standard basis vector
        }
    }
    Ok(())
}
