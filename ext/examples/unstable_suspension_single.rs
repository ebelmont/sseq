//! Computes the suspension map between different unstable Ext groups.
//!
//! Given an unstable Steenrod module $M$, compute the unstable Ext groups of $\Sigma^k M$ for
//! given $k$ up till the stable range. Each result is printed in the form
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

use std::{path::PathBuf, sync::Arc};

use algebra::module::{Module, SuspensionModule};
use ext::{
    chain_complex::{FiniteChainComplex, FreeChainComplex},
    resolution::UnstableResolution,
    resolution_homomorphism::UnstableResolutionHomomorphism,
};
use fp::vector::FpVector;
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging();

    let module = Arc::new(ext::utils::query_unstable_module_only()?);
    let save_dir = {
        let base = query::optional("Module save directory", |x| {
            core::result::Result::<PathBuf, std::convert::Infallible>::Ok(PathBuf::from(x))
        });
        move |shift| {
            base.as_ref().cloned().map(|mut x| {
                x.push(format!("suspension{shift}"));
                x
            })
        }
    };

    let shift_t = query::raw("Target sphere: ", str::parse::<i32>);

    let max = Bidegree::n_s(
        query::raw("Max n", str::parse),
        query::raw("Max s", str::parse),
    );
    let min_degree = Bidegree::s_t(0, module.min_degree());


    let shift = Bidegree::s_t(0, shift_t);
    let res_a: Arc<UnstableResolution<FiniteChainComplex<_>>> =
        Arc::new(UnstableResolution::new_with_save(
        Arc::new(FiniteChainComplex::ccdz(Arc::new(SuspensionModule::new(
            Arc::clone(&module),
            shift_t - 1,
        )))),
        save_dir(shift_t - 1),
    )?);
    let res_b: Arc<UnstableResolution<FiniteChainComplex<_>>> =
        Arc::new(UnstableResolution::new_with_save(
        Arc::new(FiniteChainComplex::ccdz(Arc::new(SuspensionModule::new(
            Arc::clone(&module),
            shift_t,
        )))),
        save_dir(shift_t),
    )?);

    res_a.compute_through_stem(max + shift);
    res_b.compute_through_stem(max + shift);

    let suspension_shift = Bidegree::s_t(0, 1);
    let hom = UnstableResolutionHomomorphism::new(
        String::from("suspension"),
        Arc::clone(&res_b),
        Arc::clone(&res_a),
        suspension_shift,
    );

    hom.extend_step_raw(
        min_degree + shift,
        Some(vec![FpVector::from_slice(module.prime(), &[1])]),
    );
    hom.extend_all();

    for n in 2 * ((min_degree + shift).n() - 1)..=(max + shift).n() {
        if n < (min_degree + shift).n() {
            continue;
        }
        for s in 0..=max.s() {
            let source = Bidegree::n_s(n, s);
            let target = source - suspension_shift;
            let source_num_gens = res_b.number_of_gens_in_bidegree(source);
            let target_num_gens = res_a.number_of_gens_in_bidegree(target);
            let m = if source_num_gens == 0 || target_num_gens == 0 {
                String::new()
            } else {
                let m = hom.get_map(target.s()).hom_k(target.t());
                if source_num_gens == target_num_gens
                    && m.iter().enumerate().all(|(n, x)| {
                        x.iter()
                            .enumerate()
                            .all(|(m, &z)| if m == n { z == 1 } else { z == 0 })
                    })
                {
                    String::new()
                } else {
                    format!(" - {m:?}")
                }
            };
            println!("{n} {s} {shift_t}: {source_num_gens}{m}", n = n - shift_t);
        }
    }

    Ok(())
}
