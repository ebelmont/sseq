#[cfg(feature = "nassau")]
compile_error!("This example does not support the nassau feature");

use std::sync::Arc;

use algebra::{module::Module, Algebra};
use ext::chain_complex::{AugmentedChainComplex, BoundedChainComplex, ChainComplex};
use hom_cochain_complex::HomCochainComplex;
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    eprintln!("This script checks for realization obstructions in Ext(M, M).");
    eprintln!("It scans the stem -2 line for nonzero groups.");
    eprintln!();

    let resolution = ext::utils::query_module_only("Module", None, false)?;
    let target_cc = resolution.target();

    if target_cc.max_s() > 1 {
        anyhow::bail!("Input must be a module, not a chain complex (max_s must be 1)");
    }

    let module = target_cc.module(0);

    let max_mod_deg = module
        .max_degree()
        .ok_or_else(|| anyhow::anyhow!("Module must be bounded (finite dimensional)"))?;
    let min_mod_deg = module.min_degree();
    let diam = max_mod_deg - min_mod_deg;

    eprintln!("Module degree range: [{min_mod_deg}, {max_mod_deg}], diameter = {diam}");

    // We need to check Ext^{s, s-2}(M, M) for s in [3, diam+2].
    // We also scan stem -1 for informational purposes.
    // s_max covers both: max of (diam + 2) for stem -2, and (diam + 2) for stem -1.
    let s_max = diam + 4;

    // Resolution range: HomCochainComplex needs resolution through
    // stem = hom_max.n() + max_mod_deg, filtration = hom_max.s() + 1
    // We need stem 0 for the self-test (Ext^{0,0}), stem -1 and stem -2 for scans.
    let hom_max = Bidegree::n_s(0, s_max);
    let res_max = hom_max + Bidegree::n_s(max_mod_deg, 1);
    resolution.compute_through_stem(res_max);
    resolution
        .algebra()
        .compute_basis(hom_max.t() + max_mod_deg + 2);

    let hom_cc = HomCochainComplex::new(Arc::new(resolution), Arc::clone(&module));
    hom_cc.compute_through_stem(hom_max);

    // Self-test: Ext^{0,0}(M, M) should be >= 1 (contains the identity)
    let ext_0_0 = hom_cc.homology_dimension(Bidegree::n_s(0, 0));
    assert!(
        ext_0_0 >= 1,
        "Self-test failed: Ext^{{0,0}}(M, M) = {ext_0_0}, expected >= 1"
    );
    eprintln!("Self-test: dim Ext^{{0,0}}(M, M) = {ext_0_0} (OK)");
    eprintln!();

    // Scan stem -2 line: Ext^{s, s-2}(M, M) for s = 3..=diam+2
    eprintln!("=== Stem -2 line (realization obstructions) ===");
    let mut obstructions = Vec::new();
    for s in 3..=diam + 2 {
        let b = Bidegree::n_s(-2, s);
        let dim = hom_cc.homology_dimension(b);
        if dim > 0 {
            eprintln!("  Ext^{{{s}, {}}}(M, M) = {dim}  <-- NONZERO", b.t());
            obstructions.push((s, b.t(), dim));
        } else {
            eprintln!("  Ext^{{{s}, {}}}(M, M) = 0", b.t());
        }
    }
    eprintln!();

    // Scan stem -1 line for informational purposes
    eprintln!("=== Stem -1 line (informational) ===");
    for s in 1..=s_max {
        let b = Bidegree::n_s(-1, s);
        let dim = hom_cc.homology_dimension(b);
        if dim > 0 {
            eprintln!("  Ext^{{{s}, {}}}(M, M) = {dim}", b.t());
        }
    }
    eprintln!();

    // Verdict
    if obstructions.is_empty() {
        println!("REALIZABLE");
        eprintln!("Verdict: All stem -2 groups vanish. No realization obstructions detected.");
    } else {
        println!("INCONCLUSIVE");
        eprintln!(
            "Verdict: Nonzero groups detected on stem -2 line at {} bidegree(s):",
            obstructions.len()
        );
        for (s, t, dim) in &obstructions {
            eprintln!("  (s, t) = ({s}, {t}), dim = {dim}");
        }
        eprintln!("Realization is obstructed or requires further analysis.");
    }

    Ok(())
}

mod hom_cochain_complex {
    use std::sync::Arc;

    use algebra::module::{
        homomorphism::{HomPullback, ModuleHomomorphism},
        HomModule, Module,
    };
    use ext::chain_complex::FreeChainComplex;
    use fp::matrix::Subquotient;
    use once::OnceBiVec;
    use sseq::coordinates::Bidegree;

    pub struct HomCochainComplex<CC: FreeChainComplex, M: Module<Algebra = CC::Algebra>> {
        source: Arc<CC>,
        target: Arc<M>,
        modules: OnceBiVec<Arc<HomModule<M>>>,
        differentials: OnceBiVec<Arc<HomPullback<M>>>,
    }

    impl<CC: FreeChainComplex, M: Module<Algebra = CC::Algebra>> HomCochainComplex<CC, M> {
        pub fn new(source: Arc<CC>, target: Arc<M>) -> Self {
            Self {
                source,
                target,
                modules: OnceBiVec::new(0),
                differentials: OnceBiVec::new(0),
            }
        }

        pub fn compute_through_stem(&self, max: Bidegree) {
            self.modules.extend(max.s() + 1, |s| {
                Arc::new(HomModule::new(
                    self.source.module(s),
                    Arc::clone(&self.target),
                ))
            });
            self.differentials.extend(max.s(), |s| {
                Arc::new(HomPullback::new(
                    Arc::clone(&self.modules[s]),
                    Arc::clone(&self.modules[s + 1]),
                    self.source.differential(s + 1),
                ))
            });
            for (s, module) in self.modules.iter() {
                module.compute_basis(max.n() + s + 1);
            }
            for (s, d) in self.differentials.iter() {
                d.compute_auxiliary_data_through_degree(max.n() + s + 1);
            }
        }

        pub fn homology_dimension(&self, b: Bidegree) -> usize {
            if b.s() == 0 {
                self.differentials[b.s()].kernel(b.t()).unwrap().dimension()
            } else {
                Subquotient::from_parts(
                    self.differentials[b.s()].kernel(b.t()).cloned().unwrap(),
                    self.differentials[b.s() - 1].image(b.t()).cloned().unwrap(),
                )
                .dimension()
            }
        }
    }
}
