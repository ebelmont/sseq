use std::sync::Arc;

use algebra::module::{
    homomorphism::{HomPullback, ModuleHomomorphism},
    HomModule, Module,
};
use fp::matrix::Subquotient;
use once::OnceBiVec;
use sseq::coordinates::Bidegree;

use super::FreeChainComplex;

/// The cochain complex `Hom_A(C_*, N)` obtained by applying `Hom_A(-, N)` to a
/// free resolution `C_*`. Its cohomology computes `Ext^{s,t}_A(M, N)`.
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

    pub fn min_degree(&self) -> i32 {
        self.modules[0].min_degree()
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

    pub fn module(&self, s: i32) -> &HomModule<M> {
        &self.modules[s]
    }

    pub fn differential(&self, s: i32) -> &HomPullback<M> {
        &self.differentials[s]
    }
}
