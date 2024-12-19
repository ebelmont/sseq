//! Computes all products in $\Mod_{C\lambda^2}$.

use std::{sync::Arc, vec};

use algebra::module::Module;
use dashmap::DashMap as HashMap;
use ext::{
    chain_complex::{ChainComplex, FreeChainComplex},
    resolution_homomorphism::ResolutionHomomorphism,
    secondary::*,
    utils::QueryModuleResolution,
};
use fp::{matrix::Matrix, prime::ValidPrime, vector::FpVector};
use itertools::Itertools;
use maybe_rayon::prelude::*;
use sseq::{coordinates::Bidegree, Sseq, SseqBasisElement, SseqBasisElementKind};

struct SecondaryHomotopyGroups {
    basis_elements: HashMap<Bidegree, Vec<SecondaryHomotopyBasisElement>>,
}

struct SecondaryHomotopyElement {
    b: Bidegree,
    classical: FpVector,
    lambda: FpVector,
}

impl SecondaryHomotopyElement {
    fn new(b: Bidegree, classical: FpVector, lambda: FpVector) -> Self {
        Self {
            b,
            classical,
            lambda,
        }
    }
}

impl std::fmt::Display for SecondaryHomotopyElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({n},{s}) ({classical} + λ {lambda})",
            n = self.b.n(),
            s = self.b.s(),
            classical = self.classical,
            lambda = self.lambda
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SecondaryHomotopyBasisElement {
    b: Bidegree,
    kind: SecondaryHomotopyBasisElementKind,
    v: FpVector,
}

impl std::fmt::Display for SecondaryHomotopyBasisElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{kind} ({n},{s}){v}",
            kind = self.kind,
            n = self.b.n(),
            s = self.b.s(),
            v = self.v
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SecondaryHomotopyBasisElementKind {
    B,
    Z,
    #[allow(non_camel_case_types)]
    λZ,
    #[allow(non_camel_case_types)]
    λE,
}

impl SecondaryHomotopyBasisElementKind {
    fn is_lambda(&self) -> bool {
        !self.is_classical()
    }

    fn is_classical(&self) -> bool {
        matches!(self, Self::B | Self::Z)
    }
}

impl std::fmt::Display for SecondaryHomotopyBasisElementKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::B => write!(f, "B"),
            Self::Z => write!(f, "Z"),
            Self::λZ => write!(f, "λZ"),
            Self::λE => write!(f, "λE"),
        }
    }
}

impl SecondaryHomotopyGroups {
    fn to_basis_elements(
        &self,
        mut elt: SecondaryHomotopyElement,
    ) -> anyhow::Result<Vec<SecondaryHomotopyBasisElement>> {
        let elt_str = format!("{}", elt);
        let all_basis_elements = self.basis_elements.get(&elt.b).unwrap();

        let mut basis_elements = Vec::new();
        for classical_basis_element in all_basis_elements.iter().filter(|e| e.kind.is_classical()) {
            if elt
                .classical
                .iter()
                .zip(classical_basis_element.v.iter())
                .any(|(c1, c2)| c1 > 0 && c2 > 0)
            {
                elt.classical.add(&classical_basis_element.v, 1);
                basis_elements.push(classical_basis_element.clone());
            }
        }
        assert!(elt.classical.is_zero());

        for lambda_basis_element in all_basis_elements.iter().filter(|e| e.kind.is_lambda()) {
            if elt
                .lambda
                .iter()
                .zip(lambda_basis_element.v.iter())
                .any(|(c1, c2)| c1 > 0 && c2 > 0)
            {
                elt.lambda.add(&lambda_basis_element.v, 1);
                basis_elements.push(lambda_basis_element.clone());
            }
        }
        if !elt.lambda.is_zero() {
            anyhow::bail!(format!(
                "Failed to express {elt_str} as a sum using basis [{all_basis_elements}]",
                all_basis_elements = all_basis_elements.iter().join(", "),
            ));
        } else {
            Ok(basis_elements)
        }
    }
}

#[derive(Clone)]
struct ProductComputationData {
    p: ValidPrime,
    resolution: Arc<QueryModuleResolution>,
    unit: Arc<QueryModuleResolution>,
    is_unit: bool,
    res_lift: Arc<SecondaryResolution<QueryModuleResolution>>,
    unit_lift: Arc<SecondaryResolution<QueryModuleResolution>>,
    res_sseq: Arc<Sseq>,
    unit_sseq: Arc<Sseq>,
    homotopy_groups: Arc<SecondaryHomotopyGroups>,
}

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging();

    let resolution = Arc::new(
        ext::utils::query_module_only("Module", Some(algebra::AlgebraType::Milnor), true).unwrap(),
    );
    let (is_unit, unit) = ext::utils::get_unit(Arc::clone(&resolution))?;
    let p = resolution.prime();

    let max = Bidegree::n_s(
        query::with_default("Max n", "30", str::parse),
        query::with_default("Max s", "7", str::parse),
    );

    let res_lift = Arc::new(SecondaryResolution::new(Arc::clone(&resolution)));

    let span = tracing::Span::current();
    let compute_nth_stem = |n| {
        let unit_lift = if is_unit {
            Arc::clone(&res_lift)
        } else {
            Arc::new(SecondaryResolution::new(Arc::clone(&unit)))
        };

        // Compute E3 page
        let res_sseq = Arc::new(res_lift.e3_page());
        let unit_sseq = if is_unit {
            Arc::clone(&res_sseq)
        } else {
            Arc::new(unit_lift.e3_page())
        };

        let homotopy_groups =
            compute_secondary_homotopy_groups(Arc::clone(&resolution), Arc::clone(&res_sseq));

        let data = ProductComputationData {
            p,
            resolution: Arc::clone(&resolution),
            unit: Arc::clone(&unit),
            is_unit,
            res_lift: Arc::clone(&res_lift),
            unit_lift,
            res_sseq: Arc::clone(&res_sseq),
            unit_sseq,
            homotopy_groups: Arc::new(homotopy_groups),
        };

        let restrict_s = std::env::var("HOMOLOGICAL_DEGREE")
            .ok()
            .and_then(|s| str::parse::<u32>(&s).ok());

        resolution
            .iter_stem()
            .skip(1)
            .filter(|b| b.n() == n && restrict_s.map_or(true, |s| b.s() == s))
            .maybe_par_bridge()
            .try_for_each(|b| -> anyhow::Result<()> {
                let _tracing_guard = span.enter();
                let r = get_r(&res_sseq, b);
                res_sseq
                    .bze_basis(b.n(), b.s() as i32, r)
                    .take_while(|e| e.kind != SseqBasisElementKind::E)
                    .maybe_par_bridge()
                    .try_for_each(|v| compute_products(v, data.clone()))
            })?;
        anyhow::Ok(())
    };

    let half_max_stem = max.n() / 2;
    for n in 0..half_max_stem {
        let current_max = Bidegree::n_s(i32::min(2 * n + 1, max.n()), max.s());
        resolution.compute_through_stem(current_max);
        res_lift.extend_all();

        compute_nth_stem(n)?;
    }

    // We can do the rest in parallel
    resolution.compute_through_stem(max);
    res_lift.extend_all();
    (half_max_stem..=max.n())
        .maybe_par_bridge()
        .try_for_each(compute_nth_stem)?;

    Ok(())
}

fn get_r(sseq: &sseq::Sseq<sseq::Adams>, b: Bidegree) -> i32 {
    let d = sseq.page_data(b.n(), b.s() as i32);
    std::cmp::min(3, d.len() - 1)
}

fn compute_secondary_homotopy_groups(
    resolution: Arc<QueryModuleResolution>,
    res_sseq: Arc<Sseq>,
) -> SecondaryHomotopyGroups {
    let basis_elements = HashMap::new();

    for b in resolution.iter_stem() {
        basis_elements.insert(b, Vec::new());
    }

    for b in resolution.iter_stem() {
        let r = get_r(&res_sseq, b);
        for elt in res_sseq.bze_basis(b.n(), b.s() as i32, r) {
            match elt.kind {
                SseqBasisElementKind::B => {
                    basis_elements
                        .get_mut(&b)
                        .unwrap()
                        .push(SecondaryHomotopyBasisElement {
                            b,
                            kind: SecondaryHomotopyBasisElementKind::B,
                            v: elt.v,
                        });
                }
                SseqBasisElementKind::Z => {
                    basis_elements
                        .get_mut(&b)
                        .unwrap()
                        .push(SecondaryHomotopyBasisElement {
                            b,
                            kind: SecondaryHomotopyBasisElementKind::Z,
                            v: elt.v.clone(),
                        });
                    if b.s() > 0 {
                        let b_prime = b - Bidegree::n_s(0, 1);
                        basis_elements.get_mut(&b_prime).unwrap().push(
                            SecondaryHomotopyBasisElement {
                                b: b_prime,
                                kind: SecondaryHomotopyBasisElementKind::λZ,
                                v: elt.v,
                            },
                        );
                    }
                }
                SseqBasisElementKind::E => {
                    let b_prime = b - Bidegree::n_s(0, 1);
                    basis_elements
                        .get_mut(&b_prime)
                        .unwrap()
                        .push(SecondaryHomotopyBasisElement {
                            b: b_prime,
                            kind: SecondaryHomotopyBasisElementKind::λE,
                            v: elt.v,
                        });
                }
            }
        }
    }

    SecondaryHomotopyGroups { basis_elements }
}

#[tracing::instrument(skip(data), fields(%elt))]
fn compute_products(elt: SseqBasisElement, data: ProductComputationData) -> anyhow::Result<()> {
    let name = format!("{}", elt);
    let shift = Bidegree::n_s(elt.x, elt.y as u32);

    if data.resolution.next_homological_degree() <= shift.s() + 2 {
        eprintln!("Adams filtration of {name} too large, skipping");
        return Ok(());
    }

    let hom = Arc::new(ResolutionHomomorphism::new(
        name.clone(),
        Arc::clone(&data.resolution),
        Arc::clone(&data.unit),
        shift,
    ));

    let matrix = Matrix::from_row(
        data.p,
        elt.v.to_owned(),
        hom.source.number_of_gens_in_bidegree(shift),
    )
    .transpose();

    hom.extend_step(shift, Some(&matrix));
    // let extend_max = shift + generator.degree() + LAMBDA_BIDEGREE;
    let res_max = Bidegree::n_s(
        data.resolution.module(0).max_computed_degree(),
        data.resolution.next_homological_degree() - 1,
    );
    // This is the maximum bidegree of the region containing the entire stem of `generator` and
    // everything to its left.
    // let extend_max = Bidegree::n_s(std::cmp::min(extend_max.n(), res_max.n()), res_max.s());
    // hom.extend_through_stem(extend_max);
    hom.extend_all();

    if !data.is_unit {
        data.unit.compute_through_stem(res_max - shift);
    }

    let hom_lift = SecondaryResolutionHomomorphism::new(
        Arc::clone(&data.res_lift),
        Arc::clone(&data.unit_lift),
        Arc::clone(&hom),
    );

    hom_lift.extend_all();

    let name = hom_lift.name();
    // Iterate through the multiplicand
    for b in data.unit.iter_stem().skip(1) {
        if (b.n(), b.s() as i32) > (elt.x, elt.y) {
            // By symmetry, it's enough to compute products with cycles in degrees at most as large
            // as the generator (in the lexicographic order). The `iter_stem` iterator returns
            // bidegrees that are lexicographically increasing.
            break;
        }

        if (b + shift).n() > 200 {
            // Stems increase, and we are only computing up to 200.
            break;
        }

        // The potential target has to be hit, and we need to have computed (the data need for) the
        // d2 that hits the potential target.
        if !data
            .resolution
            .has_computed_bidegree(b + shift + LAMBDA_BIDEGREE)
        {
            continue;
        }
        if !data
            .resolution
            .has_computed_bidegree(b + shift - Bidegree::s_t(1, 0))
        {
            continue;
        }

        if data.unit.number_of_gens_in_bidegree(b) == 0 {
            continue;
        }

        let r = get_r(data.unit_sseq.as_ref(), b);
        let bze_basis = data
            .unit_sseq
            .bze_basis(b.n(), b.s() as i32, r)
            .collect::<Vec<_>>();

        let target_num_gens = data.resolution.number_of_gens_in_bidegree(b + shift);
        let lambda_num_gens = data
            .resolution
            .number_of_gens_in_bidegree(b + shift + LAMBDA_BIDEGREE);

        if target_num_gens == 0 && lambda_num_gens == 0 {
            continue;
        }

        let b_span = tracing::info_span!("output", b = %b);
        let _b_span_guard = b_span.enter();

        // First print the products with non-surviving classes
        if target_num_gens > 0 {
            let hom_k = hom.get_map((b + shift).s()).hom_k(b.t());
            for gen in bze_basis
                .iter()
                .filter(|e| e.kind == SseqBasisElementKind::E)
            {
                let i = gen.v.first_nonzero().unwrap().0;
                if hom_k[i].iter().any(|x| *x != 0) {
                    let zero_below = FpVector::new(
                        data.p,
                        data.resolution
                            .number_of_gens_in_bidegree(b + shift - LAMBDA_BIDEGREE),
                    );
                    let answer = SecondaryHomotopyElement {
                        b: b + shift - LAMBDA_BIDEGREE,
                        classical: zero_below,
                        lambda: FpVector::from_slice(data.p, &hom_k[i]),
                    };
                    let output_element = data.homotopy_groups.to_basis_elements(answer);
                    if let Err(e) = output_element {
                        println!("Failed to express {name} [λ{gen}]: {e}");
                    } else {
                        println!(
                            "{name} λ[{gen}] = {}",
                            output_element.unwrap().iter().join(" + ")
                        );
                    }
                }
            }
        }

        let subspace_gens = bze_basis
            .iter()
            .filter(|e| e.kind != SseqBasisElementKind::E);

        // Now print the secondary products
        if subspace_gens.clone().count() == 0 {
            continue;
        }

        let mut outputs = vec![
            FpVector::new(data.p, target_num_gens + lambda_num_gens);
            subspace_gens.clone().count()
        ];

        hom_lift.hom_k(
            Some(&data.res_sseq),
            b,
            subspace_gens.clone().map(|e| e.v.as_slice()),
            outputs.iter_mut().map(FpVector::as_slice_mut),
        );

        for (gen, output) in subspace_gens
            .zip_eq(outputs)
            .filter(|(_, v)| v.iter().any(|x| x != 0))
        {
            let output = SecondaryHomotopyElement::new(
                b + shift,
                output.slice(0, target_num_gens).to_owned(),
                output
                    .slice(target_num_gens, target_num_gens + lambda_num_gens)
                    .to_owned(),
            );
            let output_element = data.homotopy_groups.to_basis_elements(output);
            if let Err(e) = output_element {
                println!("Failed to express {name} [{gen}]: {e}");
            } else {
                println!(
                    "{name} [{gen}] = {}",
                    output_element.unwrap().iter().join(" + ")
                );
            }
        }
    }
    Ok(())
}
