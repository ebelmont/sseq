//! Decoding d₂ differentials into secondary Steenrod operation equations.
//!
//! Given a bounded Steenrod module M and a seed lift h: M₂ → M₀, this module traces each
//! nonzero d₂(generator in M₂) through the chain M₂ →^{d₂} M₁ →^{d₁} M₀ →^{d₀} H*(X)
//! to express the differential as composites of Steenrod operations acting on cohomology
//! classes.

use std::{collections::BTreeMap, io};

use algebra::{
    module::{
        Module,
        homomorphism::{FreeModuleHomomorphism, ModuleHomomorphism},
    },
    pair_algebra::PairAlgebra,
    Algebra,
};
use fp::vector::FpVector;

use crate::chain_complex::{AugmentedChainComplex, BoundedChainComplex, ChainComplex, FreeChainComplex};

/// A secondary Steenrod operation equation for a bounded module.
///
/// Represents the equation: Σ [composite_ops] · x_gen = rhs
/// where the sum is over composites of Steenrod operations grouped by
/// which M₀ generator they factor through.
pub struct SecondaryEquation {
    /// Generator degree in M₂.
    pub gen_degree: i32,
    /// Generator index in M₂.
    pub gen_index: usize,
    /// Terms grouped by target M₀ generator (gen_deg, gen_idx) → list of operation strings.
    pub terms: BTreeMap<(i32, usize), Vec<String>>,
    /// Right-hand side expressed in H*(X).
    pub rhs: String,
}

impl SecondaryEquation {
    /// Write this equation in human-readable form.
    pub fn write(&self, writer: &mut dyn io::Write) -> io::Result<()> {
        let max_line = 72;
        let mut first_gen = true;
        for (&(gen_deg, gen_idx), composites) in &self.terms {
            let prefix = if first_gen { "  " } else { "+ " };
            first_gen = false;

            let mut bracket_lines: Vec<String> = Vec::new();
            let mut current_line = String::from("[ ");
            for (i, op) in composites.iter().enumerate() {
                let addition = if i == 0 {
                    op.clone()
                } else {
                    format!(" + {op}")
                };
                if current_line.len() + addition.len() > max_line {
                    bracket_lines.push(current_line);
                    current_line = format!("  + {op}");
                } else {
                    current_line.push_str(&addition);
                }
            }
            current_line.push_str(&format!(" ] · x_({gen_deg},{gen_idx})"));
            bracket_lines.push(current_line);

            for (li, line) in bracket_lines.iter().enumerate() {
                if li == 0 {
                    writeln!(writer, "{prefix}{line}")?;
                } else {
                    writeln!(writer, "  {line}")?;
                }
            }
        }
        writeln!(writer, "= {}", self.rhs)
    }

    /// Write this equation with M₀ generator names mapped to H*(X) classes.
    pub fn write_with_names(
        &self,
        writer: &mut dyn io::Write,
        gen_names: &BTreeMap<(i32, usize), String>,
    ) -> io::Result<()> {
        let max_line = 72;
        let mut first_gen = true;
        for (&(gen_deg, gen_idx), composites) in &self.terms {
            let x_name = gen_names
                .get(&(gen_deg, gen_idx))
                .map(|s| s.as_str())
                .unwrap_or("?");
            let prefix = if first_gen { "  " } else { "+ " };
            first_gen = false;

            let mut bracket_lines: Vec<String> = Vec::new();
            let mut current_line = String::from("[ ");
            for (i, op) in composites.iter().enumerate() {
                let addition = if i == 0 {
                    op.clone()
                } else {
                    format!(" + {op}")
                };
                if current_line.len() + addition.len() > max_line {
                    bracket_lines.push(current_line);
                    current_line = format!("  + {op}");
                } else {
                    current_line.push_str(&addition);
                }
            }
            current_line.push_str(&format!(" ] · {x_name}"));
            bracket_lines.push(current_line);

            for (li, line) in bracket_lines.iter().enumerate() {
                if li == 0 {
                    writeln!(writer, "{prefix}{line}")?;
                } else {
                    writeln!(writer, "  {line}")?;
                }
            }
        }
        writeln!(writer, "= {}", self.rhs)
    }
}

/// Decode all d₂ differentials into secondary Steenrod operation equations.
///
/// For each generator g in M₂ with d₂(g) ≠ 0, traces the differential through
/// M₂ →^{d₂} M₁ →^{d₁} M₀ →^{d₀} H*(X) to express the d₂ as composites of
/// Steenrod operations on cohomology classes.
///
/// # Arguments
/// - `resolution`: The resolution of the module.
/// - `seed_lift`: The lift of the seed (h: M₂ → M₀) through the augmentation.
///
/// # Type Parameters
/// - `CC`: Resolution type.
/// - `M`: Target module type (the module being resolved).
#[tracing::instrument(skip_all)]
pub fn decode_secondary_ops<CC, M>(
    resolution: &CC,
    seed_lift: &FreeModuleHomomorphism<CC::Module>,
) -> Vec<SecondaryEquation>
where
    CC: FreeChainComplex + AugmentedChainComplex<ChainMap = FreeModuleHomomorphism<M>>,
    CC::Algebra: PairAlgebra,
    M: Module<Algebra = CC::Algebra>,
    <CC as AugmentedChainComplex>::TargetComplex: BoundedChainComplex<Module = M>,
{
    let target_cc = resolution.target();
    let module = target_cc.module(0);
    let source = resolution.module(2);
    let m0 = resolution.module(0);
    let m1 = resolution.module(1);
    let d0 = resolution.chain_map(0);
    let d1 = resolution.differential(1);
    let d2_diff = resolution.differential(2);
    let p = source.prime();
    let max_src_deg = source.max_computed_degree();

    // Build M₀ generator → H*(X) name mapping via d₀
    let mut m0_gen_x_names: BTreeMap<(i32, usize), String> = BTreeMap::new();
    for gen_deg in m0.min_degree()..=m0.max_computed_degree() {
        for gen_idx in 0..m0.number_of_gens_in_degree(gen_deg) {
            let d0_out = d0.output(gen_deg, gen_idx);
            let x_str = module.element_to_string(gen_deg, d0_out.as_slice());
            m0_gen_x_names.insert((gen_deg, gen_idx), x_str);
        }
    }

    let mut equations = Vec::new();

    for d in source.min_degree()..=max_src_deg {
        let n = source.number_of_gens_in_degree(d);
        for gi in 0..n {
            let d2_out = d2_diff.output(d, gi);
            if d2_out.is_zero() {
                continue;
            }

            // Collect (outer_op · inner_op) terms grouped by M₀ generator
            let mut gen_terms: BTreeMap<(i32, usize), Vec<String>> = BTreeMap::new();

            for (idx, _coeff) in d2_out.as_slice().iter_nonzero() {
                let m1_opgen = m1.index_to_op_gen(d, idx);
                let outer = m1.algebra().basis_element_to_string(
                    m1_opgen.operation_degree,
                    m1_opgen.operation_index,
                );

                let d1_out = d1.output(m1_opgen.generator_degree, m1_opgen.generator_index);

                for (d1_idx, _) in d1_out.as_slice().iter_nonzero() {
                    let m0_opgen = m0.index_to_op_gen(m1_opgen.generator_degree, d1_idx);
                    let inner = m0.algebra().basis_element_to_string(
                        m0_opgen.operation_degree,
                        m0_opgen.operation_index,
                    );

                    let key = (m0_opgen.generator_degree, m0_opgen.generator_index);
                    let composite = match (outer.as_str(), inner.as_str()) {
                        ("1", "1") => "1".to_string(),
                        ("1", _) => inner.clone(),
                        (_, "1") => outer.clone(),
                        _ => format!("{outer}·{inner}"),
                    };
                    gen_terms.entry(key).or_default().push(composite);
                }
            }

            if gen_terms.is_empty() {
                continue;
            }

            // Compute d₀(seed(g)) for RHS
            let rhs = if d >= seed_lift.min_degree() && d < seed_lift.next_degree() {
                let h_out = seed_lift.output(d, gi);
                if h_out.is_zero() {
                    continue;
                }
                let m0_deg = d - 1; // degree = 1 for the standard setup
                let x_dim = module.dimension(m0_deg);
                let mut d0_out = FpVector::new(p, x_dim);
                d0.apply(d0_out.as_slice_mut(), 1, m0_deg, h_out.as_slice());
                if d0_out.is_zero() {
                    continue;
                }
                module.element_to_string(m0_deg, d0_out.as_slice())
            } else {
                continue;
            };

            equations.push(SecondaryEquation {
                gen_degree: d,
                gen_index: gi,
                terms: gen_terms,
                rhs,
            });
        }
    }

    equations
}

/// Build the M₀ generator → H*(X) name mapping for a resolution.
///
/// This maps each generator (gen_deg, gen_idx) of M₀ to the string representation
/// of its image under d₀ in H*(X).
pub fn m0_generator_names<CC, M>(
    resolution: &CC,
) -> BTreeMap<(i32, usize), String>
where
    CC: FreeChainComplex + AugmentedChainComplex<ChainMap = FreeModuleHomomorphism<M>>,
    CC::Algebra: PairAlgebra,
    M: Module<Algebra = CC::Algebra>,
    <CC as AugmentedChainComplex>::TargetComplex: BoundedChainComplex<Module = M>,
{
    let target_cc = resolution.target();
    let module = target_cc.module(0);
    let m0 = resolution.module(0);
    let d0 = resolution.chain_map(0);

    let mut names = BTreeMap::new();
    for gen_deg in m0.min_degree()..=m0.max_computed_degree() {
        for gen_idx in 0..m0.number_of_gens_in_degree(gen_deg) {
            let d0_out = d0.output(gen_deg, gen_idx);
            let x_str = module.element_to_string(gen_deg, d0_out.as_slice());
            names.insert((gen_deg, gen_idx), x_str);
        }
    }
    names
}
