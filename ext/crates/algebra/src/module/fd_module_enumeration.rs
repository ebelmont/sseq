use std::sync::Arc;

use bivec::BiVec;

use crate::algebra::GeneratedAlgebra;
use super::finite_dimensional_module::FiniteDimensionalModule;

/// One step in the systematic enumeration of FDModules.
pub(crate) struct EnumStep {
    input_deg: i32,
    output_deg: i32,
    has_generator: bool,
    /// Index of the generator operation (only meaningful if has_generator).
    generator_op_idx: usize,
    input_dim: usize,
    output_dim: usize,
    /// Number of free F_2 bits at this step: input_dim * output_dim if has_generator, else 0.
    num_free_bits: usize,
}

/// Build the list of enumeration steps in canonical order (input_deg HIGH-to-LOW,
/// output_deg LOW-to-HIGH), skipping pairs where either dimension is 0.
fn build_enum_steps<A: GeneratedAlgebra>(
    algebra: &A,
    graded_dim: &BiVec<usize>,
) -> Vec<EnumStep> {
    let min_deg = graded_dim.min_degree();
    let max_deg = graded_dim.len(); // one past the end
    let mut steps = Vec::new();

    for input_deg in (min_deg..max_deg).rev() {
        for output_deg in (input_deg + 1)..max_deg {
            let op_deg = output_deg - input_deg;
            let input_dim = graded_dim[input_deg];
            let output_dim = graded_dim[output_deg];

            if input_dim == 0 || output_dim == 0 {
                continue;
            }

            let gens = algebra.generators(op_deg);
            if gens.is_empty() {
                // Non-generator step: extend_actions + check_validity, no free bits
                steps.push(EnumStep {
                    input_deg,
                    output_deg,
                    has_generator: false,
                    generator_op_idx: 0,
                    input_dim,
                    output_dim,
                    num_free_bits: 0,
                });
            } else {
                for op_idx in gens {
                    steps.push(EnumStep {
                        input_deg,
                        output_deg,
                        has_generator: true,
                        generator_op_idx: op_idx,
                        input_dim,
                        output_dim,
                        num_free_bits: input_dim * output_dim,
                    });
                }
            }
        }
    }

    steps
}

/// Decode a choice integer into output vectors and set generator actions on the module.
fn apply_choice<A: GeneratedAlgebra>(
    module: &mut FiniteDimensionalModule<A>,
    step: &EnumStep,
    choice: u64,
) {
    let op_deg = step.output_deg - step.input_deg;
    for input_idx in 0..step.input_dim {
        let mut output = vec![0u32; step.output_dim];
        for out_idx in 0..step.output_dim {
            let bit_pos = input_idx * step.output_dim + out_idx;
            if (choice >> bit_pos) & 1 == 1 {
                output[out_idx] = 1;
            }
        }
        module.set_action(op_deg, step.generator_op_idx, step.input_deg, input_idx, &output);
    }
}

/// Recursively enumerate all valid FDModules by backtracking over the enumeration steps.
fn enumerate_recursive<A: GeneratedAlgebra>(
    module: FiniteDimensionalModule<A>,
    steps: &[EnumStep],
    mut step_idx: usize,
    results: &mut Vec<FiniteDimensionalModule<A>>,
    progress_step: Option<usize>,
) {
    // 1. Process forced steps (has_generator == false) in-place
    let mut module = module;
    while step_idx < steps.len() && !steps[step_idx].has_generator {
        let step = &steps[step_idx];
        module.extend_actions(step.input_deg, step.output_deg);
        if module.check_validity(step.input_deg, step.output_deg).is_err() {
            return; // prune
        }
        step_idx += 1;
    }

    // 2. If all steps processed, module is valid
    if step_idx >= steps.len() {
        results.push(module);
        return;
    }

    // 3. Generator step — enumerate all 2^num_free_bits choices
    let step = &steps[step_idx];
    let num_choices = 1u64 << step.num_free_bits;
    let progress_interval = if let Some(ps) = progress_step {
        if step_idx == ps {
            std::cmp::max(1, num_choices / 16)
        } else {
            0
        }
    } else {
        0
    };

    for choice in 0..num_choices {
        if progress_interval > 0 && choice % progress_interval == 0 {
            eprintln!(
                "  Progress: {}/{} ({:.0}%), {} valid so far",
                choice,
                num_choices,
                100.0 * choice as f64 / num_choices as f64,
                results.len(),
            );
        }

        let mut m = module.clone();
        apply_choice(&mut m, step, choice);
        m.extend_actions(step.input_deg, step.output_deg);
        if m.check_validity(step.input_deg, step.output_deg).is_err() {
            continue; // prune
        }
        enumerate_recursive(m, steps, step_idx + 1, results, progress_step);
    }
}

impl<A: GeneratedAlgebra> FiniteDimensionalModule<A> {
    /// Systematically enumerate all valid FDModules with the given graded dimensions.
    pub fn enumerate(algebra: &Arc<A>, graded_dim: &BiVec<usize>) -> Vec<Self> {
        // Ensure algebra basis is computed up to the max operation degree
        let degree_difference = graded_dim.len() - graded_dim.min_degree();
        algebra.compute_basis(degree_difference);

        let steps = build_enum_steps(&**algebra, graded_dim);

        let total_free_bits: usize = steps.iter().map(|s| s.num_free_bits).sum();
        let generator_steps: Vec<_> = steps
            .iter()
            .enumerate()
            .filter(|(_, s)| s.has_generator)
            .collect();

        eprintln!(
            "Enumeration steps: {} total, {} generator steps",
            steps.len(),
            generator_steps.len()
        );
        eprintln!(
            "Total free bits: {} (raw search space: 2^{} = {})",
            total_free_bits,
            total_free_bits,
            if total_free_bits <= 63 {
                format!("{}", 1u64 << total_free_bits)
            } else {
                format!("~2^{}", total_free_bits)
            }
        );
        if total_free_bits > 28 {
            eprintln!(
                "WARNING: Large search space (>{} candidates). This may be slow.",
                1u64 << 28
            );
        }

        // Find the first generator step for progress reporting
        let progress_step = generator_steps.first().map(|(idx, _)| *idx);

        let base_module = Self::new(
            Arc::clone(algebra),
            "enum_fd".to_string(),
            graded_dim.clone(),
        );

        let mut results = Vec::new();
        enumerate_recursive(base_module, &steps, 0, &mut results, progress_step);

        eprintln!("Enumeration complete: {} valid modules found.", results.len());
        results
    }
}
