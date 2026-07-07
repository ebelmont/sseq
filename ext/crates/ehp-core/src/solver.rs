use log::{debug, info};

use crate::constraints::ConstraintSystem;
use crate::gf2::*;
use crate::result::SATResult;

/// Solve the GF(2) constraint system using Gaussian elimination.
///
/// This replaces CryptoMiniSat entirely: since all constraints are linear
/// XOR equations over GF(2), Gaussian elimination is both simpler and faster.
///
/// Returns None if the system is inconsistent.
pub fn solve(system: &ConstraintSystem) -> Option<SATResult> {
    if system.num_vars == 0 {
        info!("solve: no variables to solve");
        return None;
    }

    let (a, b) = system.to_matrix();

    info!(
        "solve: {} variables, {} constraints",
        system.num_vars,
        system.num_constraints()
    );

    let result = gauss_solve(&a, &b);

    if !result.consistent {
        info!("solve: system is INCONSISTENT (no solution)");
        return None;
    }

    let offset = result.solution.unwrap();

    // Identify unknown variables: every variable touched by some kernel
    // vector. This is strictly more than the free columns — a pivot column
    // that depends on a free column is also undetermined (the original
    // computes this as the nonzero columns of the kernel matrix,
    // `find_zero_cols` in sat_backend.py).
    let mut touched = hashbrown::HashSet::new();
    for kv in &result.kernel {
        for idx in vec_support(kv) {
            touched.insert(idx);
        }
    }
    let unknown: Vec<usize> = touched.into_iter().collect();

    // Build echelon form of the kernel for the unknowns
    let kernel_matrix = if result.kernel.is_empty() {
        mat_zero(0, system.num_vars)
    } else {
        mat_from_rows(result.kernel.clone(), system.num_vars)
    };

    let determined = system.num_vars - unknown.len();
    info!(
        "solve: {} / {} variables determined ({} unknown)",
        determined,
        system.num_vars,
        unknown.len()
    );

    // Compute unknown degrees (unique tridegrees that have unknown entries)
    let mut unknown_degrees = Vec::new();
    let mut seen = hashbrown::HashSet::new();
    for &idx in &unknown {
        let var = &system.vars[idx];
        let key = (var.n, var.s, var.f);
        if seen.insert(key) {
            unknown_degrees.push(key);
        }
    }
    debug!("{} tridegrees have unknown entries", unknown_degrees.len());

    Some(SATResult {
        offset,
        unknown: unknown.into_iter().collect(),
        kernel: kernel_matrix,
        vars: system.vars.clone(),
        var_index: system.var_index.clone(),
    })
}
