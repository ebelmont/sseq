use log::{debug, info};

use crate::constraints::ConstraintSystem;
use crate::gf2::*;
use crate::result::SATResult;
use fp::vector::FpVector;

/// Solve the GF(2) constraint system using Gaussian elimination.
///
/// This replaces CryptoMiniSat entirely: since all constraints are linear
/// XOR equations over GF(2), Gaussian elimination is both simpler and faster.
///
/// Returns None if the system is inconsistent.
///
/// Solver selection (env `EHP_SOLVER`, read once): unset/other = the
/// original per-column elimination below; `dense` = one augmented
/// `fp::Matrix` reduced with the M4RI-based `row_reduce` (experimental —
/// same canonical RREF, so results are identical, but the reduction and
/// scans run on the optimized dense kernel); `verify` = run BOTH and
/// compare offset/kernel/unknowns byte-for-byte (loud warning on mismatch),
/// returning the classic result. Timings for each are logged at info level.
pub fn solve(system: &ConstraintSystem) -> Option<SATResult> {
    match solver_mode() {
        "dense" => return solve_dense(system),
        "verify" => {
            let t0 = std::time::Instant::now();
            let classic = solve_classic(system);
            let t_classic = t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            let dense = solve_dense(system);
            let t_dense = t1.elapsed().as_secs_f64();
            info!(
                "EHP_SOLVER=verify: classic {:.2}s, dense {:.2}s",
                t_classic, t_dense
            );
            match (&classic, &dense) {
                (None, None) => {}
                (Some(c), Some(d)) => {
                    let same = c.offset == d.offset
                        && c.kernel == d.kernel
                        && c.unknown == d.unknown;
                    if !same {
                        eprintln!(
                            "*** EHP_SOLVER=verify MISMATCH: dense solver disagrees with \
                             classic (offset {} kernel {} unknown {}) — using classic; \
                             report this ***",
                            c.offset == d.offset,
                            c.kernel == d.kernel,
                            c.unknown == d.unknown,
                        );
                    }
                }
                _ => eprintln!(
                    "*** EHP_SOLVER=verify MISMATCH: consistency disagreement \
                     (classic {:?}, dense {:?}) — using classic; report this ***",
                    classic.is_some(),
                    dense.is_some(),
                ),
            }
            return classic;
        }
        _ => {}
    }
    solve_classic(system)
}

fn solver_mode() -> &'static str {
    static MODE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODE.get_or_init(|| std::env::var("EHP_SOLVER").unwrap_or_default())
        .as_str()
}

fn solve_classic(system: &ConstraintSystem) -> Option<SATResult> {
    if system.num_vars == 0 {
        info!("solve: no variables to solve");
        return None;
    }

    info!(
        "solve: {} variables, {} constraints",
        system.num_vars,
        system.num_constraints()
    );

    let result = sparse_gauss_solve(&system.rows, &system.rhs, system.num_vars);

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
        mat_from_rows(&result.kernel, system.num_vars)
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

/// [`solve`], followed by the d²=0 one-leg linearization fixpoint (Tier 1 of
/// the nonlinear-constraint plan; `EHP_D2_LINEAR=0` disables).
///
/// After the linear solve, [`crate::constraints::make_d2_linear_rows`]
/// derives the linear consequences of d∘d = 0 from the determined leg of
/// each composable pair; they are folded in incrementally via
/// [`crate::interpage::update_sat_result`] (kernel augmentation — no
/// re-solve), which can determine further entries, enabling further rows —
/// iterated to a fixpoint.
///
/// Returns `(result, newly_determined_by_d2)`; `None` means the system is
/// genuinely inconsistent — either the base linear system, or d²=0 against
/// values it already determined (a real contradiction, since every row is a
/// true consequence of d∘d = 0 at clean degrees).
pub fn solve_with_d2(
    page: &crate::page::SATPage,
    system: &ConstraintSystem,
) -> Option<(SATResult, usize)> {
    let mut result = solve(system)?;
    if !crate::constraints::d2_linearize_enabled() {
        return Some((result, 0));
    }
    let mut emitted: hashbrown::HashSet<Vec<usize>> = hashbrown::HashSet::new();
    let mut total_new = 0usize;
    for pass in 0..16 {
        let fresh: Vec<(Vec<usize>, bool)> =
            crate::constraints::make_d2_linear_rows(page, &result)
                .into_iter()
                .filter(|(idxs, _)| emitted.insert(idxs.clone()))
                .collect();
        if fresh.is_empty() {
            break;
        }
        debug!(
            "d2 linearization pass {}: {} fresh rows",
            pass + 1,
            fresh.len()
        );
        match crate::interpage::update_sat_result(&result, &fresh, &[]) {
            Some((newly, updated)) => {
                total_new += newly.len();
                result = updated;
                if newly.is_empty() {
                    break;
                }
            }
            None => {
                info!(
                    "d2 linearization: INCONSISTENT on E_{} — d∘d = 0 contradicts \
                     the determined differentials (genuine contradiction)",
                    page.r,
                );
                return None;
            }
        }
    }
    if total_new > 0 {
        info!(
            "d2 linearization determined {} additional entries on E_{}",
            total_new, page.r,
        );
    }
    Some((result, total_new))
}

/// Experimental dense solver (`EHP_SOLVER=dense`): the whole system as ONE
/// augmented `fp::Matrix` `[A | b]`, reduced with fp's M4RI-based
/// `row_reduce` instead of the per-column elimination in `gf2.rs` (whose
/// pivot searches and consistency/kernel scans read single bits — ~rows×cols
/// `entry()` calls at t=80). RREF is canonical, so every derived quantity
/// (particular solution, kernel basis in free-column order, unknown set) is
/// IDENTICAL to the classic solver's — enforced by `EHP_SOLVER=verify`.
fn solve_dense(system: &ConstraintSystem) -> Option<SATResult> {
    if system.num_vars == 0 {
        info!("solve: no variables to solve");
        return None;
    }
    let ncols = system.num_vars;
    let nrows = system.rows.len();
    info!("solve(dense): {} variables, {} constraints", ncols, nrows);

    // Augmented [A | b].
    let mut aug = mat_zero(nrows, ncols + 1);
    for (i, row) in system.rows.iter().enumerate() {
        let mut r = vec_zero(ncols + 1);
        for &j in row {
            r.set_entry(j, 1);
        }
        if system.rhs[i] {
            r.set_entry(ncols, 1);
        }
        mat_set_row(&mut aug, i, &r);
    }

    aug.row_reduce();
    let pivots = aug.pivots().to_vec();

    // A pivot in the b column is a row [0 … 0 | 1]: inconsistent.
    if pivots[ncols] >= 0 {
        info!("solve(dense): system is INCONSISTENT (no solution)");
        return None;
    }

    let pivot_cols: Vec<usize> = (0..ncols).filter(|&c| pivots[c] >= 0).collect();
    let free_cols: Vec<usize> = (0..ncols).filter(|&c| pivots[c] < 0).collect();

    // Particular solution: pivot variables take the reduced b entries.
    let mut offset = vec_zero(ncols);
    for &c in &pivot_cols {
        let i = pivots[c] as usize;
        if aug.row(i).entry(ncols) != 0 {
            offset.set_entry(c, 1);
        }
    }

    // Kernel basis in free-column order (matching the classic construction):
    // kv_fc[fc] = 1 and kv_fc[c] = RREF[pivot_row(c)][fc]. Built in one pass
    // over the pivot rows' supports instead of per-entry probes.
    let free_pos: hashbrown::HashMap<usize, usize> =
        free_cols.iter().enumerate().map(|(k, &c)| (c, k)).collect();
    let mut kernel: Vec<FpVector> = free_cols
        .iter()
        .map(|&fc| {
            let mut v = vec_zero(ncols);
            v.set_entry(fc, 1);
            v
        })
        .collect();
    for &c in &pivot_cols {
        let i = pivots[c] as usize;
        let row = aug.row(i).to_owned();
        for j in vec_support(&row) {
            if j < ncols {
                if let Some(&k) = free_pos.get(&j) {
                    kernel[k].set_entry(c, 1);
                }
            }
        }
    }

    // Unknown = every variable touched by the kernel space (identical to the
    // classic definition; basis-independent).
    let mut touched = hashbrown::HashSet::new();
    for kv in &kernel {
        for idx in vec_support(kv) {
            touched.insert(idx);
        }
    }
    let unknown: hashbrown::HashSet<usize> = touched;

    let kernel_matrix = if kernel.is_empty() {
        mat_zero(0, ncols)
    } else {
        mat_from_rows(&kernel, ncols)
    };

    info!(
        "solve(dense): {} / {} variables determined ({} unknown)",
        ncols - unknown.len(),
        ncols,
        unknown.len()
    );

    Some(SATResult {
        offset,
        unknown,
        kernel: kernel_matrix,
        vars: system.vars.clone(),
        var_index: system.var_index.clone(),
    })
}

#[cfg(test)]
mod dense_solver_tests {
    use super::*;
    use crate::constraints::{ConstraintSystem, DiffVar};

    /// Classic and dense solvers must agree exactly (RREF is canonical).
    #[test]
    fn dense_matches_classic() {
        // 6 variables, mixed determined/underdetermined system with rhs.
        let vars: Vec<DiffVar> =
            (0..6).map(|i| DiffVar::new(2, 3, 1, 0, i as u16)).collect();
        let mut sys = ConstraintSystem::new(vars);
        for (idxs, rhs) in [
            (vec![0usize, 1], true),
            (vec![1, 2, 3], false),
            (vec![0, 2, 3], true),
            (vec![4], true),
            (vec![3, 5], false),
        ] {
            sys.rows.push(idxs);
            sys.rhs.push(rhs);
        }
        let c = solve_classic(&sys).expect("classic SAT");
        let d = solve_dense(&sys).expect("dense SAT");
        assert_eq!(c.offset, d.offset);
        assert_eq!(c.kernel, d.kernel);
        assert_eq!(c.unknown, d.unknown);

        // Inconsistent variant agrees too.
        sys.rows.push(vec![4]);
        sys.rhs.push(false); // x4 = 1 and x4 = 0
        assert!(solve_classic(&sys).is_none());
        assert!(solve_dense(&sys).is_none());
    }
}
