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
        "uf" => return solve_uf(system),
        "verify-uf" => {
            let t0 = std::time::Instant::now();
            let classic = solve_classic(system);
            let t_classic = t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            let uf = solve_uf(system);
            let t_uf = t1.elapsed().as_secs_f64();
            info!(
                "EHP_SOLVER=verify-uf: classic {:.2}s, uf {:.2}s",
                t_classic, t_uf
            );
            match (&classic, &uf) {
                (None, None) => {}
                (Some(c), Some(u)) => {
                    // Canonical invariants: unknown set; offset restricted to
                    // determined vars; kernel ROW SPACE (bases may differ).
                    let unknown_ok = c.unknown == u.unknown;
                    let mut offset_ok = true;
                    for i in 0..system.num_vars {
                        if !c.unknown.contains(&i)
                            && (c.offset.entry(i) != 0) != (u.offset.entry(i) != 0)
                        {
                            offset_ok = false;
                            break;
                        }
                    }
                    let mut ck = c.kernel.clone();
                    let mut uk = u.kernel.clone();
                    ck.row_reduce();
                    uk.row_reduce();
                    let kernel_ok = c.kernel.rows() == u.kernel.rows() && ck == uk;
                    if !(unknown_ok && offset_ok && kernel_ok) {
                        eprintln!(
                            "*** EHP_SOLVER=verify-uf MISMATCH (unknown {} offset {} kernel {}) — using classic; report this ***",
                            unknown_ok, offset_ok, kernel_ok,
                        );
                    }
                }
                _ => eprintln!(
                    "*** EHP_SOLVER=verify-uf MISMATCH: consistency disagreement (classic {:?}, uf {:?}) — using classic; report this ***",
                    classic.is_some(),
                    uf.is_some(),
                ),
            }
            return classic;
        }
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

    // Augmented [A | b], filled directly from the sparse rows.
    let mut aug = mat_zero(nrows, ncols + 1);
    for (i, row) in system.rows.iter().enumerate() {
        let mut r = aug.row_mut(i);
        for &j in row.iter() {
            r.set_entry(j as usize, 1);
        }
        if system.rhs[i] {
            r.set_entry(ncols, 1);
        }
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
        mat_from_rows(kernel, ncols)
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
            sys.add_constraint_indices(&idxs, rhs);
        }
        let c = solve_classic(&sys).expect("classic SAT");
        let d = solve_dense(&sys).expect("dense SAT");
        assert_eq!(c.offset, d.offset);
        assert_eq!(c.kernel, d.kernel);
        assert_eq!(c.unknown, d.unknown);

        // The union-find solver agrees on the canonical invariants.
        let u = solve_uf(&sys).expect("uf SAT");
        assert_eq!(c.unknown, u.unknown);
        for i in 0..6 {
            if !c.unknown.contains(&i) {
                assert_eq!(c.offset.entry(i), u.offset.entry(i), "offset var {}", i);
            }
        }
        let (mut ck, mut uk) = (c.kernel.clone(), u.kernel.clone());
        ck.row_reduce();
        uk.row_reduce();
        assert_eq!(c.kernel.rows(), u.kernel.rows());
        assert_eq!(ck, uk);

        // Inconsistent variant agrees too.
        sys.add_constraint_indices(&[4], false); // x4 = 1 and x4 = 0
        assert!(solve_classic(&sys).is_none());
        assert!(solve_dense(&sys).is_none());
        assert!(solve_uf(&sys).is_none());
    }
}

// =============================================================================
// Parity union-find solver (`EHP_SOLVER=uf`)
// =============================================================================

/// Union-find with edge parities: `find(x)` returns `(root, p)` with
/// `x = root XOR p` under the parity relations processed so far.
struct ParityUF {
    parent: Vec<u32>,
    /// Parity of the edge to the parent.
    par: Vec<bool>,
    rank: Vec<u8>,
}

impl ParityUF {
    fn new(n: usize) -> Self {
        ParityUF {
            parent: (0..n as u32).collect(),
            par: vec![false; n],
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> (usize, bool) {
        // Two passes: locate root, then compress parities along the path.
        let mut r = x;
        let mut p = false;
        while self.parent[r] as usize != r {
            p ^= self.par[r];
            r = self.parent[r] as usize;
        }
        let root = r;
        let mut cur = x;
        let mut cur_p = false;
        while self.parent[cur] as usize != cur {
            let next = self.parent[cur] as usize;
            let next_p = cur_p ^ self.par[cur];
            self.parent[cur] = root as u32;
            let old = self.par[cur];
            self.par[cur] = p ^ cur_p; // parity from cur to root
            let _ = old;
            cur = next;
            cur_p = next_p;
        }
        (root, p)
    }

    /// Impose `a XOR b = parity`. Returns false on contradiction... never —
    /// contradictions surface through assignments; a redundant consistent
    /// union is a no-op, an inconsistent one is caught by the caller folding
    /// values. This union itself cannot fail: if roots are equal the caller
    /// must check `pa ^ pb == parity`.
    fn union(&mut self, a: usize, b: usize, parity: bool) -> Result<(), ()> {
        let (ra, pa) = self.find(a);
        let (rb, pb) = self.find(b);
        if ra == rb {
            return if pa ^ pb == parity { Ok(()) } else { Err(()) };
        }
        // a = ra^pa, b = rb^pb, want a^b = parity  =>  ra ^ rb = parity^pa^pb
        let edge = parity ^ pa ^ pb;
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb as u32;
            self.par[ra] = edge;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra as u32;
            self.par[rb] = edge;
        } else {
            self.parent[rb] = ra as u32;
            self.par[rb] = edge;
            self.rank[ra] += 1;
        }
        Ok(())
    }
}

/// Parity union-find solver: exploits the measured shape of EHP constraint
/// systems (~97% of rows have support ≤ 2 — unit pins and two-variable
/// parity equations). Pins and parity edges are absorbed into a
/// `ParityUF` + per-root assignments in O(nnz α); rows with reduced support
/// ≥ 3 are folded over class representatives and solved as a SMALL dense
/// residual with the M4RI path. Peak memory is O(num_vars) plus the residual
/// square — megabytes where the dense augmented matrix would be tens of GB
/// at t=130.
///
/// The returned solution is EQUIVALENT to the classic solver's (same
/// determined variables with the same values, same unknown set, same kernel
/// row space) but the kernel BASIS and the offset's values on unknown
/// coordinates may differ — both are non-canonical choices. Everything
/// downstream (effective values, page turning, update_sat_result) depends
/// only on the invariant parts; `EHP_SOLVER=verify-uf` checks exactly those.
fn solve_uf(system: &ConstraintSystem) -> Option<SATResult> {
    if system.num_vars == 0 {
        info!("solve: no variables to solve");
        return None;
    }
    let n = system.num_vars;
    let mut uf = ParityUF::new(n);
    // Assigned value per ROOT (indexed by variable id; only roots consulted).
    let mut val: Vec<Option<bool>> = vec![None; n];

    // Residual rows (original sparse rows with support >= 3, reprocessed each
    // round through the current union-find state).
    let mut residual: Vec<(Vec<u32>, bool)> = Vec::new();

    // Reduce a row to (sorted root support, rhs) folding parities and known
    // root values. Returns None on immediate contradiction (empty & rhs=1).
    // Processing a reduced row of support 1 assigns; support 2 unions.
    #[derive(Debug)]
    enum RowFate {
        Consumed,
        Deferred(Vec<u32>, bool),
        Contradiction,
    }

    fn process_row(
        idxs: &[u32],
        rhs_in: bool,
        uf: &mut ParityUF,
        val: &mut [Option<bool>],
    ) -> RowFate {
        let mut rhs = rhs_in;
        let mut roots: Vec<u32> = Vec::with_capacity(idxs.len());
        for &i in idxs {
            let (r, p) = uf.find(i as usize);
            rhs ^= p;
            roots.push(r as u32);
        }
        roots.sort_unstable();
        // XOR-cancel duplicate roots.
        let mut folded: Vec<u32> = Vec::with_capacity(roots.len());
        let mut it = roots.into_iter().peekable();
        while let Some(r) = it.next() {
            if it.peek() == Some(&r) {
                it.next();
            } else {
                folded.push(r);
            }
        }
        // Fold assigned roots into the rhs.
        let mut live: Vec<u32> = Vec::with_capacity(folded.len());
        for r in folded {
            match val[r as usize] {
                Some(v) => rhs ^= v,
                None => live.push(r),
            }
        }
        match live.len() {
            0 => {
                if rhs {
                    RowFate::Contradiction
                } else {
                    RowFate::Consumed
                }
            }
            1 => {
                val[live[0] as usize] = Some(rhs);
                RowFate::Consumed
            }
            2 => match uf.union(live[0] as usize, live[1] as usize, rhs) {
                Ok(()) => RowFate::Consumed,
                Err(()) => RowFate::Contradiction,
            },
            _ => RowFate::Deferred(live, rhs),
        }
    }

    // Round 1: all rows. Later rounds: only the deferred residual, until it
    // stops shrinking (assignments/unions from residual reprocessing can
    // cascade).
    for (i, row) in system.rows.iter().enumerate() {
        match process_row(row, system.rhs[i], &mut uf, &mut val) {
            RowFate::Consumed => {}
            RowFate::Deferred(live, rhs) => residual.push((live, rhs)),
            RowFate::Contradiction => {
                info!("solve(uf): system is INCONSISTENT (no solution)");
                return None;
            }
        }
    }
    loop {
        let before = residual.len();
        let mut next_residual = Vec::with_capacity(residual.len());
        for (idxs, rhs) in residual.drain(..) {
            match process_row(&idxs, rhs, &mut uf, &mut val) {
                RowFate::Consumed => {}
                RowFate::Deferred(live, r) => next_residual.push((live, r)),
                RowFate::Contradiction => {
                    info!("solve(uf): system is INCONSISTENT (no solution)");
                    return None;
                }
            }
        }
        residual = next_residual;
        if residual.len() == before {
            break;
        }
    }

    // Values may have been assigned to non-root class members' roots after
    // unions; normalize: an assignment on a root that later got merged INTO
    // another root must be folded. (union() never merges two assigned roots
    // without the caller... it can: both classes unassigned at union time is
    // the common case, but a root assigned earlier can be merged under a new
    // root by a later union — fold all assignments down to current roots.)
    let mut root_val: hashbrown::HashMap<usize, bool> = hashbrown::HashMap::new();
    for i in 0..n {
        if let Some(v) = val[i] {
            let (r, p) = uf.find(i);
            let rv = v ^ p;
            if let Some(&prev) = root_val.get(&r) {
                if prev != rv {
                    info!("solve(uf): system is INCONSISTENT (no solution)");
                    return None;
                }
            } else {
                root_val.insert(r, rv);
            }
        }
    }
    // Re-check the residual against final assignments (a root may have been
    // assigned after its last reprocessing round without shrinking the count).
    let mut dense_rows: Vec<(Vec<u32>, bool)> = Vec::new();
    for (idxs, rhs_in) in &residual {
        let mut rhs = *rhs_in;
        let mut live: Vec<u32> = Vec::new();
        for &i in idxs {
            let (r, p) = uf.find(i as usize);
            rhs ^= p;
            match root_val.get(&r) {
                Some(&v) => rhs ^= v,
                None => live.push(r as u32),
            }
        }
        live.sort_unstable();
        let mut folded: Vec<u32> = Vec::with_capacity(live.len());
        let mut it = live.into_iter().peekable();
        while let Some(r) = it.next() {
            if it.peek() == Some(&r) {
                it.next();
            } else {
                folded.push(r);
            }
        }
        if folded.is_empty() {
            if rhs {
                info!("solve(uf): system is INCONSISTENT (no solution)");
                return None;
            }
            continue;
        }
        dense_rows.push((folded, rhs));
    }

    // Dense residual over the live roots.
    let mut local: hashbrown::HashMap<u32, usize> = hashbrown::HashMap::new();
    for (idxs, _) in &dense_rows {
        for &r in idxs {
            let next = local.len();
            local.entry(r).or_insert(next);
        }
    }
    let m = local.len();
    let mut local_rev: Vec<u32> = vec![0; m];
    for (&r, &li) in &local {
        local_rev[li] = r;
    }

    let mut res_offset: Vec<bool> = vec![false; m];
    let mut res_kernel: Vec<Vec<usize>> = Vec::new(); // supports in local ids
    if m > 0 {
        let mut aug = mat_zero(dense_rows.len(), m + 1);
        for (i, (idxs, rhs)) in dense_rows.iter().enumerate() {
            let mut row = aug.row_mut(i);
            for &r in idxs {
                row.set_entry(local[&r], 1);
            }
            if *rhs {
                row.set_entry(m, 1);
            }
        }
        aug.row_reduce();
        let pivots = aug.pivots().to_vec();
        if pivots[m] >= 0 {
            info!("solve(uf): system is INCONSISTENT (no solution)");
            return None;
        }
        for c in 0..m {
            if pivots[c] >= 0 && aug.row(pivots[c] as usize).entry(m) != 0 {
                res_offset[c] = true;
            }
        }
        // Kernel of the residual: free-column construction.
        for fc in 0..m {
            if pivots[fc] >= 0 {
                continue;
            }
            let mut kv = vec![fc];
            for c in 0..m {
                if pivots[c] >= 0 && aug.row(pivots[c] as usize).entry(fc) != 0 {
                    kv.push(c);
                }
            }
            res_kernel.push(kv);
        }
    }

    // Assemble the full-width result. Class members: root -> member list.
    let mut members: hashbrown::HashMap<usize, Vec<(usize, bool)>> =
        hashbrown::HashMap::new();
    for i in 0..n {
        let (r, p) = uf.find(i);
        members.entry(r).or_default().push((i, p));
    }

    let mut offset = vec_zero(n);
    let mut kernel_rows: Vec<FpVector> = Vec::new();
    let mut free_roots: Vec<usize> = Vec::new();
    for (&root, mems) in &members {
        let assigned = root_val.get(&root).copied();
        let in_residual = local.contains_key(&(root as u32));
        let base = match (assigned, in_residual) {
            (Some(v), _) => Some(v),
            (None, true) => {
                let li = local[&(root as u32)];
                // Determined by the residual iff its column is a pivot AND
                // no residual kernel vector touches it.
                if res_kernel.iter().any(|kv| kv.contains(&li)) {
                    None // undetermined via residual kernel
                } else {
                    Some(res_offset[li])
                }
            }
            (None, false) => None, // completely free class
        };
        // Particular value for the root: assigned / residual particular /
        // free choice 0. Members carry their parity relative to the root —
        // the offset must respect it even for free classes, or the parity
        // constraints themselves would be violated by the particular
        // solution.
        let root_particular = match (base, in_residual) {
            (Some(v), _) => v,
            (None, true) => res_offset[local[&(root as u32)]],
            (None, false) => {
                free_roots.push(root);
                false
            }
        };
        for &(mem, p) in mems {
            if root_particular ^ p {
                vec_set(&mut offset, mem, true);
            }
        }
    }
    // Kernel: one vector per completely-free class (class indicator), plus
    // residual kernel vectors expanded through class membership.
    free_roots.sort_unstable();
    for root in free_roots {
        let mut kv = vec_zero(n);
        for &(mem, _) in &members[&root] {
            vec_set(&mut kv, mem, true);
        }
        kernel_rows.push(kv);
    }
    for kv_local in &res_kernel {
        let mut kv = vec_zero(n);
        for &li in kv_local {
            let root = local_rev[li] as usize;
            for &(mem, _) in &members[&root] {
                vec_set(&mut kv, mem, true);
            }
        }
        kernel_rows.push(kv);
    }

    let mut touched = hashbrown::HashSet::new();
    for kv in &kernel_rows {
        for idx in vec_support(kv) {
            touched.insert(idx);
        }
    }
    let unknown: hashbrown::HashSet<usize> = touched;

    let kernel_matrix = if kernel_rows.is_empty() {
        mat_zero(0, n)
    } else {
        mat_from_rows(kernel_rows, n)
    };

    info!(
        "solve(uf): {} / {} variables determined ({} unknown; residual {} x {})",
        n - unknown.len(),
        n,
        unknown.len(),
        dense_rows.len(),
        m,
    );

    Some(SATResult {
        offset,
        unknown,
        kernel: kernel_matrix,
        vars: system.vars.clone(),
        var_index: system.var_index.clone(),
    })
}
