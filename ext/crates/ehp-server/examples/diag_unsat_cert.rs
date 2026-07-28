//! UNSAT *certificate* extraction for a user-asserted differential that makes
//! a page's constraint system inconsistent.
//!
//! Reported case: asserting d4(3,32,9)[0,0] = 1 on E4 (EHP_MAX_T=70,
//! EHP_MAX_R=4) makes E4 UNSAT, but the differential is mathematically
//! nontrivial — so some E4 constraint is WRONG. This diag:
//!
//! 1. builds the E2→E4 chain exactly like diag_sweep / the REPL startup,
//! 2. builds the E4 system with the asserted diff in `known_diffs`,
//! 3. runs GF(2) Gaussian elimination while tracking, for every working row,
//!    the set of ORIGINAL constraint rows it is a sum of; the inconsistent
//!    row (zero coefficients, RHS=1) yields the exact set of original
//!    constraints whose XOR is the contradiction (the certificate),
//! 4. maps each certificate row back to its generator (naturality square /
//!    Leibniz pair / known diff) by replaying build_constraint_system's
//!    generation order with provenance tags, verified row-for-row against
//!    the real system,
//! 5. digs one level deeper on each certificate constraint: prints the map
//!    matrices of naturality squares (flagging zero/identity-DEFAULTED data),
//!    the product blocks of Leibniz pairs (flagging MISSING blocks that
//!    `ProductTable::multiply` silently treats as zero), and the previous
//!    page's uncertainty at every participating degree (flagging degrees
//!    whose E4 data depends on an unknown d3 yet are NOT excluded).
//!
//! Scenarios (each is an independent chain + solve):
//!   S0: stock known diffs only                       (expect SAT — sanity)
//!   SA: stock + target d4(3,32,9)[0,0]=1             (expect UNSAT → cert)
//!   SB: stock + user's other 8 adds, no target       (expect SAT — user report)
//!   SC: stock + all 9 adds                           (expect UNSAT → cert)
//!   SD: replay of the REPL's INCREMENTAL cascade for the two d3 adds
//!       (patched turned map, not a full re-turn), then the 8/9-add tests on
//!       the resulting E4; also prints where it differs from a full re-turn
//!   SF: knowns from a REPL `save` file (DIAG_KNOWN_DIFFS=<path>) at every
//!       page — the faithful reproduction of a live session's accumulated
//!       state; tests the final page with the file minus/plus the target
//!
//! Run (t=50 first — much faster; then t=70 to match the user):
//!   EHP_MAX_T=50 EHP_MAX_R=4 EHP_DATA=$HOME/ehp-sat-rs/data/E2 \
//!     cargo run -p ehp-server --release --example diag_unsat_cert
//!
//! Env: DIAG_SCENARIOS (default "0ABC", any subset of "0ABCDF"),
//! DIAG_MAX_DIG (deep-dig row cap, default 40), DIAG_KNOWN_DIFFS (save file
//! for SF), EHP_RELAX_TARGET_EXCLUDE=0 to test under the strict pre-relaxation
//! exclusion semantics the user's live session was running.

use std::collections::BTreeMap;

use ehp_core::constraints::{
    self, make_known_constraints, make_leibniz_constraint_single,
    make_naturality_constraint_single, ConstraintSystem, DiffVar,
};
use ehp_core::gf2::*;
use ehp_core::map::MapKind;
use ehp_core::page::SATPage;
use ehp_core::pageturning::{self, TurnContext, TurnedBidegree};
use ehp_core::result::SATResult;
use ehp_core::tridegree::Tridegree;
use ehp_core::{io, solver};
use fp::matrix::Matrix;
use fp::vector::FpVector;
use hashbrown::{HashMap, HashSet};

const DEFAULT_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/E2");

// =============================================================================
// Provenance
// =============================================================================

#[derive(Clone, Debug)]
enum Prov {
    Nat { kind: MapKind, anchor: Tridegree },
    Leib { deg1: Tridegree, deg2: Tridegree },
    Known { var: DiffVar, val: bool },
}

impl std::fmt::Display for Prov {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Prov::Nat { kind, anchor } => write!(
                f,
                "naturality[{}] anchor ({},{},{})",
                kind.name(),
                anchor.n,
                anchor.s,
                anchor.f
            ),
            Prov::Leib { deg1, deg2 } => write!(
                f,
                "leibniz deg1=({},{},{}) deg2=({},{},{})",
                deg1.n, deg1.s, deg1.f, deg2.n, deg2.s, deg2.f
            ),
            Prov::Known { var, val } => write!(
                f,
                "known-diff d({},{},{})[{},{}] = {}",
                var.n, var.s, var.f, var.row, var.col, *val as u8
            ),
        }
    }
}

/// Replay build_constraint_system's generation order (naturality E/H/P each
/// dedup-sorted, then Leibniz dedup-sorted, then known in map order), tagging
/// every produced row with its generator(s). Verified against the real system.
fn reconstruct_provenance(
    page: &SATPage,
    cutoff: i32,
    known: &HashMap<DiffVar, bool>,
    sys: &ConstraintSystem,
) -> Result<Vec<(Vec<usize>, bool, Vec<Prov>)>, String> {
    let mut rows: Vec<(Vec<usize>, bool, Vec<Prov>)> = Vec::new();

    // --- Naturality, per map kind, in MapKind::all() order ---------------
    for kind in MapKind::all() {
        // BTreeMap over sorted index vectors = deduplicate_constraints
        // (sort each constraint, sort the list lexicographically, dedup).
        let mut dedup: BTreeMap<Vec<usize>, Vec<Prov>> = BTreeMap::new();
        for s in 0..=cutoff {
            for f in 0..=(cutoff - s) {
                for n in 2..=(s + 2) {
                    let t = Tridegree::new(n, s, f);
                    for mut c in
                        make_naturality_constraint_single(page, t, kind, &sys.var_index)
                    {
                        c.sort();
                        dedup.entry(c).or_default().push(Prov::Nat { kind, anchor: t });
                    }
                }
            }
        }
        for (c, p) in dedup {
            rows.push((c, false, p));
        }
    }

    // --- Leibniz (replicates make_leibniz_constraints' pair enumeration) --
    {
        let mut dedup: BTreeMap<Vec<usize>, Vec<Prov>> = BTreeMap::new();

        let mut degrees_by_n: HashMap<i32, Vec<(i32, i32)>> = HashMap::new();
        for (&t, &d) in &page.dimension {
            if t.s + t.f <= cutoff && d > 0 && t.f >= 1 {
                degrees_by_n.entry(t.n).or_default().push((t.s, t.f));
            }
        }
        let max_data_n = page
            .dimension
            .iter()
            .filter(|(_, &d)| d > 0)
            .map(|(t, _)| t.n)
            .max()
            .unwrap_or(cutoff);

        for s3 in 0..=cutoff {
            for f3 in 1..=(cutoff - s3) {
                for th1 in 2..=max_data_n {
                    let source_degrees = match degrees_by_n.get(&th1) {
                        Some(v) => v,
                        None => continue,
                    };
                    for &(s1, f1) in source_degrees {
                        let th2 = th1 + s1 - 1;
                        let f2 = f3 - f1;
                        let s2 = s3 - s1;
                        if th2 > max_data_n {
                            continue;
                        }
                        let deg1 = Tridegree::new(th1, s1, f1);
                        let deg2 = Tridegree::new(th2, s2, f2);
                        for mut c in make_leibniz_constraint_single(
                            page,
                            deg1,
                            deg2,
                            &sys.var_index,
                            None,
                        ) {
                            c.sort();
                            dedup.entry(c).or_default().push(Prov::Leib { deg1, deg2 });
                        }
                    }
                }
            }
        }
        for (c, p) in dedup {
            rows.push((c, false, p));
        }
    }

    // --- Known diffs (same HashMap instance ⇒ same iteration order) -------
    for (indices, val) in make_known_constraints(known, &sys.var_index) {
        let var = sys.vars[indices[0]];
        rows.push((indices, val, vec![Prov::Known { var, val }]));
    }

    // --- Verify against the real system, row for row ----------------------
    if rows.len() != sys.rows.len() {
        return Err(format!(
            "row count mismatch: reconstructed {} vs real {}",
            rows.len(),
            sys.rows.len()
        ));
    }
    for (i, (indices, rhs, _)) in rows.iter().enumerate() {
        let real: Vec<usize> = sys.rows[i].clone();
        if &real != indices || sys.rhs[i] != *rhs {
            return Err(format!(
                "row {} mismatch: reconstructed {:?} rhs={} vs real {:?} rhs={}",
                i, indices, rhs, real, sys.rhs[i]
            ));
        }
    }
    Ok(rows)
}

// =============================================================================
// Tracked Gaussian elimination → UNSAT certificate
// =============================================================================

/// `ConstraintSystem.rows` is now stored sparsely (`Vec<usize>` per row);
/// this diagnostic's tracked elimination still works over dense `FpVector`s,
/// so convert at the boundary.
fn sparse_row_to_dense(row: &[usize], ncols: usize) -> FpVector {
    let mut v = vec_zero(ncols);
    for &i in row {
        v.set_entry(i, 1);
    }
    v
}

/// Sorted-set XOR (symmetric difference).
fn xor_merge(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// Forward elimination over GF(2) with row-combination tracking. Returns the
/// smallest certificate found: a set of ORIGINAL row indices whose XOR has
/// zero coefficients and RHS 1 — i.e. sums to the contradiction 0 = 1.
/// Returns None if the system is consistent.
fn tracked_certificate(sys: &ConstraintSystem) -> Option<Vec<usize>> {
    let n = sys.rows.len();
    let ncols = sys.num_vars;
    let mut coeffs: Vec<FpVector> = sys
        .rows
        .iter()
        .map(|r| sparse_row_to_dense(r, ncols))
        .collect();
    let mut rhs: Vec<bool> = sys.rhs.clone();
    let mut combos: Vec<Vec<u32>> = (0..n).map(|i| vec![i as u32]).collect();

    let mut pivot_row = 0usize;
    for col in 0..ncols {
        let Some(p) = (pivot_row..n).find(|&i| vec_get(&coeffs[i], col)) else {
            continue;
        };
        coeffs.swap(pivot_row, p);
        rhs.swap(pivot_row, p);
        combos.swap(pivot_row, p);

        let pivot_coeffs = coeffs[pivot_row].clone();
        let pivot_rhs = rhs[pivot_row];
        let pivot_combo = combos[pivot_row].clone();
        for i in (pivot_row + 1)..n {
            if vec_get(&coeffs[i], col) {
                coeffs[i] += &pivot_coeffs;
                rhs[i] ^= pivot_rhs;
                combos[i] = xor_merge(&combos[i], &pivot_combo);
            }
        }
        pivot_row += 1;
        if pivot_row == n {
            break;
        }
    }

    // All inconsistent rows are now zero-coefficient with RHS=1;
    // pick the smallest certificate among them.
    let mut best: Option<Vec<usize>> = None;
    for i in 0..n {
        if rhs[i] && coeffs[i].is_zero() {
            let cert: Vec<usize> = combos[i].iter().map(|&x| x as usize).collect();
            if best.as_ref().map_or(true, |b| cert.len() < b.len()) {
                best = Some(cert);
            }
        }
    }

    // Sanity: verify the certificate really sums to (0 | 1).
    if let Some(ref cert) = best {
        let mut acc = vec_zero(ncols);
        let mut racc = false;
        for &i in cert {
            acc += &sparse_row_to_dense(&sys.rows[i], ncols);
            racc ^= sys.rhs[i];
        }
        assert!(
            acc.is_zero() && racc,
            "certificate FAILED verification (sum != (0|1)) — tracking bug"
        );
    }
    best
}

// =============================================================================
// Deep-dig reporting
// =============================================================================

fn mat_str(m: &Matrix, indent: &str) -> String {
    if m.rows() == 0 || m.columns() == 0 {
        return format!("{}[{}x{} empty]\n", indent, m.rows(), m.columns());
    }
    let mut s = String::new();
    for i in 0..m.rows() {
        s.push_str(indent);
        s.push('[');
        for j in 0..m.columns() {
            s.push(if mat_get(m, i, j) { '1' } else { '0' });
        }
        s.push_str("]\n");
    }
    s
}

fn stable_rep(t: Tridegree) -> Tridegree {
    if t.n > t.s + 2 {
        Tridegree::new(t.s + 2, t.s, t.f)
    } else {
        t
    }
}

/// Per-degree status line: dimension, exclusion, and previous-page uncertainty
/// (outgoing d_{r-1} at the degree / incoming d_{r-1} into the degree). If the
/// previous page had uncertainty here but the degree is NOT excluded, the
/// turned data (dims/maps/products) at this degree may be WRONG — flag it.
fn degree_status(
    label: &str,
    t: Tridegree,
    page: &SATPage,
    prev: Option<(&SATPage, &SATResult)>,
) {
    let dim = page.dim_at(t);
    let excl = page.is_excluded(t);
    let mut extra = String::new();
    let mut gap = false;
    if let Some((prev_page, prev_res)) = prev {
        let pr = prev_page.r;
        let rep = stable_rep(t);
        let out_unc = prev_res.is_tridegree_uncertain(rep);
        let in_unc = prev_res.is_tridegree_uncertain(stable_rep(t.diff_source(pr)));
        let prev_excl = prev_page.is_excluded(t);
        extra = format!(
            " | E{}: d{}-out-unknown={} d{}-in-unknown={} was-excluded={}",
            pr, pr, out_unc, pr, in_unc, prev_excl
        );
        gap = (out_unc || in_unc || prev_excl) && !excl;
    }
    println!(
        "      {:<10} ({:>3},{:>3},{:>3})  dim={} excluded={}{}{}",
        label,
        t.n,
        t.s,
        t.f,
        dim,
        excl,
        extra,
        if gap { "  <<< EXCLUSION GAP (prev-page uncertainty, not excluded)" } else { "" }
    );
}

fn map_matrix_status(page: &SATPage, kind: MapKind, t: Tridegree) -> (Matrix, &'static str) {
    let stored = page
        .maps
        .get(&kind)
        .and_then(|mt| mt.matrix_at(t))
        .is_some();
    let m = page.map_matrix(kind, t);
    let status = if stored {
        "stored"
    } else if kind == MapKind::E && t.n > t.s + 1 && page.dim_at(t) == page.dim_at(kind.target_degree(t)) {
        "DEFAULTED (stable-E identity, no stored data)"
    } else {
        "DEFAULTED (zero map, no stored data)"
    };
    (m, status)
}

fn dig_naturality(
    kind: MapKind,
    anchor: Tridegree,
    page: &SATPage,
    prev: Option<(&SATPage, &SATResult)>,
) {
    let r = page.r;
    let t = anchor;
    let (src, diff_src, tgt) = match kind {
        MapKind::P => {
            let src = Tridegree::new(2 * t.n + 1, t.s - t.n + 1, t.f - 2);
            let diff_src = Tridegree::new(2 * t.n + 1, t.s - t.n, t.f - 2 + r);
            (src, diff_src, t)
        }
        _ => (t, t.diff_target(r), kind.target_degree(t)),
    };
    let diff_tgt = tgt.diff_target(r);

    println!(
        "    square: phi_target * D[({},{},{})] = D[({},{},{})] * phi_source   (d_{})",
        src.n, src.s, src.f, tgt.n, tgt.s, tgt.f, r
    );
    degree_status("src", src, page, prev);
    degree_status("d(src)", diff_src, page, prev);
    degree_status("tgt", tgt, page, prev);
    degree_status("d(tgt)", diff_tgt, page, prev);

    let (phi_src, st1) = map_matrix_status(page, kind, src);
    println!(
        "      phi_source = {}({},{},{})  [{}x{}]  {}",
        kind.name(), src.n, src.s, src.f, phi_src.rows(), phi_src.columns(), st1
    );
    print!("{}", mat_str(&phi_src, "        "));
    let (phi_tgt, st2) = map_matrix_status(page, kind, diff_src);
    println!(
        "      phi_target = {}({},{},{})  [{}x{}]  {}",
        kind.name(), diff_src.n, diff_src.s, diff_src.f, phi_tgt.rows(), phi_tgt.columns(), st2
    );
    print!("{}", mat_str(&phi_tgt, "        "));
}

fn product_block_status(
    page: &SATPage,
    deg1: Tridegree,
    deg2: Tridegree,
    expected_tgt_dim: usize,
    label: &str,
) {
    match page.products.block(deg1, deg2) {
        Some(pm) => {
            let dim_ok = pm.tgt_dim as usize == expected_tgt_dim;
            println!(
                "      product block {} ({},{},{}) x ({},{},{}): stored [{}x{} -> tgt_dim {}]{}",
                label,
                deg1.n, deg1.s, deg1.f,
                deg2.n, deg2.s, deg2.f,
                pm.dim1, pm.dim2, pm.tgt_dim,
                if dim_ok { "" } else { "  <<< tgt_dim MISMATCH — multiply() silently returns ZERO" }
            );
            print!("{}", mat_str(&pm.matrix, "        "));
        }
        None => {
            println!(
                "      product block {} ({},{},{}) x ({},{},{}): MISSING — multiply() returns ZERO \
                 (wrong if the true product is nonzero)",
                label,
                deg1.n, deg1.s, deg1.f,
                deg2.n, deg2.s, deg2.f,
            );
        }
    }
}

fn dig_leibniz(
    deg1: Tridegree,
    deg2: Tridegree,
    page: &SATPage,
    prev: Option<(&SATPage, &SATResult)>,
) {
    let r = page.r;
    let e_deg2 = Tridegree::new(deg2.n + 1, deg2.s, deg2.f);
    let prod_deg = Tridegree::new(deg1.n, deg1.s + deg2.s, deg1.f + deg2.f);
    let prod_dr_deg = prod_deg.diff_target(r);
    let d_deg1 = deg1.diff_target(r);
    let d_deg2 = deg2.diff_target(r);
    let target_deg = Tridegree::new(deg2.n + 1, deg2.s - 1, deg2.f + r); // d(E deg2) lives here

    println!(
        "    relation: d(x·E(y)) = x·d(E(y)) + d(x)·y  with x@({},{},{}), y@({},{},{})  (d_{})",
        deg1.n, deg1.s, deg1.f, deg2.n, deg2.s, deg2.f, r
    );
    degree_status("x=deg1", deg1, page, prev);
    degree_status("y=deg2", deg2, page, prev);
    degree_status("E(y)", e_deg2, page, prev);
    degree_status("d(x)", d_deg1, page, prev);
    degree_status("d(y)", d_deg2, page, prev);
    degree_status("dE(y)", target_deg, page, prev);
    degree_status("x·E(y)", prod_deg, page, prev);
    degree_status("d(x·E(y))", prod_dr_deg, page, prev);

    let (e_at_deg2, st1) = map_matrix_status(page, MapKind::E, deg2);
    println!(
        "      E at deg2 ({},{},{})  [{}x{}]  {}",
        deg2.n, deg2.s, deg2.f, e_at_deg2.rows(), e_at_deg2.columns(), st1
    );
    print!("{}", mat_str(&e_at_deg2, "        "));
    let (e_at_d2, st2) = map_matrix_status(page, MapKind::E, d_deg2);
    println!(
        "      E at d(deg2) ({},{},{})  [{}x{}]  {}",
        d_deg2.n, d_deg2.s, d_deg2.f, e_at_d2.rows(), e_at_d2.columns(), st2
    );
    print!("{}", mat_str(&e_at_d2, "        "));

    // The three product uses inside make_leibniz_constraint_single:
    //   x·E(y):      multiply(deg1, ·, e_deg2, ·, dim(prod_deg))
    //   Ytilde rhs1: multiply(deg1, ·, target_deg, ·, dim(prod_dr_deg))
    //   Ytilde rhs2: multiply(d_deg1, ·, deg2, ·, dim(prod_dr_deg))
    product_block_status(page, deg1, e_deg2, page.dim_at(prod_deg), "x·E(y)   ");
    if deg2.s != 0 && page.dim_at(target_deg) != 0 {
        product_block_status(page, deg1, target_deg, page.dim_at(prod_dr_deg), "x·dE(y)  ");
    }
    if deg1.s != 0 && page.dim_at(d_deg1) != 0 {
        product_block_status(page, d_deg1, deg2, page.dim_at(prod_dr_deg), "d(x)·y   ");
    }
}

// =============================================================================
// Chain building & scenario analysis
// =============================================================================

struct Chain {
    /// (page, result) for E2 .. E_{max_r - 1}.
    prev: Vec<(SATPage, SATResult)>,
    /// The final page E_{max_r}.
    final_page: SATPage,
    /// Turned data of the LAST turn (E_{max_r-1} → E_{max_r}), needed to
    /// replay the REPL's *incremental* cascade in scenario SD.
    last_turned: HashMap<Tridegree, TurnedBidegree>,
}

fn build_chain(
    data: &str,
    max_t: i32,
    max_r: i32,
    extra_known: &HashMap<i32, Vec<(DiffVar, bool)>>,
    known_file: Option<&str>,
) -> Result<Chain, String> {
    let mut current = io::load_page(data, 2, max_t).map_err(|e| e.to_string())?;
    let mut prev = Vec::new();
    let mut last_turned = HashMap::new();
    loop {
        let r = current.r;
        if r >= max_r {
            return Ok(Chain { prev, final_page: current, last_turned });
        }
        let mut known =
            ehp_server::load_known_diffs(r, known_file).map_err(|e| e.to_string())?;
        for (dv, val) in extra_known.get(&r).into_iter().flatten() {
            known.insert(*dv, *val);
        }
        let cutoff = current.max_t.unwrap_or(0);
        let sys = constraints::build_constraint_system(&current, cutoff, &known);
        let num_vars = sys.num_vars;
        let result = solver::solve(&sys)
            .ok_or_else(|| format!("E_{r}: base system UNSAT/empty while building chain"))?;
        eprintln!(
            "  E_{r}: {}/{} vars determined",
            num_vars - result.unknown.len(),
            num_vars
        );
        let (next, turned) = pageturning::build_next_page(&current, &result)
            .map_err(|e| format!("E_{r}: contradiction while turning: {e}"))?;
        last_turned = turned;
        prev.push((current, result));
        current = next;
    }
}

// =============================================================================
// Scenario SD: replay of the REPL's INCREMENTAL cascade (ehp_chart.rs
// cascade_resolve): after an add on E_{r}, only `changed ∪ changed.diff_target`
// degrees are re-turned in the cached turned map before rebuilding E_{r+1}.
// Everything else in the turned map — including all STABLE COPIES (n > s+2)
// of changed stable-representative degrees, which share the rep's
// differential — keeps its pre-add homology. This function replicates that
// path exactly so the resulting E4 can be compared against a full re-turn.
// =============================================================================

/// ehp_chart.rs `effective_var_value`: unknown/absent → false.
fn effective_var_value(result: &SATResult, var: &DiffVar) -> bool {
    if let Some(&idx) = result.var_index.get(var) {
        if result.unknown.contains(&idx) {
            false
        } else {
            result.offset.entry(idx) != 0
        }
    } else {
        false
    }
}

/// ehp_chart.rs `changed_diff_tridegrees`.
fn changed_diff_tridegrees(
    old_result: Option<&SATResult>,
    new_result: &SATResult,
) -> HashSet<Tridegree> {
    let mut changed = HashSet::new();
    for var in &new_result.vars {
        let new_val = effective_var_value(new_result, var);
        let old_val = old_result.is_some_and(|r| effective_var_value(r, var));
        if new_val != old_val {
            changed.insert(var.tridegree());
        }
    }
    if let Some(old) = old_result {
        for var in &old.vars {
            if !new_result.var_index.contains_key(var) && effective_var_value(old, var) {
                changed.insert(var.tridegree());
            }
        }
    }
    changed
}

/// Replay one REPL cascade step at the previous page: re-solve it with
/// `known`, patch `turned` only at the incremental `affected` set, and
/// return the new result. Mirrors ehp_chart.rs cascade_resolve lines
/// ~598-726 for a single page step.
fn incremental_cascade_step(
    prev_page: &SATPage,
    old_result: &SATResult,
    known: &HashMap<DiffVar, bool>,
    turned: &mut HashMap<Tridegree, TurnedBidegree>,
) -> Result<SATResult, String> {
    let r = prev_page.r;
    let cutoff = prev_page.max_t.unwrap_or(0);
    let sys = constraints::build_constraint_system(prev_page, cutoff, known);
    let new_res = solver::solve(&sys)
        .ok_or_else(|| format!("E_{r} re-solve INCONSISTENT during incremental replay"))?;

    let changed = changed_diff_tridegrees(Some(old_result), &new_res);
    // affected_homology_tridegrees: t and t.diff_target(r) only.
    let mut affected: HashSet<Tridegree> = HashSet::new();
    for &t in &changed {
        affected.insert(t);
        affected.insert(t.diff_target(r));
    }
    eprintln!(
        "  [SD] E_{r} incremental step: {} changed tridegrees, {} re-turned (turned map has {} keys)",
        changed.len(),
        affected.len(),
        turned.len()
    );

    let ctx = TurnContext::new(prev_page, &new_res);
    for &t in &affected {
        if prev_page.dim_at(t) == 0 {
            turned.remove(&t);
            continue;
        }
        match ctx.get_tb(t) {
            Ok(Some(tb)) if !tb.basis.is_empty() => {
                turned.insert(t, tb);
            }
            Ok(_) => {
                turned.remove(&t);
            }
            Err(e) => return Err(format!("E_{r} incremental re-turn contradiction: {e}")),
        }
    }
    Ok(new_res)
}

/// Analyze one scenario on the final page. Returns the certificate if UNSAT.
fn analyze(
    label: &str,
    page: &SATPage,
    known: &HashMap<DiffVar, bool>,
    prev: Option<(&SATPage, &SATResult)>,
    max_dig: usize,
) -> Option<Vec<usize>> {
    println!("\n=================================================================");
    println!("SCENARIO {label}");
    println!("=================================================================");
    let cutoff = page.max_t.unwrap_or(0);
    println!(
        "E_{} page: cutoff(max_t)={} max_t={:?}, {} known diffs:",
        page.r, cutoff, page.max_t, known.len()
    );
    let mut kv: Vec<_> = known.iter().collect();
    kv.sort_by_key(|(v, _)| **v);
    for (v, val) in kv {
        let in_range = page.dim_at(Tridegree::new(v.n.min(v.s + 2), v.s, v.f)) > 0;
        println!(
            "  d{}({},{},{})[{},{}] = {}   (source dim>0: {})",
            page.r, v.n, v.s, v.f, v.row, v.col, *val as u8, in_range
        );
    }

    let sys = constraints::build_constraint_system(page, cutoff, known);
    println!("system: {} vars, {} constraints", sys.num_vars, sys.num_constraints());

    // Provenance reconstruction + verification.
    let prov = match reconstruct_provenance(page, cutoff, known, &sys) {
        Ok(p) => {
            println!("provenance reconstruction: VERIFIED (matches real system row-for-row)");
            Some(p)
        }
        Err(e) => {
            println!("provenance reconstruction FAILED: {e}");
            println!("(continuing; certificate rows will lack provenance)");
            None
        }
    };

    // Cross-check with the production solver.
    let solver_result = solver::solve(&sys);
    match &solver_result {
        Some(res) => println!(
            "solver::solve: SAT, {}/{} determined",
            sys.num_vars - res.unknown.len(),
            sys.num_vars
        ),
        None => println!("solver::solve: None (inconsistent or empty)"),
    }

    // Report the target var's determined value when SAT (context for UNSAT runs).
    if let Some(res) = &solver_result {
        let tv = DiffVar::new(3, 32, 9, 0, 0);
        if let Some(&idx) = res.var_index.get(&tv) {
            let det = !res.unknown.contains(&idx);
            println!(
                "  d4(3,32,9)[0,0]: {} (value {})",
                if det { "DETERMINED" } else { "unknown" },
                vec_get(&res.offset, idx) as u8
            );
        } else {
            println!("  d4(3,32,9)[0,0]: NOT A VARIABLE in this system");
        }
    }

    let t0 = std::time::Instant::now();
    let cert = tracked_certificate(&sys);
    println!(
        "tracked elimination: {:.1}s — {}",
        t0.elapsed().as_secs_f64(),
        match &cert {
            Some(c) => format!("INCONSISTENT, certificate of {} constraints", c.len()),
            None => "consistent".to_string(),
        }
    );

    let Some(cert) = cert else { return None };

    println!("\n--- UNSAT CERTIFICATE (XOR of these constraints = [0 = 1]) ---");
    for &idx in &cert {
        let vars: Vec<String> = sys.rows[idx]
            .iter()
            .map(|&i| {
                let v = sys.vars[i];
                format!("d{}({},{},{})[{},{}]", page.r, v.n, v.s, v.f, v.row, v.col)
            })
            .collect();
        println!("\n  row #{idx}: {} = {}", vars.join(" + "), sys.rhs[idx] as u8);
        if let Some(prov) = &prov {
            for p in &prov[idx].2 {
                println!("    from: {p}");
            }
        }
    }

    println!("\n--- DEEP DIG on certificate constraints ---");
    if let Some(prov) = &prov {
        for (k, &idx) in cert.iter().enumerate() {
            if k >= max_dig {
                println!("  ... ({} more rows, raise DIAG_MAX_DIG)", cert.len() - k);
                break;
            }
            println!("\n  row #{idx}:");
            // Dig every generator of the row (dedup identical generators).
            let mut seen = Vec::new();
            for p in &prov[idx].2 {
                let key = format!("{p}");
                if seen.contains(&key) {
                    continue;
                }
                seen.push(key);
                println!("    generator: {p}");
                match p {
                    Prov::Nat { kind, anchor } => dig_naturality(*kind, *anchor, page, prev),
                    Prov::Leib { deg1, deg2 } => dig_leibniz(*deg1, *deg2, page, prev),
                    Prov::Known { .. } => {
                        println!("      (user/stock assertion — ground truth by definition)")
                    }
                }
            }
        }
    }

    Some(cert)
}

// =============================================================================
// main
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let data = std::env::var("EHP_DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let max_t: i32 = std::env::var("EHP_MAX_T").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    let max_r: i32 = std::env::var("EHP_MAX_R").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let scenarios =
        std::env::var("DIAG_SCENARIOS").unwrap_or_else(|_| "0ABC".to_string());
    let max_dig: usize =
        std::env::var("DIAG_MAX_DIG").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
    // A REPL `save` file (`r,n,s,f,row,col,value` lines): scenario F builds
    // the whole chain with these knowns at every page — the user's exact
    // accumulated session knowledge.
    let known_file = std::env::var("DIAG_KNOWN_DIFFS").ok();

    // The user's adds (REPL `add <r> <n> <s> <f> <row> <col> <value>` form).
    let target = DiffVar::new(3, 32, 9, 0, 0); // d4, the bisected culprit
    let other_d4: Vec<DiffVar> = vec![
        DiffVar::new(3, 34, 10, 0, 0),
        DiffVar::new(3, 38, 9, 0, 0),
        DiffVar::new(3, 39, 11, 0, 0),
        DiffVar::new(3, 40, 10, 0, 1),
        DiffVar::new(3, 40, 13, 0, 0),
        DiffVar::new(3, 41, 12, 0, 0),
    ];
    let other_d3: Vec<DiffVar> = vec![
        DiffVar::new(3, 41, 15, 0, 0),
        DiffVar::new(3, 43, 16, 0, 0),
    ];

    eprintln!("data={data} max_t={max_t} max_r={max_r} scenarios={scenarios}");

    // ---- Chain A: stock knowns at every page (scenarios S0, SA) ----------
    let mut cert_a = None;
    if scenarios.contains('0') || scenarios.contains('A') {
        eprintln!("\nBuilding chain A (stock knowns)...");
        let chain = build_chain(&data, max_t, max_r, &HashMap::new(), None)?;
        let prev = chain.prev.last().map(|(p, r)| (p, r));
        let stock = ehp_server::load_known_diffs(max_r, None)?;

        if scenarios.contains('0') {
            analyze("S0: stock known diffs only", &chain.final_page, &stock, prev, max_dig);
        }
        if scenarios.contains('A') {
            let mut known = stock.clone();
            known.insert(target, true);
            cert_a = analyze(
                "SA: stock + target d4(3,32,9)[0,0]=1 ALONE",
                &chain.final_page,
                &known,
                prev,
                max_dig,
            );
        }
    }

    // ---- Chain B: user's d3 adds baked into E3 (scenarios SB, SC) --------
    if scenarios.contains('B') || scenarios.contains('C') {
        eprintln!("\nBuilding chain B (with user's two d3 adds at E3)...");
        let mut extra = HashMap::new();
        extra.insert(3, other_d3.iter().map(|&v| (v, true)).collect::<Vec<_>>());
        let chain = build_chain(&data, max_t, max_r, &extra, None)?;
        let prev = chain.prev.last().map(|(p, r)| (p, r));
        let stock = ehp_server::load_known_diffs(max_r, None)?;
        let mut with_others = stock.clone();
        for &v in &other_d4 {
            with_others.insert(v, true);
        }

        if scenarios.contains('B') {
            analyze(
                "SB: stock + user's other 8 adds (NO target)",
                &chain.final_page,
                &with_others,
                prev,
                max_dig,
            );
        }
        if scenarios.contains('C') {
            let mut known = with_others.clone();
            known.insert(target, true);
            let cert_c = analyze(
                "SC: stock + all 9 adds (target included)",
                &chain.final_page,
                &known,
                prev,
                max_dig,
            );
            if let (Some(a), Some(c)) = (&cert_a, &cert_c) {
                println!(
                    "\ncertificates SA vs SC: {}",
                    if a == c { "IDENTICAL" } else { "DIFFERENT (context-dependent)" }
                );
            }
        }
    }

    // ---- Scenario SD: REPL incremental-cascade replay ---------------------
    // The REPL does NOT rebuild E4 from scratch after a d3 add: it re-solves
    // E3 and re-turns only `changed ∪ changed.diff_target(3)` in the cached
    // turned map (cascade_resolve in ehp_chart.rs), then rebuilds E4 from
    // that PATCHED map. Everything not in the affected set — in particular
    // every STABLE COPY (n > s+2) of a changed stable-rep degree — keeps its
    // pre-add homology. This scenario replays that path with the user's two
    // d3 adds, reports where the stale E4 differs from a full re-turn, and
    // re-tests the 8-adds / 9-adds consistency on the stale page.
    if scenarios.contains('D') {
        eprintln!("\nBuilding chain for SD (stock knowns, keeping turned data)...");
        let chain = build_chain(&data, max_t, max_r, &HashMap::new(), None)?;
        let (e3_page, e3_stock_res) =
            chain.prev.last().ok_or("SD: no previous page in chain")?;
        let mut turned = chain.last_turned.clone();

        // Replay: the six d4 adds only touch E4's known-diff map (no page
        // change). The two d3 adds each trigger an E3 cascade — apply them
        // one at a time, exactly like typed REPL adds.
        let mut e3_known = ehp_server::load_known_diffs(max_r - 1, None)?;
        let mut last: Option<SATResult> = None;
        for &dv in &other_d3 {
            e3_known.insert(dv, true);
            let prev_res: &SATResult = last.as_ref().unwrap_or(e3_stock_res);
            let res = incremental_cascade_step(e3_page, prev_res, &e3_known, &mut turned)?;
            last = Some(res);
        }
        let e3_new = last.ok_or("SD: no d3 adds to replay")?;

        // Rebuild E4 from the PATCHED turned map (the REPL's next_ps.page).
        let mut e4_stale = pageturning::build_page_from_turned(e3_page, &turned);
        let (ex, tonly) = pageturning::make_next_exclude_set(e3_page, &e3_new);
        e4_stale.exclude_set = ex;
        e4_stale.target_only_exclude = tonly;

        // Compare against a FULL re-turn from the same E3 result.
        match pageturning::build_next_page(e3_page, &e3_new) {
            Ok((e4_fresh, _)) => {
                let mut stale: Vec<(Tridegree, usize, usize)> = Vec::new();
                for (&t, &d) in &e4_stale.dimension {
                    if e4_fresh.dim_at(t) != d {
                        stale.push((t, d, e4_fresh.dim_at(t)));
                    }
                }
                for (&t, &d) in &e4_fresh.dimension {
                    if !e4_stale.dimension.contains_key(&t) && d > 0 {
                        stale.push((t, 0, d));
                    }
                }
                stale.sort();
                println!(
                    "\n[SD] incremental-vs-full E4 dimension mismatches: {}",
                    stale.len()
                );
                for (t, ds, df) in stale.iter().take(80) {
                    println!(
                        "  ({:>3},{:>3},{:>3}): incremental dim {} vs full dim {}{}",
                        t.n, t.s, t.f, ds, df,
                        if t.n > t.s + 2 { "   [stable copy]" } else { "" }
                    );
                }
                if stale.len() > 80 {
                    println!("  ... ({} more)", stale.len() - 80);
                }
            }
            Err(e) => println!("\n[SD] full re-turn hit contradiction: {e}"),
        }

        let stock = ehp_server::load_known_diffs(max_r, None)?;
        let mut with_others = stock.clone();
        for &v in &other_d4 {
            with_others.insert(v, true);
        }
        analyze(
            "SD-B: STALE incremental E4 + other 8 adds (NO target)",
            &e4_stale,
            &with_others,
            Some((e3_page, &e3_new)),
            max_dig,
        );
        let mut known = with_others.clone();
        known.insert(target, true);
        analyze(
            "SD-C: STALE incremental E4 + all 9 adds (target included)",
            &e4_stale,
            &known,
            Some((e3_page, &e3_new)),
            max_dig,
        );
    }

    // ---- Scenario SF: the user's SAVED session knowns (REPL `save` file) --
    // Builds the chain with the file's knowns at EVERY page (their sweep
    // forcings, E2/E3 adds, everything), then tests the final page with the
    // file's knowns minus/plus the target diff. This is the faithful
    // reproduction once the user provides their save file:
    //   DIAG_KNOWN_DIFFS=path/to/save.csv DIAG_SCENARIOS=F \
    //     EHP_MAX_T=70 EHP_MAX_R=4 cargo run -p ehp-server --release \
    //     --example diag_unsat_cert
    if scenarios.contains('F') {
        let Some(path) = known_file.as_deref() else {
            eprintln!("DIAG_SCENARIOS=F requires DIAG_KNOWN_DIFFS=<save file>");
            return Ok(());
        };
        eprintln!("\nBuilding chain F (knowns from {path} at every page)...");
        let chain = build_chain(&data, max_t, max_r, &HashMap::new(), Some(path))?;
        let prev = chain.prev.last().map(|(p, r)| (p, r));
        let file_known = ehp_server::load_known_diffs(max_r, Some(path))?;

        let mut without = file_known.clone();
        let had_target = without.remove(&target).is_some();
        analyze(
            &format!(
                "SF-B: saved knowns MINUS target (file contained target: {had_target})"
            ),
            &chain.final_page,
            &without,
            prev,
            max_dig,
        );
        let mut with = without.clone();
        with.insert(target, true);
        analyze(
            "SF-C: saved knowns PLUS target d4(3,32,9)[0,0]=1",
            &chain.final_page,
            &with,
            prev,
            max_dig,
        );
    }

    // ---- Scenarios R0-R3: the user's ACTUAL saved session state -----------
    // DIAG_KNOWN_DIFFS=<save file>, DIAG_SCENARIOS=R.
    //   R0: saved state as-is (baseline; the file postdates the undo)
    //   R1: saved state + target d4(3,32,9)[0,0]=1 alone
    //   R2: saved state + the full 9-add batch (the two d3s are already in
    //       the file, so this adds the 7 d4s incl. the target)
    //   R3: saved state MINUS the suspect d2(36,35,2)=1 + the batch — the
    //       closest match to the historical state at contradiction time
    if scenarios.contains('R') {
        let Some(path) = known_file.as_deref() else {
            eprintln!("DIAG_SCENARIOS=R requires DIAG_KNOWN_DIFFS=<save file>");
            return Ok(());
        };

        // --- chain with the saved knowns at every page (R0, R1, R2) -------
        eprintln!("\nBuilding chain R (saved knowns at every page from {path})...");
        let chain = build_chain(&data, max_t, max_r, &HashMap::new(), Some(path))?;
        let prev = chain.prev.last().map(|(p, r)| (p, r));
        let base = ehp_server::load_known_diffs(max_r, Some(path))?;

        analyze("R0: saved state as-is (baseline)", &chain.final_page, &base, prev, max_dig);

        let mut r1 = base.clone();
        r1.insert(target, true);
        analyze(
            "R1: saved state + target d4(3,32,9)[0,0]=1 ALONE",
            &chain.final_page,
            &r1,
            prev,
            max_dig,
        );

        let mut r2 = base.clone();
        for &v in &other_d4 {
            r2.insert(v, true);
        }
        r2.insert(target, true);
        analyze(
            "R2: saved state + full 9-add batch (7 d4s; d3s already saved)",
            &chain.final_page,
            &r2,
            prev,
            max_dig,
        );

        // --- R3: rebuild the chain WITHOUT the suspect d2(36,35,2)=1 -------
        let suspect_line = "2,36,35,2,0,0,1";
        let contents = std::fs::read_to_string(path)?;
        let filtered: String = contents
            .lines()
            .filter(|l| l.trim() != suspect_line)
            .map(|l| format!("{l}\n"))
            .collect();
        let removed = contents.lines().count() - filtered.lines().count();
        let filtered_path = std::env::temp_dir().join("diag_unsat_cert_no_suspect_d2.csv");
        std::fs::write(&filtered_path, &filtered)?;
        eprintln!(
            "\nBuilding chain R3 (saved knowns MINUS d2(36,35,2)=1 — removed {removed} line(s))..."
        );
        let filtered_str = filtered_path.to_string_lossy().to_string();
        let chain3 = build_chain(&data, max_t, max_r, &HashMap::new(), Some(&filtered_str))?;
        let prev3 = chain3.prev.last().map(|(p, r)| (p, r));
        let base3 = ehp_server::load_known_diffs(max_r, Some(&filtered_str))?;

        analyze(
            "R3-base: saved state minus d2(36,35,2), no batch",
            &chain3.final_page,
            &base3,
            prev3,
            max_dig,
        );
        let mut r3 = base3.clone();
        for &v in &other_d4 {
            r3.insert(v, true);
        }
        r3.insert(target, true);
        analyze(
            "R3: saved state minus d2(36,35,2) + full 9-add batch",
            &chain3.final_page,
            &r3,
            prev3,
            max_dig,
        );
    }

    Ok(())
}
