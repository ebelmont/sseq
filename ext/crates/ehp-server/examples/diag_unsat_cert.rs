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
//!
//! Run (t=50 first — much faster; then t=70 to match the user):
//!   EHP_MAX_T=50 EHP_MAX_R=4 EHP_DATA=$HOME/ehp-sat-rs/data/E2 \
//!     cargo run -p ehp-server --release --example diag_unsat_cert
//!
//! Env: DIAG_SCENARIOS (default "0ABC", any subset), DIAG_MAX_DIG (default 40,
//! cap on deep-dug certificate rows).

use std::collections::BTreeMap;

use ehp_core::constraints::{
    self, make_known_constraints, make_leibniz_constraint_single,
    make_naturality_constraint_single, ConstraintSystem, DiffVar,
};
use ehp_core::gf2::*;
use ehp_core::map::MapKind;
use ehp_core::page::SATPage;
use ehp_core::pageturning;
use ehp_core::result::SATResult;
use ehp_core::tridegree::Tridegree;
use ehp_core::{io, solver};
use fp::matrix::Matrix;
use fp::vector::FpVector;
use hashbrown::HashMap;

const DEFAULT_DATA: &str = concat!(env!("HOME"), "/ehp-sat-rs/data/E2.ehp");

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
        let real: Vec<usize> = vec_support(&sys.rows[i]).collect();
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
    let mut coeffs: Vec<FpVector> = sys.rows.clone();
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
            acc += &sys.rows[i];
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
}

fn build_chain(
    data: &str,
    max_t: i32,
    max_r: i32,
    extra_known: &HashMap<i32, Vec<(DiffVar, bool)>>,
) -> Result<Chain, String> {
    let mut current = io::load_page(data, 2, max_t).map_err(|e| e.to_string())?;
    let mut prev = Vec::new();
    loop {
        let r = current.r;
        if r >= max_r {
            return Ok(Chain { prev, final_page: current });
        }
        let mut known = ehp_server::load_known_diffs(r, None).map_err(|e| e.to_string())?;
        for (dv, val) in extra_known.get(&r).into_iter().flatten() {
            known.insert(*dv, *val);
        }
        let cutoff = current.max_s.unwrap_or(0);
        let sys = constraints::build_constraint_system(&current, cutoff, &known);
        let num_vars = sys.num_vars;
        let result = solver::solve(&sys)
            .ok_or_else(|| format!("E_{r}: base system UNSAT/empty while building chain"))?;
        eprintln!(
            "  E_{r}: {}/{} vars determined",
            num_vars - result.unknown.len(),
            num_vars
        );
        let (next, _turned) = pageturning::build_next_page(&current, &result)
            .map_err(|e| format!("E_{r}: contradiction while turning: {e}"))?;
        prev.push((current, result));
        current = next;
    }
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
    let cutoff = page.max_s.unwrap_or(0);
    println!(
        "E_{} page: cutoff(max_s)={} max_t={:?}, {} known diffs:",
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
        let vars: Vec<String> = vec_support(&sys.rows[idx])
            .map(|i| {
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
        let chain = build_chain(&data, max_t, max_r, &HashMap::new())?;
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
        let chain = build_chain(&data, max_t, max_r, &extra)?;
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

    Ok(())
}
