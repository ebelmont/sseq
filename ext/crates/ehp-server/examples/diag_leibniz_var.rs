//! Diagnose why a specific d_r variable is not determined by any Leibniz
//! constraint: locate the variable, count the constraints touching it, then
//! enumerate EVERY Leibniz pair (deg1, deg2) that could reference it (via the
//! LHS product term, the RHS1 deg2 term, or the RHS2 deg1 term) and replay the
//! guard checks of `make_leibniz_constraint_single`, printing exactly which
//! guard rejected each pair (or how many constraints it emitted).
//!
//! Run:
//!   EHP_DATA=$HOME/ehp-sat-rs/data/E2 EHP_MAX_T=50 \
//!   DIAG_N=43 DIAG_S=34 DIAG_F=4 DIAG_R=3 \
//!   cargo run -p ehp-server --release --example diag_leibniz_var

use ehp_core::constraints::{self, make_leibniz_constraint_single, DiffVar};
use ehp_core::gf2::*;
use ehp_core::io;
use ehp_core::map::MapKind;
use ehp_core::page::SATPage;
use ehp_core::pageturning;
use ehp_core::solver;
use ehp_core::tridegree::Tridegree;
use hashbrown::{HashMap, HashSet};

const DEFAULT_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/E2");

fn stable_rep(t: Tridegree) -> Tridegree {
    if t.n > t.s + 2 {
        Tridegree::new(t.s + 2, t.s, t.f)
    } else {
        t
    }
}

fn env_i32(name: &str, default: i32) -> i32 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// How a candidate pair could reference the target variable.
#[derive(Clone, Copy, PartialEq)]
enum Touch {
    Lhs,  // prod_deg folds to the var degree (LHS: D[prod] entries)
    Rhs1, // deg2 folds to the var degree (RHS1: D[deg2] entries)
    Rhs2, // deg1 folds to the var degree (RHS2: D[deg1] entries)
}

impl Touch {
    fn name(self) -> &'static str {
        match self {
            Touch::Lhs => "LHS(prod=var)",
            Touch::Rhs1 => "RHS1(deg2=var)",
            Touch::Rhs2 => "RHS2(deg1=var)",
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let data = std::env::var("EHP_DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let max_t: i32 = env_i32("EHP_MAX_T", 50);
    let diag_n = env_i32("DIAG_N", 43);
    let diag_s = env_i32("DIAG_S", 34);
    let diag_f = env_i32("DIAG_F", 4);
    let diag_r = env_i32("DIAG_R", 3);

    eprintln!("Target: d_{diag_r} at (n={diag_n}, s={diag_s}, f={diag_f}), max_t={max_t}");
    eprintln!("Loading E_2 (max_t={max_t})...");
    let mut current = io::load_page(&data, 2, max_t)?;

    // Build pages up to E_{diag_r} (loop shape from diag_sweep.rs).
    while current.r < diag_r {
        let r = current.r;
        let known = ehp_server::load_known_diffs(r, None)?;
        let cutoff = current.max_s.unwrap_or(0);
        let sys = constraints::build_constraint_system(&current, cutoff, &known);
        let num_vars = sys.num_vars;
        let result = solver::solve(&sys).ok_or_else(|| format!("E_{r} UNSAT"))?;
        eprintln!(
            "E_{r}: {}/{} vars determined",
            num_vars - result.unknown.len(),
            num_vars
        );
        let (next, _turned) = pageturning::build_next_page(&current, &result)?;
        current = next;
    }
    let page: SATPage = current;
    let r = page.r;
    assert_eq!(r, diag_r);

    let known = ehp_server::load_known_diffs(r, None)?;
    let cutoff = page.max_s.unwrap_or(0);
    eprintln!("\nE_{r}: cutoff (max_s) = {cutoff}, max_t = {:?}", page.max_t);
    let sys = constraints::build_constraint_system(&page, cutoff, &known);
    let res = solver::solve(&sys);
    match &res {
        Some(res) => eprintln!(
            "E_{r}: {}/{} vars determined, {} constraints",
            sys.num_vars - res.unknown.len(),
            sys.num_vars,
            sys.num_constraints()
        ),
        None => eprintln!("E_{r}: UNSAT"),
    }

    // ------------------------------------------------------------------
    // (3) Variable status
    // ------------------------------------------------------------------
    let full = Tridegree::new(diag_n, diag_s, diag_f);
    let rep = Tridegree::new(diag_n.min(diag_s + 2), diag_s, diag_f);
    let tgt = rep.diff_target(r);
    eprintln!("\n=== Variable status ===");
    eprintln!(
        "full degree {full}: dim = {}, is_excluded = {}",
        page.dim_at(full),
        page.is_excluded(full)
    );
    eprintln!(
        "stable rep {rep}: dim = {}, is_excluded = {}",
        page.dim_at(rep),
        page.is_excluded(rep)
    );
    eprintln!(
        "d_{r} target {tgt} (rep {}): dim = {}, is_excluded = {}",
        stable_rep(tgt),
        page.dim_at(tgt),
        page.is_excluded(tgt)
    );

    let src_dim = page.dim_at(rep);
    let tgt_dim = page.dim_at(tgt);
    let mut var_indices: HashSet<usize> = HashSet::new();
    let mut any_var = false;
    for row in 0..tgt_dim.max(1) as u16 {
        for col in 0..src_dim.max(1) as u16 {
            let dv = DiffVar::new(rep.n, rep.s, rep.f, row, col);
            if let Some(&idx) = sys.var_index.get(&dv) {
                any_var = true;
                var_indices.insert(idx);
                let status = match &res {
                    Some(res) => {
                        if res.unknown.contains(&idx) {
                            "UNKNOWN".to_string()
                        } else {
                            format!("determined = {}", vec_get(&res.offset, idx) as u8)
                        }
                    }
                    None => "UNSAT".to_string(),
                };
                let touching = sys
                    .rows
                    .iter()
                    .filter(|row_vec| vec_get(row_vec, idx))
                    .count();
                eprintln!(
                    "  var d_{r}{rep}[{row},{col}] = #{idx}: {status}; {touching} constraints touch it"
                );
            }
        }
    }
    if !any_var {
        eprintln!(
            "  NOT A VARIABLE (make_basis skipped the degree: excluded={}, target excluded={}, src_dim={}, tgt_dim={})",
            page.is_excluded(rep),
            page.is_excluded(tgt),
            src_dim,
            tgt_dim
        );
    }

    // Was any Leibniz pair recorded as skipped-because-excluded at this degree?
    eprintln!("\n=== excluded_leibniz entries mentioning the degree ===");
    for key in [rep, stable_rep(tgt)] {
        match sys.excluded_leibniz.get(&key) {
            Some(pairs) => {
                eprintln!("  key {key}: {} skipped pairs recorded", pairs.len());
                for (d1, d2) in pairs.iter().take(15) {
                    eprintln!("    deg1={d1} deg2={d2}");
                }
                if pairs.len() > 15 {
                    eprintln!("    ... ({} more)", pairs.len() - 15);
                }
            }
            None => eprintln!("  key {key}: none"),
        }
    }

    // ------------------------------------------------------------------
    // (4) Enumerate every Leibniz pair that could reference the variable.
    // ------------------------------------------------------------------
    // Rebuild the exact ingredients of make_leibniz_constraints.
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
    eprintln!("\nmax_data_n = {max_data_n}, cutoff = {cutoff}");

    // n-values that fold to the variable's stable rep n.
    let matches_var_n = |n: i32| -> bool { n.min(diag_s + 2) == rep.n };

    // Collect candidates: pair -> touch modes.
    let mut cands: HashMap<(Tridegree, Tridegree), Vec<Touch>> = HashMap::new();
    let push = |d1: Tridegree, d2: Tridegree, t: Touch, m: &mut HashMap<(Tridegree, Tridegree), Vec<Touch>>| {
        let e = m.entry((d1, d2)).or_default();
        if !e.contains(&t) {
            e.push(t);
        }
    };

    for (&th1, degs) in &degrees_by_n {
        for &(s1, f1) in degs {
            let th2 = th1 + s1 - 1;
            // Case A: product degree == (th1, diag_s, diag_f) folding to rep.
            if matches_var_n(th1) {
                let s2 = diag_s - s1;
                let f2 = diag_f - f1;
                if s2 >= 0 && f2 >= 0 {
                    push(
                        Tridegree::new(th1, s1, f1),
                        Tridegree::new(th2, s2, f2),
                        Touch::Lhs,
                        &mut cands,
                    );
                }
            }
            // Case B: deg2 == (th2, diag_s, diag_f) folding to rep.
            if matches_var_n(th2) {
                push(
                    Tridegree::new(th1, s1, f1),
                    Tridegree::new(th2, diag_s, diag_f),
                    Touch::Rhs1,
                    &mut cands,
                );
            }
            // Case C: deg1 == (th1, diag_s, diag_f) folding to rep.
            if matches_var_n(th1) && (s1, f1) == (diag_s, diag_f) {
                let th2c = th1 + diag_s - 1;
                for (&t2, &d2) in &page.dimension {
                    if t2.n == th2c && d2 > 0 {
                        push(Tridegree::new(th1, s1, f1), t2, Touch::Rhs2, &mut cands);
                    }
                }
            }
        }
    }

    let mut pairs: Vec<((Tridegree, Tridegree), Vec<Touch>)> = cands.into_iter().collect();
    pairs.sort_by_key(|((d1, d2), _)| (d1.n, d1.s, d1.f, d2.n, d2.s, d2.f));
    eprintln!("\n=== {} candidate Leibniz pairs could reference the var ===", pairs.len());

    let mut reject_counts: HashMap<String, usize> = HashMap::new();
    let mut detailed = 0usize;
    let mut emitted_with_var = 0usize;

    for ((deg1, deg2), touches) in &pairs {
        let (deg1, deg2) = (*deg1, *deg2);
        let is_h1_pair =
            (deg1.s, deg1.f) == (1, 1) || (deg2.s, deg2.f) == (1, 1);

        let s3 = deg1.s + deg2.s;
        let f3 = deg1.f + deg2.f;
        let prod_deg = Tridegree::new(deg1.n, s3, f3);
        let prod_dr_deg = prod_deg.diff_target(r);
        let e_deg2 = Tridegree::new(deg2.n + 1, deg2.s, deg2.f);
        let d_deg1 = deg1.diff_target(r);
        let d_deg2 = deg2.diff_target(r);
        let all_trideg = [deg1, deg2, e_deg2, d_deg2, d_deg1, prod_deg, prod_dr_deg];

        // --- Enumeration check: would build_constraint_system's loops
        //     (make_leibniz_constraints) ever visit this pair? ---
        let mut enum_fail: Vec<String> = Vec::new();
        if !(2..=max_data_n).contains(&deg1.n) {
            enum_fail.push(format!("th1={} outside 2..=max_data_n", deg1.n));
        }
        if deg2.n > max_data_n {
            enum_fail.push(format!("th2={} > max_data_n={}", deg2.n, max_data_n));
        }
        if deg1.f < 1 || deg1.s + deg1.f > cutoff || page.dim_at(deg1) == 0 {
            enum_fail.push(format!(
                "deg1 not in degrees_by_n (f1={}, s1+f1={}, dim={})",
                deg1.f,
                deg1.s + deg1.f,
                page.dim_at(deg1)
            ));
        }
        if !(0..=cutoff).contains(&s3) || f3 < 1 || f3 > cutoff - s3 {
            enum_fail.push(format!("(s3,f3)=({s3},{f3}) outside loop range (cutoff {cutoff})"));
        }
        if deg2.s < 0 || deg2.f < 0 {
            enum_fail.push(format!("deg2 negative ({},{})", deg2.s, deg2.f));
        }

        // --- Guard replay, in make_leibniz_constraint_single order ---
        let mut guard_fail: Option<String> = None;
        if !page.is_in_computed_polygon_source(deg2) {
            guard_fail = Some("polygon(deg2)".into());
        } else if page.dim_at(deg2) == 0 {
            guard_fail = Some("dim(deg2)=0".into());
        } else if !page.is_in_computed_polygon_source(prod_deg)
            || !page.is_in_computed_polygon_source(prod_dr_deg)
        {
            guard_fail = Some(format!(
                "polygon(prod {prod_deg} / prod_dr {prod_dr_deg}) [s+f = {}/{} vs max_t {:?}]",
                prod_deg.s + prod_deg.f,
                prod_dr_deg.s + prod_dr_deg.f,
                page.max_t
            ));
        } else if page.dim_at(prod_dr_deg) == 0 {
            guard_fail = Some(format!("dim(prod_dr {prod_dr_deg})=0"));
        } else {
            let excluded: Vec<String> = all_trideg
                .iter()
                .filter(|&&td| page.is_excluded(td))
                .map(|td| format!("{td}~rep{}", stable_rep(*td)))
                .collect();
            if !excluded.is_empty() {
                guard_fail = Some(format!("EXCLUDED degrees: {}", excluded.join(", ")));
            } else if page.dim_at(deg1) == 0 {
                guard_fail = Some("dim(deg1)=0".into());
            }
        }

        let touch_names: Vec<&str> = touches.iter().map(|t| t.name()).collect();

        if let Some(fail) = &guard_fail {
            let key = if let Some(rest) = enum_fail.first() {
                format!("NOT-ENUMERATED [{rest}] + guard [{fail}]")
            } else {
                format!("guard [{fail}]")
            };
            *reject_counts.entry(short_reason(fail, &enum_fail)).or_default() += 1;
            // Detail only for h1 pairs and late-guard (exclusion) rejects.
            if is_h1_pair || fail.starts_with("EXCLUDED") {
                detailed += 1;
                if detailed <= 200 {
                    eprintln!(
                        "{} deg1={deg1} deg2={deg2} [{}] REJECTED: {key}",
                        if is_h1_pair { "[h1]" } else { "    " },
                        touch_names.join("+")
                    );
                }
            }
            continue;
        }
        if !enum_fail.is_empty() {
            *reject_counts
                .entry(format!("NOT-ENUMERATED: {}", enum_fail.join("; ")))
                .or_default() += 1;
            detailed += 1;
            eprintln!(
                "{} deg1={deg1} deg2={deg2} [{}] NOT ENUMERATED by build_constraint_system: {}",
                if is_h1_pair { "[h1]" } else { "    " },
                touch_names.join("+"),
                enum_fail.join("; ")
            );
            // Still fall through to see what it WOULD emit.
        }

        // Guards pass — run the real generator.
        let cons = make_leibniz_constraint_single(&page, deg1, deg2, &sys.var_index, None);
        let n_with_var = cons
            .iter()
            .filter(|c| c.iter().any(|i| var_indices.contains(i)))
            .count();

        // Deep analysis of term structure for the skip()/vacuous cases.
        let dim1 = page.dim_at(deg1);
        let dim2 = page.dim_at(deg2);
        let prod_dr_dim = page.dim_at(prod_dr_deg);
        let e_matrix = page.map_matrix(MapKind::E, deg2);
        let e_stored = page.maps[&MapKind::E].matrix_at(deg2).is_some();

        let mut prod_nonzero = false;
        for i2 in 0..dim2 {
            let e_elt2 = mat_vec_mul(&e_matrix, &vec_basis(dim2, i2));
            for i1 in 0..dim1 {
                let p = page.products.multiply(
                    deg1,
                    &vec_basis(dim1, i1),
                    e_deg2,
                    &e_elt2,
                    page.dim_at(prod_deg),
                );
                if !p.is_zero() {
                    prod_nonzero = true;
                }
            }
        }

        // RHS1 term: Ytilde(deg1, basis of target_deg) combined with E at d_deg2
        let target_deg = Tridegree::new(deg2.n + 1, deg2.s - 1, deg2.f + r);
        let t1_dim = page.dim_at(target_deg);
        let mut ytil1_nonzero = false;
        if deg2.s != 0 && t1_dim > 0 {
            for i1 in 0..dim1 {
                for j in 0..t1_dim {
                    let p = page.products.multiply(
                        deg1,
                        &vec_basis(dim1, i1),
                        target_deg,
                        &vec_basis(t1_dim, j),
                        prod_dr_dim,
                    );
                    if !p.is_zero() {
                        ytil1_nonzero = true;
                    }
                }
            }
        }
        let e_at_d_stored = page.maps[&MapKind::E].matrix_at(d_deg2).is_some();
        let e_at_d_zero = mat_is_zero(&page.map_matrix(MapKind::E, d_deg2));

        // RHS2 term: Ytilde(basis of d_deg1, deg2)
        let t2_dim = page.dim_at(d_deg1);
        let mut ytil2_nonzero = false;
        if deg1.s != 0 && t2_dim > 0 {
            for j in 0..t2_dim {
                for i2 in 0..dim2 {
                    let p = page.products.multiply(
                        d_deg1,
                        &vec_basis(t2_dim, j),
                        deg2,
                        &vec_basis(dim2, i2),
                        prod_dr_dim,
                    );
                    if !p.is_zero() {
                        ytil2_nonzero = true;
                    }
                }
            }
        }

        // Variable-block presence for the three referenced differentials.
        let block = |t: Tridegree| -> bool {
            let rp = stable_rep(t);
            sys.var_index
                .contains_key(&DiffVar::new(rp.n, rp.s, rp.f, 0, 0))
        };
        let prod_block = block(prod_deg);
        let deg1_block = block(deg1);
        let deg2_block = block(deg2);

        let mut notes: Vec<String> = Vec::new();
        if cons.is_empty() {
            if prod_nonzero && !prod_block {
                notes.push(format!(
                    "SKIP(): LHS product nonzero but no var block at prod rep {}",
                    stable_rep(prod_deg)
                ));
            }
            if ytil1_nonzero && !e_at_d_zero && !deg2_block {
                notes.push(format!(
                    "SKIP(): RHS1 (Ytil1*E) nonzero but no var block at deg2 rep {}",
                    stable_rep(deg2)
                ));
            }
            if ytil2_nonzero && !deg1_block {
                notes.push(format!(
                    "SKIP(): RHS2 Ytil2 nonzero but no var block at deg1 rep {}",
                    stable_rep(deg1)
                ));
            }
            if notes.is_empty() {
                notes.push("vacuous: all terms zero (product/Ytilde all zero or cancel)".into());
            }
        }

        let interesting = n_with_var > 0 || is_h1_pair || cons.is_empty();
        if n_with_var > 0 {
            emitted_with_var += cons.len().min(n_with_var);
        }
        if interesting {
            detailed += 1;
            eprintln!(
                "{} deg1={deg1} deg2={deg2} [{}] dims(d1={dim1},d2={dim2},prod={},prod_dr={prod_dr_dim}) \
                 constraints={} WITH_VAR={} | prod!=0:{} ytil1!=0:{} ytil2!=0:{} \
                 E@deg2 stored:{} E@d(deg2) stored:{}(zero:{}) blocks(prod:{},deg1:{},deg2:{}) {}",
                if is_h1_pair { "[h1]" } else { "    " },
                touch_names.join("+"),
                page.dim_at(prod_deg),
                cons.len(),
                n_with_var,
                prod_nonzero,
                ytil1_nonzero,
                ytil2_nonzero,
                e_stored,
                e_at_d_stored,
                e_at_d_zero,
                prod_block,
                deg1_block,
                deg2_block,
                notes.join("; ")
            );
        } else {
            *reject_counts
                .entry("emitted constraints but none touch the var".into())
                .or_default() += 1;
        }
    }

    eprintln!("\n=== Rejection summary ===");
    let mut counts: Vec<(&String, &usize)> = reject_counts.iter().collect();
    counts.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
    for (reason, count) in counts {
        eprintln!("  {count:5}  {reason}");
    }
    eprintln!("\nPairs emitting constraints containing the var: {emitted_with_var}");

    // ------------------------------------------------------------------
    // (5) h1-specific report: where does h1 live relative to this class?
    // ------------------------------------------------------------------
    eprintln!("\n=== h1 landscape around the target class ===");
    eprintln!("(h_i lives at (n+s, hi_stem, 1); pairs put the h_i factor at th2 = th1+s1-1)");
    for nn in [diag_n + diag_s - 2, diag_n + diag_s - 1, diag_n + diag_s] {
        let h1t = Tridegree::new(nn, 1, 1);
        eprintln!(
            "  ({nn},1,1): dim = {}, excluded = {}, polygon_src = {}",
            page.dim_at(h1t),
            page.is_excluded(h1t),
            page.is_in_computed_polygon_source(h1t)
        );
    }
    // The two canonical h1 pairs:
    // (a) our class times h1: deg1 = full, deg2 = (n+s-1, 1, 1)
    // (b) x times h1 = our class: deg1 = (n, s-1, f-1), deg2 = (n+s-2, 1, 1)
    let pa = (full, Tridegree::new(diag_n + diag_s - 1, 1, 1));
    let pb = (
        Tridegree::new(diag_n, diag_s - 1, diag_f - 1),
        Tridegree::new(diag_n + diag_s - 2, 1, 1),
    );
    for (label, (d1, d2)) in [("class*h1 (var via RHS2)", pa), ("x*h1=class (var via LHS)", pb)] {
        eprintln!(
            "  {label}: deg1={d1} (dim {}) deg2={d2} (dim {})",
            page.dim_at(d1),
            page.dim_at(d2)
        );
        // product presence checks on this page
        let e_d2 = Tridegree::new(d2.n + 1, d2.s, d2.f);
        let prod = Tridegree::new(d1.n, d1.s + d2.s, d1.f + d2.f);
        if page.dim_at(d1) > 0 && page.dim_at(e_d2) > 0 {
            let p = page.products.multiply(
                d1,
                &vec_basis(page.dim_at(d1), 0),
                e_d2,
                &vec_basis(page.dim_at(e_d2), 0),
                page.dim_at(prod),
            );
            eprintln!(
                "    product (basis0 x basis0) at {prod}: {}",
                if p.is_zero() { "ZERO/missing" } else { "NONZERO" }
            );
        }
    }

    // ------------------------------------------------------------------
    // (6) Counterfactual: if the var's degree (and its diff target) were NOT
    // excluded, would the constraint system determine it? (Diagnosis only —
    // clones the page, removes the exclusions, rebuilds, re-solves.)
    // ------------------------------------------------------------------
    if page.is_excluded(rep) || page.is_excluded(tgt) {
        eprintln!("\n=== Counterfactual: un-exclude {rep} (and target) and re-solve ===");
        let mut page2 = page.clone();
        page2.exclude_set.remove(&rep);
        page2.exclude_set.remove(&stable_rep(tgt));
        let sys2 = constraints::build_constraint_system(&page2, cutoff, &known);
        eprintln!(
            "  rebuilt: {} vars ({} before), {} constraints ({} before)",
            sys2.num_vars,
            sys.num_vars,
            sys2.num_constraints(),
            sys.num_constraints()
        );
        match solver::solve(&sys2) {
            None => eprintln!("  UNSAT after un-excluding (contradiction)"),
            Some(res2) => {
                for row in 0..tgt_dim.max(1) as u16 {
                    for col in 0..src_dim.max(1) as u16 {
                        let dv = DiffVar::new(rep.n, rep.s, rep.f, row, col);
                        if let Some(&idx) = sys2.var_index.get(&dv) {
                            let touching = sys2
                                .rows
                                .iter()
                                .filter(|row_vec| vec_get(row_vec, idx))
                                .count();
                            if res2.unknown.contains(&idx) {
                                eprintln!(
                                    "  var d_{r}{rep}[{row},{col}]: still UNKNOWN ({touching} constraints touch it)"
                                );
                            } else {
                                eprintln!(
                                    "  var d_{r}{rep}[{row},{col}]: DETERMINED = {} ({touching} constraints touch it)",
                                    vec_get(&res2.offset, idx) as u8
                                );
                            }
                        } else {
                            eprintln!("  var d_{r}{rep}[{row},{col}]: still not a variable");
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Collapse a guard failure into a short bucket name for the summary.
fn short_reason(fail: &str, enum_fail: &[String]) -> String {
    let base = if fail.starts_with("EXCLUDED") {
        "guard: excluded degree".to_string()
    } else if fail.starts_with("polygon(prod") {
        "guard: prod/prod_dr out of polygon".to_string()
    } else if fail.starts_with("polygon(deg2)") {
        "guard: deg2 out of polygon".to_string()
    } else if fail.starts_with("dim(prod_dr") {
        "guard: dim(prod_dr)=0".to_string()
    } else {
        format!("guard: {fail}")
    };
    if enum_fail.is_empty() {
        base
    } else {
        format!("{base} (also not enumerated)")
    }
}
