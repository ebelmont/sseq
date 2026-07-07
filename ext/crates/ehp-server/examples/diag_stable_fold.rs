//! Headless check of the stable-fold invariant under the REPL's incremental
//! cascade re-turn (the stale-stable-copy soundness bug).
//!
//! Replays exactly what `cascade_resolve` in ehp_chart.rs does after an `add`
//! at a stable-rep degree: re-solve, compute the changed/affected degrees,
//! patch the cached turned map at those degrees only, rebuild the next page,
//! then verify dim(n,s,f) == dim(s+2,s,f) for every past-stable key.
//!
//! - `DIAG_EXPAND=1` (default): affected set is expanded across the stable
//!   fold (the fix) — the invariant must HOLD (PASS).
//! - `DIAG_EXPAND=0`: old rep-only affected set — the invariant must FIRE
//!   (FAIL lines), proving both the bug and the guardrail detector.
//!
//! Run: `EHP_MAX_T=50 cargo run -p ehp-server --release --example diag_stable_fold`

use ehp_core::constraints::{self, DiffVar};
use ehp_core::page::SATPage;
use ehp_core::result::SATResult;
use ehp_core::tridegree::Tridegree;
use ehp_core::{io, pageturning, solver};
use hashbrown::{HashMap, HashSet};

const DEFAULT_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/E2");

/// Turning-effective value (mirrors ehp_chart.rs::effective_var_value).
fn eff(res: &SATResult, var: &DiffVar) -> bool {
    match res.var_index.get(var) {
        Some(&i) => !res.unknown.contains(&i) && res.offset.entry(i) != 0,
        None => false,
    }
}

/// Stable-fold violations (mirrors ehp_chart.rs::stable_fold_violations).
fn fold_violations(page: &SATPage) -> Vec<(Tridegree, usize, usize)> {
    let mut v: Vec<(Tridegree, usize, usize)> = page
        .dimension
        .iter()
        .filter(|(t, _)| t.n > t.s + 2)
        .filter_map(|(&t, &d)| {
            let rep_dim = page.dim_at(Tridegree::new(t.s + 2, t.s, t.f));
            (rep_dim != d).then_some((t, d, rep_dim))
        })
        .collect();
    v.sort_by_key(|(t, _, _)| *t);
    v
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let data = std::env::var("EHP_DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let max_t: i32 = std::env::var("EHP_MAX_T").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    let expand = std::env::var("DIAG_EXPAND").map(|v| v != "0").unwrap_or(true);

    eprintln!("Loading E_2 (max_t={max_t}, expand={expand})...");
    let page = io::load_page(&data, 2, max_t)?;
    let cutoff = page.max_s.unwrap_or(0);

    // Base solve (empty known diffs).
    let known: HashMap<DiffVar, bool> = HashMap::new();
    let system = constraints::build_constraint_system(&page, cutoff, &known);
    let result = solver::solve(&system).expect("E2 base solve failed");
    eprintln!(
        "E_2: {}/{} vars determined",
        system.num_vars - result.unknown.len(),
        system.num_vars,
    );

    // Startup sanity: the fold invariant must hold on the loaded page and on
    // a freshly (fully) turned next page.
    let v_src = fold_violations(&page);
    let (base_next, turned) = pageturning::build_next_page(&page, &result)?;
    let v_full = fold_violations(&base_next);
    eprintln!(
        "Sanity: fold violations — loaded E_2: {}, full-turned E_3: {}",
        v_src.len(),
        v_full.len(),
    );
    if !v_src.is_empty() || !v_full.is_empty() {
        eprintln!("UNEXPECTED: invariant does not hold at startup; aborting.");
        std::process::exit(2);
    }

    // Past-stable copies by (s, f), from the page keys (as SourceIndex does).
    let mut copies_by_sf: HashMap<(i32, i32), Vec<Tridegree>> = HashMap::new();
    for &t in page.page.keys() {
        if t.n > t.s + 2 {
            copies_by_sf.entry((t.s, t.f)).or_default().push(t);
        }
    }

    // Pick an UNKNOWN d_2 variable at a stable rep (n == s+2) that has
    // past-stable copies — asserting it =1 changes the fold's homology.
    let mut candidates: Vec<DiffVar> = result
        .unknown
        .iter()
        .map(|&i| result.vars[i])
        .filter(|v| v.n == v.s + 2 && copies_by_sf.contains_key(&(v.s, v.f)))
        .collect();
    candidates.sort();
    let var = *candidates.first().expect("no unknown stable-rep d2 var with copies found");
    eprintln!(
        "Asserting d_2({},{},{})[{},{}] = 1 ({} candidates; {} copies at (s,f))",
        var.n, var.s, var.f, var.row, var.col,
        candidates.len(),
        copies_by_sf[&(var.s, var.f)].len(),
    );

    // Re-solve with the assertion (the REPL `add` path).
    let mut known2 = known.clone();
    known2.insert(var, true);
    let system2 = constraints::build_constraint_system(&page, cutoff, &known2);
    let result2 = solver::solve(&system2).expect("mutated solve came back UNSAT — pick another var");

    // Incremental re-turn, exactly as cascade_resolve does.
    let mut changed: HashSet<Tridegree> = HashSet::new();
    for v in &result2.vars {
        if eff(&result2, v) != eff(&result, v) {
            changed.insert(Tridegree::new(v.n, v.s, v.f));
        }
    }
    let mut affected: HashSet<Tridegree> = HashSet::new();
    for &t in &changed {
        affected.insert(t);
        affected.insert(t.diff_target(2));
    }
    if expand {
        // THE FIX: stable-fold expansion (mirrors ehp_chart.rs::expand_stable_fold).
        let seeds: Vec<Tridegree> =
            affected.iter().copied().filter(|t| t.n >= t.s + 2).collect();
        for t in seeds {
            affected.insert(Tridegree::new(t.s + 2, t.s, t.f));
            if let Some(copies) = copies_by_sf.get(&(t.s, t.f)) {
                affected.extend(copies.iter().copied());
            }
        }
    }
    eprintln!(
        "{} changed tridegrees, {} affected for re-turning",
        changed.len(),
        affected.len(),
    );

    let mut turned = turned;
    let ctx = pageturning::TurnContext::new(&page, &result2);
    for &t in &affected {
        if page.dim_at(t) == 0 {
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
            Err(e) => {
                eprintln!("d∘d ≠ 0 contradiction at {:?} — pick another var", e.degree);
                std::process::exit(2);
            }
        }
    }
    let rebuilt = pageturning::build_page_from_turned(&page, &turned);

    // Did the mutation actually change the fold's dims? (Otherwise the
    // scenario proves nothing.)
    let rep_col = Tridegree::new(var.s + 2, var.s, var.f);
    let tgt = rep_col.diff_target(2);
    eprintln!(
        "E_3 dims: source rep {} -> {} (was {}), target {} -> {} (was {})",
        rep_col,
        rebuilt.dim_at(rep_col),
        base_next.dim_at(rep_col),
        tgt,
        rebuilt.dim_at(tgt),
        base_next.dim_at(tgt),
    );

    let v = fold_violations(&rebuilt);
    if v.is_empty() {
        println!("PASS (expand={expand}): stable fold coherent after incremental rebuild.");
    } else {
        println!(
            "FOLD VIOLATIONS (expand={expand}): {} past-stable degree(s) disagree with their rep:",
            v.len(),
        );
        for (t, d, rep_dim) in v.iter().take(10) {
            println!(
                "  dim({},{},{}) = {} but rep dim({},{},{}) = {}",
                t.n, t.s, t.f, d, t.s + 2, t.s, t.f, rep_dim,
            );
        }
        if v.len() > 10 {
            println!("  ... and {} more", v.len() - 10);
        }
        std::process::exit(1);
    }
    Ok(())
}
