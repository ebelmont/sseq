//! Headless `sweep <r>`: build pages, then trial-error sweep every unknown
//! d_r (both values), propagating through all built pages. Exercises the
//! multi-step overlay path (the stale-coordinate panic reproduced here).
//!
//! Run: `EHP_MAX_T=50 DIAG_R=3 cargo run -p ehp-server --release --example diag_sweep`

use ehp_core::constraints::{self, ExcludedLeibniz};
use ehp_core::interpage::{self, InterpagePage};
use ehp_core::page::SATPage;
use ehp_core::pageturning::{self, TurnedBidegree};
use ehp_core::result::SATResult;
use ehp_core::tridegree::Tridegree;
use ehp_core::{io, solver};
use hashbrown::HashMap;
use std::time::Instant;

const DEFAULT_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/E2");

struct Page {
    page: SATPage,
    result: SATResult,
    excluded_leibniz: ExcludedLeibniz,
    turned: Option<HashMap<Tridegree, TurnedBidegree>>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let data = std::env::var("EHP_DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let max_t: i32 = std::env::var("EHP_MAX_T").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    let max_r: i32 = std::env::var("EHP_MAX_R").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
    let start_r: i32 = std::env::var("DIAG_R").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let min_stem: i32 = std::env::var("DIAG_MIN_STEM").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let max_stem: i32 = std::env::var("DIAG_MAX_STEM").ok().and_then(|s| s.parse().ok()).unwrap_or(i32::MAX);

    eprintln!("Loading E_2 (max_t={max_t})...");
    let mut current = io::load_page(&data, 2, max_t)?;
    let mut pages: Vec<Page> = Vec::new();

    loop {
        let r = current.r;
        let known = ehp_server::load_known_diffs(r, None)?;
        let cutoff = current.max_t.unwrap_or(0);
        let sys = constraints::build_constraint_system(&current, cutoff, &known);
        let num_vars = sys.num_vars;
        let excluded_leibniz = sys.excluded_leibniz.clone();
        let Some(result) = solver::solve(&sys) else {
            eprintln!("E_{r}: UNSAT base system, stopping");
            break;
        };
        eprintln!(
            "E_{r}: {}/{} vars determined",
            num_vars - result.unknown.len(),
            num_vars
        );

        let done = num_vars == 0 || r >= max_r;
        let mut entry = Page { page: current.clone(), result, excluded_leibniz, turned: None };
        if !done {
            match pageturning::build_next_page(&entry.page, &entry.result) {
                Ok((next, turned)) => {
                    entry.turned = Some(turned);
                    pages.push(entry);
                    current = next;
                    continue;
                }
                Err(e) => {
                    eprintln!("E_{r}: contradiction while turning: {e}");
                    pages.push(entry);
                    break;
                }
            }
        }
        pages.push(entry);
        break;
    }

    let Some(idx) = pages.iter().position(|p| p.page.r == start_r) else {
        eprintln!("no E_{start_r} page built");
        return Ok(());
    };

    let ip: Vec<InterpagePage> = pages[idx..]
        .iter()
        .map(|p| InterpagePage {
            page: &p.page,
            result: &p.result,
            excluded_leibniz: &p.excluded_leibniz,
        })
        .collect();

    let n_unknown = ip[0].result.unknown.len();
    eprintln!(
        "\nSweeping {} unknown d_{} vars (× 2 values) through E_{}...",
        n_unknown,
        start_r,
        ip.last().map(|p| p.page.r).unwrap_or(start_r),
    );
    let t0 = Instant::now();
    let findings = interpage::trial_error_sweep(&ip, min_stem, max_stem, |done, total| {
        if done % 100 == 0 || done == total {
            eprintln!("  {}/{} trials ({:.1}s)", done, total, t0.elapsed().as_secs_f64());
        }
    });
    eprintln!("Sweep finished in {:.1}s.", t0.elapsed().as_secs_f64());

    if findings.is_empty() {
        eprintln!("No contradictions — no values forced.");
    } else {
        eprintln!("Forced values ({}):", findings.len());
        for f in &findings {
            eprintln!(
                "  d_{}({},{},{})[{},{}] = {} forced ({} when assuming {})",
                start_r, f.var.n, f.var.s, f.var.f, f.var.row, f.var.col,
                (!f.contradicted_value) as u8, f.error, f.contradicted_value as u8,
            );
        }
    }

    Ok(())
}
