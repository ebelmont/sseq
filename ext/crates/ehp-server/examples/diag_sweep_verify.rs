//! Parity harness for sweep optimizations: run every trial of a headless
//! `sweep <r>` (like diag_sweep) but print each trial's full outcome — the
//! complete sorted learned-diff list or the error — so two builds can be
//! diffed line-by-line for identical behavior.
//!
//! Run: `EHP_MAX_T=50 DIAG_R=3 cargo run -p ehp-server --release --example diag_sweep_verify > out.log`

use ehp_core::constraints::{self, ExcludedLeibniz};
use ehp_core::interpage::{self, InterpagePage};
use ehp_core::page::SATPage;
use ehp_core::result::SATResult;
use ehp_core::{io, solver};
use rayon::prelude::*;
use std::time::Instant;

const DEFAULT_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/E2");

struct Page {
    page: SATPage,
    result: SATResult,
    excluded_leibniz: ExcludedLeibniz,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
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
        let cutoff = current.max_s.unwrap_or(0);
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
        let entry = Page { page: current.clone(), result, excluded_leibniz };
        if !done {
            match pageturning::build_next_page(&entry.page, &entry.result) {
                Ok((next, _turned)) => {
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

    let mut unknown_vars: Vec<constraints::DiffVar> = ip[0]
        .result
        .unknown
        .iter()
        .map(|&i| ip[0].result.vars[i])
        .filter(|v| v.s >= min_stem && v.s < max_stem)
        .collect();
    unknown_vars.sort();
    let jobs: Vec<(constraints::DiffVar, bool)> = unknown_vars
        .iter()
        .flat_map(|&v| [(v, false), (v, true)])
        .collect();
    eprintln!("Verifying {} trials on E_{start_r}...", jobs.len());

    let t0 = Instant::now();
    let mut lines: Vec<String> = jobs
        .par_iter()
        .map(|&(var, value)| {
            let head = format!(
                "trial d{start_r}({},{},{})[{},{}]={}",
                var.n, var.s, var.f, var.row, var.col, value as u8
            );
            match interpage::try_diffs(&ip, &[(var, value)]) {
                Ok(learned) => {
                    let mut items: Vec<String> = learned
                        .iter()
                        .map(|l| {
                            format!(
                                "r{}:({},{},{})[{},{}]={}",
                                l.r, l.var.n, l.var.s, l.var.f, l.var.row, l.var.col,
                                l.value as u8
                            )
                        })
                        .collect();
                    items.sort();
                    format!("{head} OK {} {}", items.len(), items.join(";"))
                }
                Err(e) => format!("{head} ERR {e}"),
            }
        })
        .collect();
    lines.sort();
    for l in &lines {
        println!("{l}");
    }
    eprintln!("Done in {:.1}s.", t0.elapsed().as_secs_f64());
    if let Some(report) = interpage::trial_stats::report_and_reset() {
        eprintln!("[timing] trial stages: {}", report);
    }
    Ok(())
}

use ehp_core::pageturning;
