//! Headless replica of the ehp_chart REPL startup: load E2, then
//! solve → turn → solve … up to EHP_MAX_R, with per-page timing and
//! SAT/UNSAT reporting. No charts, no Python — fast repro harness for
//! startup UNSAT and startup-performance questions.
//!
//! Env (same as the REPL): EHP_DATA, EHP_MAX_T, EHP_R, EHP_MAX_R,
//! EHP_OUTSIDE_DIFFS, EHP_RELAX_TARGET_EXCLUDE.

use std::time::Instant;

use ehp_core::{constraints, io, pageturning, solver};

const DEFAULT_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/E2");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let data_path = std::env::var("EHP_DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let max_t: i32 = std::env::var("EHP_MAX_T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let start_r: i32 = std::env::var("EHP_R")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let max_r: i32 = std::env::var("EHP_MAX_R")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    eprintln!("diag_startup: data={} max_t={} max_r={}", data_path, max_t, max_r);
    let t_all = Instant::now();

    let t0 = Instant::now();
    let mut current_page = io::load_page(&data_path, start_r, max_t)?;
    eprintln!("Loaded E_{} in {:.2}s", start_r, t0.elapsed().as_secs_f64());

    let mut unsat_page: Option<i32> = None;
    loop {
        let r = current_page.r;
        let t1 = Instant::now();
        let known_diffs = ehp_server::load_known_diffs_for_page(&current_page, None)?;
        let n_known = known_diffs.len();
        let cutoff = current_page.max_t.unwrap_or(0);
        let system = constraints::build_constraint_system(&current_page, cutoff, &known_diffs);
        let t_build = t1.elapsed().as_secs_f64();
        let num_vars = system.num_vars;
        let t2 = Instant::now();
        let result = solver::solve_with_d2(&current_page, &system).map(|(res, d2n)| {
            if d2n > 0 {
                eprintln!("E_{}: d2-linearization determined {} more entries", r, d2n);
            }
            res
        });
        let t_solve = t2.elapsed().as_secs_f64();

        match result {
            Some(ref res) => {
                let determined = num_vars - res.unknown.len();
                eprintln!(
                    "E_{}: SAT {}/{} vars determined | {} known diffs | build {:.2}s solve {:.2}s",
                    r, determined, num_vars, n_known, t_build, t_solve,
                );
            }
            None if num_vars == 0 => {
                eprintln!("E_{}: no differential variables", r);
            }
            None => {
                eprintln!(
                    "E_{}: *** UNSAT *** ({} vars, {} known diffs, build {:.2}s solve {:.2}s)",
                    r, num_vars, n_known, t_build, t_solve,
                );
                unsat_page = Some(r);
            }
        }

        if result.is_none() || num_vars == 0 || r >= max_r {
            break;
        }

        let t3 = Instant::now();
        let (next_page, _turned) =
            match pageturning::build_next_page(&current_page, result.as_ref().unwrap()) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("E_{}: contradiction while turning: {}", r, e);
                    break;
                }
            };
        let nonzero = next_page.dimension.values().filter(|&&d| d > 0).count();
        eprintln!(
            "Turned E_{} -> E_{}: {} tridegrees ({:.2}s)",
            r,
            r + 1,
            nonzero,
            t3.elapsed().as_secs_f64(),
        );
        if nonzero == 0 {
            break;
        }
        current_page = next_page;
    }

    eprintln!("total: {:.2}s", t_all.elapsed().as_secs_f64());
    match unsat_page {
        Some(r) => {
            eprintln!("RESULT: UNSAT at E_{}", r);
            std::process::exit(2);
        }
        None => eprintln!("RESULT: all pages SAT"),
    }
    Ok(())
}
