//! Reproduce the interpage propagation failure: assuming a value for the
//! uncertain d_3 at S^6 (34, 10) should contradict on E_4 (so the diff is
//! forced nonzero), but `try` reports nothing.
//!
//! Builds pages exactly like ehp_chart (solve, turn, solve, ...) without any
//! chart generation, then runs `try_diffs` for both values of every unknown
//! d_3 variable at (6, 34, 10), printing per-stage diagnostics.
//!
//! Run: `EHP_MAX_T=50 RUST_LOG=debug cargo run -p ehp-server --release --example diag_interpage`

use ehp_core::constraints::{self, DiffVar, ExcludedLeibniz};
use ehp_core::interpage::{self, InterpagePage};
use ehp_core::page::SATPage;
use ehp_core::pageturning::{self, TurnedBidegree};
use ehp_core::result::SATResult;
use ehp_core::tridegree::Tridegree;
use ehp_core::{io, solver};
use hashbrown::HashMap;

const DEFAULT_DATA: &str = concat!(env!("HOME"), "/ehp-sat-rs/data/E2.ehp");

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

    let target = Tridegree::new(
        std::env::var("DIAG_N").ok().and_then(|s| s.parse().ok()).unwrap_or(6),
        std::env::var("DIAG_S").ok().and_then(|s| s.parse().ok()).unwrap_or(34),
        std::env::var("DIAG_F").ok().and_then(|s| s.parse().ok()).unwrap_or(10),
    );
    let start_r: i32 = std::env::var("DIAG_R").ok().and_then(|s| s.parse().ok()).unwrap_or(3);

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
            "E_{r}: {}/{} vars determined, {} excluded degrees",
            num_vars - result.unknown.len(),
            num_vars,
            current.exclude_set.len()
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

    // Status of the target degree on the starting page.
    let ps = &pages[idx];
    eprintln!(
        "\n=== E_{start_r} at ({}, {}, {}): dim {}, target dim {}, excluded {}",
        target.n, target.s, target.f,
        ps.page.dim_at(target),
        ps.page.dim_at(target.diff_target(start_r)),
        ps.page.is_excluded(target),
    );
    let unknown_here: Vec<DiffVar> = ps
        .result
        .unknown
        .iter()
        .map(|&i| ps.result.vars[i])
        .filter(|v| v.n == target.n && v.s == target.s && v.f == target.f)
        .collect();
    eprintln!("unknown d_{start_r} vars at that degree: {unknown_here:?}");
    if unknown_here.is_empty() {
        eprintln!("nothing to try — degree fully determined or not a variable");
        return Ok(());
    }

    // Exclusion status of the surrounding degrees on the next page.
    if let Some(np) = pages.get(idx + 1) {
        for d in [target, target.diff_target(start_r)] {
            eprintln!(
                "E_{} at ({}, {}, {}): dim {}, excluded {}",
                np.page.r, d.n, d.s, d.f, np.page.dim_at(d), np.page.is_excluded(d),
            );
        }
    }

    let ip: Vec<InterpagePage> = pages[idx..]
        .iter()
        .map(|p| InterpagePage {
            page: &p.page,
            result: &p.result,
            excluded_leibniz: &p.excluded_leibniz,
        })
        .collect();

    for &var in &unknown_here {
        for val in [false, true] {
            eprintln!(
                "\n--- try d_{start_r}({},{},{})[{},{}] = {} ---",
                var.n, var.s, var.f, var.row, var.col, val as u8
            );
            match interpage::try_diffs(&ip, &[(var, val)]) {
                Ok(learned) => {
                    eprintln!("CONSISTENT, {} newly determined:", learned.len());
                    for ld in learned.iter().take(30) {
                        eprintln!(
                            "  d_{}({},{},{})[{},{}] = {}",
                            ld.r, ld.var.n, ld.var.s, ld.var.f, ld.var.row, ld.var.col,
                            ld.value as u8
                        );
                    }
                }
                Err(e) => eprintln!("CONTRADICTION: {e}"),
            }
        }
    }

    Ok(())
}
