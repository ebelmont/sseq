//! Walk E_2 up through E_{EHP_R_MAX} ONCE, dumping each page's own
//! constraint system (output/rust_E{r}_data, same format as
//! diag_dump_constraints) and each page's own rank/products/E/H/P tables
//! (output/rust_E{r}_page, same format as diag_dump_page) as it passes
//! through r = 2..=EHP_R_MAX.
//!
//! diag_dump_constraints and diag_dump_page each independently re-walk
//! from E_2 on every invocation (no cross-invocation caching -- see
//! notes/SESSION_3STAGE_E2E5_t100.md), so producing E2/E3/E4/E5 data by
//! calling them once per target page (EHP_R=2, then 3, then 4, then 5)
//! redundantly re-solves every earlier page over and over (E2's ~40min
//! solve alone got redone 4 times). This binary does the whole walk once
//! and dumps every intermediate page as it goes.
//!
//! Run: EHP_MAX_T=<t> EHP_R_MAX=<r> cargo run -p ehp-server --release --example diag_dump_page_tower

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use ehp_core::constraints;
use ehp_core::constraints::DiffVar;
use ehp_core::gf2;
use ehp_core::io;
use ehp_core::map::MapKind;
use ehp_core::page::SATPage;
use ehp_core::pageturning;
use ehp_core::result::SATResult;
use ehp_core::solver;

const DEFAULT_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/E2");

fn write_lines(path: &Path, lines: impl Iterator<Item = String>) -> std::io::Result<usize> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    let mut count = 0;
    for line in lines {
        writeln!(w, "{line}")?;
        count += 1;
    }
    Ok(count)
}

fn format_indices(indices: &[usize]) -> String {
    let parts: Vec<String> = indices.iter().map(|i| i.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

/// Dump page `r`'s own constraint system (output/rust_E{r}_data), mirroring
/// diag_dump_constraints.rs exactly. Returns the solve result so the caller
/// can turn the page onward without re-solving.
fn dump_page_data(
    page: &SATPage,
    known_diffs: &hashbrown::HashMap<DiffVar, bool>,
) -> Result<SATResult, Box<dyn std::error::Error>> {
    let r = page.r;
    let cutoff = page.max_t.unwrap_or(0);
    let system = constraints::build_constraint_system(page, cutoff, known_diffs);
    debug_assert_eq!(system.vars.len(), system.num_vars);
    eprintln!("[E_{r}] Variable count: {}", system.num_vars);

    let mut naturality: Vec<(MapKind, Vec<Vec<usize>>)> = Vec::new();
    for map_kind in MapKind::all() {
        let nat = constraints::make_naturality_constraints(page, cutoff, map_kind, &system.var_index);
        naturality.push((map_kind, nat));
    }
    let (leibniz, _excluded_leibniz) = constraints::make_leibniz_constraints(page, cutoff, &system.var_index);
    let known_constraints = constraints::make_known_constraints(known_diffs, &system.var_index);

    let out_dir = format!("output/rust_E{r}_data");
    std::fs::create_dir_all(&out_dir)?;
    let out_dir = Path::new(&out_dir);

    // Dump exclude_list before solving, so it's available even if the solve is UNSAT.
    let mut exclude_sorted: Vec<_> = page.exclude_set.iter().copied().collect();
    exclude_sorted.sort();
    write_lines(
        &out_dir.join("exclude_list"),
        exclude_sorted.iter().map(|t| format!("({}, {}, {})", t.n, t.s, t.f)),
    )?;

    eprintln!("[E_{r}] Solving...");
    let (result, _d2n) = solver::solve_with_d2(page, &system)
        .ok_or_else(|| format!("E_{r} is UNSAT (inconsistent system)"))?;

    let n_diffs = write_lines(
        &out_dir.join("diffs"),
        system.vars.iter().map(|v| format!("({}, {}, {}, {}, {})", v.n, v.s, v.f, v.row, v.col)),
    )?;
    let n_offset = write_lines(
        &out_dir.join("offset"),
        (0..system.num_vars)
            .filter(|&i| gf2::vec_get(&result.offset, i))
            .map(|i| i.to_string()),
    )?;
    let mut unknown_sorted: Vec<usize> = result.unknown.iter().copied().collect();
    unknown_sorted.sort_unstable();
    let n_unknown = write_lines(&out_dir.join("unknown"), unknown_sorted.iter().map(|i| i.to_string()))?;

    for (map_kind, cons) in &naturality {
        let filename = format!("{}_constraints", map_kind.name());
        let n = write_lines(&out_dir.join(&filename), cons.iter().map(|c| format_indices(c)))?;
        eprintln!("[E_{r}] {filename}: {n} constraints");
    }

    let n_y = write_lines(&out_dir.join("Y_constraints"), leibniz.iter().map(|c| format_indices(c)))?;
    let n_known = write_lines(
        &out_dir.join("known_constraints"),
        known_constraints.iter().map(|(indices, rhs)| format!("{} = {}", format_indices(indices), *rhs as u8)),
    )?;
    let n_exclude = write_lines(
        &out_dir.join("exclude_list"),
        exclude_sorted.iter().map(|t| format!("({}, {}, {})", t.n, t.s, t.f)),
    )?;

    eprintln!("[E_{r}] Wrote {out_dir:?}: diffs {n_diffs}, offset {n_offset}, unknown {n_unknown}, Y {n_y}, known {n_known}, exclude {n_exclude}");
    Ok(result)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let data = std::env::var("EHP_DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let max_t: i32 = std::env::var("EHP_MAX_T").ok().and_then(|s| s.parse().ok()).unwrap_or(80);
    let r_max: i32 = std::env::var("EHP_R_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(5);

    eprintln!("Loading E_2 (max_t={max_t})...");
    let mut current_page: SATPage = io::load_page(&data, 2, max_t)?;

    loop {
        let r = current_page.r;
        if r > r_max {
            break;
        }

        let page_dir = format!("output/rust_E{r}_page");
        io::save_page_json(&current_page, &page_dir)?;
        eprintln!("[E_{r}] Wrote {page_dir}");

        let known_diffs = ehp_server::load_known_diffs_for_page(&current_page, None)?;
        let result = dump_page_data(&current_page, &known_diffs)?;

        if r == r_max {
            break;
        }

        let (next_page, _turned) = pageturning::build_next_page(&current_page, &result)
            .map_err(|e| format!("contradiction while turning E_{r} -> E_{}: {e}", r + 1))?;
        current_page = next_page;
    }

    eprintln!("Done: walked E_2 through E_{r_max} in a single pass.");
    Ok(())
}
