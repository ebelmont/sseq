//! Dump the Rust EHP constraint system for one page in the same textual
//! format as the Python reference's `E{r}_data/` directories, split by
//! constraint family (E/H/P naturality, Y = Leibniz, known), so the two can
//! be diffed directly by (n,s,f,row,col) content.
//!
//! Run: EHP_MAX_T=<t> EHP_R=<r> cargo run -p ehp-server --release --example diag_dump_constraints

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use ehp_core::constraints;
use ehp_core::gf2;
use ehp_core::io;
use ehp_core::map::MapKind;
use ehp_core::page::SATPage;
use ehp_core::pageturning;
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let data = std::env::var("EHP_DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let max_t: i32 = std::env::var("EHP_MAX_T").ok().and_then(|s| s.parse().ok()).unwrap_or(80);
    let target_r: i32 = std::env::var("EHP_R").ok().and_then(|s| s.parse().ok()).unwrap_or(2);

    eprintln!("Loading E_2 (max_t={max_t})...");
    let mut current_page: SATPage = io::load_page(&data, 2, max_t)?;

    let (page, known_diffs) = loop {
        let r = current_page.r;
        let known_diffs = ehp_server::load_known_diffs_for_page(&current_page, None)?;
        let cutoff = current_page.max_t.unwrap_or(0);

        if r == target_r {
            break (current_page, known_diffs);
        }
        if r > target_r {
            return Err(format!("walked past E_{target_r} (stopped at E_{r}) without hitting it exactly").into());
        }

        let system = constraints::build_constraint_system(&current_page, cutoff, &known_diffs);
        if system.num_vars == 0 {
            return Err(format!("E_{r} has no differential variables (all classes permanent) before reaching E_{target_r}").into());
        }
        let (result, _d2n) = solver::solve_with_d2(&current_page, &system)
            .ok_or_else(|| format!("E_{r} is UNSAT (inconsistent system) while walking to E_{target_r}"))?;

        let (next_page, _turned) = pageturning::build_next_page(&current_page, &result)
            .map_err(|e| format!("contradiction while turning E_{r} -> E_{}: {e}", r + 1))?;
        current_page = next_page;
    };

    eprintln!("Building constraints for E_{target_r}...");
    let cutoff = page.max_t.unwrap_or(0);
    let system = constraints::build_constraint_system(&page, cutoff, &known_diffs);
    debug_assert_eq!(system.vars.len(), system.num_vars);
    eprintln!("Variable count: {}", system.num_vars);

    let mut naturality: Vec<(MapKind, Vec<Vec<usize>>)> = Vec::new();
    for map_kind in MapKind::all() {
        let nat = constraints::make_naturality_constraints(&page, cutoff, map_kind, &system.var_index);
        naturality.push((map_kind, nat));
    }
    let (leibniz, _excluded_leibniz) = constraints::make_leibniz_constraints(&page, cutoff, &system.var_index);
    let known_constraints = constraints::make_known_constraints(&known_diffs, &system.var_index);

    eprintln!("Solving E_{target_r}...");
    let (result, _d2n) = solver::solve_with_d2(&page, &system)
        .ok_or_else(|| format!("E_{target_r} is UNSAT (inconsistent system)"))?;

    let out_dir = format!("output/rust_E{target_r}_data");
    std::fs::create_dir_all(&out_dir)?;
    let out_dir = Path::new(&out_dir);

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
    let n_unknown = write_lines(
        &out_dir.join("unknown"),
        unknown_sorted.iter().map(|i| i.to_string()),
    )?;

    for (map_kind, constraints) in &naturality {
        let filename = format!("{}_constraints", map_kind.name());
        let n = write_lines(&out_dir.join(&filename), constraints.iter().map(|c| format_indices(c)))?;
        eprintln!("{filename}: {n} constraints");
    }

    let n_y = write_lines(
        &out_dir.join("Y_constraints"),
        leibniz.iter().map(|c| format_indices(c)),
    )?;

    let n_known = write_lines(
        &out_dir.join("known_constraints"),
        known_constraints
            .iter()
            .map(|(indices, rhs)| format!("{} = {}", format_indices(indices), *rhs as u8)),
    )?;

    let mut exclude_sorted: Vec<_> = page.exclude_set.iter().copied().collect();
    exclude_sorted.sort();
    let n_exclude = write_lines(
        &out_dir.join("exclude_list"),
        exclude_sorted.iter().map(|t| format!("({}, {}, {})", t.n, t.s, t.f)),
    )?;

    eprintln!("Wrote {out_dir:?}:");
    eprintln!("  diffs: {n_diffs}");
    eprintln!("  offset: {n_offset}");
    eprintln!("  unknown: {n_unknown}");
    eprintln!("  Y_constraints: {n_y}");
    eprintln!("  known_constraints: {n_known}");
    eprintln!("  exclude_list: {n_exclude}");

    Ok(())
}
