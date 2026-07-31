//! Headless check of the hidden-EHP-value propagation engine (Toda's
//! P(a∘E²b) = P(a)∘b over the terminal page's composition products).
//!
//! Builds pages exactly like ehp_chart (solve, turn, solve, ...) without any
//! chart generation, asserts one hidden value on the terminal page, runs the
//! propagation fixpoint, and prints every deduced value with provenance.
//!
//! Defaults to the motivating example: the class at (11,13,3) supports a
//! hidden P hitting (5,17,6) (nominal target filtration 5, so δ=1).
//!
//! Run: `EHP_MAX_T=50 cargo run -p ehp-server --release --example diag_hidden`
//! Env: DIAG_KIND (default P), DIAG_N/S/F/IDX (source, default 11 13 3 0),
//!      DIAG_TN/TS/TF/TIDX (target, default 5 17 6 0), EHP_MAX_R (default 5).
//!
//! Page source, in order of preference (both reuse paths are cwd-relative —
//! run from the same directory as the REPL sessions):
//! 1. `EHP_SNAPSHOT=<name>` — `snapshots/<name>/pages` verbatim (the SAVED,
//!    post-mutation state; no config-hash check, manifest not consulted, so
//!    pass the matching EHP_MAX_R if the snapshot's differs from the default).
//! 2. Warm-start cache — hit when EHP_MAX_T / EHP_MAX_R / data / solver env
//!    match a previous REPL startup (the STARTUP, pre-mutation state).
//! 3. Cold build (solve + turn like ehp_chart) — minutes at large max_t.

use ehp_core::element::Element;
use ehp_core::hidden;
use ehp_core::map::MapKind;
use ehp_core::page::SATPage;
use ehp_core::tridegree::Tridegree;
use ehp_core::{cache, constraints, io, pageturning, solver};

const DEFAULT_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/E2");

fn env_i32(name: &str, default: i32) -> i32 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// Terminal page from a snapshot / the warm-start cache / a cold build.
fn load_terminal_page(
    data: &str,
    max_t: i32,
    max_r: i32,
) -> Result<SATPage, Box<dyn std::error::Error>> {
    let start_r = env_i32("EHP_R", 2);

    if let Ok(name) = std::env::var("EHP_SNAPSHOT") {
        if !name.is_empty() {
            let dir = std::path::Path::new("snapshots").join(&name).join("pages");
            match cache::load_pages_from_dir(&dir, start_r, max_r) {
                Some(mut pages) if !pages.is_empty() => {
                    let page = pages.pop().unwrap().page;
                    eprintln!(
                        "Loaded snapshot '{}' (saved post-mutation state): terminal page E_{}",
                        name, page.r,
                    );
                    return Ok(page);
                }
                _ => eprintln!(
                    "Snapshot '{}' not loadable from {} (wrong cwd or EHP_MAX_R?) — falling back.",
                    name,
                    dir.display(),
                ),
            }
        }
    }

    if cache::cache_mode() != cache::CacheMode::Off {
        let hash = cache::config_hash(data, start_r, max_t, max_r);
        if let Some(mut pages) = cache::load_startup_cache(&hash, start_r, max_r) {
            if !pages.is_empty() {
                let page = pages.pop().unwrap().page;
                eprintln!(
                    "Warm-start cache HIT ({}): terminal page E_{} (startup, pre-mutation state)",
                    hash, page.r,
                );
                return Ok(page);
            }
        }
        eprintln!(
            "Warm-start cache miss (hash {}) — cold build. For a hit, run with the same env \
             as the session that built the cache (or use EHP_SNAPSHOT=<name>).",
            hash,
        );
    }

    eprintln!("Loading E_2 (max_t={max_t})...");
    let mut current = io::load_page(data, 2, max_t)?;
    loop {
        let r = current.r;
        let known = ehp_server::load_known_diffs(r, None)?;
        let cutoff = current.max_s.unwrap_or(0);
        let sys = constraints::build_constraint_system(&current, cutoff, &known);
        let Some(result) = solver::solve(&sys) else {
            eprintln!("E_{r}: UNSAT base system, stopping here");
            break;
        };
        eprintln!(
            "E_{r}: {}/{} vars determined",
            sys.num_vars - result.unknown.len(),
            sys.num_vars,
        );
        if sys.num_vars == 0 || r >= max_r {
            break;
        }
        match pageturning::build_next_page(&current, &result) {
            Ok((next, _)) => current = next,
            Err(e) => {
                eprintln!("E_{r}: contradiction while turning: {e}");
                break;
            }
        }
    }
    Ok(current)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let data = std::env::var("EHP_DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let max_t = env_i32("EHP_MAX_T", 50);
    let max_r = env_i32("EHP_MAX_R", 5);

    let kind = match std::env::var("DIAG_KIND").as_deref() {
        Ok("E") | Ok("e") => MapKind::E,
        Ok("H") | Ok("h") => MapKind::H,
        _ => MapKind::P,
    };
    let src_deg = Tridegree::new(env_i32("DIAG_N", 11), env_i32("DIAG_S", 13), env_i32("DIAG_F", 3));
    let tgt_deg =
        Tridegree::new(env_i32("DIAG_TN", 5), env_i32("DIAG_TS", 17), env_i32("DIAG_TF", 6));
    let src_idx = env_i32("DIAG_IDX", 0) as usize;
    let tgt_idx = env_i32("DIAG_TIDX", 0) as usize;

    let page = load_terminal_page(&data, max_t, max_r)?;
    eprintln!("\nTerminal page: E_{} (EHP_HIDDEN enabled: {})", page.r, hidden::hidden_enabled());

    let (src_dim, tgt_dim) = (page.dim_at(src_deg), page.dim_at(tgt_deg));
    eprintln!(
        "Asserting hidden {}({}) [dim {}] = ({}) [dim {}]",
        kind, src_deg, src_dim, tgt_deg, tgt_dim,
    );
    if src_dim == 0 || tgt_dim == 0 {
        eprintln!("A degree has dimension 0 on E_{} — nothing to assert.", page.r);
        eprintln!("(At small EHP_MAX_T the example degrees may be outside the s+f window.)");
        return Ok(());
    }

    let source = Element::basis(src_deg, src_dim, src_idx);
    let target = Element::basis(tgt_deg, tgt_dim, tgt_idx);
    let mut store = hidden::HiddenStore::new();
    match store.assert_value(&page, kind, source, target) {
        Ok(true) => {}
        Ok(false) => unreachable!("fresh store"),
        Err(e) => {
            eprintln!("REJECTED: {e}");
            return Ok(());
        }
    }

    let b_sphere = src_deg.n + src_deg.s - 2;
    let cand_degs: Vec<Tridegree> = page
        .dimension
        .iter()
        .filter(|(t, &d)| t.n == b_sphere && d > 0)
        .map(|(&t, _)| t)
        .collect();
    eprintln!("Candidate b degrees at sphere {}: {}", b_sphere, cand_degs.len());

    // Gate statistics for the P rule, so a 0-deduction run distinguishes
    // "no all-nonzero partners" from a plumbing bug.
    if kind == MapKind::P {
        let (mut cb_block, mut ae2b_block, mut both, mut e2b_nz, mut x_nz, mut y_nz) =
            (0, 0, 0, 0, 0, 0);
        for &deg_b in &cand_degs {
            let deg_eb = MapKind::E.target_degree(deg_b);
            let deg_e2b = MapKind::E.target_degree(deg_eb);
            let has_cb = page.products.has_block(tgt_deg, deg_b);
            let has_ae2b = page.products.has_block(src_deg, deg_e2b);
            cb_block += has_cb as usize;
            ae2b_block += has_ae2b as usize;
            if !(has_cb && has_ae2b) {
                continue;
            }
            both += 1;
            let dim_b = page.dim_at(deg_b);
            let x_deg = Tridegree::new(src_deg.n, src_deg.s + deg_b.s, src_deg.f + deg_b.f);
            let y_deg = Tridegree::new(tgt_deg.n, tgt_deg.s + deg_b.s, tgt_deg.f + deg_b.f);
            for i in 0..dim_b {
                let b = Element::basis(deg_b, dim_b, i);
                let eb = page.map_matrix_ref(MapKind::E, deg_b).apply_vec(&b.vec);
                if eb.is_zero() {
                    continue;
                }
                let e2b = page.map_matrix_ref(MapKind::E, deg_eb).apply_vec(&eb);
                if e2b.is_zero() {
                    continue;
                }
                e2b_nz += 1;
                let source = Element::basis(src_deg, src_dim, src_idx);
                let x = page.products.multiply(
                    src_deg, &source.vec, deg_e2b, &e2b, page.dim_at(x_deg),
                );
                if !x.is_zero() {
                    x_nz += 1;
                }
                let target = Element::basis(tgt_deg, tgt_dim, tgt_idx);
                let y =
                    page.products.multiply(tgt_deg, &target.vec, deg_b, &b.vec, page.dim_at(y_deg));
                if !y.is_zero() {
                    y_nz += 1;
                }
            }
        }
        eprintln!(
            "P-rule gates over basis b's: P(a)∘b block {}, a∘E²b block {}, both {}; \
             E²b≠0 {}, a∘E²b≠0 {}, P(a)∘b≠0 {}",
            cb_block, ae2b_block, both, e2b_nz, x_nz, y_nz,
        );
    }

    let t0 = std::time::Instant::now();
    let (deduced, warnings) = store.propagate(&page);
    eprintln!(
        "\n{} hidden values deduced in {:.2}s:",
        deduced.len(),
        t0.elapsed().as_secs_f64(),
    );
    for w in &warnings {
        eprintln!("  note: {w}");
    }
    for v in store.iter() {
        let tag = if v.is_asserted() { "asserted" } else { "deduced " };
        eprintln!("  [{tag}] {v}");
    }

    Ok(())
}
