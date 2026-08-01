//! Interactive EHP spectral sequence chart viewer with REPL.
//!
//! Loads the EHP page (binary or CSV), solves constraints, turns pages automatically
//! (E_2 -> E_3 -> E_4 -> ...) until no differential variables remain, writes
//! SeqSee-compatible CSV for each page, generates per-sphere HTML charts for all
//! pages, then enters an interactive REPL for editing differentials across all pages.
//!
//! Clicking two nodes in a chart (in d_r mode) copies the corresponding `add`
//! command to the clipboard — paste it into the REPL to apply.
//!
//! # Usage
//!
//! ```sh
//! cargo run -p ehp-server --release --example ehp_chart
//! ```
//!
//! # Environment variables
//!
//! - `EHP_DATA`       — path to data file or CSV directory (default:
//!                      ~/ehp-sat-rs/data/E2 — the canonical E2 CSVs, a
//!                      byte-identical copy of ~/EHP_SAT/data/E2)
//! - `EHP_MAX_T`      — max total degree s+f (default: 20)
//! - `EHP_FIBERVIEWS` — "0" skips the fiber-sequence charts (one per base
//!                      sphere N: S^N → ΩS^{N+1} → ΩS^{2N+1})
//! - `EHP_R`          — starting page number (default: 2)
//! - `SEQSEE_THEME`   — chart palette: "dark"/"light" (Catppuccin) or
//!                      "teak"/"linen" (Finn Juhl dark/light). Default: "dark".
//! - `EHP_SEQSEE`     — SeqSee scripts directory (contains main.py). Default:
//!                      the vendored copy at ext/seqsee (deps from its
//!                      requirements.txt), else an external ~/seqsee checkout.
//! - `EHP_PYTHON`     — python used to run the SeqSee scripts (default:
//!                      python3 for the vendored copy, the poetry venv's
//!                      python for external checkouts)
//! - `SEQSEE_DIR`     — legacy: external SeqSee checkout containing seqsee_new/

use std::collections::BTreeSet;
use std::io::{BufRead, Write as _};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use rayon::prelude::*;

use ehp_core::constraints::{self, DiffVar};
use ehp_core::hidden;
use ehp_core::interpage::{self, InterpagePage};
use ehp_core::io;
use ehp_core::map::MapKind;
use ehp_core::page::SATPage;
use ehp_core::pageturning::{self, TurnedBidegree};
use ehp_core::result::SATResult;
use ehp_core::seqsee;
use ehp_core::solver;
use ehp_core::tridegree::Tridegree;
use hashbrown::{HashMap, HashSet};

// The canonical E2 data: CSV directory kept byte-identical to
// ~/EHP_SAT/data/E2 (the reference copy). Loading the CSVs directly avoids
// .ehp binaries with a stale max-total cutoff baked in at convert time —
// the CSV loader's only cutoff is the runtime s+f <= EHP_MAX_T.
const DEFAULT_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/E2");
const DEFAULT_MAX_T: i32 = 20;

// =============================================================================
// Multi-page state
// =============================================================================

/// State for one page of the spectral sequence.
struct PageState {
    page: SATPage,
    result: Option<SATResult>,
    known_diffs: HashMap<DiffVar, bool>,
    n_values: BTreeSet<i32>,
    /// Stems with (unstable) classes — one stem chart each.
    stem_values: BTreeSet<i32>,
    /// Base spheres N (N and N+1 both with classes) — one fiber-sequence
    /// chart each (S^N → ΩS^{N+1} → ΩS^{2N+1}).
    fiber_values: BTreeSet<i32>,
    /// Homology data from turning this page into the next.
    /// Stored for incremental re-turning when a differential changes.
    turned: Option<HashMap<Tridegree, TurnedBidegree>>,
    /// Leibniz pairs skipped because a degree was excluded — used by the
    /// interpage machinery (`try`/`sweep`) to re-activate them.
    excluded_leibniz: constraints::ExcludedLeibniz,
    /// Why `result` is `None`, when it is (e.g. "re-solve was INCONSISTENT
    /// after `add 2 ...`"). Surfaced by `sweep`/`try`/`interpage` so a page
    /// with no solve result explains itself instead of failing opaquely.
    unsat_reason: Option<String>,
}

/// Records what a mutation changed so it can be undone.
struct UndoEntry {
    r: i32,
    var: DiffVar,
    /// The previous value: `None` if the var wasn't in known_diffs, `Some(v)` if it was.
    prev: Option<bool>,
}

/// Base spheres N with classes on both S^N and S^{N+1} — the fiber-sequence
/// charts worth drawing (capped like n_values).
fn compute_fiber_values(page: &SATPage, max_t: i32) -> BTreeSet<i32> {
    let ns: BTreeSet<i32> = page
        .dimension
        .iter()
        .filter(|(_, &d)| d > 0)
        .map(|(&t, _)| t.n)
        .collect();
    ns.iter()
        .copied()
        .filter(|&n| n >= 2 && n < max_t && ns.contains(&(n + 1)))
        .collect()
}

/// EHP_FIBERVIEWS != "0" (default on): generate the fiber-sequence charts.
fn fiberviews_enabled() -> bool {
    std::env::var("EHP_FIBERVIEWS").map(|v| v != "0").unwrap_or(true)
}

// =============================================================================
// Named snapshots
// -----------------------------------------------------------------------------
// A snapshot is a self-contained, human-named copy of a whole session: the
// solved page chain (serialized like the warm-start cache) + the generated
// charts (cloned) + a manifest. Unlike the warm-start cache it is NOT keyed on
// the fragile config hash — you load it by name (`snapshot load <name>` or
// EHP_SNAPSHOT=<name> at startup), which bypasses the hash and the chart
// freshness stamp entirely: no solve, no page-turn, no chart regeneration.
// Layout:  snapshots/<name>/{pages/, charts/, manifest.json}
// On APFS the chart clone is copy-on-write (`cp -c`): instant, zero extra
// space until either copy diverges — essential given multi-GB chart dirs.
// =============================================================================

/// Immutable config of the running session, captured once in `main`. The
/// snapshot-save command reads it so it doesn't have to be threaded through
/// the whole REPL dispatch.
struct SessionCfg {
    start_r: i32,
    max_t: i32,
    max_r: i32,
    theme: String,
    cache_hash: String,
}

static SESSION: OnceLock<SessionCfg> = OnceLock::new();

/// Parsed `manifest.json` of a snapshot.
struct SnapshotManifest {
    start_r: i32,
    max_t: i32,
    max_r: i32,
    theme: String,
    cache_hash: String,
    pages: usize,
    charts: usize,
    mutations: usize,
    created: u64,
}

fn snapshots_root() -> PathBuf {
    PathBuf::from("snapshots")
}

fn snapshot_dir(name: &str) -> PathBuf {
    snapshots_root().join(name)
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Format a unix timestamp as local "YYYY-MM-DD HH:MM" via `date` (darwin/BSD).
/// Falls back to the raw epoch on any failure — display-only, never fatal.
fn fmt_time(secs: u64) -> String {
    std::process::Command::new("date")
        .arg("-r")
        .arg(secs.to_string())
        .arg("+%Y-%m-%d %H:%M")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("epoch {}", secs))
}

/// Recursively clone `src` to `dst`, preferring APFS copy-on-write (`cp -c`)
/// so a multi-GB charts dir copies instantly with no extra space until
/// modified. Falls back to a plain recursive copy off-APFS.
fn clone_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    if dst.exists() {
        std::fs::remove_dir_all(dst)?;
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let cow = std::process::Command::new("cp")
        .arg("-c")
        .arg("-R")
        .arg(src)
        .arg(dst)
        .status();
    if matches!(&cow, Ok(s) if s.success()) {
        return Ok(());
    }
    // Fallback: clean any partial dst, then a plain recursive copy.
    if dst.exists() {
        std::fs::remove_dir_all(dst)?;
    }
    let status = std::process::Command::new("cp")
        .arg("-R")
        .arg(src)
        .arg(dst)
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "recursive copy failed",
        ))
    }
}

/// Count top-level `.html` files in a charts directory (headline chart count).
fn count_html(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| {
                    e.path().extension().and_then(|x| x.to_str()) == Some("html")
                })
                .count()
        })
        .unwrap_or(0)
}

// =============================================================================
// Last-session recall (`-- last` / `-- charts`)
// =============================================================================

/// Every env var that shapes a session (config hash inputs, display, data).
/// Recorded verbatim in `output/.ehp_last.json` at each successful startup;
/// `-- last` replays them so nobody has to remember EHP_MAX_T etc.
const SESSION_ENV_VARS: &[&str] = &[
    "EHP_DATA",
    "EHP_MAX_T",
    "EHP_R",
    "EHP_MAX_R",
    "SEQSEE_THEME",
    "EHP_SNAPSHOT",
    "EHP_SOLVER",
    "EHP_RELAX_TARGET_EXCLUDE",
    "EHP_OUTSIDE_DIFFS",
    "EHP_OUTSIDE_PARITY",
    "EHP_OUTSIDE_SKIP",
    "EHP_D2_LINEAR",
    "EHP_MAPVIEWS",
    "EHP_FIBERVIEWS",
];

fn last_session_path() -> PathBuf {
    PathBuf::from("output").join(".ehp_last.json")
}

fn record_last_session(cache_hash: &str) {
    let mut env = serde_json::Map::new();
    for var in SESSION_ENV_VARS {
        if let Ok(v) = std::env::var(var) {
            env.insert(var.to_string(), serde_json::Value::String(v));
        }
    }
    let v = serde_json::json!({
        "version": 1,
        "env": env,
        "cache_hash": cache_hash,
        "created": now_secs(),
    });
    if let Ok(s) = serde_json::to_string_pretty(&v) {
        let _ = std::fs::write(last_session_path(), s);
    }
}

/// Replay the recorded env (setting recorded vars, clearing unrecorded ones)
/// so the startup that follows reproduces the last session exactly — same
/// config hash, so it lands on the warm cache and archived charts.
fn apply_last_session() -> Result<String, String> {
    let raw = std::fs::read_to_string(last_session_path()).map_err(|_| {
        "no session recorded yet (output/.ehp_last.json) — start one normally first".to_string()
    })?;
    let v: serde_json::Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    let env = v
        .get("env")
        .and_then(|e| e.as_object())
        .cloned()
        .unwrap_or_default();
    let mut summary = Vec::new();
    for var in SESSION_ENV_VARS {
        match env.get(*var).and_then(|x| x.as_str()) {
            Some(val) => {
                std::env::set_var(var, val);
                summary.push(format!("{}={}", var, val));
            }
            None => std::env::remove_var(var),
        }
    }
    Ok(summary.join(" "))
}

/// Best chart index for the last session: its snapshot's charts, else the
/// warm-cache archive, else whatever is live in output/charts.
fn last_session_charts_index() -> Option<PathBuf> {
    let raw = std::fs::read_to_string(last_session_path()).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    if let Some(name) = v.pointer("/env/EHP_SNAPSHOT").and_then(|x| x.as_str()) {
        let p = snapshot_dir(name).join("charts").join("index.html");
        if p.exists() {
            return Some(p);
        }
    }
    if let Some(hash) = v.get("cache_hash").and_then(|x| x.as_str()) {
        let p = ehp_core::cache::cache_dir(hash).join("charts").join("index.html");
        if p.exists() {
            return Some(p);
        }
    }
    let live = PathBuf::from("output").join("charts").join("index.html");
    live.exists().then_some(live)
}

fn valid_snapshot_name(name: &str) -> bool {
    !name.is_empty()
        && !name.contains('/')
        && !name.contains('\\')
        && name != "."
        && name != ".."
        && !name.starts_with('.')
}

fn write_manifest(dir: &Path, m: &SnapshotManifest) -> std::io::Result<()> {
    let v = serde_json::json!({
        "version": 1,
        "start_r": m.start_r,
        "max_t": m.max_t,
        "max_r": m.max_r,
        "theme": m.theme,
        "cache_hash": m.cache_hash,
        "pages": m.pages,
        "charts": m.charts,
        "mutations": m.mutations,
        "created": m.created,
    });
    std::fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&v)?)
}

fn load_snapshot_manifest(name: &str) -> Result<SnapshotManifest, String> {
    let path = snapshot_dir(name).join("manifest.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("bad manifest json: {}", e))?;
    let geti = |k: &str, d: i64| v.get(k).and_then(|x| x.as_i64()).unwrap_or(d);
    let getu = |k: &str, d: u64| v.get(k).and_then(|x| x.as_u64()).unwrap_or(d);
    Ok(SnapshotManifest {
        start_r: geti("start_r", 2) as i32,
        max_t: geti("max_t", 20) as i32,
        max_r: geti("max_r", 5) as i32,
        theme: v
            .get("theme")
            .and_then(|x| x.as_str())
            .unwrap_or("dark")
            .to_string(),
        cache_hash: v
            .get("cache_hash")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        pages: getu("pages", 0) as usize,
        charts: getu("charts", 0) as usize,
        mutations: getu("mutations", 0) as usize,
        created: getu("created", 0),
    })
}

/// Serialize the live (post-mutation) page chain + clone the charts into a
/// named snapshot. Returns the written manifest.
fn save_snapshot(
    name: &str,
    pages: &[PageState],
    mutations: usize,
    force: bool,
) -> Result<SnapshotManifest, String> {
    if !valid_snapshot_name(name) {
        return Err(format!(
            "invalid snapshot name '{}' (no '/', '\\', leading '.', or '..')",
            name
        ));
    }
    let cfg = SESSION
        .get()
        .ok_or("session config not initialized (internal error)")?;
    let dir = snapshot_dir(name);
    if dir.exists() && !force {
        return Err(format!(
            "snapshot '{}' already exists — `snapshot save {} force` to overwrite",
            name, name
        ));
    }
    if dir.exists() {
        std::fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
    }
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    // 1. Solved page chain WITH all applied mutations (the warm-start cache
    //    dir only holds the startup state, so re-serialize from live `pages`).
    let cached: Vec<ehp_core::cache::CachedPage> = pages
        .iter()
        .map(|ps| ehp_core::cache::CachedPage {
            page: ps.page.clone(),
            result: ps.result.clone(),
            known_diffs: ps.known_diffs.clone(),
            excluded_leibniz: ps.excluded_leibniz.clone(),
        })
        .collect();
    ehp_core::cache::save_pages_to_dir(&dir.join("pages"), &cached, Some(&cfg.cache_hash))
        .map_err(|e| format!("saving pages: {}", e))?;

    // 2. Charts (the expensive artifact) — APFS copy-on-write clone.
    let charts_src = PathBuf::from("output").join("charts");
    let chart_count = if charts_src.exists() {
        clone_dir(&charts_src, &dir.join("charts")).map_err(|e| format!("cloning charts: {}", e))?;
        count_html(&dir.join("charts"))
    } else {
        0
    };

    let manifest = SnapshotManifest {
        start_r: cfg.start_r,
        max_t: cfg.max_t,
        max_r: cfg.max_r,
        theme: cfg.theme.clone(),
        cache_hash: cfg.cache_hash.clone(),
        pages: pages.len(),
        charts: chart_count,
        mutations,
        created: now_secs(),
    };
    write_manifest(&dir, &manifest).map_err(|e| format!("writing manifest: {}", e))?;
    Ok(manifest)
}

/// Restore a snapshot's charts into `output/charts` and load its solved page
/// chain into `PageState`s. Bypasses the config hash and the chart stamp.
fn restore_snapshot(name: &str, m: &SnapshotManifest) -> Result<Vec<PageState>, String> {
    let dir = snapshot_dir(name);
    let charts_snap = dir.join("charts");
    if charts_snap.exists() {
        std::fs::create_dir_all("output").map_err(|e| e.to_string())?;
        clone_dir(&charts_snap, &PathBuf::from("output").join("charts"))
            .map_err(|e| format!("restoring charts: {}", e))?;
    }
    let cached = ehp_core::cache::load_pages_from_dir(&dir.join("pages"), m.start_r, m.max_r)
        .ok_or_else(|| {
            "snapshot has no readable solved pages (cache-version mismatch — re-save it)".to_string()
        })?;
    Ok(cached_to_states(cached, m.max_t))
}

/// Re-exec this binary with `EHP_SNAPSHOT=<name>` so the startup restore path
/// loads the snapshot cleanly (swapping the whole page set + charts). On unix
/// this replaces the process image; it only returns if exec fails.
#[cfg(unix)]
fn reexec_with_snapshot(name: &str) {
    use std::os::unix::process::CommandExt;
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("current_exe failed: {}", e);
            return;
        }
    };
    eprintln!("Restarting into snapshot '{}'...", name);
    let err = std::process::Command::new(exe)
        .env("EHP_SNAPSHOT", name)
        .args(std::env::args().skip(1))
        .exec();
    eprintln!("exec failed: {}", err);
}

#[cfg(not(unix))]
fn reexec_with_snapshot(name: &str) {
    eprintln!(
        "In-REPL load needs a unix exec; restart with EHP_SNAPSHOT={}.",
        name
    );
}

/// Print all snapshots with their config + provenance.
fn list_snapshots() {
    let root = snapshots_root();
    let mut names: Vec<String> = match std::fs::read_dir(&root) {
        Ok(rd) => rd
            .flatten()
            .filter(|e| e.path().join("manifest.json").exists())
            .filter_map(|e| e.file_name().into_string().ok())
            .collect(),
        Err(_) => Vec::new(),
    };
    names.sort();
    if names.is_empty() {
        eprintln!("No snapshots. Create one with `snapshot save <name>`.");
        return;
    }
    eprintln!("Snapshots ({}):", names.len());
    for name in names {
        match load_snapshot_manifest(&name) {
            Ok(m) => eprintln!(
                "  {:<20} t={} r={}..{} theme={} charts={} muts={}  [{}]",
                name,
                m.max_t,
                m.start_r,
                m.max_r,
                m.theme,
                m.charts,
                m.mutations,
                fmt_time(m.created),
            ),
            Err(e) => eprintln!("  {:<20} (unreadable manifest: {})", name, e),
        }
    }
}

/// Convert cached/restored pages into REPL `PageState`s (shared by the
/// warm-start cache-hit path and snapshot restore). Prints per-page
/// determined-var counts.
fn cached_to_states(cached: Vec<ehp_core::cache::CachedPage>, max_t: i32) -> Vec<PageState> {
    let mut pages = Vec::new();
    for cp in cached {
        let n_values: BTreeSet<i32> = cp
            .page
            .dimension
            .iter()
            .filter(|(_, &d)| d > 0)
            .map(|(&t, _)| t.n)
            .filter(|&n| n < max_t)
            .collect();
        let stem_values: BTreeSet<i32> = cp
            .page
            .dimension
            .iter()
            .filter(|(&t, &d)| d > 0 && t.n <= t.s + 2)
            .map(|(&t, _)| t.s)
            .collect();
        let fiber_values = compute_fiber_values(&cp.page, max_t);
        let unsat_reason = if cp.result.is_some() {
            None
        } else {
            Some(format!("cached E_{} has no solve result", cp.page.r))
        };
        let r = cp.page.r;
        if let Some(ref res) = cp.result {
            eprintln!(
                "E_{}: {}/{} vars determined (cached)",
                r,
                res.vars.len() - res.unknown.len(),
                res.vars.len(),
            );
        }
        pages.push(PageState {
            page: cp.page,
            result: cp.result,
            known_diffs: cp.known_diffs,
            n_values,
            stem_values,
            fiber_values,
            turned: None,
            excluded_leibniz: cp.excluded_leibniz,
            unsat_reason,
        });
    }

    // The cache doesn't store the turned-page chains (quotient/lift maps per
    // tridegree), but `push_forward` needs them — without them the prior-page
    // uncertainty overlays (faint dashed d_k edges on later pages) silently
    // vanish from every chart rewrite in a warm-started session. Recompute
    // them here; turning is the cheap part of startup (it's the solve and
    // chart generation the cache exists to skip). The last page needs none
    // (nothing is pushed past it).
    let t0 = Instant::now();
    let n_pages = pages.len();
    let turned_maps: Vec<_> = pages[..n_pages.saturating_sub(1)]
        .par_iter()
        .map(|ps| {
            let res = ps.result.as_ref()?;
            ehp_core::pageturning::turn_page(&ps.page, res, ps.page.r + 1, ps.page.max_t).ok()
        })
        .collect();
    for (ps, turned) in pages.iter_mut().zip(turned_maps) {
        ps.turned = turned;
    }
    eprintln!(
        "Recomputed turned-page chains in {:.2}s (prior-page uncertainty overlays)",
        t0.elapsed().as_secs_f64(),
    );
    pages
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // CLI modes (after `--`): `last` replays the most recent session's
    // recorded env verbatim — no need to remember EHP_MAX_T etc.; `charts`
    // just opens the last session's charts in the browser (no solve, no REPL).
    match std::env::args().nth(1).as_deref() {
        Some("charts") => {
            return match last_session_charts_index() {
                Some(p) => {
                    let p = std::fs::canonicalize(&p).unwrap_or(p);
                    if std::env::var("EHP_NO_OPEN").map(|v| v != "0").unwrap_or(false) {
                        eprintln!("Charts at {} (EHP_NO_OPEN set — not opening)", p.display());
                    } else {
                        eprintln!("Opening {}", p.display());
                        open::that(&p)?;
                    }
                    Ok(())
                }
                None => {
                    Err("no charts recorded yet — start a session normally (or `-- last`) first"
                        .into())
                }
            };
        }
        Some("last") => match apply_last_session() {
            Ok(s) if s.is_empty() => eprintln!("Resuming last session (all defaults)"),
            Ok(s) => eprintln!("Resuming last session: {}", s),
            Err(e) => eprintln!("`last` unavailable: {} — continuing with current env.", e),
        },
        Some(other) => eprintln!(
            "Unknown argument '{}' (expected `last` or `charts`) — ignoring.",
            other
        ),
        None => {}
    }

    let data_path = std::env::var("EHP_DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let mut max_t: i32 = std::env::var("EHP_MAX_T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_T);
    let mut start_r: i32 = std::env::var("EHP_R")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    // Highest page to compute; stop turning after E_{max_r} (default 7).
    // Default terminal page E5 (the user's usual working ceiling); the E-infinity
    // fiber overlays (uncertain-diff coloring, hidden-EHP candidates) land on this
    // max page. Override with EHP_MAX_R to turn more pages.
    let mut max_r: i32 = std::env::var("EHP_MAX_R")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let mut theme = std::env::var("SEQSEE_THEME").unwrap_or_else(|_| "dark".into());

    // Snapshot startup (EHP_SNAPSHOT=<name>): load a named session verbatim.
    // Its manifest OVERRIDES max_t/start_r/max_r/theme so every downstream
    // step (CSV write, chart-stamp, config summary) is consistent with the
    // restored state — regardless of what env this process was launched with.
    let snapshot_name = std::env::var("EHP_SNAPSHOT").ok().filter(|s| !s.is_empty());
    let mut snapshot_manifest: Option<SnapshotManifest> = None;
    if let Some(name) = &snapshot_name {
        match load_snapshot_manifest(name) {
            Ok(m) => {
                start_r = m.start_r;
                max_t = m.max_t;
                max_r = m.max_r;
                theme = m.theme.clone();
                snapshot_manifest = Some(m);
            }
            Err(e) => {
                eprintln!(
                    "Snapshot '{}' not loaded: {} — continuing with normal startup.",
                    name, e
                );
            }
        }
    }

    let solver_name = std::env::var("EHP_SOLVER")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "uf".into());

    eprintln!("EHP Chart REPL (multi-page)");
    if let Some(name) = &snapshot_name {
        if snapshot_manifest.is_some() {
            eprintln!("  snapshot: {} (config from manifest)", name);
        }
    }
    eprintln!("  data:   {}", data_path);
    eprintln!("  start:  E_{}", start_r);
    eprintln!("  max t:  {}", max_t);
    eprintln!("  max r:  {}", max_r);
    eprintln!("  theme:  {}", theme);
    eprintln!("  solver: {}", solver_name);
    eprintln!();

    // 0. Warm-start cache (EHP_CACHE=0 disables): the startup chain is a
    // deterministic function of the data + semantics flags; a hash match
    // loads pages/results and skips load+build+solve+turn entirely.
    // A snapshot forces its recorded hash (the pages/charts belong to it), so
    // the chart-freshness stamp downstream matches without recomputing.
    let cache_hash = match &snapshot_manifest {
        Some(m) => m.cache_hash.clone(),
        None => ehp_core::cache::config_hash(&data_path, start_r, max_t, max_r),
    };
    let cache_mode = ehp_core::cache::cache_mode();
    let mut pages: Vec<PageState> = Vec::new();
    let mut snapshot_loaded = false;

    // Snapshot restore takes priority over the warm-start cache: load solved
    // pages + charts by name, skipping solve/turn/regen entirely.
    if let Some(m) = &snapshot_manifest {
        let name = snapshot_name.as_ref().unwrap();
        let t0 = Instant::now();
        match restore_snapshot(name, m) {
            Ok(states) => {
                pages = states;
                snapshot_loaded = true;
                eprintln!(
                    "Loaded snapshot '{}': {} pages, {} charts in {:.2}s — no solve, no regen.",
                    name,
                    pages.len(),
                    m.charts,
                    t0.elapsed().as_secs_f64(),
                );
            }
            Err(e) => eprintln!("Snapshot restore failed: {} — falling back to cold startup", e),
        }
    }

    if pages.is_empty() && cache_mode == ehp_core::cache::CacheMode::Use {
        if let Some(cached) = ehp_core::cache::load_startup_cache(&cache_hash, start_r, max_r) {
            let t0 = Instant::now();
            pages = cached_to_states(cached, max_t);
            eprintln!(
                "Warm-start cache HIT ({}): {} pages in {:.2}s",
                cache_hash,
                pages.len(),
                t0.elapsed().as_secs_f64(),
            );
        } else {
            eprintln!("Warm-start cache miss ({}) — cold startup", cache_hash);
        }
    }
    let from_cache = !pages.is_empty();

    if !from_cache {
    // 1. Load starting page
    let t0 = Instant::now();
    let page = io::load_page(&data_path, start_r, max_t)?;
    eprintln!("Loaded E_{} in {:.2}s", start_r, t0.elapsed().as_secs_f64());

    // 2. Build all pages: solve, turn, solve, turn, ...
    let mut current_page = page;

    loop {
        let r = current_page.r;
        let t1 = Instant::now();
        let known_diffs = ehp_server::load_known_diffs_for_page(&current_page, None)?;
        let cutoff = current_page.max_s.unwrap_or(0);
        let system = constraints::build_constraint_system(&current_page, cutoff, &known_diffs);
        let num_vars = system.num_vars;
        let result = solver::solve_with_d2(&current_page, &system).map(|(res, d2n)| {
            if d2n > 0 {
                eprintln!("  E_{}: d\u{b2}=0 linearization determined {} more entries", r, d2n);
            }
            res
        });

        if let Some(ref res) = result {
            let determined = num_vars - res.unknown.len();
            eprintln!(
                "E_{}: {}/{} vars determined ({:.2}s)",
                r, determined, num_vars,
                t1.elapsed().as_secs_f64(),
            );
        } else if num_vars == 0 {
            eprintln!("E_{}: no differential variables (all classes permanent)", r);
        } else {
            eprintln!("E_{}: no solution (inconsistent system)", r);
        }

        let n_values: BTreeSet<i32> = current_page
            .dimension
            .iter()
            .filter(|(_, &d)| d > 0)
            .map(|(&t, _)| t.n)
            .filter(|&n| n < max_t)
            .collect();
        let stem_values: BTreeSet<i32> = current_page
            .dimension
            .iter()
            .filter(|(&t, &d)| d > 0 && t.n <= t.s + 2)
            .map(|(&t, _)| t.s)
            .collect();
        let fiber_values = compute_fiber_values(&current_page, max_t);

        let has_result = result.is_some();
        let unsat_reason = if has_result {
            None
        } else if num_vars == 0 {
            Some(format!("E_{} has no differential variables (all classes permanent)", r))
        } else {
            Some(format!("the E_{} startup solve was INCONSISTENT", r))
        };
        pages.push(PageState {
            page: current_page,
            result,
            known_diffs,
            n_values,
            stem_values,
            fiber_values,
            turned: None,
            excluded_leibniz: system.excluded_leibniz,
            unsat_reason,
        });

        // Stop if no vars (all permanent) or no result
        if num_vars == 0 || !has_result {
            break;
        }

        // Stop once we've computed the requested top page (E_{max_r}).
        if r >= max_r {
            eprintln!("Reached E_{} (max r), stopping.", r);
            break;
        }

        // Turn page
        let t2 = Instant::now();
        let (next_page, turned) = {
            let prev = pages.last().unwrap();
            let prev_result = prev.result.as_ref().unwrap();
            match pageturning::build_next_page(&prev.page, prev_result) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("E_{}: contradiction while turning: {}", r, e);
                    break;
                }
            }
        };
        pages.last_mut().unwrap().turned = Some(turned);
        let nonzero = next_page.dimension.values().filter(|&&d| d > 0).count();
        eprintln!(
            "Turned E_{} -> E_{}: {} tridegrees ({:.2}s)",
            r, r + 1, nonzero,
            t2.elapsed().as_secs_f64(),
        );

        if nonzero == 0 {
            eprintln!("E_{}: empty page, stopping", r + 1);
            break;
        }

        current_page = next_page;
    }

    if cache_mode != ehp_core::cache::CacheMode::Off
        && pages.iter().all(|ps| ps.result.is_some())
    {
        let cached: Vec<ehp_core::cache::CachedPage> = pages
            .iter()
            .map(|ps| ehp_core::cache::CachedPage {
                page: ps.page.clone(),
                result: ps.result.clone(),
                known_diffs: ps.known_diffs.clone(),
                excluded_leibniz: ps.excluded_leibniz.clone(),
            })
            .collect();
        match ehp_core::cache::save_startup_cache(&cache_hash, &cached) {
            Ok(()) => eprintln!("Warm-start cache saved ({})", cache_hash),
            Err(e) => eprintln!("Warm-start cache save failed: {}", e),
        }
    }
    } // end cold startup

    eprintln!();
    eprintln!(
        "Computed {} pages: E_{} through E_{}",
        pages.len(),
        start_r,
        start_r + pages.len() as i32 - 1,
    );

    // Fiber E-infinity coloring (Task 4): expose the highest page r being
    // generated to the ehp_batch.py / jsonmaker subprocesses via an env var
    // they inherit. jsonmaker only reads it in fiber mode on r == max page,
    // coloring nodes involved in an UNCERTAIN Adams differential with their
    // d_r color; sphere/stem charts ignore it. Set once here so every chart
    // generation (initial + regen, all in this process) sees it.
    if let Some(max_page) = pages.iter().map(|ps| ps.page.r).max() {
        std::env::set_var("EHP_MAX_PAGE", max_page.to_string());
    }

    // 3. Write CSV for each page
    let out_dir = PathBuf::from("output");
    std::fs::create_dir_all(&out_dir)?;

    // Remove page CSVs from previous runs (a smaller run writes fewer pages,
    // and leftovers from a bigger run would linger alongside them).
    if let Ok(entries) = std::fs::read_dir(&out_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("ehp_E") && name.ends_with(".csv") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }

    let mut csv_paths: Vec<PathBuf> = Vec::new();
    for ps in &pages {
        let r = ps.page.r;
        let csv_path = out_dir.join(format!("ehp_E{}.csv", r));
        let mut file = std::fs::File::create(&csv_path)?;
        seqsee::write_ehp_csv(&ps.page, ps.result.as_ref(), &ps.known_diffs, &mut file)?;
        csv_paths.push(csv_path);
    }
    eprintln!("Wrote {} CSV files to output/", csv_paths.len());

    let charts_dir = out_dir.join("charts");
    std::fs::create_dir_all(&charts_dir)?;

    // Chart freshness stamp: charts on disk were generated from a specific
    // startup state + display config. On a warm-start cache hit with a
    // matching stamp, the charts already depict exactly this state — skip
    // clearing and regeneration entirely. The stamp is deleted on the first
    // applied mutation (regen_affected_charts), so post-mutation states
    // always regenerate.
    let want_mapviews = std::env::var("EHP_MAPVIEWS").map(|v| v != "0").unwrap_or(true);
    let want_fiberviews = fiberviews_enabled();

    // Capture the session config so `snapshot save` can record it without
    // threading everything through the REPL dispatch.
    let _ = SESSION.set(SessionCfg {
        start_r,
        max_t,
        max_r,
        theme: theme.clone(),
        cache_hash: cache_hash.clone(),
    });

    let stamp_path = charts_dir.join(".state_stamp");
    let expected_stamp = format!(
        "{}|theme={}|mapviews={}|fiberviews={}",
        cache_hash, theme, want_mapviews, want_fiberviews
    );
    // A restored snapshot brings its own charts — never regenerate them.
    let mut charts_fresh = snapshot_loaded
        || (from_cache
            && std::fs::read_to_string(&stamp_path)
                .map(|c| c.trim() == expected_stamp)
                .unwrap_or(false));
    if charts_fresh {
        if snapshot_loaded {
            eprintln!("Snapshot charts restored — skipping regeneration.");
        } else {
            eprintln!("Charts on disk match this state (stamp) — skipping regeneration.");
        }
    }

    // Charts archived alongside the warm-start cache: on a page-cache hit
    // whose on-disk charts don't match this state (cleared by another run, or
    // depicting a different config), clone the archived set back instead of
    // regenerating — chart generation is the slow part of startup.
    let cached_charts = ehp_core::cache::cache_dir(&cache_hash).join("charts");
    if !charts_fresh && from_cache {
        let archived_ok = std::fs::read_to_string(cached_charts.join(".state_stamp"))
            .map(|c| c.trim() == expected_stamp)
            .unwrap_or(false);
        if archived_ok {
            let t0 = Instant::now();
            match clone_dir(&cached_charts, &charts_dir) {
                Ok(()) => {
                    charts_fresh = true;
                    eprintln!(
                        "Restored {} archived charts from the warm-start cache in {:.2}s — skipping regeneration.",
                        count_html(&charts_dir),
                        t0.elapsed().as_secs_f64(),
                    );
                }
                Err(e) => eprintln!("Chart restore from cache failed ({}) — regenerating.", e),
            }
        }
    }

    // Remove chart artifacts from previous runs. Files are keyed only by
    // (sphere, page), so anything this run doesn't regenerate — a different
    // max_t, fewer pages, or a failed sphere — would otherwise linger and be
    // reachable via the index and WASD navigation.
    if !charts_fresh {
    let stale = clean_charts_dir(&charts_dir);
    if stale > 0 {
        eprintln!("Cleared {} chart files from previous runs", stale);
    }
    } // end !charts_fresh (clear)

    // Canonicalize paths
    let csv_paths: Vec<PathBuf> = csv_paths
        .iter()
        .map(|p| std::fs::canonicalize(p).unwrap_or_else(|_| p.clone()))
        .collect();
    let charts_dir = std::fs::canonicalize(&charts_dir)?;

    // 4. Find SeqSee directory
    let seqsee_dir = find_seqsee_dir();
    let seqsee_dir = match seqsee_dir {
        Some(d) => {
            eprintln!("Using SeqSee at {}", d.display());
            d
        }
        None => {
            eprintln!("\nSeqSee not found. CSVs written to output/.");
            eprintln!(
                "Set EHP_SEQSEE (or restore the vendored copy at {})",
                VENDORED_SEQSEE
            );
            return Ok(());
        }
    };

    // 5. Generate charts for all pages (sphere + stem views)
    if !charts_fresh {
    let mut all_chart_files: Vec<(i32, i32, PathBuf)> = Vec::new(); // (r, n, path)
    let mut all_stem_files: Vec<(i32, i32)> = Vec::new(); // (r, k)
    let mut all_fiber_files: Vec<(i32, i32)> = Vec::new(); // (r, n)
    for (i, ps) in pages.iter().enumerate() {
        let r = ps.page.r;
        eprintln!("Generating E_{} charts...", r);
        let chart_files = generate_all_charts(
            &ps.n_values,
            &csv_paths[i],
            &charts_dir,
            &seqsee_dir,
            &theme,
            r,
        )?;

        let ok_stems = generate_stem_charts(
            &ps.stem_values,
            &csv_paths[i],
            &charts_dir,
            &seqsee_dir,
            &theme,
            r,
        );

        let ok_fibers = generate_fiber_charts(
            &ps.fiber_values,
            &csv_paths[i],
            &charts_dir,
            &seqsee_dir,
            &theme,
            r,
        );

        // Inject minimap data (E/H/P maps per sphere)
        let map_count = inject_map_info(&charts_dir, &ps.page, &ps.n_values, r);
        eprintln!(
            "  {} sphere charts ({} with map info), {} stem charts, {} fiber charts",
            chart_files.len(),
            map_count,
            ok_stems.len(),
            ok_fibers.len(),
        );
        let failed = ps.n_values.len().saturating_sub(chart_files.len());
        if failed > 0 {
            eprintln!(
                "  WARNING: {} E_{} sphere charts FAILED to generate (see FAILED lines above)",
                failed, r,
            );
        }

        for (n, path) in chart_files {
            all_chart_files.push((r, n, path));
        }
        for k in ok_stems {
            all_stem_files.push((r, k));
        }
        for n in ok_fibers {
            all_fiber_files.push((r, n));
        }
    }

    // Stem/fiber chart scripts (WASD nav + class dims), then data overlays
    // for everything (diff edges + class dims need the injected markers).
    print_chart_timing("startup generation");
    timed_phase("inject stem scripts", || inject_stem_scripts(&charts_dir));
    timed_phase("inject fiber scripts", || inject_fiber_scripts(&charts_dir));
    timed_phase("inject diff/class overlays", || {
        for i in 0..pages.len() {
            fast_update_diffs(&charts_dir, &pages, i);
        }
    });
    if want_fiberviews {
        eprintln!(
            "Generated {} fiber-sequence charts (EHP_FIBERVIEWS=0 to skip)",
            all_fiber_files.len(),
        );
    }

    // Pre-generate the split-screen map views (J-map style) so the e/h/p
    // keys and WASD work immediately. EHP_MAPVIEWS=0 skips this.
    if want_mapviews && seqsee_dir.join("ehp_batch.py").exists() {
        let r_range = (
            pages.first().map(|p| p.page.r).unwrap_or(2),
            pages.last().map(|p| p.page.r).unwrap_or(2),
        );
        let t_mv = Instant::now();
        let mut mv_ok = 0usize;
        let mut mv_total = 0usize;
        for ps in &pages {
            if ps.result.is_none() {
                continue;
            }
            let (ok, total) =
                pregenerate_mapviews(ps, &charts_dir, &seqsee_dir, &theme, ps.page.r, r_range);
            mv_ok += ok;
            mv_total += total;
        }
        refresh_mapview_nav(&charts_dir, r_range);
        for ps in &pages {
            inject_map_info(&charts_dir, &ps.page, &ps.n_values, ps.page.r);
        }
        eprintln!(
            "Pre-generated {}/{} split-screen map views in {:.1}s (EHP_MAPVIEWS=0 to skip)",
            mv_ok,
            mv_total,
            t_mv.elapsed().as_secs_f64(),
        );
        print_chart_timing("startup mapviews");
    }

    if all_chart_files.is_empty() {
        eprintln!("No charts generated.");
        return Ok(());
    }

    // Generate combined index page
    generate_index_html(&charts_dir, &all_chart_files, &all_stem_files, &all_fiber_files, start_r, max_t, &theme, &pages)?;

    // Stamp the charts as depicting exactly this startup state.
    let _ = std::fs::write(&stamp_path, &expected_stamp);
    } // end !charts_fresh (generate)

    // Archive the startup charts alongside the warm-start cache so future
    // startups with this config restore them instead of regenerating (even
    // after other runs clear output/charts). APFS clone: instant, ~0 extra
    // bytes. Skipped for snapshots (their charts live in snapshots/<name>).
    // The archive is startup-state only — mutations delete the on-disk stamp,
    // and only stamped chart sets are ever archived or restored.
    if !snapshot_loaded && cache_mode == ehp_core::cache::CacheMode::Use {
        let cache_root = ehp_core::cache::cache_dir(&cache_hash);
        let archived_ok = std::fs::read_to_string(cached_charts.join(".state_stamp"))
            .map(|c| c.trim() == expected_stamp)
            .unwrap_or(false);
        if cache_root.exists()
            && !archived_ok
            && std::fs::read_to_string(&stamp_path)
                .map(|c| c.trim() == expected_stamp)
                .unwrap_or(false)
        {
            match clone_dir(&charts_dir, &cached_charts) {
                Ok(()) => eprintln!(
                    "Charts archived to {} — its index.html opens directly, no REPL needed.",
                    cached_charts.display(),
                ),
                Err(e) => eprintln!("Chart archive failed: {}", e),
            }
        }
    }

    // Right-click view-jump menu in every chart (sphere/stem/fiber).
    // Unconditional: idempotent (marker refresh), and restored/stamp-fresh
    // charts generated by an older binary need the script added too.
    timed_phase("inject view menu", || inject_view_menu(&charts_dir));

    // Same for the fiber script: on the charts-fresh skip path the archived
    // charts may carry an older-generation script (no hidden-value overlay /
    // Shift+V) — the injector's upgrade path replaces it in place.
    timed_phase("inject fiber scripts (refresh)", || inject_fiber_scripts(&charts_dir));

    // Hidden EHP map values (EXPERIMENTAL, EHP_HIDDEN=0 disables): load the
    // persisted assertions, re-run Toda propagation against the terminal
    // page, and inject the fiber-chart overlay. Runs on every startup path
    // (cold, warm cache, snapshot) — snapshot chart HTML may carry stale
    // overlay data, and this overwrites it.
    let mut hidden = hidden_startup(&pages, &data_path, &charts_dir);

    // 6. Open browser (EHP_NO_OPEN=1 skips — for scripted/headless runs)
    let index_path = charts_dir.join("index.html");
    if std::env::var("EHP_NO_OPEN").map(|v| v != "0").unwrap_or(false) {
        eprintln!("\nCharts at {} (EHP_NO_OPEN set — not opening)", index_path.display());
    } else {
        eprintln!("\nOpening {}", index_path.display());
        if let Err(e) = open::that(&index_path) {
            eprintln!("Could not open browser: {}", e);
            eprintln!("Open {} in your browser", index_path.display());
        }
    }

    // Record this session so `-- last` / `-- charts` (the `ehp` launcher) can
    // reopen it without anyone remembering the env. Written only after a
    // fully successful startup.
    record_last_session(&cache_hash);

    // 7. Enter REPL
    eprintln!();
    eprintln!("Commands: add|zero|toggle|remove <r> <n> <s> <f> <row> <col>");
    eprintln!("          try <r> <n> <s> <f> <row> <col> <0|1>, sweep <r> [min_stem [max_stem]]");
    eprintln!("          interpage [r], propagate on|off");
    eprintln!("          mapview <E|H|P> <source_n> [r]  |  mapview all [r]");
    eprintln!("          why <r> <n> <s> <f> (why is this differential un/determined?)");
    eprintln!("          undo [r n s f row col], list, status, regen [r [n]], save <path>, outside [retry|status], quit");
    eprintln!("          hidden <E|H|P> <n> <s> <f> <idx> <tn> <ts> <tf> <tidx>  (assert a hidden map value on the");
    eprintln!("          terminal page; idx accepts sums like 0+2; Toda P(a∘E²b)=P(a)∘b propagates it)");
    eprintln!("          hidden list | hidden remove <same args> | hidden undo  (EHP_HIDDEN=0 disables)");
    eprintln!("Tip: click two nodes in a chart to copy an `add` command; shift-click extra");
    eprintln!("     targets first for a sum. ';'-separated commands solve as one batch.");
    eprintln!("Chart keys: Shift+M maps minimap | e/h/p open map view | Shift+E/H/P highlight map image");
    eprintln!("            (run `mapview all` first so e/h/p and WASD land on generated views)");
    eprint!("> ");

    let mut undo_stack: Vec<UndoEntry> = Vec::new();
    // cmd string -> JSON array of differentials deduced by propagating it.
    let mut prop_log: HashMap<String, serde_json::Value> = HashMap::new();
    // Whether mutations propagate immediately (cascade after every add) or
    // are deferred until an explicit `interpage`.
    let mut auto_prop = true;
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let quit = process_stdin_cmd(
            &line,
            &mut pages,
            &mut undo_stack,
            &csv_paths,
            &charts_dir,
            &seqsee_dir,
            &theme,
            &mut prop_log,
            &mut auto_prop,
            &mut hidden,
        );
        if quit {
            break;
        }
        eprint!("> ");
    }

    Ok(())
}

// =============================================================================
// Hidden EHP map values (EXPERIMENTAL — Toda composition propagation)
// -----------------------------------------------------------------------------
// A pure overlay on the terminal page (see ehp_core::hidden for the module
// contract): user-asserted hidden E/H/P values propagate through Toda's
// P(a∘E²b) = P(a)∘b over the page's composition products (nonzero products
// only), are persisted to <data_dir>/hidden_EHP.csv (asserted rows only;
// deliberately NOT part of the warm-start cache hash — it cannot affect the
// solve), and are drawn as dotted overlay edges in the fiber charts.
// EHP_HIDDEN=0 disables everything. Nothing here touches solver / cascade /
// snapshot state; snapshots need no changes (the CSV lives outside output/
// and startup re-injects over any stale snapshot chart HTML).
// =============================================================================

/// All hidden-value session state, bound to the TERMINAL page (`pages.last()`).
struct HiddenState {
    store: hidden::HiddenStore,
    csv_path: PathBuf,
}

/// Startup: load persisted assertions, re-propagate against the terminal
/// page (deduced values are never persisted — recomputing them here is what
/// makes the CSV robust to basis drift), and inject the fiber-chart overlay.
fn hidden_startup(pages: &[PageState], data_path: &str, charts_dir: &Path) -> HiddenState {
    let mut state = HiddenState {
        store: hidden::HiddenStore::new(),
        csv_path: hidden::hidden_csv_path(data_path),
    };
    if !hidden::hidden_enabled() {
        return state;
    }
    let Some(ps) = pages.last() else { return state };
    let csv_path = state.csv_path.clone();
    for w in state.store.load_csv(&csv_path, &ps.page, ps.page.r) {
        eprintln!("  {}", w);
    }
    let (_, rule_warnings) = state.store.propagate(&ps.page);
    for w in &rule_warnings {
        eprintln!("  note: {}", w);
    }
    let asserted = state.store.asserted().count();
    if asserted > 0 || !state.store.quarantined.is_empty() {
        eprintln!(
            "Hidden EHP values: {} asserted loaded from {}, {} deduced by Toda propagation, {} quarantined.",
            asserted,
            csv_path.display(),
            state.store.deduced().count(),
            state.store.quarantined.len(),
        );
    }
    inject_hidden_data(charts_dir, pages, &state);
    state
}

/// After any differential mutation's cascade: the terminal page basis may
/// have changed, so re-validate every assertion (quarantining what no longer
/// parses) and re-deduce, then refresh the chart overlay. The CSV is
/// deliberately NOT rewritten here — quarantined rows persist on disk and
/// are re-admitted when the page state recovers (e.g. after the diff is
/// undone).
fn hidden_after_mutation(pages: &[PageState], hidden: &mut HiddenState, charts_dir: &Path) {
    if !hidden::hidden_enabled() || hidden.store.is_empty() {
        return;
    }
    let Some(ps) = pages.last() else { return };
    for w in hidden.store.rebuild(&ps.page, ps.page.r) {
        eprintln!("  {}", w);
    }
    inject_hidden_data(charts_dir, pages, hidden);
}

/// Fiber-chart base sphere N whose chart shows a hidden value of `kind` from
/// source sphere `n` (source and target spheres are both in
/// `triple_spheres(N)`).
fn hidden_fiber_base(kind: MapKind, src_n: i32) -> Option<i32> {
    match kind {
        MapKind::E => Some(src_n),                               // S^N → S^{N+1}
        MapKind::H => Some(src_n - 1),                           // S^{N+1} → S^{2N+1}
        MapKind::P => (src_n % 2 == 1).then(|| (src_n - 1) / 2), // S^{2N+1} → S^N
        MapKind::Lh0 => None,
    }
}

/// Deduced hidden values → prop-log JSON entries. `"hidden": true` makes the
/// log panel print the label instead of a d_r line.
fn hidden_to_json(pages: &[PageState], values: &[hidden::HiddenValue]) -> Vec<serde_json::Value> {
    let Some(ps) = pages.last() else { return Vec::new() };
    let r = ps.page.r;
    values
        .iter()
        .filter_map(|v| {
            let (sd, td) = (v.source.degree, v.target.degree);
            let (src_dim, tgt_dim) = (ps.page.dim_at(sd), ps.page.dim_at(td));
            if src_dim == 0 || tgt_dim == 0 {
                return None;
            }
            let src_i = ehp_core::gf2::vec_support(&v.source.vec).next()?;
            let tgt_i = ehp_core::gf2::vec_support(&v.target.vec).next()?;
            let chart = hidden_fiber_base(v.kind, sd.n)
                .filter(|b| ps.fiber_values.contains(b))
                .map(|b| serde_json::json!(format!("fiber{}_E{}.html", b, r)))
                .unwrap_or(serde_json::Value::Null);
            Some(serde_json::json!({
                "hidden": true,
                "kind": v.kind.name(),
                "delta": v.delta(),
                "label": v.to_string(),
                "srcId": seqsee::gen_name(sd.n, sd.s, sd.f, src_i, src_dim),
                "tgtId": seqsee::gen_name(td.n, td.s, td.f, tgt_i, tgt_dim),
                "chart": chart,
            }))
        })
        .collect()
}

/// Inject (or refresh) the hidden-value overlay data between
/// `/*HIDDENDATA*/…/*ENDHIDDENDATA*/` markers in the terminal page's fiber
/// charts. Every chart gets its (possibly empty) entry list, so stale
/// overlays clear. Entries carry node-id lists for every summand; the JS
/// draws one dotted line per (src, tgt) support pair. Returns charts updated.
fn inject_hidden_data(charts_dir: &Path, pages: &[PageState], hidden: &HiddenState) -> usize {
    if !hidden::hidden_enabled() {
        return 0;
    }
    let Some(ps) = pages.last() else { return 0 };
    let r = ps.page.r;
    let mut by_base: HashMap<i32, Vec<serde_json::Value>> = HashMap::new();
    let mut undrawn = 0usize;
    for v in hidden.store.iter() {
        let (sd, td) = (v.source.degree, v.target.degree);
        let (src_dim, tgt_dim) = (ps.page.dim_at(sd), ps.page.dim_at(td));
        let base = hidden_fiber_base(v.kind, sd.n).filter(|b| ps.fiber_values.contains(b));
        let (Some(base), true) = (base, src_dim > 0 && tgt_dim > 0) else {
            undrawn += 1;
            continue;
        };
        let ids = |deg: Tridegree, vec: &fp::vector::FpVector, dim: usize| -> Vec<String> {
            ehp_core::gf2::vec_support(vec)
                .map(|i| seqsee::gen_name(deg.n, deg.s, deg.f, i, dim))
                .collect()
        };
        by_base.entry(base).or_default().push(serde_json::json!({
            "kind": v.kind.name(),
            "delta": v.delta(),
            "asserted": v.is_asserted(),
            "label": v.to_string(),
            "src": ids(sd, &v.source.vec, src_dim),
            "tgt": ids(td, &v.target.vec, tgt_dim),
        }));
    }
    let mut updated = 0;
    for &base in &ps.fiber_values {
        let data =
            serde_json::Value::Array(by_base.remove(&base).unwrap_or_default()).to_string();
        let path = charts_dir.join(format!("fiber{}_E{}.html", base, r));
        let Ok(html) = std::fs::read_to_string(&path) else { continue };
        if let Some(new_html) = replace_marker(&html, "/*HIDDENDATA*/", "/*ENDHIDDENDATA*/", &data)
        {
            if std::fs::write(&path, new_html).is_ok() {
                updated += 1;
            }
        }
    }
    if undrawn > 0 {
        eprintln!(
            "  note: {} hidden values fall outside every generated fiber chart (still listed by `hidden list`).",
            undrawn,
        );
    }
    updated
}

/// Shared tail of `hidden undo` / `hidden remove`: rebuild the deductions
/// from the remaining assertions, rewrite the CSV, drop the stale hidden
/// prop-log entries, refresh charts.
fn finish_hidden_removal(
    pages: &[PageState],
    hidden: &mut HiddenState,
    charts_dir: &Path,
    prop_log: &mut HashMap<String, serde_json::Value>,
) {
    let Some(ps) = pages.last() else { return };
    for w in hidden.store.rebuild(&ps.page, ps.page.r) {
        eprintln!("  {}", w);
    }
    if let Err(e) = hidden.store.save_csv(&hidden.csv_path, ps.page.r) {
        eprintln!("  WARNING: could not write {}: {}", hidden.csv_path.display(), e);
    }
    prop_log.retain(|k, _| !k.starts_with("hidden "));
    inject_propagation(charts_dir, pages, prop_log);
    inject_hidden_data(charts_dir, pages, hidden);
    eprintln!(
        "  Remaining: {} asserted, {} deduced. Refresh browser.",
        hidden.store.asserted().count(),
        hidden.store.deduced().count(),
    );
}

/// The `hidden` REPL command family:
/// `hidden <E|H|P> <n> <s> <f> <idx> <tn> <ts> <tf> <tidx>` (idx accepts
/// `+`-joined sums like `0+2`), `hidden list`, `hidden remove <same args>`,
/// `hidden undo`.
fn hidden_cmd(
    parts: &[&str],
    pages: &[PageState],
    hidden: &mut HiddenState,
    charts_dir: &Path,
    prop_log: &mut HashMap<String, serde_json::Value>,
) {
    if !hidden::hidden_enabled() {
        eprintln!("Hidden values are disabled (EHP_HIDDEN=0).");
        return;
    }
    let Some(ps) = pages.last() else { return };
    let page = &ps.page;
    let r = page.r;
    match parts.get(1).copied() {
        Some("list") => {
            eprintln!("Hidden EHP values (terminal page E_{}):", r);
            let mut any = false;
            for v in hidden.store.asserted() {
                eprintln!("  [asserted]    {}", v);
                any = true;
            }
            for v in hidden.store.deduced() {
                eprintln!("  [deduced]     {}", v);
                any = true;
            }
            for q in &hidden.store.quarantined {
                eprintln!("  [quarantined] {}({}) = {} — {}", q.kind, q.source, q.target, q.reason);
                any = true;
            }
            if !any {
                eprintln!("  (none — assert one with `hidden P <n> <s> <f> <idx> <tn> <ts> <tf> <tidx>`)");
            }
        }
        Some("undo") => match hidden.store.pop_asserted() {
            Some(v) => {
                eprintln!("Removed hidden {}", v);
                finish_hidden_removal(pages, hidden, charts_dir, prop_log);
            }
            None => eprintln!("No asserted hidden values to undo."),
        },
        Some("remove") => {
            let (kind, source, target) = match hidden::parse_hidden_args(page, &parts[2..]) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("hidden remove: {}", e);
                    return;
                }
            };
            if hidden.store.remove_asserted(kind, &source.to_string(), &target.to_string()) {
                eprintln!("Removed hidden {}({}) = {}", kind, source, target);
                finish_hidden_removal(pages, hidden, charts_dir, prop_log);
            } else {
                eprintln!(
                    "Not an asserted hidden value (deduced ones can't be removed directly — \
                     remove the assertion they came from)."
                );
            }
        }
        _ => {
            let (kind, source, target) = match hidden::parse_hidden_args(page, &parts[1..]) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("hidden: {}", e);
                    return;
                }
            };
            match hidden.store.assert_value(page, kind, source.clone(), target.clone()) {
                Ok(true) => {}
                Ok(false) => {
                    eprintln!("Already recorded.");
                    return;
                }
                Err(e) => {
                    eprintln!("hidden: {}", e);
                    return;
                }
            }
            let nominal = kind.target_degree(source.degree);
            eprintln!(
                "Recorded hidden {}({}) = {} (δ={} above the nominal target {}).",
                kind,
                source,
                target,
                target.degree.f - nominal.f,
                nominal,
            );
            let (deduced, warnings) = hidden.store.propagate(page);
            for w in &warnings {
                eprintln!("  note: {}", w);
            }
            for v in &deduced {
                eprintln!("  deduced: {}", v);
            }
            if deduced.is_empty() {
                eprintln!("  No new values deduced (no all-nonzero composition partners found).");
            }
            if let Err(e) = hidden.store.save_csv(&hidden.csv_path, r) {
                eprintln!("  WARNING: could not write {}: {}", hidden.csv_path.display(), e);
            } else {
                eprintln!("  Persisted to {}.", hidden.csv_path.display());
            }
            let cmd_key = format!("hidden {}", parts[1..].join(" "));
            let entries = hidden_to_json(pages, &deduced);
            prop_log.insert(cmd_key, serde_json::Value::Array(entries));
            inject_propagation(charts_dir, pages, prop_log);
            let updated = inject_hidden_data(charts_dir, pages, hidden);
            if updated > 0 {
                eprintln!(
                    "  Updated {} fiber charts. Refresh browser to see the overlay.",
                    updated,
                );
            }
        }
    }
}

// =============================================================================
// REPL command processing
// =============================================================================

/// Find the index of a page by its r value.
fn page_index(pages: &[PageState], r: i32) -> Option<usize> {
    pages.iter().position(|ps| ps.page.r == r)
}

/// Compare two solve results and return tridegrees where the differential
/// matrix used for turning changed. Turning uses partial matrices (determined
/// entries kept, unknown entries zero), so a var's turning-effective value is
/// its determined value, or 0 when unknown/absent.
fn changed_diff_tridegrees(
    old_result: Option<&SATResult>,
    new_result: &SATResult,
) -> HashSet<Tridegree> {
    let mut changed = HashSet::new();

    // Check all vars in new result
    for var in &new_result.vars {
        let new_val = effective_var_value(new_result, var);
        let old_val = old_result.is_some_and(|r| effective_var_value(r, var));
        if new_val != old_val {
            changed.insert(var.tridegree());
        }
    }

    // Check vars in old result that disappeared (shouldn't normally happen
    // during cascade since the page doesn't change, but handle defensively)
    if let Some(old) = old_result {
        for var in &old.vars {
            if !new_result.var_index.contains_key(var) && effective_var_value(old, var) {
                changed.insert(var.tridegree());
            }
        }
    }

    changed
}

/// Turning-effective value of a differential variable: false if unknown or
/// absent, otherwise its determined value.
fn effective_var_value(result: &SATResult, var: &DiffVar) -> bool {
    if let Some(&idx) = result.var_index.get(var) {
        if result.unknown.contains(&idx) {
            false
        } else {
            result.offset.entry(idx) != 0
        }
    } else {
        false
    }
}

/// Given tridegrees where diff matrices changed, compute the set of tridegrees
/// whose homology is affected and needs re-turning.
///
/// For each changed source tridegree t:
/// - Homology at t changes (d_out changed)
/// - Homology at diff_target(t, r) changes (d_in changed)
fn affected_homology_tridegrees(changed: &HashSet<Tridegree>, r: i32) -> HashSet<Tridegree> {
    let mut affected = HashSet::new();
    for &t in changed {
        affected.insert(t);
        affected.insert(t.diff_target(r));
    }
    affected
}

/// Expand a set of re-turn degrees across the stable fold: for any degree at
/// or past the stable edge (n >= s+2), include its stable rep (s+2, s, f) and
/// every past-stable copy (n' > s+2, same (s, f)) present on the page.
///
/// `changed`/`affected` degrees come from DiffVars keyed at stable-REP
/// degrees (n = min(n, s+2)), but the turned cache / dimensions / products
/// are keyed by ACTUAL degrees. Copies share the rep's differential data, so
/// their homology changes in lockstep with the rep's; leaving them
/// un-re-turned gives the rebuilt next page fresh homology at the rep and
/// STALE homology at the copies, and constraint generation (E-naturality
/// identity squares across the near-stable fold; h_i Leibniz pairs whose
/// second factor lives at sphere th2 = n+s-1) then reads the stale copies and
/// emits wrong constraints — up to false contradictions against correct user
/// assertions. This mirrors the stable-copy expansion in ehp-core's interpage
/// path (`turn_page_local` + `SourceIndex::stable_by_sf`).
///
/// The (s, f) → copies index is one O(page keys) scan per call — trivial next
/// to the constraint re-solve, and the pages it indexes are rebuilt every
/// cascade iteration anyway, so caching it in `PageState` would just add an
/// invalidation hazard.
fn expand_stable_fold(page: &SATPage, degrees: HashSet<Tridegree>) -> HashSet<Tridegree> {
    // Only build the index if something can actually expand.
    if !degrees.iter().any(|t| t.n >= t.s + 2) {
        return degrees;
    }
    let mut copies_by_sf: HashMap<(i32, i32), Vec<Tridegree>> = HashMap::new();
    for &t in page.page.keys() {
        if t.n > t.s + 2 {
            copies_by_sf.entry((t.s, t.f)).or_default().push(t);
        }
    }
    let mut out = degrees;
    let seeds: Vec<Tridegree> = out.iter().copied().filter(|t| t.n >= t.s + 2).collect();
    for t in seeds {
        out.insert(Tridegree::new(t.s + 2, t.s, t.f));
        if let Some(copies) = copies_by_sf.get(&(t.s, t.f)) {
            out.extend(copies.iter().copied());
        }
    }
    out
}

/// Stable-fold invariant: every past-stable degree (n > s+2) has the same
/// dimension as its stable rep (s+2, s, f). Returns violations as
/// `(degree, dim, rep_dim)`, sorted. One pass over `dimension` — cheap enough
/// to run after every cascade rebuild as a corruption guardrail.
fn stable_fold_violations(page: &SATPage) -> Vec<(Tridegree, usize, usize)> {
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

/// A differential newly determined (or changed) by a cascade re-solve.
struct DeducedDiff {
    r: i32,
    var: DiffVar,
    value: bool,
}

/// What a cascade re-solve did: the differentials it deduced, plus — per page
/// r — the tridegrees whose quotient basis was recomputed there. Charts
/// covering those degrees have stale node sets (the CLASSDIMS fade shortcut
/// cannot express *which* class died, only how many survive), so they are
/// regenerated from the fresh homology by `regen_affected_charts`.
struct CascadeOutcome {
    deduced: Vec<DeducedDiff>,
    /// page r → re-turned tridegrees on that page.
    affected_degrees: HashMap<i32, HashSet<Tridegree>>,
}

/// Bitwise matrix equality.
fn mat_eq(a: &fp::matrix::Matrix, b: &fp::matrix::Matrix) -> bool {
    a.rows() == b.rows()
        && a.columns() == b.columns()
        && ehp_core::gf2::mat_raw_words(a) == ehp_core::gf2::mat_raw_words(b)
}

/// Full content comparison of two pages: dimensions, exclusions, products,
/// and maps. Used to decide whether a cascade rebuild actually changed
/// anything a later page depends on.
/// Describe up to `limit` structural differences between two pages (for the
/// cascade-patch verifier's mismatch report). `a` = patched, `b` = full.
fn page_structural_diff(a: &SATPage, b: &SATPage, limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    for (&t, &d) in &b.dimension {
        if a.dim_at(t) != d {
            out.push(format!("dim({},{},{}) patched {} vs full {}", t.n, t.s, t.f, a.dim_at(t), d));
        }
    }
    for (&t, &d) in &a.dimension {
        if b.dimension.get(&t).copied().unwrap_or(0) != d {
            out.push(format!("dim({},{},{}) patched {} vs full {}", t.n, t.s, t.f, d, b.dim_at(t)));
        }
    }
    for (&(d1, d2), pb) in b.products.iter_blocks() {
        match a.products.block(d1, d2) {
            Some(pa) if pa == pb => {}
            Some(_) => out.push(format!("product block ({},{},{})x({},{},{}) differs", d1.n, d1.s, d1.f, d2.n, d2.s, d2.f)),
            None => out.push(format!("product block ({},{},{})x({},{},{}) missing in patched", d1.n, d1.s, d1.f, d2.n, d2.s, d2.f)),
        }
        if out.len() >= limit {
            return out;
        }
    }
    for (&(d1, d2), _) in a.products.iter_blocks() {
        if b.products.block(d1, d2).is_none() {
            out.push(format!("product block ({},{},{})x({},{},{}) extra in patched", d1.n, d1.s, d1.f, d2.n, d2.s, d2.f));
        }
        if out.len() >= limit {
            return out;
        }
    }
    for kind in MapKind::all_with_lh0() {
        let (Some(ma), Some(mb)) = (a.maps.get(&kind), b.maps.get(&kind)) else { continue };
        for (t, mat_b) in &mb.matrices {
            match ma.matrices.get(t) {
                Some(mat_a) if mat_a.as_ref() == mat_b.as_ref() => {}
                Some(_) => out.push(format!("{} map at ({},{},{}) differs", kind, t.n, t.s, t.f)),
                None => out.push(format!("{} map at ({},{},{}) missing in patched", kind, t.n, t.s, t.f)),
            }
            if out.len() >= limit {
                return out;
            }
        }
        for t in ma.matrices.keys() {
            if !mb.matrices.contains_key(t) {
                out.push(format!("{} map at ({},{},{}) extra in patched", kind, t.n, t.s, t.f));
            }
            if out.len() >= limit {
                return out;
            }
        }
    }
    out
}

/// Structural page content only — dimensions, products, maps — WITHOUT the
/// exclusion sets (the cascade compares those separately via the
/// symmetric-difference it already computes). The distinction matters: the
/// next page's structural content is a pure function of (this page's
/// structural content, the turned data), so when neither changed the rebuild
/// is skippable and only the exclusion sets need recomputing.
fn page_structural_equal(a: &SATPage, b: &SATPage) -> bool {
    if a.dimension != b.dimension {
        return false;
    }
    if a.products.num_blocks() != b.products.num_blocks() {
        return false;
    }
    for (&(d1, d2), pa) in a.products.iter_blocks() {
        let Some(pb) = b.products.block(d1, d2) else { return false };
        if pa.dim1 != pb.dim1 || pa.dim2 != pb.dim2 || pa.tgt_dim != pb.tgt_dim {
            return false;
        }
        if pa != pb {
            return false;
        }
    }
    for kind in MapKind::all() {
        let (Some(ma), Some(mb)) = (a.maps.get(&kind), b.maps.get(&kind)) else {
            return a.maps.get(&kind).is_none() == b.maps.get(&kind).is_none();
        };
        if ma.matrices.len() != mb.matrices.len() {
            return false;
        }
        for (t, mat_a) in &ma.matrices {
            let Some(mat_b) = mb.matrices.get(t) else { return false };
            if mat_a.as_ref() != mat_b.as_ref() {
                return false;
            }
        }
    }
    true
}

/// Map of determined variables to their values.
fn determined_map(res: &SATResult) -> HashMap<DiffVar, bool> {
    res.vars
        .iter()
        .enumerate()
        .filter(|(i, _)| !res.unknown.contains(i))
        .map(|(i, v)| (*v, res.offset.entry(i) != 0))
        .collect()
}

/// EHP_CASCADE_PATCH != "0" (default ON): rebuild cascade pages incrementally
/// — recompute only the induced maps/products/dims touching the tridegrees
/// whose induction inputs changed — instead of re-inducing the whole page
/// (the dominant cascade cost at scale). `EHP_CASCADE_PATCH=0` restores the
/// full rebuild everywhere.
fn cascade_patch_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("EHP_CASCADE_PATCH").map_or(true, |v| v != "0"))
}

/// EHP_CASCADE_VERIFY == "1": run BOTH the incremental patch and the full
/// rebuild on every cascade step and cross-check them structurally
/// (expensive — for validating the patch path on a real workload).
fn cascade_verify() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("EHP_CASCADE_VERIFY").map_or(false, |v| v == "1"))
}

/// EHP_CONSENSUS != "0" (default ON): `interpage try` applies both-worlds
/// consensus determinations (a var forced to the same value whether a switch
/// is 0 or 1 holds unconditionally). `EHP_CONSENSUS=0` restores
/// contradiction-only forcing.
fn consensus_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("EHP_CONSENSUS").map_or(true, |v| v != "0"))
}

/// Cascade re-solve: re-solve page at `start_idx`, then incrementally re-turn
/// and re-solve subsequent pages. Stops early if no differential matrices
/// change — unless `force` is set (an explicit `interpage` run), in which
/// case every page is re-solved, re-turned, and rebuilt to the end.
/// `solve_through` is the highest page INDEX with directly mutated
/// known_diffs: pages up to and including it are always re-solved (an early
/// stop below it would silently drop a mutation applied on a later page).
/// Returns every differential that became determined (or changed value)
/// relative to the previous solve — the deductions from the mutation — plus
/// the re-turned degrees per page (whose charts need real regeneration).
fn cascade_resolve(
    pages: &mut [PageState],
    start_idx: usize,
    solve_through: usize,
    force: bool,
) -> CascadeOutcome {
    let mut deduced = Vec::new();
    let mut affected_degrees: HashMap<i32, HashSet<Tridegree>> = HashMap::new();
    // Degrees on the *next* page whose dimensions changed when it was rebuilt
    // — their homology (and everything downstream) must be re-turned even if
    // no differential value changed there.
    let mut carry: HashSet<Tridegree> = HashSet::new();
    // Whether the previous iteration's rebuild changed the page content
    // (dims, exclusions, products, maps) at all. Product/map changes don't
    // show up in diff-value comparison but still propagate downstream.
    let mut prev_content_changed = false;
    // Degrees where the current page's structural content (dims/products/
    // maps — NOT exclusions) changed when it was last rebuilt/patched — the
    // extra dirty set for patching the NEXT page (content changes here alter
    // its induction inputs at exactly these degrees). `None` = unknown delta
    // (a full rebuild that changed things): the next page must be fully
    // rebuilt too.
    let mut content_dirty: Option<HashSet<Tridegree>> = Some(HashSet::new());
    for i in start_idx..pages.len() {
        let r = pages[i].page.r;

        // Save old result for comparison
        let old_result = pages[i].result.take();

        // Re-solve
        let cutoff = pages[i].page.max_s.unwrap_or(0);
        let system =
            constraints::build_constraint_system(&pages[i].page, cutoff, &pages[i].known_diffs);
        let num_vars = system.num_vars;
        let result = solver::solve_with_d2(&pages[i].page, &system).map(|(res, d2n)| {
            if d2n > 0 {
                eprintln!("  E_{}: d\u{b2}=0 linearization determined {} more entries", r, d2n);
            }
            res
        });

        if let Some(ref res) = result {
            let determined = num_vars - res.unknown.len();
            eprintln!("  E_{}: {}/{} vars determined", r, determined, num_vars);
        } else if num_vars == 0 {
            eprintln!("  E_{}: no differential variables", r);
        } else {
            eprintln!("  E_{}: INCONSISTENT", r);
        }

        // Collect newly determined (or value-changed) differentials, and
        // track every DETERMINATION-STATUS change (unknown ↔ determined,
        // value flips). A var going unknown → determined-ZERO changes no
        // turning-effective matrix (unknown already turns as 0), but it DOES
        // change the next page's exclusion set and this page's uncertainty
        // display — so it must keep the cascade alive and regenerate the
        // charts whose baked nulldif/exclusion dashes it invalidates.
        let mut determination_changed = false;
        let mut det_changed_degs: HashSet<Tridegree> = HashSet::new();
        if let Some(ref new_res) = result {
            let old_det = old_result
                .as_ref()
                .map(determined_map)
                .unwrap_or_default();
            let mark = |var: &DiffVar, det_degs: &mut HashSet<Tridegree>| {
                let src = Tridegree::new(var.n, var.s, var.f);
                det_degs.insert(src);
                det_degs.insert(src.diff_target(r));
            };
            for (vi, var) in new_res.vars.iter().enumerate() {
                if new_res.unknown.contains(&vi) {
                    if old_det.contains_key(var) {
                        // Lost determination (e.g. after an undo): its dashes
                        // must come back.
                        determination_changed = true;
                        mark(var, &mut det_changed_degs);
                    }
                    continue;
                }
                let val = new_res.offset.entry(vi) != 0;
                if old_det.get(var) != Some(&val) {
                    deduced.push(DeducedDiff { r, var: *var, value: val });
                    determination_changed = true;
                    mark(var, &mut det_changed_degs);
                }
            }
            // Vars that vanished from the system entirely.
            for var in old_det.keys() {
                if !new_res.var_index.contains_key(var) {
                    determination_changed = true;
                    mark(var, &mut det_changed_degs);
                }
            }
        }
        // Refresh this page's charts where determination changed: the
        // generation-time dashed (nulldif / excluded-degree) lines only clear
        // on a real regen with a fresh CSV — the DIFFDATA overlay alone
        // leaves the baked dashes behind. Expanded across the stable fold so
        // copy-sphere charts refresh with their rep.
        if !det_changed_degs.is_empty() {
            let folded = expand_stable_fold(&pages[i].page, det_changed_degs);
            affected_degrees.entry(r).or_default().extend(folded);
        }

        // TODO(unsat-quarantine): before reporting INCONSISTENT as ground
        // truth, re-verify against a freshly built (non-incremental) page
        // chain, so stale incremental state can never manufacture a false
        // contradiction against a correct user assertion. Not built yet.
        pages[i].unsat_reason = match &result {
            Some(_) => None,
            None if num_vars == 0 => Some(format!(
                "E_{} has no differential variables (all classes permanent)", r,
            )),
            None => Some(format!(
                "the E_{} re-solve was INCONSISTENT — a recorded differential contradicts \
                 the constraints (`undo` to restore)", r,
            )),
        };
        pages[i].result = result;
        pages[i].excluded_leibniz = system.excluded_leibniz;

        // If there's no next page or no result, we're done
        if i + 1 >= pages.len() {
            break;
        }
        let Some(ref new_res) = pages[i].result else {
            break;
        };

        // Check which diff matrices actually changed, plus degrees whose
        // dimensions changed when this page was rebuilt (carried from the
        // previous iteration). Even with no value/dimension change, a rebuilt
        // page whose products/maps differ must keep cascading.
        let mut changed = changed_diff_tridegrees(old_result.as_ref(), new_res);
        changed.extend(carry.drain());
        if changed.is_empty()
            && !prev_content_changed
            && !determination_changed
            && !force
            && i >= solve_through
        {
            eprintln!(
                "  E_{}: nothing changed (values, dims, products, maps, determination) — cascade stops",
                r,
            );
            break;
        }

        let affected = affected_homology_tridegrees(&changed, r);
        // THE FIX (stale stable copies): expand across the stable fold so
        // past-stable copies of any changed rep are re-turned too — otherwise
        // the rebuilt page mixes fresh rep homology with stale copy homology
        // and constraint generation emits wrong (even falsely contradictory)
        // constraints. See `expand_stable_fold`.
        let affected = expand_stable_fold(&pages[i].page, affected);
        eprintln!(
            "  E_{}: {} tridegrees changed, {} affected for re-turning",
            r,
            changed.len(),
            affected.len(),
        );
        // The quotient basis on the *next* page is recomputed at these
        // degrees — its charts there can no longer be trusted index-wise.
        affected_degrees
            .entry(r + 1)
            .or_default()
            .extend(affected.iter().copied());

        // Split for borrow checker: need mutable access to pages[i] and pages[i+1]
        let (left, right) = pages.split_at_mut(i + 1);
        let curr = &mut left[i];
        let next_ps = &mut right[0];
        let new_res = curr.result.as_ref().unwrap();

        // Get or initialize turned data
        let mut turned_fresh = false;
        if curr.turned.is_none() {
            // Fallback: full turn (shouldn't happen normally, startup initializes it)
            let new_max_t = curr.page.max_t.map(|t| t - 1);
            match pageturning::turn_page(&curr.page, new_res, r + 1, new_max_t) {
                Ok(t) => {
                    curr.turned = Some(t);
                    // The delta vs whatever the next page was built from is
                    // unknown — incremental patching is off the table.
                    turned_fresh = true;
                }
                Err(e) => {
                    eprintln!("  E_{}: CONTRADICTION while turning: {} — consider `undo`", r, e);
                    return CascadeOutcome { deduced, affected_degrees };
                }
            }
        }
        let turned = curr.turned.as_mut().unwrap();

        // Incrementally re-turn only affected tridegrees (with the original's
        // uncertainty semantics: whole differential zeroed at uncertain degrees)
        let ctx = pageturning::TurnContext::new(&curr.page, new_res);
        for &t in &affected {
            if curr.page.dim_at(t) == 0 {
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
                    eprintln!("  E_{}: CONTRADICTION: {} — consider `undo`", r, e);
                    return CascadeOutcome { deduced, affected_degrees };
                }
            }
        }

        // Dirty set for incremental patching of the next page: the degrees
        // whose turned data was just re-computed (`affected`) plus the
        // degrees where THIS page's structural content changed when it was
        // itself rebuilt/patched (they alter the induction inputs). `None`
        // means fall back to the full rebuild.
        let patch_dirty: Option<HashSet<Tridegree>> =
            if force || !cascade_patch_enabled() || turned_fresh {
                None
            } else {
                content_dirty.take().map(|mut cd| {
                    cd.extend(affected.iter().copied());
                    cd
                })
            };

        // FAST PATH: empty dirty set — the next page's structural content is
        // byte-identical, so skip the rebuild entirely; only the exclusion
        // sets (a function of which entries are UNKNOWN, not of the
        // turning-effective values) can differ.
        if patch_dirty.as_ref().is_some_and(|d| d.is_empty()) && !cascade_verify() {
            let (next_exclude, next_target_only) =
                pageturning::make_next_exclude_set(&curr.page, new_res);
            let excl_changed: Vec<Tridegree> = next_exclude
                .symmetric_difference(&next_ps.page.exclude_set)
                .copied()
                .collect();
            let excl_any = !excl_changed.is_empty()
                || next_target_only != next_ps.page.target_only_exclude;
            if !excl_changed.is_empty() {
                let e = affected_degrees.entry(r + 1).or_default();
                for t in excl_changed {
                    e.insert(t);
                    e.insert(t.diff_target(r + 1));
                }
            }
            next_ps.page.exclude_set = next_exclude;
            next_ps.page.target_only_exclude = next_target_only;
            eprintln!(
                "  E_{}: structurally unchanged — skipped page rebuild ({})",
                r + 1,
                if excl_any { "exclusions updated" } else { "exclusions unchanged" },
            );
            carry.clear();
            prev_content_changed = excl_any;
            content_dirty = Some(HashSet::new());
            if !excl_any && i + 1 > solve_through {
                // The next page's inputs are bit-identical (structure,
                // exclusions, and its own known_diffs untouched), so its
                // re-solve — and everything downstream — is a provable
                // no-op. Stop instead of paying the solves.
                eprintln!(
                    "  E_{}: inputs unchanged — downstream re-solves skipped, cascade stops",
                    r + 1,
                );
                break;
            }
            continue;
        }

        // Rebuild the next page: incrementally patch a cheap Arc-sharing
        // clone of the old one (only maps/products/dims touching the dirty
        // degrees are recomputed), or fall back to the full rebuild.
        // EHP_CASCADE_VERIFY=1 runs both and cross-checks.
        let mut patch_result: Option<pageturning::PatchOutcome> = None;
        let mut next_page = if let Some(dirty) = &patch_dirty {
            let t_patch = Instant::now();
            let mut np = next_ps.page.overlay_clone();
            let po = pageturning::patch_page_from_turned(&curr.page, turned, dirty, &mut np);
            if timing_enabled() {
                eprintln!(
                    "  [timing] E_{} cascade patch: {:.2}s ({} dirty → {} changed)",
                    r + 1,
                    t_patch.elapsed().as_secs_f64(),
                    dirty.len(),
                    po.changed.len(),
                );
            }
            let mut use_patch = true;
            if cascade_verify() {
                let full = pageturning::build_page_from_turned(&curr.page, turned);
                if page_structural_equal(&np, &full) {
                    eprintln!(
                        "  E_{}: cascade patch verified against full rebuild ({} dirty → {} changed)",
                        r + 1,
                        dirty.len(),
                        po.changed.len(),
                    );
                } else {
                    eprintln!(
                        "  E_{}: CASCADE-PATCH MISMATCH vs full rebuild ({} dirty) — using the \
                         full rebuild; please report (EHP_CASCADE_PATCH=0 disables patching)",
                        r + 1,
                        dirty.len(),
                    );
                    for line in page_structural_diff(&np, &full, 8) {
                        eprintln!("    {}", line);
                    }
                    np = full;
                    use_patch = false;
                }
            }
            if use_patch {
                patch_result = Some(po);
            }
            np
        } else {
            pageturning::build_page_from_turned(&curr.page, turned)
        };

        // GUARDRAIL (stable-fold invariant): every past-stable degree must
        // share its rep's dimension. A violation means the incremental
        // re-turn missed a degree (the stale-stable-copy bug class) — warn
        // loudly and self-heal with a full, non-incremental page turn instead
        // of letting the corruption feed constraint generation.
        let violations = stable_fold_violations(&next_page);
        if !violations.is_empty() {
            eprintln!(
                "  E_{}: STABLE-FOLD VIOLATION after incremental rebuild — {} past-stable degree(s) \
                 disagree with their stable rep:",
                r + 1,
                violations.len(),
            );
            for (t, d, rep_dim) in violations.iter().take(5) {
                eprintln!(
                    "    dim({},{},{}) = {} but rep dim({},{},{}) = {}",
                    t.n, t.s, t.f, d, t.s + 2, t.s, t.f, rep_dim,
                );
            }
            if violations.len() > 5 {
                eprintln!("    ... and {} more", violations.len() - 5);
            }
            eprintln!(
                "    Self-healing with a full E_{} page turn (non-incremental). \
                 This is an incremental-cascade bug — please report it.",
                r,
            );
            let new_max_t = curr.page.max_t.map(|t| t - 1);
            match pageturning::turn_page(&curr.page, new_res, r + 1, new_max_t) {
                Ok(full) => {
                    *turned = full;
                    next_page = pageturning::build_page_from_turned(&curr.page, turned);
                    patch_result = None;
                    let still = stable_fold_violations(&next_page);
                    if !still.is_empty() {
                        eprintln!(
                            "    WARNING: {} violation(s) persist after the full turn — the E_{} \
                             source data itself is fold-inconsistent.",
                            still.len(),
                            r,
                        );
                    }
                }
                Err(e) => {
                    eprintln!(
                        "  E_{}: CONTRADICTION during self-heal full turn: {} — consider `undo`",
                        r, e,
                    );
                    return CascadeOutcome { deduced, affected_degrees };
                }
            }
        }

        // Recompute the next page's exclusions: degrees whose bounding d_r is
        // still unknown stay excluded; degrees that just became determined are
        // un-excluded so constraints there become active.
        let (next_exclude, next_target_only) =
            pageturning::make_next_exclude_set(&curr.page, new_res);
        // Exclusion-status changes alter the chart display on their own: an
        // excluded degree's uncertainty is drawn as dashes to ALL potential
        // targets (seqsee exclusion fallback) even when no variable exists
        // there — so an un-excluded (or newly excluded) degree needs its
        // charts regenerated with a fresh CSV, exactly like a dimension
        // change.
        let excl_any = {
            let excl_changed: Vec<Tridegree> = next_exclude
                .symmetric_difference(&next_ps.page.exclude_set)
                .copied()
                .collect();
            let any = !excl_changed.is_empty()
                || next_target_only != next_ps.page.target_only_exclude;
            if !excl_changed.is_empty() {
                let e = affected_degrees.entry(r + 1).or_default();
                for t in excl_changed {
                    e.insert(t);
                    e.insert(t.diff_target(r + 1));
                }
            }
            any
        };
        next_page.exclude_set = next_exclude;
        next_page.target_only_exclude = next_target_only;
        let next_n_values: BTreeSet<i32> = next_page
            .dimension
            .iter()
            .filter(|(_, &d)| d > 0)
            .map(|(&t, _)| t.n)
            .collect();
        // Record where the rebuilt page's dimensions differ from the old one:
        // homology at those degrees, at the sources mapping into them, and at
        // their targets must be re-turned on the next iteration.
        let r_next = r + 1;
        let dims_changed: HashSet<Tridegree> = match &patch_result {
            // The patch touched ONLY its dirty degrees, so its dims-changed
            // record is complete — no page-wide scan needed.
            Some(po) => po.dims_changed.iter().copied().collect(),
            None => {
                let mut dims_changed: HashSet<Tridegree> = HashSet::new();
                for (&t, &d) in &next_page.dimension {
                    if next_ps.page.dim_at(t) != d {
                        dims_changed.insert(t);
                    }
                }
                for (&t, &d) in &next_ps.page.dimension {
                    if next_page.dim_at(t) != d {
                        dims_changed.insert(t);
                    }
                }
                dims_changed
            }
        };
        carry = dims_changed
            .iter()
            .flat_map(|&t| [t, t.diff_source(r_next)])
            .collect();
        // Chart regen: dimension changes anywhere on the rebuilt page (incl.
        // from a self-heal full turn) invalidate those charts too.
        affected_degrees
            .entry(r_next)
            .or_default()
            .extend(dims_changed.iter().copied());
        if !dims_changed.is_empty() {
            eprintln!(
                "  E_{}: {} tridegrees changed dimension after rebuild",
                r_next,
                dims_changed.len(),
            );
        }
        // Flags + the dirty set for the NEXT page's patch. On the patch path
        // the outcome's changed set is exact; a full rebuild has an unknown
        // delta unless the structural compare says nothing changed.
        let structural_changed;
        (structural_changed, content_dirty) = match patch_result {
            Some(po) => (!po.changed.is_empty(), Some(po.changed)),
            None => {
                if page_structural_equal(&next_ps.page, &next_page) {
                    (false, Some(HashSet::new()))
                } else {
                    (true, None)
                }
            }
        };
        prev_content_changed = structural_changed || excl_any;

        next_ps.stem_values = next_page
            .dimension
            .iter()
            .filter(|(&t, &d)| d > 0 && t.n <= t.s + 2)
            .map(|(&t, _)| t.s)
            .collect();
        next_ps.fiber_values = compute_fiber_values(&next_page, i32::MAX);
        next_ps.page = next_page;
        next_ps.n_values = next_n_values;
    }
    CascadeOutcome { deduced, affected_degrees }
}

/// Process a single REPL command line. Returns `true` if the REPL should exit.
fn process_stdin_cmd(
    line: &str,
    pages: &mut [PageState],
    undo_stack: &mut Vec<UndoEntry>,
    csv_paths: &[PathBuf],
    charts_dir: &Path,
    seqsee_dir: &Path,
    theme: &str,
    prop_log: &mut HashMap<String, serde_json::Value>,
    auto_prop: &mut bool,
    hidden: &mut HiddenState,
) -> bool {
    let line = line.trim();
    if line.is_empty() {
        return false;
    }

    // Multi-mutation line: several ';'-separated add/zero/toggle/remove
    // commands applied together with a single cascade at the end. This is how
    // sum-target differentials arrive from shift-click, and how a batch of
    // differentials can be pasted and solved in one go.
    if line.contains(';') {
        let segs: Vec<&str> = line.split(';').map(str::trim).filter(|s| !s.is_empty()).collect();
        let all_mutations = segs.len() > 1
            && segs.iter().all(|s| {
                matches!(
                    s.split_whitespace().next().unwrap_or(""),
                    "add" | "zero" | "toggle" | "remove"
                )
            });
        if all_mutations {
            process_multi_mutation(
                &segs, pages, undo_stack, csv_paths, charts_dir, seqsee_dir, theme, prop_log,
                *auto_prop, hidden,
            );
            return false;
        }
    }

    let parts: Vec<&str> = line.split_whitespace().collect();
    let cmd = parts[0];

    match cmd {
        "quit" | "exit" | "q" => {
            eprintln!("Goodbye.");
            return true;
        }

        "add" | "zero" | "toggle" | "remove" => {
            if parts.len() < 7 {
                eprintln!("Usage: {} <r> <n> <s> <f> <row> <col>", cmd);
                return false;
            }
            let Ok(r) = parts[1].parse::<i32>() else {
                eprintln!("Invalid r: {}", parts[1]);
                return false;
            };
            let Ok(n) = parts[2].parse::<i32>() else {
                eprintln!("Invalid n: {}", parts[2]);
                return false;
            };
            let Ok(s) = parts[3].parse::<i32>() else {
                eprintln!("Invalid s: {}", parts[3]);
                return false;
            };
            let Ok(f) = parts[4].parse::<i32>() else {
                eprintln!("Invalid f: {}", parts[4]);
                return false;
            };
            let Ok(row) = parts[5].parse::<u16>() else {
                eprintln!("Invalid row: {}", parts[5]);
                return false;
            };
            let Ok(col) = parts[6].parse::<u16>() else {
                eprintln!("Invalid col: {}", parts[6]);
                return false;
            };

            let Some(idx) = page_index(pages, r) else {
                eprintln!("No page E_{} loaded", r);
                return false;
            };

            let n_var = n.min(s + 2);
            let dv = DiffVar::new(n_var, s, f, row, col);
            let prev = pages[idx].known_diffs.get(&dv).copied();

            // Validate the entry is representable on this page; refuse
            // impossible adds instead of recording a no-op.
            if cmd != "remove" {
                match check_diff_addable(&pages[idx].page, r, &dv) {
                    Err(reason) => {
                        eprintln!("Cannot set d_{}[({},{},{})] row={} col={}: {}", r, n, s, f, row, col, reason);
                        return false;
                    }
                    Ok(Some(note)) => eprintln!("Note: {}", note),
                    Ok(None) => {}
                }
            }

            match cmd {
                "add" => {
                    pages[idx].known_diffs.insert(dv, true);
                    eprintln!("Set d_{}[({},{},{})] row={} col={} = 1", r, n, s, f, row, col);
                }
                "zero" => {
                    pages[idx].known_diffs.insert(dv, false);
                    eprintln!("Set d_{}[({},{},{})] row={} col={} = 0", r, n, s, f, row, col);
                }
                "toggle" => {
                    let new_val = match prev {
                        Some(v) => !v,
                        None => true,
                    };
                    pages[idx].known_diffs.insert(dv, new_val);
                    eprintln!(
                        "Toggled d_{}[({},{},{})] row={} col={} -> {}",
                        r, n, s, f, row, col,
                        if new_val { 1 } else { 0 }
                    );
                }
                "remove" => {
                    if pages[idx].known_diffs.remove(&dv).is_some() {
                        eprintln!(
                            "Removed d_{}[({},{},{})] row={} col={} from known diffs",
                            r, n, s, f, row, col
                        );
                    } else {
                        eprintln!("Not found in known diffs");
                        return false;
                    }
                }
                _ => unreachable!(),
            }

            // Record undo entry
            undo_stack.push(UndoEntry { r, var: dv, prev });

            if !*auto_prop {
                eprintln!("  Recorded (propagation deferred — run `interpage` to propagate).");
                return false;
            }

            // Cascade re-solve from this page through all later pages,
            // recording every differential deduced from this mutation.
            let outcome = cascade_resolve(pages, idx, idx, false);
            let deduced = outcome.deduced;
            let cmd_key = format!("{} {} {} {} {} {} {}", cmd, r, n, s, f, row, col);

            // Check for inconsistency on the target page. The cascade already
            // recorded whether the missing result means UNSAT (vs. no vars);
            // pin the reason to this mutation so later `sweep`/`try` errors
            // say what to undo. A single mutation is its own bisect: the
            // pre-mutation system solved, so this mutation is the culprit.
            if pages[idx].result.is_none()
                && pages[idx]
                    .unsat_reason
                    .as_deref()
                    .is_some_and(|m| m.contains("INCONSISTENT"))
            {
                let applied = pages[idx].known_diffs.get(&dv).copied();
                let desc = mutation_desc(r, &dv, applied);
                pages[idx].unsat_reason = Some(format!(
                    "the E_{} re-solve was INCONSISTENT after `{}`; {} is individually inconsistent",
                    r, cmd_key, desc,
                ));
                eprintln!("  WARNING: E_{} is now INCONSISTENT.", r);
                eprintln!(
                    "  {} is individually inconsistent — the engine claims this value is impossible \
                     (given the previously recorded diffs).",
                    desc,
                );
                eprintln!(
                    "  Recover with: undo {} {} {} {} {} {}   (or `remove {} {} {} {} {} {}`)",
                    r, n, s, f, row, col, r, n, s, f, row, col,
                );
                eprintln!(
                    "  If you believe this differential is mathematically correct, this is an engine bug — report it."
                );
            }

            // Regenerate charts whose homology basis changed (the CLASSDIMS
            // fade shortcut cannot show which class died — see BUG notes).
            regen_affected_charts(
                pages, &outcome.affected_degrees, csv_paths, charts_dir, seqsee_dir, theme,
            );

            let entries = deduced_to_json(pages, &deduced, &[(r, dv)]);
            if !entries.is_empty() {
                eprintln!(
                    "  {} differentials deduced — click the log entry in a chart to browse them",
                    entries.len(),
                );
            }
            prop_log.insert(cmd_key, serde_json::Value::Array(entries));
            inject_propagation(charts_dir, pages, prop_log);

            // Fast-update diff overlays for all affected pages
            let mut total_updated = 0;
            for i in idx..pages.len() {
                total_updated += fast_update_diffs(charts_dir, pages, i);
            }
            if total_updated > 0 {
                eprintln!("  Updated {} charts. Refresh browser to see changes.", total_updated);
            }

            // The cascade may have changed the terminal page's basis — hidden
            // values re-validate + re-deduce against the new state.
            hidden_after_mutation(pages, hidden, charts_dir);
        }

        "undo" => {
            // `undo` — revert the last mutation.
            // `undo <r> <n> <s> <f> <row> <col>` — targeted: revert the most
            // recent mutation of that differential, wherever it sits in the
            // stack. Both paths share the revert body below.
            let entry = if parts.len() > 1 {
                let nums: Option<Vec<i64>> = (parts.len() >= 7)
                    .then(|| parts[1..7].iter().map(|p| p.parse().ok()).collect())
                    .flatten();
                let Some(nums) = nums else {
                    eprintln!("Usage: undo — revert last mutation;");
                    eprintln!("       undo <r> <n> <s> <f> <row> <col> — revert most recent mutation of that differential");
                    return false;
                };
                let (r, n, s, f) = (nums[0] as i32, nums[1] as i32, nums[2] as i32, nums[3] as i32);
                let (row, col) = (nums[4] as u16, nums[5] as u16);
                let dv = DiffVar::new(n.min(s + 2), s, f, row, col);
                // Most recent matching entry = highest stack index.
                let Some(pos) = undo_stack
                    .iter()
                    .rposition(|e| e.r == r && e.var == dv)
                else {
                    eprintln!(
                        "No recorded mutation of d_{}({},{},{})[{},{}] to undo.",
                        r, dv.n, dv.s, dv.f, dv.row, dv.col,
                    );
                    eprintln!(
                        "  (If the diff was never recorded as a mutation, use `remove {} {} {} {} {} {}`.)",
                        r, n, s, f, row, col,
                    );
                    return false;
                };
                Some(undo_stack.remove(pos))
            } else {
                undo_stack.pop()
            };

            if let Some(entry) = entry {
                let r = entry.r;
                let dv = entry.var;
                let Some(idx) = page_index(pages, r) else {
                    eprintln!("Page E_{} no longer loaded", r);
                    return false;
                };

                match entry.prev {
                    Some(v) => {
                        pages[idx].known_diffs.insert(dv, v);
                        eprintln!(
                            "Reverted d_{}[({},{},{})] row={} col={} back to {}",
                            r, dv.n, dv.s, dv.f, dv.row, dv.col,
                            if v { 1 } else { 0 },
                        );
                    }
                    None => {
                        pages[idx].known_diffs.remove(&dv);
                        eprintln!(
                            "Reverted d_{}[({},{},{})] row={} col={} (removed from known diffs)",
                            r, dv.n, dv.s, dv.f, dv.row, dv.col,
                        );
                    }
                }

                // Cascade re-solve from this page; the undone command's
                // propagation record is stale, so drop it.
                let outcome = cascade_resolve(pages, idx, idx, false);
                regen_affected_charts(
                    pages, &outcome.affected_degrees, csv_paths, charts_dir, seqsee_dir, theme,
                );
                prop_log.retain(|key, _| {
                    let nums: Vec<i64> = key
                        .split_whitespace()
                        .skip(1)
                        .filter_map(|p| p.parse().ok())
                        .collect();
                    !(nums.len() == 6
                        && nums[0] as i32 == r
                        && (nums[1] as i32).min(nums[2] as i32 + 2) == dv.n
                        && nums[2] as i32 == dv.s
                        && nums[3] as i32 == dv.f
                        && nums[4] as u16 == dv.row
                        && nums[5] as u16 == dv.col)
                });
                inject_propagation(charts_dir, pages, prop_log);

                // Fast-update overlays
                let mut total_updated = 0;
                for i in idx..pages.len() {
                    total_updated += fast_update_diffs(charts_dir, pages, i);
                }
                if total_updated > 0 {
                    eprintln!("  Updated {} charts. Refresh browser to see changes.", total_updated);
                }

                // Undoing a diff can restore quarantined hidden assertions.
                hidden_after_mutation(pages, hidden, charts_dir);
            } else {
                eprintln!("Nothing to undo.");
            }
        }

        "hidden" => {
            hidden_cmd(&parts, pages, hidden, charts_dir, prop_log);
        }

        "list" => {
            let mut any = false;
            for ps in pages.iter() {
                if ps.known_diffs.is_empty() {
                    continue;
                }
                any = true;
                let r = ps.page.r;
                let mut entries: Vec<_> = ps.known_diffs.iter().collect();
                entries.sort_by_key(|(dv, _)| (dv.n, dv.s, dv.f, dv.row, dv.col));
                eprintln!("d_{} ({} entries):", r, entries.len());
                for (dv, val) in &entries {
                    eprintln!(
                        "  ({},{},{}) row={} col={} = {}",
                        dv.n, dv.s, dv.f, dv.row, dv.col,
                        if **val { 1 } else { 0 },
                    );
                }
            }
            if !any {
                eprintln!("No known differentials.");
            }
        }

        "status" => {
            for ps in pages.iter() {
                let r = ps.page.r;
                let cutoff = ps.page.max_s.unwrap_or(0);
                let system =
                    constraints::build_constraint_system(&ps.page, cutoff, &ps.known_diffs);
                let num_vars = system.num_vars;

                if let Some(ref res) = ps.result {
                    let determined = num_vars - res.unknown.len();
                    eprintln!(
                        "E_{} | {}/{} vars determined | {} unknown tridegrees | {} known diffs | {} spheres",
                        r, determined, num_vars,
                        res.unknown_tridegrees().len(),
                        ps.known_diffs.len(),
                        ps.n_values.len(),
                    );
                } else if num_vars == 0 {
                    eprintln!(
                        "E_{} | no variables (all permanent) | {} known diffs | {} spheres",
                        r, ps.known_diffs.len(), ps.n_values.len(),
                    );
                } else {
                    eprintln!(
                        "E_{} | INCONSISTENT | {} known diffs | {} spheres",
                        r, ps.known_diffs.len(), ps.n_values.len(),
                    );
                }
            }
        }

        "why" => {
            // why <r> <n> <s> <f> — explain a differential's status and, if it
            // is exclusion-suppressed, which lower-page unknowns caused that.
            if parts.len() < 5 {
                eprintln!("Usage: why <r> <n> <s> <f>");
                eprintln!(
                    "Explains d_r at (n,s,f): determined / genuinely underdetermined / \
                     suppressed by an exclusion — tracing the exclusion to the unknown \
                     lower-page differential entries that created it."
                );
                return false;
            }
            let nums: Option<Vec<i64>> = parts[1..5].iter().map(|p| p.parse().ok()).collect();
            let Some(nums) = nums else {
                eprintln!("Invalid numeric argument");
                return false;
            };
            let (r, n, s, f) = (nums[0] as i32, nums[1] as i32, nums[2] as i32, nums[3] as i32);
            let Some(idx) = page_index(pages, r) else {
                eprintln!("No page E_{} loaded", r);
                return false;
            };
            why_differential(pages, idx, n, s, f);
        }

        "regen" => {
            // regen [r [n]]
            let target_r = if parts.len() > 1 {
                match parts[1].parse::<i32>() {
                    Ok(r) => Some(r),
                    Err(_) => {
                        eprintln!("Invalid page number: {}", parts[1]);
                        return false;
                    }
                }
            } else {
                None
            };

            let sphere_n = if parts.len() > 2 {
                match parts[2].parse::<i32>() {
                    Ok(n) => Some(n),
                    Err(_) => {
                        eprintln!("Invalid sphere number: {}", parts[2]);
                        return false;
                    }
                }
            } else {
                None
            };

            match target_r {
                Some(r) => {
                    let Some(idx) = page_index(pages, r) else {
                        eprintln!("No page E_{} loaded", r);
                        return false;
                    };
                    eprintln!("Re-solving E_{}...", r);
                    let result = solve_and_export(&pages[idx].page, &pages[idx].known_diffs, r, &csv_paths[idx]);
                    pages[idx].unsat_reason = result.is_none().then(|| {
                        format!(
                            "the E_{} re-solve during `regen` produced no result \
                             (INCONSISTENT, or no differential variables)", r,
                        )
                    });
                    pages[idx].result = result;

                    match sphere_n {
                        Some(n) => {
                            if pages[idx].n_values.contains(&n) {
                                let _ = regenerate_sphere(
                                    n, &csv_paths[idx], charts_dir, seqsee_dir, theme, r,
                                );
                                // Fast-update diff overlay
                                fast_update_diffs(charts_dir, pages, idx);
                                inject_map_info(charts_dir, &pages[idx].page, &pages[idx].n_values, r);
                                eprintln!("Regenerated S^{} E_{}. Reload browser.", n, r);
                            } else {
                                eprintln!("Sphere S^{} not in E_{} data", n, r);
                            }
                        }
                        None => {
                            let _ = regenerate_all(
                                &pages[idx].n_values,
                                &csv_paths[idx],
                                charts_dir,
                                seqsee_dir,
                                theme,
                                r,
                            );
                            let _ = generate_stem_charts(
                                &pages[idx].stem_values,
                                &csv_paths[idx],
                                charts_dir,
                                seqsee_dir,
                                theme,
                                r,
                            );
                            let _ = generate_fiber_charts(
                                &pages[idx].fiber_values,
                                &csv_paths[idx],
                                charts_dir,
                                seqsee_dir,
                                theme,
                                r,
                            );
                            inject_stem_scripts(charts_dir);
                            inject_fiber_scripts(charts_dir);
                            inject_view_menu(charts_dir);
                            inject_hidden_data(charts_dir, pages, hidden);
                            fast_update_diffs(charts_dir, pages, idx);
                            inject_map_info(charts_dir, &pages[idx].page, &pages[idx].n_values, r);
                            eprintln!("Regenerated all E_{} charts (incl. stems). Reload browser.", r);
                        }
                    }
                }
                None => {
                    // Regenerate all pages
                    for i in 0..pages.len() {
                        let r = pages[i].page.r;
                        eprintln!("Re-solving E_{}...", r);
                        let result =
                            solve_and_export(&pages[i].page, &pages[i].known_diffs, r, &csv_paths[i]);
                        pages[i].unsat_reason = result.is_none().then(|| {
                            format!(
                                "the E_{} re-solve during `regen` produced no result \
                                 (INCONSISTENT, or no differential variables)", r,
                            )
                        });
                        pages[i].result = result;
                        let _ = regenerate_all(
                            &pages[i].n_values,
                            &csv_paths[i],
                            charts_dir,
                            seqsee_dir,
                            theme,
                            r,
                        );
                        let _ = generate_stem_charts(
                            &pages[i].stem_values,
                            &csv_paths[i],
                            charts_dir,
                            seqsee_dir,
                            theme,
                            r,
                        );
                        let _ = generate_fiber_charts(
                            &pages[i].fiber_values,
                            &csv_paths[i],
                            charts_dir,
                            seqsee_dir,
                            theme,
                            r,
                        );
                        inject_map_info(charts_dir, &pages[i].page, &pages[i].n_values, r);
                    }
                    inject_stem_scripts(charts_dir);
                    inject_fiber_scripts(charts_dir);
                    inject_view_menu(charts_dir);
                    inject_hidden_data(charts_dir, pages, hidden);
                    for i in 0..pages.len() {
                        fast_update_diffs(charts_dir, pages, i);
                    }
                    eprintln!("Regenerated all charts (incl. stems). Reload browser.");
                }
            }
            print_chart_timing("regen command");
        }

        "save" => {
            if parts.len() < 2 {
                eprintln!("Usage: save <path>");
                return false;
            }
            let path = parts[1];
            match save_all_known_diffs(pages, path) {
                Ok(total) => eprintln!("Saved {} diffs to {}", total, path),
                Err(e) => eprintln!("Save failed: {}", e),
            }
        }

        "outside" => {
            // outside [retry]  — re-attempt loading outside-diff rows that
            //   were pruned at startup ("degree excluded (partial-quotient
            //   basis)" etc.): uncertainties resolved since then (mutations,
            //   `interpage try`) may have made their degrees trustworthy.
            //   Newly admissible rows are applied like add/zero (undo entry
            //   each, one cascade per pass) and the scan repeats until a pass
            //   applies nothing — each cascade can un-exclude more degrees
            //   downstream, admitting further rows.
            // outside status — dry run: report what would load, change nothing.
            let sub = parts.get(1).copied().unwrap_or("retry");
            if sub != "retry" && sub != "status" {
                eprintln!("Usage: outside [retry|status]");
                return false;
            }
            let dry = sub == "status";
            let dir = match std::env::var("EHP_OUTSIDE_DIFFS") {
                Ok(d) if !d.is_empty() => d,
                _ => {
                    eprintln!("EHP_OUTSIDE_DIFFS is not set — no outside rows to retry.");
                    return false;
                }
            };

            let mut pass = 0usize;
            let mut total_applied = 0usize;
            loop {
                pass += 1;
                let mut admissible: Vec<(usize, DiffVar, bool)> = Vec::new();
                let mut conflicts: Vec<(i32, DiffVar, bool, bool)> = Vec::new();
                let mut still_pruned = 0usize;
                let mut already = 0usize;
                for idx in 0..pages.len() {
                    let r = pages[idx].page.r;
                    let mut rows = match ehp_server::load_outside_diffs(r, &dir) {
                        Ok(rows) => rows,
                        Err(e) => {
                            eprintln!("  E_{}: outside load failed: {}", r, e);
                            continue;
                        }
                    };
                    // Prune against the CURRENT page state — this is the
                    // whole point: exclusions may have cleared since startup.
                    still_pruned +=
                        ehp_core::constraints::prune_outside_diffs(&pages[idx].page, &mut rows)
                            .len();
                    for (dv, val) in rows {
                        match pages[idx].known_diffs.get(&dv) {
                            Some(_) => already += 1, // enforced (or user-overridden): keep as is
                            None => {
                                // A solver-determined opposite value would make the
                                // cascade UNSAT — surface it instead of applying.
                                let determined_opposite =
                                    pages[idx].result.as_ref().and_then(|res| {
                                        res.var_index.get(&dv).and_then(|&vi| {
                                            (!res.unknown.contains(&vi))
                                                .then(|| res.offset.entry(vi) != 0)
                                        })
                                    });
                                match determined_opposite {
                                    Some(v) if v != val => conflicts.push((r, dv, val, v)),
                                    _ => admissible.push((idx, dv, val)),
                                }
                            }
                        }
                    }
                }

                for (r, dv, want, have) in &conflicts {
                    eprintln!(
                        "  CONFLICT: outside d_{}({},{},{})[{},{}] = {} but the solver determined {} — not applied",
                        r, dv.n, dv.s, dv.f, dv.row, dv.col, *want as u8, *have as u8,
                    );
                }

                if dry {
                    eprintln!(
                        "outside status: {} rows would load now, {} still pruned, {} already enforced/overridden, {} conflict(s).",
                        admissible.len(),
                        still_pruned,
                        already,
                        conflicts.len(),
                    );
                    if !admissible.is_empty() {
                        eprintln!("  Run `outside retry` to apply (each row gets an undo entry).");
                    }
                    return false;
                }
                if admissible.is_empty() {
                    if pass == 1 {
                        eprintln!(
                            "No newly admissible outside rows ({} still pruned, {} already enforced).",
                            still_pruned, already,
                        );
                    } else {
                        eprintln!(
                            "[pass {}] nothing further — {} row(s) applied in total ({} still pruned).",
                            pass, total_applied, still_pruned,
                        );
                    }
                    break;
                }

                eprintln!(
                    "[pass {}] {} outside row(s) now admissible — applying + cascading...",
                    pass,
                    admissible.len(),
                );
                let first_idx = admissible.iter().map(|(i, _, _)| *i).min().unwrap();
                let last_idx = admissible.iter().map(|(i, _, _)| *i).max().unwrap();
                let mut sources: Vec<(i32, DiffVar)> = Vec::new();
                for &(idx, dv, val) in &admissible {
                    let r = pages[idx].page.r;
                    pages[idx].known_diffs.insert(dv, val);
                    undo_stack.push(UndoEntry { r, var: dv, prev: None });
                    sources.push((r, dv));
                    eprintln!(
                        "  d_{}({},{},{})[{},{}] = {}",
                        r, dv.n, dv.s, dv.f, dv.row, dv.col, val as u8,
                    );
                }
                total_applied += admissible.len();

                let outcome = cascade_resolve(pages, first_idx, last_idx, false);
                if let Some(ps) = pages.iter().find(|p| {
                    p.result.is_none()
                        && p.unsat_reason.as_deref().is_some_and(|m| m.contains("INCONSISTENT"))
                }) {
                    eprintln!(
                        "  WARNING: E_{} is now INCONSISTENT — an applied outside row contradicts \
                         the current state. `undo` reverts the most recent application(s).",
                        ps.page.r,
                    );
                }
                regen_affected_charts(
                    pages, &outcome.affected_degrees, csv_paths, charts_dir, seqsee_dir, theme,
                );
                let entries = deduced_to_json(pages, &outcome.deduced, &sources);
                if !entries.is_empty() {
                    eprintln!(
                        "  {} differential(s) deduced — click the log entry in a chart to browse them",
                        entries.len(),
                    );
                }
                prop_log.insert(
                    format!("outside retry (pass {})", pass),
                    serde_json::Value::Array(entries),
                );
                inject_propagation(charts_dir, pages, prop_log);
                let mut total_updated = 0;
                for i in first_idx..pages.len() {
                    total_updated += fast_update_diffs(charts_dir, pages, i);
                }
                if total_updated > 0 {
                    eprintln!("  Updated {} charts.", total_updated);
                }

                if pass >= 10 {
                    eprintln!("  Stopping after 10 passes (still converging?) — rerun `outside retry`.");
                    break;
                }
            }
            hidden_after_mutation(pages, hidden, charts_dir);
        }

        "snapshot" | "snap" => {
            // snapshot save <name> [force] | snapshot list | snapshot load <name>
            let sub = parts.get(1).copied().unwrap_or("");
            match sub {
                "save" => {
                    let Some(name) = parts.get(2) else {
                        eprintln!("Usage: snapshot save <name> [force]");
                        return false;
                    };
                    let force = parts.get(3).map(|s| *s == "force").unwrap_or(false);
                    // Mutation count = undo entries applied this session.
                    match save_snapshot(name, pages, undo_stack.len(), force) {
                        Ok(m) => {
                            eprintln!(
                                "Saved snapshot '{}' (clone, ~0 extra bytes): {} pages, {} charts, {} mutations.",
                                name, m.pages, m.charts, m.mutations
                            );
                            // Point last-session recall (`ehp` / `-- last`) at
                            // this snapshot: a plain resume reopens the saved
                            // state, mutations included, not the startup state.
                            std::env::set_var("EHP_SNAPSHOT", name);
                            record_last_session(&m.cache_hash);
                            eprintln!(
                                "  Reload later with:  EHP_SNAPSHOT={} cargo run -p ehp-server --release --example ehp_chart",
                                name
                            );
                            eprintln!("  or in-REPL:  snapshot load {}", name);
                        }
                        Err(e) => eprintln!("snapshot save failed: {}", e),
                    }
                }
                "list" | "ls" => list_snapshots(),
                "load" => {
                    let Some(name) = parts.get(2) else {
                        eprintln!("Usage: snapshot load <name>");
                        return false;
                    };
                    if !snapshot_dir(name).join("manifest.json").exists() {
                        eprintln!("No snapshot '{}' (see `snapshot list`).", name);
                        return false;
                    }
                    // Loading swaps the entire page set + charts, so re-exec
                    // this binary with EHP_SNAPSHOT set — a clean restart that
                    // reuses the startup restore path (no solve, no regen).
                    reexec_with_snapshot(name);
                    // reexec_with_snapshot only returns if exec failed:
                    eprintln!(
                        "Could not re-exec. Restart manually:  EHP_SNAPSHOT={} cargo run -p ehp-server --release --example ehp_chart",
                        name
                    );
                }
                _ => {
                    eprintln!("Usage: snapshot save <name> [force] | snapshot list | snapshot load <name>");
                }
            }
        }

        "mapview" => {
            // mapview <E|H|P> <source_n> [r]   |   mapview all [r]
            if parts.len() < 2 {
                eprintln!("Usage: mapview <E|H|P> <source_n> [r]  |  mapview all [r]");
                eprintln!("Opens a side-by-side view of the map from S^source_n (defaults to the first page).");
                return false;
            }
            let r_range = (
                pages.first().map(|p| p.page.r).unwrap_or(2),
                pages.last().map(|p| p.page.r).unwrap_or(2),
            );

            if parts[1].eq_ignore_ascii_case("all") {
                // Bulk-generate every map view for one page so keyboard
                // navigation (e/h/p in charts, WASD between views) lands on
                // existing files.
                let r = parts
                    .get(2)
                    .and_then(|p| p.parse::<i32>().ok())
                    .unwrap_or(r_range.0);
                let Some(idx) = page_index(pages, r) else {
                    eprintln!("No page E_{} loaded", r);
                    return false;
                };
                eprintln!("Generating map views for E_{}...", r);
                let t0 = Instant::now();
                let (ok, total) =
                    pregenerate_mapviews(&pages[idx], charts_dir, seqsee_dir, theme, r, r_range);
                eprintln!(
                    "Generated {}/{} map views in {:.1}s",
                    ok,
                    total,
                    t0.elapsed().as_secs_f64(),
                );
                // Link everything up now that the files exist: WASD targets in
                // the views, and view availability for the charts' e/h/p keys.
                refresh_mapview_nav(charts_dir, r_range);
                inject_map_info(charts_dir, &pages[idx].page, &pages[idx].n_values, r);
                eprintln!("Reload charts so e/h/p keys pick up the generated views.");
                return false;
            }

            if parts.len() < 3 {
                eprintln!("Usage: mapview <E|H|P> <source_n> [r]  |  mapview all [r]");
                return false;
            }
            let kind = match parts[1].to_ascii_uppercase().as_str() {
                "E" => MapKind::E,
                "H" => MapKind::H,
                "P" => MapKind::P,
                other => {
                    eprintln!("Unknown map: {} (use E, H, or P)", other);
                    return false;
                }
            };
            let Ok(n) = parts[2].parse::<i32>() else {
                eprintln!("Invalid sphere: {}", parts[2]);
                return false;
            };
            let r = parts
                .get(3)
                .and_then(|p| p.parse::<i32>().ok())
                .unwrap_or_else(|| pages[0].page.r);

            let Some(idx) = page_index(pages, r) else {
                eprintln!("No page E_{} loaded", r);
                return false;
            };
            if !map_domain_n(kind, n) {
                eprintln!(
                    "{} is not defined on S^{} ({}).",
                    kind.name(),
                    n,
                    match kind {
                        MapKind::P => "needs odd n >= 5",
                        _ => "needs n >= 2",
                    },
                );
                return false;
            }
            let target_n = map_target_n(kind, n);
            let ps = &pages[idx];
            if !ps.n_values.contains(&n) {
                eprintln!("Sphere S^{} not in E_{} data", n, r);
                return false;
            }
            if !ps.n_values.contains(&target_n) {
                eprintln!(
                    "Target sphere S^{} (of {} from S^{}) not in E_{} data",
                    target_n, kind.name(), n, r,
                );
                return false;
            }

            eprintln!(
                "Generating side-by-side {}: S^{} → S^{} (E_{})...",
                kind.name(), n, target_n, r,
            );
            match generate_map_sidebyside(
                &ps.page, kind, n, target_n, charts_dir, seqsee_dir, theme, r, r_range,
            ) {
                Ok(path) => {
                    eprintln!("Generated {}", path.display());
                    refresh_mapview_nav(charts_dir, r_range);
                    inject_map_info(charts_dir, &ps.page, &ps.n_values, r);
                    if let Err(e) = open::that(&path) {
                        eprintln!("Could not open browser: {} — open it manually.", e);
                    }
                }
                Err(e) => eprintln!("mapview failed: {}", e),
            }
        }

        "propagate" => {
            match parts.get(1).copied() {
                Some("off") => {
                    *auto_prop = false;
                    eprintln!("Propagation deferred: add/zero/toggle only record; run `interpage` to solve.");
                }
                Some("on") => {
                    *auto_prop = true;
                    eprintln!("Propagation immediate: every mutation cascades right away.");
                }
                _ => eprintln!(
                    "Usage: propagate on|off   (currently {})",
                    if *auto_prop { "on" } else { "off" },
                ),
            }
        }

        "interpage" => {
            // interpage try [min_stem [max_stem]] — automated trial-and-error
            // to a fixpoint: sweep both values of every unknown differential
            // on every page, apply each forced value (contradiction ⇒ the
            // opposite value holds), cascade the consequences, and repeat the
            // whole pass until nothing new is forced.
            if parts.get(1).copied() == Some("try") {
                let min_stem: i32 = parts.get(2).and_then(|p| p.parse().ok()).unwrap_or(0);
                let max_stem: i32 = parts.get(3).and_then(|p| p.parse().ok()).unwrap_or(i32::MAX);
                run_interpage_try(
                    pages, undo_stack, csv_paths, charts_dir, seqsee_dir, theme, prop_log,
                    min_stem, max_stem,
                );
                hidden_after_mutation(pages, hidden, charts_dir);
                return false;
            }

            // interpage [r_start] — propagate all recorded differentials
            // through every page (the on-demand form of the original
            // run_interpage flow; the same math runs, driven by the session's
            // known diffs instead of a trial file).
            let start_idx = match parts.get(1).and_then(|p| p.parse::<i32>().ok()) {
                Some(r) => match page_index(pages, r) {
                    Some(i) => i,
                    None => {
                        eprintln!("No page E_{} loaded", r);
                        return false;
                    }
                },
                None => 0,
            };
            eprintln!(
                "Interpage propagation from E_{} through E_{}...",
                pages[start_idx].page.r,
                pages.last().map(|p| p.page.r).unwrap_or(0),
            );
            let t0 = Instant::now();
            let outcome = cascade_resolve(pages, start_idx, start_idx, true);
            let deduced = outcome.deduced;
            eprintln!(
                "  {} differentials newly determined (or changed) in {:.1}s",
                deduced.len(),
                t0.elapsed().as_secs_f64(),
            );
            regen_affected_charts(
                pages, &outcome.affected_degrees, csv_paths, charts_dir, seqsee_dir, theme,
            );
            for d in deduced.iter().take(40) {
                eprintln!(
                    "  d_{}({},{},{})[{},{}] = {}",
                    d.r, d.var.n, d.var.s, d.var.f, d.var.row, d.var.col, d.value as u8,
                );
            }
            if deduced.len() > 40 {
                eprintln!(
                    "  ... and {} more (browse via the chart log panel)",
                    deduced.len() - 40,
                );
            }

            let entries = deduced_to_json(pages, &deduced, &[]);
            prop_log.insert("interpage".to_string(), serde_json::Value::Array(entries));
            inject_propagation(charts_dir, pages, prop_log);
            for i in start_idx..pages.len() {
                fast_update_diffs(charts_dir, pages, i);
                inject_map_info(charts_dir, &pages[i].page, &pages[i].n_values, pages[i].page.r);
            }
            for ps in pages[start_idx..].iter() {
                if ps.result.is_none() && !ps.known_diffs.is_empty() {
                    eprintln!(
                        "  WARNING: E_{} is INCONSISTENT — some recorded differential is contradictory (`undo` to revert).",
                        ps.page.r,
                    );
                }
            }
            hidden_after_mutation(pages, hidden, charts_dir);
            eprintln!("Charts updated — refresh browser.");
        }

        "try" => {
            // try <r> <n> <s> <f> <row> <col> <val>
            if parts.len() < 8 {
                eprintln!("Usage: try <r> <n> <s> <f> <row> <col> <0|1>");
                eprintln!("Assumes d_r(n,s,f)[row,col] = val and propagates through all pages.");
                return false;
            }
            let nums: Option<Vec<i64>> = parts[1..8].iter().map(|p| p.parse().ok()).collect();
            let Some(nums) = nums else {
                eprintln!("Invalid numeric argument");
                return false;
            };
            let (r, n, s, f) = (nums[0] as i32, nums[1] as i32, nums[2] as i32, nums[3] as i32);
            let (row, col, val) = (nums[4] as u16, nums[5] as u16, nums[6] != 0);

            let Some(idx) = page_index(pages, r) else {
                eprintln!("No page E_{} loaded", r);
                return false;
            };
            let Some(ip) = interpage_slice(pages, idx) else {
                eprintln!("{}", no_result_msg(&pages[idx]));
                return false;
            };
            if idx + ip.len() < pages.len() {
                eprintln!(
                    "  NOTE: propagation limited to E_{} — {}",
                    ip.last().map(|p| p.page.r).unwrap_or(r),
                    no_result_msg(&pages[idx + ip.len()]),
                );
            }
            let var = DiffVar::new(n, s, f, row, col);
            match interpage::try_diffs(&ip, &[(var, val)]) {
                Ok(learned) => {
                    eprintln!(
                        "Assuming d_{}({},{},{})[{},{}] = {} is CONSISTENT through E_{}.",
                        r, n, s, f, row, col, val as u8,
                        ip.last().map(|p| p.page.r).unwrap_or(r),
                    );
                    if learned.is_empty() {
                        eprintln!("No new differentials determined.");
                    } else {
                        eprintln!("Newly determined ({}):", learned.len());
                        for ld in &learned {
                            eprintln!(
                                "  d_{}({},{},{})[{},{}] = {}",
                                ld.r, ld.var.n, ld.var.s, ld.var.f,
                                ld.var.row, ld.var.col, ld.value as u8,
                            );
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "CONTRADICTION ({}): d_{}({},{},{})[{},{}] = {} is impossible — forced to {}.",
                        e, r, n, s, f, row, col, val as u8, (!val) as u8,
                    );
                    eprintln!(
                        "Apply with: {} {} {} {} {} {} {}",
                        if val { "zero" } else { "add" },
                        r, n, s, f, row, col,
                    );
                }
            }
        }

        "sweep" => {
            // sweep <r> [min_stem [max_stem]]
            if parts.len() < 2 {
                eprintln!("Usage: sweep <r> [min_stem [max_stem]]");
                eprintln!("Tries both values of every unknown d_r; contradictions force the opposite value.");
                return false;
            }
            let Ok(r) = parts[1].parse::<i32>() else {
                eprintln!("Invalid r: {}", parts[1]);
                return false;
            };
            let min_stem: i32 = parts.get(2).and_then(|p| p.parse().ok()).unwrap_or(0);
            let max_stem: i32 = parts.get(3).and_then(|p| p.parse().ok()).unwrap_or(i32::MAX);

            let Some(idx) = page_index(pages, r) else {
                eprintln!("No page E_{} loaded", r);
                return false;
            };
            let Some(ip) = interpage_slice(pages, idx) else {
                eprintln!("{}", no_result_msg(&pages[idx]));
                return false;
            };
            if idx + ip.len() < pages.len() {
                eprintln!(
                    "  NOTE: propagation limited to E_{} — {}",
                    ip.last().map(|p| p.page.r).unwrap_or(r),
                    no_result_msg(&pages[idx + ip.len()]),
                );
            }
            let n_unknown = ip[0].result.unknown.len();
            eprintln!(
                "Sweeping unknown d_{} vars (stems {}..{}): {} vars × 2 values, propagating through E_{}...",
                r, min_stem,
                if max_stem == i32::MAX { "∞".to_string() } else { max_stem.to_string() },
                n_unknown,
                ip.last().map(|p| p.page.r).unwrap_or(r),
            );
            let t0 = Instant::now();
            let sweep_outcome =
                interpage::trial_error_sweep_full(&ip, min_stem, max_stem, |done, total| {
                    if done % 20 == 0 || done == total {
                        eprintln!("  {}/{} trials done", done, total);
                    }
                });
            let findings = sweep_outcome.contradictions;
            eprintln!("Sweep finished in {:.1}s.", t0.elapsed().as_secs_f64());

            // Both-worlds consensus (report-only here; `interpage try`
            // applies them): same value forced whether the switch is 0 or 1.
            if !sweep_outcome.consensus.is_empty() {
                let mut seen: HashSet<(i32, DiffVar, bool)> = HashSet::new();
                eprintln!("Determined in BOTH worlds of some unknown (unconditional):");
                for c in &sweep_outcome.consensus {
                    if seen.insert((c.r, c.var, c.value)) {
                        eprintln!(
                            "  d_{}({},{},{})[{},{}] = {} (via d_{}({},{},{})[{},{}])",
                            c.r, c.var.n, c.var.s, c.var.f, c.var.row, c.var.col,
                            c.value as u8,
                            c.via_r, c.via.n, c.via.s, c.via.f, c.via.row, c.via.col,
                        );
                    }
                }
                eprintln!("  (run `interpage try` to apply these automatically)");
            }

            if findings.is_empty() {
                eprintln!("No contradictions found — no values forced.");
            } else {
                // A var contradicted for BOTH values means the base system is bad.
                let mut by_var: HashMap<DiffVar, Vec<&interpage::SweepFinding>> = HashMap::new();
                for fnd in &findings {
                    by_var.entry(fnd.var).or_default().push(fnd);
                }
                let mut lines: Vec<String> = Vec::new();
                let mut vars_sorted: Vec<&DiffVar> = by_var.keys().collect();
                vars_sorted.sort();
                for var in vars_sorted {
                    let fs = &by_var[var];
                    if fs.len() == 2 {
                        lines.push(format!(
                            "d_{}({},{},{})[{},{}]: BOTH values contradict — base system inconsistent!",
                            r, var.n, var.s, var.f, var.row, var.col,
                        ));
                    } else {
                        let fnd = fs[0];
                        lines.push(format!(
                            "d_{}({},{},{})[{},{}] = {} forced ({} when assuming {})",
                            r, var.n, var.s, var.f, var.row, var.col,
                            (!fnd.contradicted_value) as u8,
                            fnd.error,
                            fnd.contradicted_value as u8,
                        ));
                    }
                }
                eprintln!("Forced values ({}):", lines.len());
                for l in &lines {
                    eprintln!("  {}", l);
                }
                let log_path = format!("output/sweep_E{}.log", r);
                if let Err(e) = std::fs::write(&log_path, lines.join("\n") + "\n") {
                    eprintln!("Could not write {}: {}", log_path, e);
                } else {
                    eprintln!("Written to {}", log_path);
                }
            }
        }

        _ => {
            eprintln!("Unknown command: {}", cmd);
            eprintln!("Commands: add|zero|toggle|remove <r> <n> <s> <f> <row> <col>");
            eprintln!("          try <r> <n> <s> <f> <row> <col> <0|1>, sweep <r> [min_stem [max_stem]]");
            eprintln!("          interpage [r], interpage try [min_stem [max_stem]], propagate on|off");
            eprintln!("          mapview <E|H|P> <source_n> [r]  |  mapview all [r]");
            eprintln!("          why <r> <n> <s> <f> (explain a differential's status)");
            eprintln!("          undo [r n s f row col], list, status, regen [r [n]], save <path>, outside [retry|status], quit");
    eprintln!("          hidden <E|H|P> <n> <s> <f> <idx> <tn> <ts> <tf> <tidx>  (assert a hidden map value on the");
    eprintln!("          terminal page; idx accepts sums like 0+2; Toda P(a∘E²b)=P(a)∘b propagates it)");
    eprintln!("          hidden list | hidden remove <same args> | hidden undo  (EHP_HIDDEN=0 disables)");
            eprintln!("          snapshot save <name> [force] | snapshot list | snapshot load <name>");
        }
    }

    false
}

/// Human-readable description of one applied mutation (`None` = removed).
fn mutation_desc(r: i32, dv: &DiffVar, applied: Option<bool>) -> String {
    match applied {
        Some(v) => format!(
            "d_{}({},{},{})[{},{}]={}",
            r, dv.n, dv.s, dv.f, dv.row, dv.col, v as u8,
        ),
        None => format!(
            "remove d_{}({},{},{})[{},{}]",
            r, dv.n, dv.s, dv.f, dv.row, dv.col,
        ),
    }
}

/// A mutation batch left `page`'s system INCONSISTENT — bisect to name the
/// culprit(s) instead of making the user do it by hand.
///
/// Strategy: re-solve once per mutation with known_diffs = the pre-batch
/// snapshot + that single mutation ("individually inconsistent" — the engine
/// claims the value is impossible on its own). If no single mutation is bad,
/// fall back to a greedy prefix scan (apply cumulatively in batch order) and
/// name the first mutation that tips the system.
///
/// Prints a full report (with `undo` recovery hints and a soundness note) and
/// returns a short culprit summary for `unsat_reason`, or `None` if the
/// inconsistency could not be reproduced page-locally.
fn bisect_inconsistent_mutations(
    page: &SATPage,
    r: i32,
    base_known: &HashMap<DiffVar, bool>,
    muts: &[(DiffVar, Option<bool>)],
) -> Option<String> {
    let cutoff = page.max_s.unwrap_or(0);
    // true = the system with these known diffs is INCONSISTENT.
    let inconsistent_with = |known: &HashMap<DiffVar, bool>| -> bool {
        let system = constraints::build_constraint_system(page, cutoff, known);
        system.num_vars > 0 && solver::solve(&system).is_none()
    };
    fn apply(known: &mut HashMap<DiffVar, bool>, dv: &DiffVar, val: &Option<bool>) {
        match val {
            Some(v) => {
                known.insert(*dv, *v);
            }
            None => {
                known.remove(dv);
            }
        }
    }
    let undo_hint =
        |dv: &DiffVar| format!("undo {} {} {} {} {} {}", r, dv.n, dv.s, dv.f, dv.row, dv.col);

    eprintln!(
        "  Bisecting {} mutation(s) on E_{} to find the culprit (one re-solve each)...",
        muts.len(),
        r,
    );
    let t0 = Instant::now();

    // Pass 1: each mutation alone on top of the pre-batch known diffs.
    let mut individually_bad: Vec<usize> = Vec::new();
    for (i, (dv, val)) in muts.iter().enumerate() {
        let mut known = base_known.clone();
        apply(&mut known, dv, val);
        if inconsistent_with(&known) {
            individually_bad.push(i);
        }
    }
    if !individually_bad.is_empty() {
        for &i in &individually_bad {
            let (dv, val) = &muts[i];
            eprintln!(
                "  {} is individually inconsistent — the engine claims this value is impossible.",
                mutation_desc(r, dv, *val),
            );
            eprintln!("    Recover with: {}", undo_hint(dv));
        }
        eprintln!(
            "  If you believe the differential is mathematically correct, this is an engine bug — report it."
        );
        eprintln!("  (Bisect took {:.1}s.)", t0.elapsed().as_secs_f64());
        let (dv, val) = &muts[individually_bad[0]];
        return Some(format!(
            "{} is individually inconsistent{}",
            mutation_desc(r, dv, *val),
            if individually_bad.len() > 1 {
                format!(" (+{} more, see bisect report)", individually_bad.len() - 1)
            } else {
                String::new()
            },
        ));
    }

    // Pass 2: no single culprit — greedy prefix scan for the first mutation
    // that tips the cumulative system.
    let mut known = base_known.clone();
    for (dv, val) in muts {
        apply(&mut known, dv, val);
        if inconsistent_with(&known) {
            eprintln!(
                "  No single mutation is inconsistent on its own; the batch is inconsistent only in combination."
            );
            eprintln!(
                "  First tipping mutation (cumulative scan in batch order): {}",
                mutation_desc(r, dv, *val),
            );
            eprintln!("    Recover with: {}", undo_hint(dv));
            eprintln!(
                "  If you believe these differentials are mathematically correct, this is an engine bug — report it."
            );
            eprintln!("  (Bisect took {:.1}s.)", t0.elapsed().as_secs_f64());
            return Some(format!(
                "inconsistent only in combination; first tipping mutation: {}",
                mutation_desc(r, dv, *val),
            ));
        }
    }

    eprintln!(
        "  Bisect could not reproduce the inconsistency page-locally in {:.1}s (it may involve \
         constraints from a cascaded page) — no single culprit identified.",
        t0.elapsed().as_secs_f64(),
    );
    None
}

/// Apply several mutations (`add`/`zero`/`toggle`/`remove` segments) with a
/// single cascade re-solve and injection pass at the end. Sum-target
/// differentials are several `add`s sharing a source column.
fn process_multi_mutation(
    segs: &[&str],
    pages: &mut [PageState],
    undo_stack: &mut Vec<UndoEntry>,
    csv_paths: &[PathBuf],
    charts_dir: &Path,
    seqsee_dir: &Path,
    theme: &str,
    prop_log: &mut HashMap<String, serde_json::Value>,
    auto_prop: bool,
    hidden: &mut HiddenState,
) {
    // Parse and validate everything before mutating anything.
    struct Mutation {
        cmd: String,
        idx: usize,
        r: i32,
        dv: DiffVar,
    }
    let mut muts: Vec<Mutation> = Vec::new();
    for seg in segs {
        let p: Vec<&str> = seg.split_whitespace().collect();
        if p.len() < 7 {
            eprintln!("Invalid segment (need `cmd r n s f row col`): {}", seg);
            return;
        }
        let nums: Option<Vec<i64>> = p[1..7].iter().map(|x| x.parse().ok()).collect();
        let Some(nums) = nums else {
            eprintln!("Invalid numbers in segment: {}", seg);
            return;
        };
        let (r, n, s, f) = (nums[0] as i32, nums[1] as i32, nums[2] as i32, nums[3] as i32);
        let (row, col) = (nums[4] as u16, nums[5] as u16);
        let Some(idx) = page_index(pages, r) else {
            eprintln!("No page E_{} loaded (segment: {})", r, seg);
            return;
        };
        let dv = DiffVar::new(n.min(s + 2), s, f, row, col);
        if p[0] != "remove" {
            match check_diff_addable(&pages[idx].page, r, &dv) {
                Err(reason) => {
                    eprintln!("Cannot apply `{}`: {}", seg, reason);
                    return;
                }
                Ok(Some(note)) => eprintln!("Note: {}", note),
                Ok(None) => {}
            }
        }
        muts.push(Mutation { cmd: p[0].to_string(), idx, r, dv });
    }

    // Pre-batch known-diff snapshots per mutated page — needed to bisect for
    // the culprit if the batch lands INCONSISTENT.
    let mut pre_known: HashMap<usize, HashMap<DiffVar, bool>> = HashMap::new();
    for m in &muts {
        pre_known
            .entry(m.idx)
            .or_insert_with(|| pages[m.idx].known_diffs.clone());
    }

    // Apply all mutations, recording undo entries (undo reverts one at a time).
    let mut min_idx = usize::MAX;
    let mut added: Vec<(i32, DiffVar)> = Vec::new();
    // Per mutated page: the mutations as applied (`None` = removed), in batch
    // order — the bisect input.
    let mut applied_by_page: HashMap<usize, Vec<(DiffVar, Option<bool>)>> = HashMap::new();
    for m in &muts {
        let prev = pages[m.idx].known_diffs.get(&m.dv).copied();
        match m.cmd.as_str() {
            "add" => {
                pages[m.idx].known_diffs.insert(m.dv, true);
            }
            "zero" => {
                pages[m.idx].known_diffs.insert(m.dv, false);
            }
            "toggle" => {
                let v = prev.map(|v| !v).unwrap_or(true);
                pages[m.idx].known_diffs.insert(m.dv, v);
            }
            "remove" => {
                pages[m.idx].known_diffs.remove(&m.dv);
            }
            _ => unreachable!(),
        }
        applied_by_page
            .entry(m.idx)
            .or_default()
            .push((m.dv, pages[m.idx].known_diffs.get(&m.dv).copied()));
        undo_stack.push(UndoEntry { r: m.r, var: m.dv, prev });
        added.push((m.r, m.dv));
        min_idx = min_idx.min(m.idx);
    }
    if !auto_prop {
        eprintln!(
            "Recorded {} mutations (propagation deferred — run `interpage` to propagate).",
            muts.len(),
        );
        return;
    }
    eprintln!("Applied {} mutations; re-solving once...", muts.len());

    // One cascade for the whole batch.
    let max_idx = muts.iter().map(|m| m.idx).max().unwrap_or(min_idx);
    let outcome = cascade_resolve(pages, min_idx, max_idx, false);
    let deduced = outcome.deduced;
    let cmd_key = segs.join("; ");

    // Any mutated page that came back INCONSISTENT: bisect its own mutations
    // so the error names the culprit instead of the whole batch. The batch
    // stays applied (no auto-undo) — the user decides what to revert.
    let mut bad_pages: Vec<usize> = applied_by_page
        .keys()
        .copied()
        .filter(|&i| {
            pages[i].result.is_none()
                && pages[i]
                    .unsat_reason
                    .as_deref()
                    .is_some_and(|m| m.contains("INCONSISTENT"))
        })
        .collect();
    bad_pages.sort_unstable();
    for i in bad_pages {
        let r_bad = pages[i].page.r;
        eprintln!(
            "  WARNING: E_{} is now INCONSISTENT — the batch stays applied; `undo` the culprit to recover.",
            r_bad,
        );
        let culprit = bisect_inconsistent_mutations(
            &pages[i].page,
            r_bad,
            &pre_known[&i],
            &applied_by_page[&i],
        );
        pages[i].unsat_reason = Some(match culprit {
            Some(c) => format!(
                "the E_{} re-solve was INCONSISTENT after `{}`; {}",
                r_bad, cmd_key, c,
            ),
            None => format!(
                "the E_{} re-solve was INCONSISTENT after `{}` (`undo` {} times to restore)",
                r_bad,
                cmd_key,
                muts.len(),
            ),
        });
    }

    regen_affected_charts(
        pages, &outcome.affected_degrees, csv_paths, charts_dir, seqsee_dir, theme,
    );

    let entries = deduced_to_json(pages, &deduced, &added);
    if !entries.is_empty() {
        eprintln!(
            "  {} differentials deduced — click the log entry in a chart to browse them",
            entries.len(),
        );
    }
    prop_log.insert(cmd_key, serde_json::Value::Array(entries));
    inject_propagation(charts_dir, pages, prop_log);

    let mut total_updated = 0;
    for i in min_idx..pages.len() {
        total_updated += fast_update_diffs(charts_dir, pages, i);
    }
    if total_updated > 0 {
        eprintln!("  Updated {} charts. Refresh browser to see changes.", total_updated);
    }

    // The cascade may have changed the terminal page's basis — hidden values
    // re-validate + re-deduce against the new state.
    hidden_after_mutation(pages, hidden, charts_dir);
}

/// Check whether a differential entry can be asserted on this page.
///
/// `Err(reason)` — impossible (no classes / index out of range), refuse.
/// `Ok(Some(note))` — allowed, but the degree is uncertainty-excluded: the
/// value is enforced as ground truth while automatic constraints there stay
/// disabled.
fn check_diff_addable(page: &SATPage, r: i32, dv: &DiffVar) -> Result<Option<String>, String> {
    let t = Tridegree::new(dv.n, dv.s, dv.f);
    let tgt = t.diff_target(r);
    let src_dim = page.dim_at(t);
    let tgt_dim = page.dim_at(tgt);
    if src_dim == 0 {
        return Err(format!("no classes at ({},{},{}) on E_{}", t.n, t.s, t.f, r));
    }
    if tgt_dim == 0 {
        return Err(format!(
            "no classes at the d_{} target ({},{},{}) on E_{}",
            r, tgt.n, tgt.s, tgt.f, r,
        ));
    }
    if dv.col as usize >= src_dim {
        return Err(format!("source index {} out of range (dim {})", dv.col, src_dim));
    }
    if dv.row as usize >= tgt_dim {
        return Err(format!("target index {} out of range (dim {})", dv.row, tgt_dim));
    }
    if page.is_excluded(t) || page.is_excluded(tgt) {
        return Ok(Some(format!(
            "({},{},{}) or its target is uncertainty-excluded on E_{} (an underlying d_{} there is \
             undetermined). The value is enforced as ground truth, but derived constraints at that \
             degree stay disabled — consider `try {} ...` to test assumptions, or resolve the \
             lower-page uncertainty first.",
            t.n, t.s, t.f, r, r - 1, r - 1,
        )));
    }
    Ok(None)
}

/// Build the interpage view starting at `start_idx`: consecutive pages that
/// all have a solve result (propagation stops at the first page without one).
fn interpage_slice(pages: &[PageState], start_idx: usize) -> Option<Vec<InterpagePage<'_>>> {
    let mut out = Vec::new();
    for ps in &pages[start_idx..] {
        let Some(ref result) = ps.result else { break };
        out.push(InterpagePage {
            page: &ps.page,
            result,
            excluded_leibniz: &ps.excluded_leibniz,
        });
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// `interpage try [min_stem [max_stem]]` — automated trial-and-error to a
/// fixpoint. For every page (lowest first), sweep both values of every
/// unknown differential; a contradiction forces the opposite value, which is
/// recorded like an `add`/`zero` (known diff + undo entry) and cascaded
/// immediately, so later sweeps see the consequences. Whole passes repeat
/// until one forces nothing new. Results go to the propagation log, the
/// charts, and `output/interpage_try.log`.
fn run_interpage_try(
    pages: &mut [PageState],
    undo_stack: &mut Vec<UndoEntry>,
    csv_paths: &[PathBuf],
    charts_dir: &Path,
    seqsee_dir: &Path,
    theme: &str,
    prop_log: &mut HashMap<String, serde_json::Value>,
    min_stem: i32,
    max_stem: i32,
) {
    let t0 = Instant::now();
    let last_r = pages.last().map(|p| p.page.r).unwrap_or(0);
    eprintln!(
        "Automated interpage trial-and-error (stems {}..{}), propagating through E_{}.",
        min_stem,
        if max_stem == i32::MAX { "∞".to_string() } else { max_stem.to_string() },
        last_r,
    );
    eprintln!("Each pass tries both values of every unknown differential — this can take a long time at large max_t.");

    let mut log_lines: Vec<String> = Vec::new();
    let mut all_deduced: Vec<DeducedDiff> = Vec::new();
    let mut all_affected: HashMap<i32, HashSet<Tridegree>> = HashMap::new();
    let mut total_forced = 0usize;
    let mut pass = 0usize;
    // EHP_TIMING bookkeeping: per-(page, var) contradiction outcomes of the
    // previous pass, to measure how many trials a pass repeats with an
    // IDENTICAL outcome — the empirical ceiling for influence-based trial
    // skipping (a trial's outcome can only change if the pass in between
    // changed state its propagation can reach).
    let mut prev_outcomes: HashMap<(i32, DiffVar), (bool, bool)> = HashMap::new();
    let mut sweep_secs = 0f64;
    let mut cascade_secs = 0f64;
    let mut total_trials = 0usize;

    loop {
        pass += 1;
        let mut forced_this_pass = 0usize;

        for idx in 0..pages.len() {
            let r = pages[idx].page.r;

            // Sweep against the current state (immutable borrow ends before
            // the forced values are applied below).
            let (findings, consensus, trialed) = {
                let Some(ip) = interpage_slice(pages, idx) else {
                    eprintln!("[pass {}] skipping E_{}: {}", pass, r, no_result_msg(&pages[idx]));
                    continue;
                };
                let trialed: Vec<DiffVar> = ip[0]
                    .result
                    .unknown
                    .iter()
                    .map(|&i| ip[0].result.vars[i])
                    .filter(|v| v.s >= min_stem && v.s < max_stem)
                    .collect();
                if trialed.is_empty() {
                    continue;
                }
                eprintln!(
                    "[pass {}] E_{}: sweeping {} unknown vars × 2 values...",
                    pass,
                    r,
                    trialed.len(),
                );
                let t1 = Instant::now();
                let outcome =
                    interpage::trial_error_sweep_full(&ip, min_stem, max_stem, |done, total| {
                        if done % 50 == 0 || done == total {
                            eprintln!("  E_{}: {}/{} trials", r, done, total);
                        }
                    });
                eprintln!("  E_{}: sweep done in {:.1}s", r, t1.elapsed().as_secs_f64());
                sweep_secs += t1.elapsed().as_secs_f64();
                total_trials += 2 * trialed.len();
                if timing_enabled() {
                    if let Some(report) = interpage::trial_stats::report_and_reset() {
                        eprintln!("  [timing] E_{} trial stages: {}", r, report);
                    }
                }
                (outcome.contradictions, outcome.consensus, trialed)
            };

            // Outcome-repeat measurement (EHP_TIMING): compare this sweep's
            // per-var contradiction pattern with the previous pass's.
            if timing_enabled() {
                let contradicted: HashSet<(DiffVar, bool)> =
                    findings.iter().map(|f| (f.var, f.contradicted_value)).collect();
                let mut repeated = 0usize;
                let mut compared = 0usize;
                let mut cur: Vec<((i32, DiffVar), (bool, bool))> = Vec::new();
                for &var in &trialed {
                    let outcome = (
                        contradicted.contains(&(var, false)),
                        contradicted.contains(&(var, true)),
                    );
                    if let Some(prev) = prev_outcomes.get(&(r, var)) {
                        compared += 1;
                        if *prev == outcome {
                            repeated += 1;
                        }
                    }
                    cur.push(((r, var), outcome));
                }
                prev_outcomes.extend(cur);
                if compared > 0 {
                    eprintln!(
                        "  [timing] E_{} pass {}: {}/{} re-trialed vars had IDENTICAL outcomes \
                         to the previous pass (skippable ceiling)",
                        r, pass, repeated, compared,
                    );
                }
            }
            if findings.is_empty() && consensus.is_empty() {
                continue;
            }

            // Group per variable: one contradiction forces the opposite
            // value; both values contradicting means the base system itself
            // is inconsistent — report, don't apply.
            let mut by_var: HashMap<DiffVar, Vec<&interpage::SweepFinding>> = HashMap::new();
            for f in &findings {
                by_var.entry(f.var).or_default().push(f);
            }
            let mut vars_sorted: Vec<DiffVar> = by_var.keys().copied().collect();
            vars_sorted.sort();

            let mut applied = 0usize;
            // Highest page index that received a value this batch — the
            // cascade must re-solve at least through it (consensus values
            // can land on pages beyond the swept one).
            let mut max_applied_idx = idx;
            for var in vars_sorted {
                let fs = &by_var[&var];
                if fs.len() == 2 {
                    let line = format!(
                        "[pass {}] d_{}({},{},{})[{},{}]: BOTH values contradict — base system inconsistent, not applied!",
                        pass, r, var.n, var.s, var.f, var.row, var.col,
                    );
                    eprintln!("  {}", line);
                    log_lines.push(line);
                    continue;
                }
                let forced = !fs[0].contradicted_value;
                let line = format!(
                    "[pass {}] d_{}({},{},{})[{},{}] = {} forced ({} when assuming {})",
                    pass, r, var.n, var.s, var.f, var.row, var.col,
                    forced as u8, fs[0].error, fs[0].contradicted_value as u8,
                );
                eprintln!("  {}", line);
                log_lines.push(line);

                let prev = pages[idx].known_diffs.get(&var).copied();
                pages[idx].known_diffs.insert(var, forced);
                undo_stack.push(UndoEntry { r, var, prev });
                applied += 1;
            }

            // Both-worlds consensus: a var forced to the SAME value in the
            // two consistent worlds of some switch holds unconditionally
            // (case analysis — see trial_error_sweep_full). Dedupe by
            // (r, var); conflicting conclusions from different switches mean
            // the base system is inconsistent — report, apply neither.
            if consensus_enabled() && !consensus.is_empty() {
                let mut chosen: HashMap<(i32, DiffVar), &interpage::ConsensusFinding> =
                    HashMap::new();
                let mut conflicted: HashSet<(i32, DiffVar)> = HashSet::new();
                for c in &consensus {
                    match chosen.get(&(c.r, c.var)) {
                        Some(prev) if prev.value != c.value => {
                            conflicted.insert((c.r, c.var));
                        }
                        Some(_) => {}
                        None => {
                            chosen.insert((c.r, c.var), c);
                        }
                    }
                }
                for &(cr, cv) in &conflicted {
                    let line = format!(
                        "[pass {}] d_{}({},{},{})[{},{}]: CONFLICTING both-worlds values via \
                         different switches — base system inconsistent, not applied!",
                        pass, cr, cv.n, cv.s, cv.f, cv.row, cv.col,
                    );
                    eprintln!("  {}", line);
                    log_lines.push(line);
                }
                let mut keys: Vec<(i32, DiffVar)> = chosen
                    .keys()
                    .filter(|k| !conflicted.contains(*k))
                    .copied()
                    .collect();
                keys.sort();
                for key in keys {
                    let c = chosen[&key];
                    let Some(cidx) = page_index(pages, c.r) else { continue };
                    let prev = pages[cidx].known_diffs.get(&c.var).copied();
                    if prev == Some(c.value) {
                        continue; // already recorded
                    }
                    if prev == Some(!c.value) {
                        let line = format!(
                            "[pass {}] d_{}({},{},{})[{},{}]: both-worlds value {} CONTRADICTS \
                             the recorded value — not applied, check the recorded diff!",
                            pass, c.r, c.var.n, c.var.s, c.var.f, c.var.row, c.var.col,
                            c.value as u8,
                        );
                        eprintln!("  {}", line);
                        log_lines.push(line);
                        continue;
                    }
                    let line = format!(
                        "[pass {}] d_{}({},{},{})[{},{}] = {} determined (same value in both \
                         worlds of d_{}({},{},{})[{},{}])",
                        pass, c.r, c.var.n, c.var.s, c.var.f, c.var.row, c.var.col,
                        c.value as u8, c.via_r, c.via.n, c.via.s, c.via.f, c.via.row, c.via.col,
                    );
                    eprintln!("  {}", line);
                    log_lines.push(line);
                    pages[cidx].known_diffs.insert(c.var, c.value);
                    undo_stack.push(UndoEntry { r: c.r, var: c.var, prev });
                    max_applied_idx = max_applied_idx.max(cidx);
                    applied += 1;
                }
            }
            if applied == 0 {
                continue;
            }
            forced_this_pass += applied;
            total_forced += applied;

            // One cascade for this page's batch of forced values; later
            // sweeps (and passes) run against the updated state.
            let t_c = Instant::now();
            let outcome = cascade_resolve(pages, idx, max_applied_idx, false);
            cascade_secs += t_c.elapsed().as_secs_f64();
            if !outcome.deduced.is_empty() {
                eprintln!("  cascade deduced {} further differentials", outcome.deduced.len());
            }
            all_deduced.extend(outcome.deduced);
            for (r_aff, degs) in outcome.affected_degrees {
                all_affected.entry(r_aff).or_default().extend(degs);
            }
        }

        if forced_this_pass == 0 {
            eprintln!(
                "[pass {}] nothing new forced — fixpoint reached ({} forced total, {:.1}s).",
                pass,
                total_forced,
                t0.elapsed().as_secs_f64(),
            );
            if timing_enabled() {
                eprintln!(
                    "  [timing] interpage try: {} passes, {} trials, sweeps {:.1}s, \
                     cascades {:.1}s, total {:.1}s",
                    pass,
                    total_trials,
                    sweep_secs,
                    cascade_secs,
                    t0.elapsed().as_secs_f64(),
                );
            }
            break;
        }
        eprintln!("[pass {}] {} values forced; running another pass...", pass, forced_this_pass);
    }

    // Persist the findings and refresh the charts.
    if !log_lines.is_empty() {
        let _ = std::fs::create_dir_all("output");
        let log_path = "output/interpage_try.log";
        if let Err(e) = std::fs::write(log_path, log_lines.join("\n") + "\n") {
            eprintln!("Could not write {}: {}", log_path, e);
        } else {
            eprintln!("Findings written to {}", log_path);
        }
    }
    if total_forced > 0 {
        regen_affected_charts(pages, &all_affected, csv_paths, charts_dir, seqsee_dir, theme);
        let entries = deduced_to_json(pages, &all_deduced, &[]);
        prop_log.insert("interpage try".to_string(), serde_json::Value::Array(entries));
        inject_propagation(charts_dir, pages, prop_log);
        for i in 0..pages.len() {
            fast_update_diffs(charts_dir, pages, i);
            inject_map_info(charts_dir, &pages[i].page, &pages[i].n_values, pages[i].page.r);
        }
        for ps in pages.iter() {
            if ps.result.is_none() && !ps.known_diffs.is_empty() {
                eprintln!(
                    "  WARNING: E_{} is INCONSISTENT — some recorded differential is contradictory (`undo` to revert).",
                    ps.page.r,
                );
            }
        }
        eprintln!("Charts updated — refresh browser.");
        eprintln!("(Each forced value has an undo entry; `undo` reverts them one at a time.)");
    } else {
        eprintln!("No values forced — nothing to apply.");
    }
}

// =============================================================================
// Solve and export
// =============================================================================

/// Re-solve constraints and write updated CSV. Returns the solve result.
fn solve_and_export(
    page: &SATPage,
    known_diffs: &HashMap<DiffVar, bool>,
    r: i32,
    csv_path: &Path,
) -> Option<SATResult> {
    let cutoff = page.max_s.unwrap_or(0);
    let system = constraints::build_constraint_system(page, cutoff, known_diffs);
    let result = solver::solve_with_d2(page, &system).map(|(res, _)| res);

    if let Some(ref res) = result {
        let determined = system.num_vars - res.unknown.len();
        eprintln!("  E_{}: {}/{} vars determined", r, determined, system.num_vars);
    } else if system.num_vars > 0 {
        eprintln!("  E_{}: INCONSISTENT", r);
    }

    // Write updated CSV
    match std::fs::File::create(csv_path) {
        Ok(mut file) => {
            if let Err(e) = seqsee::write_ehp_csv(page, result.as_ref(), known_diffs, &mut file) {
                eprintln!("  CSV write error: {}", e);
            }
        }
        Err(e) => eprintln!("  CSV create error: {}", e),
    }

    result
}

// =============================================================================
// Chart generation
// =============================================================================

/// Generate (or regenerate) the HTML chart for a single sphere.
/// `why` command: explain the status of d_r at (n, s, f) on `pages[idx]`.
fn why_differential(pages: &[PageState], idx: usize, n_raw: i32, s: i32, f: i32) {
    let ps = &pages[idx];
    let r = ps.page.r;
    let n = n_raw.min(s + 2);
    if n != n_raw {
        eprintln!("(n folded to the stable representative: {} -> {})", n_raw, n);
    }
    let src = Tridegree::new(n, s, f);
    let tgt = src.diff_target(r);
    let sdim = ps.page.dim_at(src);
    let tdim = ps.page.dim_at(tgt);
    eprintln!(
        "d_{}({},{},{}) -> ({},{},{}) on E_{}: source dim {}, target dim {}",
        r, src.n, src.s, src.f, tgt.n, tgt.s, tgt.f, r, sdim, tdim,
    );
    if sdim == 0 || tdim == 0 {
        eprintln!("  => trivially ZERO (a 0-dimensional side); no variables needed.");
        return;
    }
    let Some(res) = ps.result.as_ref() else {
        eprintln!(
            "  E_{} has no solve result ({}).",
            r,
            ps.unsat_reason.as_deref().unwrap_or("unsolved"),
        );
        return;
    };

    if res.var_index.contains_key(&DiffVar::new(n, s, f, 0, 0)) {
        // Variables exist: report each entry.
        let mut unknown_entries = Vec::new();
        for row in 0..tdim as u16 {
            for col in 0..sdim as u16 {
                let dv = DiffVar::new(n, s, f, row, col);
                match res.var_index.get(&dv) {
                    Some(&vi) if res.unknown.contains(&vi) => unknown_entries.push((row, col)),
                    Some(&vi) => {
                        let v = res.offset.entry(vi) != 0;
                        eprintln!("  [{},{}] determined = {}", row, col, v as u8);
                    }
                    None => eprintln!("  [{},{}] (no variable)", row, col),
                }
            }
        }
        if unknown_entries.is_empty() {
            eprintln!("  => fully DETERMINED.");
        } else {
            let list: Vec<String> = unknown_entries
                .iter()
                .map(|(a, b)| format!("[{},{}]", a, b))
                .collect();
            eprintln!(
                "  entries {} UNKNOWN — variables exist but no constraint pins them \
                 and no trial contradiction forces them: genuinely underdetermined.",
                list.join(" "),
            );
            eprintln!(
                "  (`try {} {} {} {} <row> <col> <0|1>` tests a value; if `interpage try` \
                 already reached its fixpoint, both values are consistent.)",
                r, n, s, f,
            );
        }
        return;
    }

    // No variables: exclusion (or bounds).
    eprintln!("  no d_{} variables exist at this degree:", r);
    let mut any_excluded = false;
    for (label, deg) in [("source", src), ("target", tgt)] {
        if ps.page.is_excluded(deg) {
            any_excluded = true;
            let kind = if ps.page.is_excluded_target_only(deg) {
                "target-only"
            } else {
                "hard"
            };
            eprintln!(
                "  - {} ({},{},{}) is EXCLUDED ({}):",
                label, deg.n, deg.s, deg.f, kind,
            );
            explain_exclusion(pages, idx, deg, 6, 4);
        }
    }
    if !any_excluded {
        eprintln!("  - neither degree is excluded: outside the polygon/cutoff bounds.");
        return;
    }
    eprintln!(
        "  => the differential is UNCERTAIN (suppressed, not determined zero). \
         Determining the unknown entries above un-excludes it on the next re-solve."
    );
}

/// Print, indented by `indent`, the previous-page unknowns and carried-forward
/// exclusions that put `deg` in `pages[idx]`'s exclude set. Mirrors
/// `make_next_exclude_set`: an input degree x (an unknown-d_{r-1} source, a
/// prior excluded degree, or its incoming-d pre-image (n, s+1, f-r)) excludes
/// x itself (if unstable) and stable_rep(x.diff_target(r-1)).
fn explain_exclusion(pages: &[PageState], idx: usize, deg: Tridegree, depth: usize, indent: usize) {
    let pad = " ".repeat(indent);
    if idx == 0 {
        eprintln!("{}(exclusion on the base page — unexpected)", pad);
        return;
    }
    if depth == 0 {
        eprintln!("{}(trace depth limit reached)", pad);
        return;
    }
    let prev = &pages[idx - 1];
    let pr = prev.page.r;
    let fold = |t: Tridegree| {
        if t.n > t.s + 2 {
            Tridegree::new(t.s + 2, t.s, t.f)
        } else {
            t
        }
    };
    let mut found = false;

    // 1. Unknown differentials on the previous page.
    if let Some(res) = prev.result.as_ref() {
        let mut by_src: HashMap<Tridegree, Vec<(u16, u16)>> = HashMap::new();
        for &vi in res.unknown.iter() {
            let v = &res.vars[vi];
            by_src
                .entry(Tridegree::new(v.n, v.s, v.f))
                .or_default()
                .push((v.row, v.col));
        }
        let mut srcs: Vec<_> = by_src.keys().copied().collect();
        srcs.sort();
        for src in srcs {
            let hits_as_source = src == deg;
            let hits_as_target = fold(src.diff_target(pr)) == deg;
            if !hits_as_source && !hits_as_target {
                continue;
            }
            found = true;
            let mut entries = by_src[&src].clone();
            entries.sort();
            let list: Vec<String> = entries
                .iter()
                .map(|(a, b)| format!("[{},{}]", a, b))
                .collect();
            eprintln!(
                "{}unknown d_{}({},{},{}) {} — entries {}",
                pad,
                pr,
                src.n,
                src.s,
                src.f,
                if hits_as_source {
                    "OUT of this degree"
                } else {
                    "hitting this degree"
                },
                list.join(" "),
            );
        }
    } else {
        eprintln!("{}E_{} below has no solve result — cannot trace", pad, pr);
    }

    // 2. Carried-forward exclusions from the previous page.
    let mut carried: Vec<Tridegree> = Vec::new();
    for &pdeg in prev.page.exclude_set.iter() {
        for x in [pdeg, Tridegree::new(pdeg.n, pdeg.s + 1, pdeg.f - pr)] {
            let matches = (x.n <= x.s + 2 && x == deg)
                || fold(x.diff_target(pr)) == deg;
            if matches && !carried.contains(&pdeg) {
                carried.push(pdeg);
            }
        }
    }
    carried.sort();
    for pdeg in carried {
        found = true;
        eprintln!(
            "{}carried forward from E_{}'s exclusion of ({},{},{}):",
            pad, pr, pdeg.n, pdeg.s, pdeg.f,
        );
        explain_exclusion(pages, idx - 1, pdeg, depth - 1, indent + 2);
    }

    if !found {
        eprintln!(
            "{}(no matching unknown or carried entry found on E_{} — possible stale \
             exclude set; try `regen` / a fresh solve)",
            pad, pr,
        );
    }
}

fn regenerate_sphere(
    n: i32,
    csv_path: &Path,
    charts_dir: &Path,
    seqsee_dir: &Path,
    theme: &str,
    r: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let csv_abs = std::fs::canonicalize(csv_path)?;
    let html_path = charts_dir.join(format!("S{}_E{}.html", n, r));
    run_seqsee_pipeline(n, &csv_abs, &html_path, seqsee_dir, theme, r)
}

/// Regenerate HTML charts for all spheres (in parallel).
fn regenerate_all(
    n_values: &BTreeSet<i32>,
    csv_path: &Path,
    charts_dir: &Path,
    seqsee_dir: &Path,
    theme: &str,
    r: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let spheres: Vec<i32> = n_values.iter().copied().collect();
    let errors: Vec<(i32, String)> = spheres
        .par_iter()
        .filter_map(|&n| {
            regenerate_sphere(n, csv_path, charts_dir, seqsee_dir, theme, r)
                .err()
                .map(|e| (n, e.to_string()))
        })
        .collect();
    for (n, e) in errors {
        eprintln!("  S^{} E_{}: error: {}", n, r, e);
    }
    Ok(())
}

/// Regenerate — from the freshly re-solved state — every sphere/stem chart
/// whose node set may have changed during a cascade
/// (`CascadeOutcome::affected_degrees`).
///
/// This is the real fix for the CLASSDIMS shortcut being wrong about *which*
/// class dies: the engine's quotient re-indexes survivors, so the only
/// faithful chart after a dimension change is one re-laid-out from the new
/// homology (fresh CSV → SeqSee). Charts the cascade didn't touch keep the
/// cheap in-place CLASSDIMS/DIFFDATA update.
///
/// Returns the number of charts regenerated. Every affected chart is
/// regenerated no matter how many (the generation is batched and parallel —
/// a near-full set costs about the same as startup chart generation), so the
/// fade-last-class shortcut never has to stand in for real layout; it
/// survives only as the display of last resort when chart generation itself
/// fails (e.g. no usable python).
fn regen_affected_charts(
    pages: &[PageState],
    affected: &HashMap<i32, HashSet<Tridegree>>,
    csv_paths: &[PathBuf],
    charts_dir: &Path,
    seqsee_dir: &Path,
    theme: &str,
) -> usize {
    // The charts no longer depict the cached startup state: drop the
    // freshness stamp so the next warm start regenerates.
    let _ = std::fs::remove_file(charts_dir.join(".state_stamp"));
    // Collect (page idx, spheres, stems, fibers) worth regenerating.
    let mut plan: Vec<(usize, BTreeSet<i32>, BTreeSet<i32>, BTreeSet<i32>)> = Vec::new();
    let mut total = 0usize;
    for (&r, degrees) in affected {
        let Some(idx) = page_index(pages, r) else { continue };
        let ps = &pages[idx];
        let spheres: BTreeSet<i32> = degrees
            .iter()
            .map(|t| t.n)
            .filter(|n| ps.n_values.contains(n))
            .collect();
        let stems: BTreeSet<i32> = degrees
            .iter()
            .filter(|t| t.n <= t.s + 2)
            .map(|t| t.s)
            .filter(|k| ps.stem_values.contains(k))
            .collect();
        // A degree on sphere m appears on fiber charts with base N = m,
        // N = m - 1 (via S^{N+1}) or N = (m-1)/2 (via S^{2N+1}).
        let fibers: BTreeSet<i32> = if fiberviews_enabled() {
            degrees
                .iter()
                .flat_map(|t| {
                    let m = t.n;
                    let mut bases = vec![m, m - 1];
                    if m >= 5 && m % 2 == 1 {
                        bases.push((m - 1) / 2);
                    }
                    bases
                })
                .filter(|n| ps.fiber_values.contains(n))
                .collect()
        } else {
            BTreeSet::new()
        };
        if spheres.is_empty() && stems.is_empty() && fibers.is_empty() {
            continue;
        }
        total += spheres.len() + stems.len() + fibers.len();
        plan.push((idx, spheres, stems, fibers));
    }
    if plan.is_empty() {
        return 0;
    }
    if total > 80 {
        eprintln!(
            "  {} charts affected — regenerating all of them (batched, parallel; \
             a large set takes a few minutes, comparable to startup)...",
            total,
        );
    }
    plan.sort_by_key(|(idx, _, _, _)| *idx);

    let t0 = Instant::now();
    // The plan entries are one-per-page and fully independent (own CSV, own
    // chart files) — run them in parallel. Serially, 4 pages × 3 chart modes
    // meant 12 back-to-back batch invocations, and with small affected sets
    // most chunks pay the per-process CSV parse; page-level parallelism
    // overlaps them.
    let results: Vec<(usize, bool, bool)> = plan
        .par_iter()
        .map(|(idx, spheres, stems, fibers)| {
            let ps = &pages[*idx];
            let r = ps.page.r;
            // Fresh CSV first: the SeqSee pipeline reads it.
            match std::fs::File::create(&csv_paths[*idx]) {
                Ok(mut file) => {
                    if let Err(e) = seqsee::write_ehp_csv(
                        &ps.page, ps.result.as_ref(), &ps.known_diffs, &mut file,
                    ) {
                        eprintln!("  E_{}: CSV write error: {} — skipping chart regen", r, e);
                        return (0, false, false);
                    }
                }
                Err(e) => {
                    eprintln!("  E_{}: CSV create error: {} — skipping chart regen", r, e);
                    return (0, false, false);
                }
            }
            let mut regenerated = 0usize;
            match generate_all_charts(spheres, &csv_paths[*idx], charts_dir, seqsee_dir, theme, r)
            {
                Ok(files) => regenerated += files.len(),
                Err(e) => eprintln!("  E_{}: chart regen error: {}", r, e),
            }
            if !stems.is_empty() {
                let ok =
                    generate_stem_charts(stems, &csv_paths[*idx], charts_dir, seqsee_dir, theme, r);
                regenerated += ok.len();
            }
            if !fibers.is_empty() {
                let ok = generate_fiber_charts(
                    fibers, &csv_paths[*idx], charts_dir, seqsee_dir, theme, r,
                );
                regenerated += ok.len();
            }
            // Fresh charts need their map-minimap data back.
            inject_map_info(charts_dir, &ps.page, &ps.n_values, r);
            (regenerated, !stems.is_empty(), !fibers.is_empty())
        })
        .collect();
    let regenerated: usize = results.iter().map(|(n, _, _)| n).sum();
    let any_stems = results.iter().any(|&(_, s, _)| s);
    let any_fibers = results.iter().any(|&(_, _, f)| f);
    if any_stems {
        inject_stem_scripts(charts_dir);
    }
    if any_fibers {
        inject_fiber_scripts(charts_dir);
    }
    inject_view_menu(charts_dir);
    if regenerated > 0 {
        let desc: Vec<String> = plan
            .iter()
            .map(|(idx, spheres, stems, fibers)| {
                let r = pages[*idx].page.r;
                let mut parts: Vec<String> = spheres.iter().map(|n| format!("S{}", n)).collect();
                parts.extend(stems.iter().map(|k| format!("stem{}", k)));
                parts.extend(fibers.iter().map(|n| format!("fiber{}", n)));
                format!("E_{}: {}", r, parts.join(" "))
            })
            .collect();
        eprintln!(
            "  Regenerated {} affected charts in {:.1}s ({})",
            regenerated,
            t0.elapsed().as_secs_f64(),
            desc.join("; "),
        );
    }
    print_chart_timing("mutation regen");
    regenerated
}

/// Explain a missing solve result (`sweep`/`try`/`interpage` failure paths).
fn no_result_msg(ps: &PageState) -> String {
    match &ps.unsat_reason {
        Some(reason) => format!("E_{} has no solve result: {}", ps.page.r, reason),
        None => format!("E_{} has no solve result", ps.page.r),
    }
}

/// Base command for `ehp_batch.py` (venv python or poetry fallback).
fn batch_cmd(seqsee_dir: &Path) -> std::process::Command {
    let mut cmd = match seqsee_python(seqsee_dir) {
        Some(py) => {
            let mut c = std::process::Command::new(py);
            c.arg("ehp_batch.py");
            c
        }
        None => {
            let mut c = std::process::Command::new("poetry");
            c.args(["run", "python", "ehp_batch.py"]);
            c
        }
    };
    cmd.current_dir(seqsee_dir)
        .stderr(std::process::Stdio::null());
    cmd
}

/// EHP_TIMING set (any non-empty value): print chart-generation timing
/// breakdowns (per-phase aggregates from the batch scripts' TIMESUM lines,
/// plus injection-pass stopwatches). Propagates to the python children,
/// which then also emit per-item TIME lines. Diagnostic-only: no effect on
/// any generated file.
fn timing_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("EHP_TIMING").map_or(false, |v| !v.is_empty() && v != "0"))
}

/// Per-mode chart-generation timing aggregate: (items, jsonmaker s,
/// render s, output bytes). Summed across all parallel batch chunks since
/// the last [`print_chart_timing`]; the per-chunk times are CPU-side
/// (chunks run concurrently, so their sum exceeds wall time).
type ChartTimingMap = HashMap<String, (usize, f64, f64, u64)>;
static CHART_TIMING: OnceLock<std::sync::Mutex<ChartTimingMap>> = OnceLock::new();

fn note_chart_timing(mode: &str, items: usize, json_s: f64, render_s: f64, bytes: u64) {
    let m = CHART_TIMING.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut m = m.lock().unwrap();
    let e = m.entry(mode.to_string()).or_insert((0, 0.0, 0.0, 0));
    e.0 += items;
    e.1 += json_s;
    e.2 += render_s;
    e.3 += bytes;
}

/// Print (and reset) the accumulated chart-generation timing, when
/// EHP_TIMING is set. `context` labels the phase (startup, regen, ...).
fn print_chart_timing(context: &str) {
    let Some(m) = CHART_TIMING.get() else { return };
    let mut m = m.lock().unwrap();
    if m.is_empty() {
        return;
    }
    if timing_enabled() {
        let mut modes: Vec<_> = m.iter().collect();
        modes.sort_by(|a, b| a.0.cmp(b.0));
        for (mode, (items, json_s, render_s, bytes)) in modes {
            eprintln!(
                "  [timing] {} {}: {} charts, jsonmaker {:.1}s + render {:.1}s (cpu-side sums), \
                 {:.1} MB written",
                context,
                mode,
                items,
                json_s,
                render_s,
                *bytes as f64 / 1e6,
            );
        }
    }
    m.clear();
}

/// Run `f`, printing its wall time under EHP_TIMING.
fn timed_phase<T>(label: &str, f: impl FnOnce() -> T) -> T {
    if !timing_enabled() {
        return f();
    }
    let t0 = Instant::now();
    let out = f();
    eprintln!("  [timing] {}: {:.2}s", label, t0.elapsed().as_secs_f64());
    out
}

/// Parse the batch script's per-item report lines, returning the OK ids.
/// `TIMESUM` lines (one per chunk) feed the EHP_TIMING aggregate; unknown
/// prefixes are ignored.
fn parse_batch_report(stdout: &[u8], what: &str, r: i32) -> Vec<i32> {
    let mut oks = Vec::new();
    for line in String::from_utf8_lossy(stdout).lines() {
        if let Some(rest) = line.strip_prefix("OK ") {
            if let Ok(n) = rest.trim().parse::<i32>() {
                oks.push(n);
            }
        } else if let Some(rest) = line.strip_prefix("FAIL ") {
            let mut it = rest.trim().splitn(2, ' ');
            let n = it.next().unwrap_or("?");
            let reason = it.next().unwrap_or("unknown error");
            eprintln!("  {}{} E_{}: FAILED ({})", what, n, r, reason);
        } else if let Some(rest) = line.strip_prefix("TIMESUM ") {
            // "TIMESUM <mode> items=<k> json=<s> render=<s> bytes=<n>"
            let mut mode = "";
            let (mut items, mut json_s, mut render_s, mut bytes) = (0usize, 0f64, 0f64, 0u64);
            for (i, tok) in rest.split_whitespace().enumerate() {
                if i == 0 {
                    mode = tok;
                } else if let Some(v) = tok.strip_prefix("items=") {
                    items = v.parse().unwrap_or(0);
                } else if let Some(v) = tok.strip_prefix("json=") {
                    json_s = v.parse().unwrap_or(0.0);
                } else if let Some(v) = tok.strip_prefix("render=") {
                    render_s = v.parse().unwrap_or(0.0);
                } else if let Some(v) = tok.strip_prefix("bytes=") {
                    bytes = v.parse().unwrap_or(0);
                }
            }
            if !mode.is_empty() {
                note_chart_timing(mode, items, json_s, render_s, bytes);
            }
        }
        // Per-item "TIME ..." lines from the batch script are deliberately NOT
        // forwarded (hundreds of lines per regen drowned the interesting
        // output); the per-mode TIMESUM aggregates above carry the signal.
    }
    oks
}

/// Run one `ehp_batch.py` process over a chunk of slice values (spheres or
/// stems), returning the values it generated successfully.
fn run_batch_chunk(
    seqsee_dir: &Path,
    csv_abs: &Path,
    charts_dir: &Path,
    theme: &str,
    r: i32,
    mode: &str, // "sphere" | "stem" | "fiber"
    values: &[i32],
) -> Vec<i32> {
    let mut cmd = batch_cmd(seqsee_dir);
    cmd.arg(mode)
        .arg(csv_abs)
        .arg(charts_dir)
        .arg(theme)
        .arg(r.to_string());
    for v in values {
        cmd.arg(v.to_string());
    }

    let out = match cmd.output() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("  batch chunk failed to start: {}", e);
            return Vec::new();
        }
    };
    let what = if mode == "sphere" {
        "S^"
    } else if mode == "fiber" {
        "fiber "
    } else {
        "stem "
    };
    parse_batch_report(&out.stdout, what, r)
}

/// Generate charts for a list of slice values via chunked batch processes.
fn generate_charts_batch(
    mode: &str,
    values: &[i32],
    csv_abs: &Path,
    charts_dir: &Path,
    seqsee_dir: &Path,
    theme: &str,
    r: i32,
) -> Vec<i32> {
    if values.is_empty() {
        return Vec::new();
    }
    let nchunks = rayon::current_num_threads().clamp(1, values.len());
    let chunk_size = values.len().div_ceil(nchunks);
    let mut oks: Vec<i32> = values
        .par_chunks(chunk_size)
        .flat_map(|chunk| run_batch_chunk(seqsee_dir, csv_abs, charts_dir, theme, r, mode, chunk))
        .collect();
    oks.sort_unstable();
    oks
}

/// Generate all charts initially. Uses `ehp_batch.py` (one python process per
/// CPU-sized chunk of spheres) when available, falling back to two processes
/// per sphere. Returns (n, path) pairs for successfully generated charts.
fn generate_all_charts(
    n_values: &BTreeSet<i32>,
    csv_path: &Path,
    charts_dir: &Path,
    seqsee_dir: &Path,
    theme: &str,
    r: i32,
) -> Result<Vec<(i32, PathBuf)>, Box<dyn std::error::Error>> {
    let csv_abs = std::fs::canonicalize(csv_path)?;
    // Resolve the venv python up front so parallel workers don't race the
    // one-time `poetry env info` lookup.
    let _ = seqsee_python(seqsee_dir);

    let spheres: Vec<i32> = n_values.iter().copied().collect();
    if spheres.is_empty() {
        return Ok(Vec::new());
    }

    let ok_spheres: Vec<i32> = if seqsee_dir.join("ehp_batch.py").exists() {
        // Batched: chunks run in parallel, spheres within a chunk share one
        // python process (startup + imports amortized).
        generate_charts_batch("sphere", &spheres, &csv_abs, charts_dir, seqsee_dir, theme, r)
    } else {
        // Fallback: per-sphere pipeline (two subprocesses each), in parallel.
        spheres
            .par_iter()
            .filter_map(|&n| {
                let html_path = charts_dir.join(format!("S{}_E{}.html", n, r));
                match run_seqsee_pipeline(n, &csv_abs, &html_path, seqsee_dir, theme, r) {
                    Ok(()) => Some(n),
                    Err(e) => {
                        eprintln!("  S^{} E_{}: FAILED ({})", n, r, e);
                        None
                    }
                }
            })
            .collect()
    };

    // The batch script writes raw HTML; inject the interactive script here.
    // (The per-sphere fallback already injected it.)
    let mut chart_files: Vec<(i32, PathBuf)> = ok_spheres
        .par_iter()
        .filter_map(|&n| {
            let html_path = charts_dir.join(format!("S{}_E{}.html", n, r));
            if !html_path.exists() {
                return None;
            }
            let html = std::fs::read_to_string(&html_path).ok()?;
            if !html.contains("/*DIFFDATA*/") {
                if let Err(e) = inject_click_script(&html_path, r, theme) {
                    eprintln!("  S^{} E_{}: inject failed ({})", n, r, e);
                    return None;
                }
            }
            Some((n, html_path))
        })
        .collect();
    chart_files.sort_by_key(|(n, _)| *n);

    Ok(chart_files)
}

/// Generate stem-view charts (one per stem, x = n; see STEM_VIEW_SPEC.md) and
/// inject the stem chart script (WASD nav + live class dims). Returns the
/// stems generated successfully.
fn generate_stem_charts(
    stem_values: &BTreeSet<i32>,
    csv_path: &Path,
    charts_dir: &Path,
    seqsee_dir: &Path,
    theme: &str,
    r: i32,
) -> Vec<i32> {
    let Ok(csv_abs) = std::fs::canonicalize(csv_path) else {
        return Vec::new();
    };
    if !seqsee_dir.join("ehp_batch.py").exists() {
        eprintln!("  stem charts need ehp_batch.py in the SeqSee dir — skipping");
        return Vec::new();
    }
    let stems: Vec<i32> = stem_values.iter().copied().collect();
    generate_charts_batch("stem", &stems, &csv_abs, charts_dir, seqsee_dir, theme, r)
}

/// Generate fiber-sequence charts (one per base sphere N, showing the triple
/// S^N → ΩS^{N+1} → ΩS^{2N+1}; see fiber_spec.md). Returns the base spheres
/// generated successfully. EHP_FIBERVIEWS=0 disables.
fn generate_fiber_charts(
    fiber_values: &BTreeSet<i32>,
    csv_path: &Path,
    charts_dir: &Path,
    seqsee_dir: &Path,
    theme: &str,
    r: i32,
) -> Vec<i32> {
    if !fiberviews_enabled() {
        return Vec::new();
    }
    let Ok(csv_abs) = std::fs::canonicalize(csv_path) else {
        return Vec::new();
    };
    if !seqsee_dir.join("ehp_batch.py").exists() {
        eprintln!("  fiber charts need ehp_batch.py in the SeqSee dir — skipping");
        return Vec::new();
    }
    let ns: Vec<i32> = fiber_values.iter().copied().collect();
    generate_charts_batch("fiber", &ns, &csv_abs, charts_dir, seqsee_dir, theme, r)
}

/// Parse `stem{k}_E{r}.html` into (r, k).
fn parse_stem_filename(name: &str) -> Option<(i32, i32)> {
    let stem = name.strip_prefix("stem")?.strip_suffix(".html")?;
    let (k, r) = stem.split_once("_E")?;
    Some((r.parse().ok()?, k.parse().ok()?))
}

/// Inject (or refresh) the stem-chart script in every stem chart on disk:
/// WASD navigation between stem charts (unified convention: w/s = stem ∓/+,
/// a/d = page ∓/+) and the live CLASSDIMS overlay. Scans the charts dir so
/// navigation reflects exactly the files that exist.
fn inject_stem_scripts(charts_dir: &Path) {
    let Ok(entries) = std::fs::read_dir(charts_dir) else { return };
    let stem_files: Vec<(i32, i32)> = entries
        .flatten()
        .filter_map(|e| parse_stem_filename(&e.file_name().to_string_lossy()))
        .collect();
    let have: HashSet<(i32, i32)> = stem_files.iter().copied().collect();
    for &(r, k) in &stem_files {
        let path = charts_dir.join(format!("stem{}_E{}.html", k, r));
        let Ok(html) = std::fs::read_to_string(&path) else { continue };

        let nb = |kk: i32, rr: i32| -> serde_json::Value {
            if have.contains(&(rr, kk)) {
                serde_json::json!(format!("stem{}_E{}.html", kk, rr))
            } else {
                serde_json::Value::Null
            }
        };
        // Unified WASD convention: w/s (up/down) = stem -/+, a/d (left/right)
        // = page -/+.
        let nav = serde_json::json!({
            "up": nb(k - 1, r),
            "down": nb(k + 1, r),
            "left": nb(k, r - 1),
            "right": nb(k, r + 1),
        });

        if html.contains("/*STEMNAV*/") {
            // Refresh the NAV data in place.
            let (sm, em) = ("/*STEMNAV*/", "/*ENDSTEMNAV*/");
            if let (Some(s), Some(e)) = (html.find(sm), html.find(em)) {
                let so = s + sm.len();
                let new_html = format!("{}{}{}", &html[..so], nav, &html[e..]);
                let _ = std::fs::write(&path, new_html);
            }
            continue;
        }

        let script = format!(
            r#"<script>
(function() {{
  const NAV = /*STEMNAV*/{nav}/*ENDSTEMNAV*/;
  const CLASS_DIMS = /*CLASSDIMS*/null/*ENDCLASSDIMS*/;
  const PAGE_R = {r};
  const STEM_K = {k};
  // Live dims shared with the click script (shift-tap zero needs the target
  // degree's current dimension).
  window.EHP_CLASS_DIMS = CLASS_DIMS;

  // The template restores `seqsee_viewport` from sessionStorage on every
  // load; saving here before WASD navigation keeps the pan/zoom fixed while
  // stepping stems/pages (it only jumped because the injected nav never
  // saved).
  function saveViewport() {{
    try {{
      if (window.panZoom) {{
        const p = window.panZoom.getPan();
        sessionStorage.setItem('seqsee_viewport',
          JSON.stringify({{ x: p.x, y: p.y, zoom: window.panZoom.getZoom() }}));
      }}
    }} catch (err) {{}}
  }}

  window.addEventListener('keydown', (e) => {{
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'Escape') {{ cancelPending('cancelled'); return; }}
    const dir = {{w: 'up', s: 'down', a: 'left', d: 'right'}}[e.key.toLowerCase()];
    if (!dir) return;
    window.__ehpShiftTap = false;  // shift+WASD is navigation, not a zero-tap
    e.preventDefault();
    e.stopImmediatePropagation();
    const url = NAV[dir];
    if (url) {{ saveViewport(); location.href = url; }}
  }}, true);

  function parseNodeId(id) {{
    const m = (id || '').match(/^S(-?\d+)_(-?\d+)_(-?\d+)(?:_(\d+))?$/);
    if (!m) return null;
    return {{ n: parseInt(m[1]), s: parseInt(m[2]), f: parseInt(m[3]),
             idx: m[4] !== undefined ? parseInt(m[4]) : 0 }};
  }}

  function applyClassDims() {{
    if (!CLASS_DIMS) return;
    const dead = new Set();
    document.querySelectorAll('#nodes-group circle, #nodes-group rect').forEach(el => {{
      const p = parseNodeId(el.id);
      if (!p) return;
      const key = p.n + '_' + p.s + '_' + p.f;
      const dim = (key in CLASS_DIMS) ? CLASS_DIMS[key] : 0;
      const isDead = p.idx >= dim;
      el.style.opacity = isDead ? '0.15' : '';
      if (isDead) dead.add(el.id);
    }});
    document.querySelectorAll('#edges-group [data-source]').forEach(el => {{
      const s = el.getAttribute('data-source');
      const t = el.getAttribute('data-target');
      el.style.opacity = (dead.has(s) || (t && dead.has(t))) ? '0.1' : '';
    }});
  }}

  // ==========================================================================
  // Add-differential flow from "?" (uncertain) classes.
  //
  // Click an uncertain-SOURCE class (data-uncertain="src"|"both") on this
  // stem chart: the pending source is stored in sessionStorage and the chart
  // navigates to stem k-1 on the same page — the target stem of a d_r. That
  // chart highlights the target bidegree (same n, f + r); click or
  // shift-click classes there to build the `add` batch, which is copied to
  // the clipboard for pasting into the REPL (each segment
  // `add r n s f row col`, row = target index, col = source index).
  // Escape cancels anywhere.
  // ==========================================================================
  const PENDING_KEY = 'ehp_stem_pending';
  let pendingTargets = [];

  function readPending() {{
    try {{ return JSON.parse(sessionStorage.getItem(PENDING_KEY) || 'null'); }}
    catch (err) {{ return null; }}
  }}

  function cancelPending(msg) {{
    if (sessionStorage.getItem(PENDING_KEY)) {{
      sessionStorage.removeItem(PENDING_KEY);
      document.querySelectorAll('.ehp-target-hl, .ehp-target-sel').forEach(el => {{
        el.classList.remove('ehp-target-hl');
        el.classList.remove('ehp-target-sel');
      }});
      pendingTargets = [];
      if (msg) setStemStatus('Add-differential ' + msg);
    }}
  }}

  function setStemStatus(text) {{
    let bar = document.getElementById('ehp-stem-status');
    if (!bar) {{
      bar = document.createElement('div');
      bar.id = 'ehp-stem-status';
      bar.style.cssText = 'position:fixed;top:8px;left:50%;transform:translateX(-50%);' +
        'background:rgba(0,0,0,0.78);color:#fff;padding:6px 14px;border-radius:6px;' +
        'font:13px sans-serif;z-index:10000;pointer-events:none;max-width:80%;';
      document.body.appendChild(bar);
    }}
    bar.textContent = text;
    bar.style.display = text ? 'block' : 'none';
  }}

  function copyCmd(cmd, desc) {{
    navigator.clipboard.writeText(cmd).then(() => {{
      setStemStatus('Copied: ' + cmd + '  (' + desc + ') — paste into REPL');
    }}).catch(() => {{
      window.prompt('Paste into REPL:', cmd);
      setStemStatus(cmd);
    }});
  }}

  function initPendingTarget() {{
    const pending = readPending();
    if (!pending || pending.r !== PAGE_R || pending.s - 1 !== STEM_K) return;
    const tn = pending.n, ts = pending.s - 1, tf = pending.f + PAGE_R;
    let count = 0;
    document.querySelectorAll('#nodes-group circle, #nodes-group rect').forEach(el => {{
      const p = parseNodeId(el.id);
      if (p && p.n === tn && p.s === ts && p.f === tf) {{
        el.classList.add('ehp-target-hl');
        count++;
      }}
    }});
    if (count === 0) {{
      setStemStatus('d_' + PAGE_R + '(' + pending.name + '): no classes at target (' +
                    tn + ',' + ts + ',' + tf + ') on this chart — Escape cancels');
      return;
    }}
    setStemStatus('d_' + PAGE_R + '(' + pending.name + '): click a highlighted target' +
                  ' — shift-click to build a sum, Escape cancels');
  }}

  function onStemNodeClick(e) {{
    const el = e.currentTarget;
    const p = parseNodeId(el.id);
    if (!p) return;
    const pending = readPending();

    // Target-selection mode: this chart is the pending source's target stem
    // and the clicked class sits in the target bidegree.
    if (pending && pending.r === PAGE_R && pending.s - 1 === STEM_K &&
        p.n === pending.n && p.s === pending.s - 1 && p.f === pending.f + PAGE_R) {{
      if (!pendingTargets.includes(p.idx)) {{
        pendingTargets.push(p.idx);
        el.classList.add('ehp-target-sel');
      }}
      if (e.shiftKey) {{
        setStemStatus('d_' + PAGE_R + '(' + pending.name + ') = ' +
                      pendingTargets.map(i => '[' + i + ']').join(' + ') +
                      ' + …  — shift-click more targets, plain click the last one');
        return;
      }}
      const segs = pendingTargets.map(row =>
        'add ' + PAGE_R + ' ' + pending.n + ' ' + pending.s + ' ' + pending.f +
        ' ' + row + ' ' + pending.idx);
      const cmd = segs.join('; ');
      cancelPending(null);
      copyCmd(cmd, 'd_' + PAGE_R + '(' + pending.name + ')');
      return;
    }}

    // Otherwise: start a new flow from an uncertain-source class.
    const unc = el.getAttribute('data-uncertain');
    if (unc === 'src' || unc === 'both') {{
      const tgtUrl = 'stem' + (p.s - 1) + '_E' + PAGE_R + '.html';
      sessionStorage.setItem(PENDING_KEY, JSON.stringify({{
        r: PAGE_R, n: p.n, s: p.s, f: p.f, idx: p.idx, name: el.id
      }}));
      saveViewport();
      location.href = tgtUrl;
    }} else if (unc === 'tgt') {{
      setStemStatus(el.id + ' is a possible TARGET of an uncertain differential' +
                    ' — start from the source class (one stem up)');
    }}
  }}

  function initStem() {{
    applyClassDims();
    const st = document.createElement('style');
    st.textContent =
      '.ehp-target-hl {{ stroke: #f5a623 !important; stroke-width: 4px !important; }}' +
      '.ehp-target-sel {{ stroke: #2ecc71 !important; stroke-width: 4px !important; }}' +
      '#nodes-group [data-uncertain] {{ cursor: pointer; }}';
    document.head.appendChild(st);
    document.querySelectorAll('#nodes-group circle, #nodes-group rect').forEach(el => {{
      el.addEventListener('click', onStemNodeClick);
    }});
    initPendingTarget();
    // The template's Sphere/Stem toggle is a placeholder; from a stem chart
    // it navigates back to a sphere chart of this page.
    const viewBtn = document.getElementById('sphere-toggle');
    if (viewBtn) {{
      const fresh = viewBtn.cloneNode(true);
      fresh.removeAttribute('onclick');
      viewBtn.parentNode.replaceChild(fresh, viewBtn);
      fresh.addEventListener('click', () => {{
        const n = window.prompt('Open sphere view — sphere number:');
        if (n !== null && n.trim() !== '' && !isNaN(parseInt(n))) {{
          sessionStorage.removeItem('seqsee_viewport'); // cross-view jump
          location.href = 'S' + parseInt(n) + '_E{r}.html';
        }}
      }});
    }}
    // "EHP" button: jump to the fiber-sequence view (a stem chart spans many
    // spheres, so prompt for the base sphere).
    const ctr = document.getElementById('controls-container');
    if (ctr) {{
      const ehpBtn = document.createElement('button');
      ehpBtn.className = 'control-button';
      ehpBtn.textContent = 'EHP';
      ehpBtn.title = 'Open EHP fiber-sequence view';
      ehpBtn.addEventListener('click', () => {{
        const n = window.prompt('Open EHP fiber view — base sphere n:');
        if (n !== null && n.trim() !== '' && !isNaN(parseInt(n))) {{
          sessionStorage.removeItem('seqsee_viewport'); // cross-view jump
          location.href = 'fiber' + parseInt(n) + '_E{r}.html';
        }}
      }});
      ctr.appendChild(ehpBtn);
    }}
  }}
  if (document.readyState === 'loading') {{
    window.addEventListener('DOMContentLoaded', initStem);
  }} else {{
    initStem();
  }}
}})();
</script>
</body>"#,
        );
        let new_html = html.replace("</body>", &script);
        let _ = std::fs::write(&path, new_html);
    }
}

/// Parse `fiber{n}_E{r}.html` into (r, n).
fn parse_fiber_filename(name: &str) -> Option<(i32, i32)> {
    let stem = name.strip_prefix("fiber")?.strip_suffix(".html")?;
    let (n, r) = stem.split_once("_E")?;
    Some((r.parse().ok()?, n.parse().ok()?))
}

/// Inject (or refresh) the fiber-chart script in every fiber chart on disk:
/// WASD navigation between fiber charts (unified convention: w/s = base
/// sphere ∓/+, a/d = page ∓/+), the live CLASSDIMS overlay, and
/// click-to-highlight of a class with its outgoing (shift: also incoming)
/// fiber-sequence edges. Fiber charts are read-only — no add-differential
/// flow. Scans the charts dir so navigation reflects exactly the files that
/// exist.
fn inject_fiber_scripts(charts_dir: &Path) {
    let Ok(entries) = std::fs::read_dir(charts_dir) else { return };
    let fiber_files: Vec<(i32, i32)> = entries
        .flatten()
        .filter_map(|e| parse_fiber_filename(&e.file_name().to_string_lossy()))
        .collect();
    let have: HashSet<(i32, i32)> = fiber_files.iter().copied().collect();
    for &(r, n) in &fiber_files {
        let path = charts_dir.join(format!("fiber{}_E{}.html", n, r));
        let Ok(html) = std::fs::read_to_string(&path) else { continue };

        let nb = |nn: i32, rr: i32| -> serde_json::Value {
            if have.contains(&(rr, nn)) {
                serde_json::json!(format!("fiber{}_E{}.html", nn, rr))
            } else {
                serde_json::Value::Null
            }
        };
        // Unified WASD convention: w/s (up/down) = base sphere -/+, a/d
        // (left/right) = page -/+.
        let nav = serde_json::json!({
            "up": nb(n - 1, r),
            "down": nb(n + 1, r),
            "left": nb(n, r - 1),
            "right": nb(n, r + 1),
        });

        if html.contains("/*FIBERNAV*/") && html.contains("/*HIDDENDATA*/") {
            // Current-generation script: refresh the NAV data in place (the
            // HIDDENDATA payload is refreshed separately by
            // `inject_hidden_data`).
            let (sm, em) = ("/*FIBERNAV*/", "/*ENDFIBERNAV*/");
            if let (Some(s), Some(e)) = (html.find(sm), html.find(em)) {
                let so = s + sm.len();
                let new_html = format!("{}{}{}", &html[..so], nav, &html[e..]);
                let _ = std::fs::write(&path, new_html);
            }
            continue;
        }

        let hidden_js = FIBER_HIDDEN_JS;
        let script = format!(
            r#"<script>
(function() {{
  const NAV = /*FIBERNAV*/{nav}/*ENDFIBERNAV*/;
  const CLASS_DIMS = /*CLASSDIMS*/null/*ENDCLASSDIMS*/;
  const HIDDEN_EDGES = /*HIDDENDATA*/[]/*ENDHIDDENDATA*/;
  window.EHP_CLASS_DIMS = CLASS_DIMS;

  // The template restores `seqsee_viewport` from sessionStorage on every
  // load; saving here before WASD navigation keeps the pan/zoom fixed while
  // stepping base spheres/pages.
  function saveViewport() {{
    try {{
      if (window.panZoom) {{
        const p = window.panZoom.getPan();
        sessionStorage.setItem('seqsee_viewport',
          JSON.stringify({{ x: p.x, y: p.y, zoom: window.panZoom.getZoom() }}));
      }}
    }} catch (err) {{}}
  }}

  window.addEventListener('keydown', (e) => {{
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'Escape') {{ if (hiddenMode) exitHiddenMode(); clearFocus(); return; }}
    if (e.key === 'V' && e.shiftKey) {{
      e.preventDefault();
      e.stopImmediatePropagation();
      toggleHiddenMode();
      return;
    }}
    const dir = {{w: 'up', s: 'down', a: 'left', d: 'right'}}[e.key.toLowerCase()];
    if (!dir) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    const url = NAV[dir];
    if (url) {{ saveViewport(); location.href = url; }}
  }}, true);

  function parseNodeId(id) {{
    const m = (id || '').match(/^S(-?\d+)_(-?\d+)_(-?\d+)(?:_(\d+))?$/);
    if (!m) return null;
    return {{ n: parseInt(m[1]), s: parseInt(m[2]), f: parseInt(m[3]),
             idx: m[4] !== undefined ? parseInt(m[4]) : 0 }};
  }}

  function applyClassDims() {{
    if (!CLASS_DIMS) return;
    const dead = new Set();
    document.querySelectorAll('#nodes-group circle, #nodes-group rect').forEach(el => {{
      const p = parseNodeId(el.id);
      if (!p) return;
      const key = p.n + '_' + p.s + '_' + p.f;
      const dim = (key in CLASS_DIMS) ? CLASS_DIMS[key] : 0;
      const isDead = p.idx >= dim;
      el.style.opacity = isDead ? '0.15' : '';
      if (isDead) dead.add(el.id);
    }});
    document.querySelectorAll('#edges-group [data-source]').forEach(el => {{
      const s = el.getAttribute('data-source');
      const t = el.getAttribute('data-target');
      el.style.opacity = (dead.has(s) || (t && dead.has(t))) ? '0.1' : '';
    }});
  }}
{hidden_js}
  // ==========================================================================
  // Click-to-highlight: click a class to focus it, its outgoing E/H/P edges
  // (data-source == id) and their targets; shift-click also includes incoming
  // edges (data-target == id) and their sources. Everything else fades.
  // Click empty space or Escape clears.
  // ==========================================================================
  let focusId = null;

  function clearFocus() {{
    focusId = null;
    document.querySelectorAll('.fiber-focus, .faded').forEach(el => {{
      el.classList.remove('fiber-focus');
      el.classList.remove('faded');
    }});
  }}

  function onFiberNodeClick(e) {{
    if (hiddenMode) {{ onHiddenClick(e); return; }}
    const el = e.currentTarget;
    if (!el.id) return;
    e.stopPropagation();
    if (focusId === el.id && !e.shiftKey) {{ clearFocus(); return; }}
    clearFocus();
    focusId = el.id;
    const nodes = new Set([focusId]);
    const focusEdges = new Set();
    document.querySelectorAll('#edges-group [data-source]').forEach(ln => {{
      const s = ln.getAttribute('data-source');
      const t = ln.getAttribute('data-target');
      if (s === focusId) {{
        focusEdges.add(ln);
        if (t) nodes.add(t);
      }} else if (e.shiftKey && t === focusId) {{
        focusEdges.add(ln);
        nodes.add(s);
      }}
    }});
    document.querySelectorAll('#nodes-group circle, #nodes-group rect').forEach(nd => {{
      nd.classList.add(nodes.has(nd.id) ? 'fiber-focus' : 'faded');
    }});
    document.querySelectorAll('#edges-group [data-source]').forEach(ln => {{
      ln.classList.add(focusEdges.has(ln) ? 'fiber-focus' : 'faded');
    }});
  }}

  function initFiber() {{
    applyClassDims();
    drawHiddenOverlay();
    const st = document.createElement('style');
    st.textContent =
      '#nodes-group circle, #nodes-group rect {{ cursor: pointer; }}' +
      '#nodes-group .fiber-focus {{ stroke-width: 3px !important; }}' +
      '#nodes-group .hidden-sel {{ stroke: #ff6ea6 !important; stroke-width: 3px !important; }}' +
      '#nodes-group .faded, #edges-group .faded {{ opacity: 0.15; }}';
    document.head.appendChild(st);
    document.querySelectorAll('#nodes-group circle, #nodes-group rect').forEach(el => {{
      el.addEventListener('click', onFiberNodeClick);
    }});
    // Node handlers stop propagation, so any click that reaches the document
    // was on empty space.
    document.addEventListener('click', () => clearFocus());
    // The template's Sphere/Stem toggle is a placeholder; from a fiber chart
    // it navigates back to the base sphere's chart on this page.
    const viewBtn = document.getElementById('sphere-toggle');
    if (viewBtn) {{
      const fresh = viewBtn.cloneNode(true);
      fresh.removeAttribute('onclick');
      viewBtn.parentNode.replaceChild(fresh, viewBtn);
      fresh.addEventListener('click', (e) => {{
        e.stopPropagation();
        // Cross-view jump: the saved viewport is meaningless on the sphere
        // chart's layout — clear it so the Adams chart opens at its default
        // framing instead of a fiber-chart pan/zoom.
        sessionStorage.removeItem('seqsee_viewport');
        location.href = 'S{n}_E{r}.html';
      }});
    }}
  }}
  if (document.readyState === 'loading') {{
    window.addEventListener('DOMContentLoaded', initFiber);
  }} else {{
    initFiber();
  }}
}})();
</script>"#,
        );

        if let Some(nav_pos) = html.find("/*FIBERNAV*/") {
            // Older-generation script (no HIDDENDATA): replace the enclosing
            // <script> block with the fresh one so already-generated charts
            // (warm cache, snapshots) pick up the hidden overlay + Shift+V
            // without a regen. The FIBERNAV marker only ever lives in the
            // script this function injected, so the span is unambiguous; if
            // the structure doesn't match, skip rather than corrupt.
            let (Some(start), Some(end_rel)) =
                (html[..nav_pos].rfind("<script>"), html[nav_pos..].find("</script>"))
            else {
                eprintln!(
                    "  note: {} has an unrecognized fiber script — run `regen` to upgrade it.",
                    path.display(),
                );
                continue;
            };
            let end = nav_pos + end_rel + "</script>".len();
            let new_html = format!("{}{}{}", &html[..start], script, &html[end..]);
            let _ = std::fs::write(&path, new_html);
            continue;
        }

        let new_html = html.replace("</body>", &format!("{}\n</body>", script));
        let _ = std::fs::write(&path, new_html);
    }
}

/// Hidden-EHP-value JS for the fiber charts (plain constant — no `format!`
/// placeholders, so braces need no escaping; spliced into the fiber script's
/// IIFE, where `parseNodeId` etc. are in scope). Two features:
/// - `drawHiddenOverlay()`: dotted overlay edges for the injected
///   HIDDEN_EDGES (asserted = full opacity, deduced = faint), drawn above the
///   E/H/P edges and below the nodes, with a tooltip label.
/// - Shift+V assert mode: click a source class, then the target (shift-click
///   accumulates a target sum, plain click finishes), and a `hidden <kind> …`
///   REPL command lands on the clipboard. The kind is inferred from the
///   sphere relation (E: n+1, H: 2n−1, P: (n−1)/2) and pre-validated (nominal
///   stem, δ ≥ 1); the REPL re-validates authoritatively.
const FIBER_HIDDEN_JS: &str = r#"
  // ==========================================================================
  // Hidden EHP map values: overlay + Shift+V assert mode (EHP_HIDDEN feature).
  // ==========================================================================
  const HIDDEN_COLOR = '#ff6ea6';  // matches the fiber_hidden candidate pink

  function nodeCenter(id) {
    const el = document.getElementById(id);
    if (!el) return null;
    if (el.tagName === 'circle')
      return { x: parseFloat(el.getAttribute('cx')), y: parseFloat(el.getAttribute('cy')) };
    const x = parseFloat(el.getAttribute('x') || '0');
    const y = parseFloat(el.getAttribute('y') || '0');
    return { x: x + parseFloat(el.getAttribute('width') || '0') / 2,
             y: y + parseFloat(el.getAttribute('height') || '0') / 2 };
  }

  function drawHiddenOverlay() {
    const edges = document.querySelector('#edges-group');
    if (!edges) return;
    const old = document.getElementById('hidden-overlay');
    if (old) old.remove();
    if (!HIDDEN_EDGES || !HIDDEN_EDGES.length) return;
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.id = 'hidden-overlay';
    HIDDEN_EDGES.forEach(h => {
      (h.src || []).forEach(sid => (h.tgt || []).forEach(tid => {
        const a = nodeCenter(sid), b = nodeCenter(tid);
        if (!a || !b) return;
        const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        ln.setAttribute('x1', a.x); ln.setAttribute('y1', a.y);
        ln.setAttribute('x2', b.x); ln.setAttribute('y2', b.y);
        ln.setAttribute('stroke', HIDDEN_COLOR);
        ln.setAttribute('stroke-width', '1.6');
        ln.setAttribute('stroke-dasharray', '2,3');
        ln.setAttribute('opacity', h.asserted ? '0.95' : '0.5');
        ln.classList.add('hidden-overlay-edge');
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = h.label || (h.kind + ' (hidden, d=' + h.delta + ')');
        ln.appendChild(title);
        g.appendChild(ln);
      }));
    });
    // Above the E/H/P edges, below the nodes.
    edges.parentNode.insertBefore(g, edges.nextSibling);
  }

  let hiddenMode = false;
  let hiddenSrc = null;    // { p, id }
  let hiddenTgts = [];     // [{ p, id }] — summands of the target

  function setHiddenStatus(text) {
    let bar = document.getElementById('hidden-status');
    if (!bar) {
      bar = document.createElement('div');
      bar.id = 'hidden-status';
      bar.style.cssText = 'position:fixed;top:8px;left:50%;transform:translateX(-50%);' +
        'background:#222;color:#ffd7e6;border:1px solid ' + HIDDEN_COLOR + ';' +
        'padding:4px 12px;border-radius:6px;font:12px sans-serif;z-index:1000;' +
        'max-width:80%;text-align:center;';
      document.body.appendChild(bar);
    }
    bar.textContent = text;
    bar.style.display = text ? 'block' : 'none';
  }

  function clearHiddenSel() {
    hiddenSrc = null;
    hiddenTgts = [];
    document.querySelectorAll('.hidden-sel').forEach(el => el.classList.remove('hidden-sel'));
  }

  function exitHiddenMode() {
    hiddenMode = false;
    clearHiddenSel();
    setHiddenStatus('');
  }

  function toggleHiddenMode() {
    if (hiddenMode) { exitHiddenMode(); return; }
    hiddenMode = true;
    setHiddenStatus('hidden-value mode: click the source class, then the target ' +
      '(shift-click accumulates a target sum; Shift+V or Esc exits)');
  }

  function hiddenCopy(cmd) {
    navigator.clipboard.writeText(cmd).then(() => {
      setHiddenStatus('Copied: ' + cmd + ' — paste into REPL (click next source; Shift+V exits)');
    }).catch(() => {
      window.prompt('Paste into REPL:', cmd);
      setHiddenStatus(cmd);
    });
  }

  function finishHiddenAssert() {
    const s = hiddenSrc.p, t = hiddenTgts[0].p;
    let kind = null, nomS = null, nomF = null;
    if (t.n === s.n + 1) { kind = 'E'; nomS = s.s; nomF = s.f; }
    else if (t.n === 2 * s.n - 1) { kind = 'H'; nomS = s.s - s.n + 1; nomF = s.f - 1; }
    else if (s.n % 2 === 1 && s.n >= 5 && t.n === (s.n - 1) / 2) {
      kind = 'P'; nomS = s.s + (s.n - 1) / 2 - 1; nomF = s.f + 2;
    }
    let err = null;
    if (!kind) err = 'S' + t.n + ' is not the E/H/P target sphere of S' + s.n;
    else if (t.s !== nomS) err = kind + ' target stem must be ' + nomS + ' (got ' + t.s + ')';
    else if (t.f < nomF + 1) err = 'a hidden ' + kind + ' value must land at filtration >= ' +
      (nomF + 1) + ' (nominal target filtration is ' + nomF + ')';
    if (err) { setHiddenStatus(err + ' — selection cleared'); clearHiddenSel(); return; }
    const tidx = hiddenTgts.map(x => x.p.idx).join('+');
    const cmd = 'hidden ' + kind + ' ' + s.n + ' ' + s.s + ' ' + s.f + ' ' + s.idx +
      ' ' + t.n + ' ' + t.s + ' ' + t.f + ' ' + tidx;
    clearHiddenSel();
    hiddenCopy(cmd);
  }

  function onHiddenClick(e) {
    const el = e.currentTarget;
    if (!el.id) return;
    e.stopPropagation();
    const p = parseNodeId(el.id);
    if (!p) return;
    if (!hiddenSrc) {
      hiddenSrc = { p: p, id: el.id };
      el.classList.add('hidden-sel');
      setHiddenStatus('Source ' + el.id +
        ' — click the target (shift-click first to accumulate a sum; sums in the ' +
        'SOURCE are typed in the REPL directly, e.g. idx 0+1)');
      return;
    }
    const first = hiddenTgts[0];
    if (first && (p.n !== first.p.n || p.s !== first.p.s || p.f !== first.p.f)) {
      setHiddenStatus('Target summands must share a degree — Esc restarts');
      return;
    }
    hiddenTgts.push({ p: p, id: el.id });
    el.classList.add('hidden-sel');
    if (e.shiftKey) {
      setHiddenStatus('Target sum: ' + hiddenTgts.map(x => x.id).join(' + ') +
        ' — plain-click the last summand to finish');
      return;
    }
    finishHiddenAssert();
  }
"#;

/// Parse `S{n}_E{r}.html` into (r, n). Strict: bare integers only, so map
/// views (`map_*.html`) and any suffixed variants are not treated as sphere
/// charts.
fn parse_sphere_filename(name: &str) -> Option<(i32, i32)> {
    let stem = name.strip_prefix('S')?.strip_suffix(".html")?;
    let (n, r) = stem.split_once("_E")?;
    Some((r.parse().ok()?, n.parse().ok()?))
}

/// Inject (or refresh) the right-click view-jump menu in every chart on disk
/// (sphere, stem, AND fiber charts): right-clicking a class offers jumps to
/// that class's other views — its sphere chart, its stem chart, and the
/// fiber-sequence charts where its sphere is the E/H/P source. For a class
/// on sphere m those are unique: fiber{m} (E: S^m → S^{m+1}), fiber{m-1}
/// (H: S^m → S^{2m-1}), and fiber{(m-1)/2} for odd m (P: S^m → S^{(m-1)/2}).
/// Availability is scanned from the charts dir per page, so charts that were
/// never generated are simply not offered. Navigation lands with
/// `#focus=<nodeId>`, which every injected copy also handles: highlight the
/// class and pan to it once panZoom is ready.
fn inject_view_menu(charts_dir: &Path) {
    let Ok(entries) = std::fs::read_dir(charts_dir) else { return };
    // r -> sorted list of available n / k, per chart kind.
    let mut spheres: HashMap<i32, Vec<i32>> = HashMap::new();
    let mut stems: HashMap<i32, Vec<i32>> = HashMap::new();
    let mut fibers: HashMap<i32, Vec<i32>> = HashMap::new();
    // (filename, mode, r, current n-or-k)
    let mut files: Vec<(String, &'static str, i32, i32)> = Vec::new();
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if let Some((r, n)) = parse_sphere_filename(&name) {
            spheres.entry(r).or_default().push(n);
            files.push((name, "sphere", r, n));
        } else if let Some((r, k)) = parse_stem_filename(&name) {
            stems.entry(r).or_default().push(k);
            files.push((name, "stem", r, k));
        } else if let Some((r, n)) = parse_fiber_filename(&name) {
            fibers.entry(r).or_default().push(n);
            files.push((name, "fiber", r, n));
        }
    }
    for v in spheres.values_mut() {
        v.sort_unstable();
    }
    for v in stems.values_mut() {
        v.sort_unstable();
    }
    for v in fibers.values_mut() {
        v.sort_unstable();
    }
    let empty: Vec<i32> = Vec::new();

    for (name, mode, r, cur) in files {
        let path = charts_dir.join(&name);
        let Ok(html) = std::fs::read_to_string(&path) else { continue };
        let view = serde_json::json!({
            "mode": mode,
            "r": r,
            "cur": cur,
            "spheres": spheres.get(&r).unwrap_or(&empty),
            "stems": stems.get(&r).unwrap_or(&empty),
            "fibers": fibers.get(&r).unwrap_or(&empty),
        });

        if html.contains("/*VIEWNAV*/") {
            // Refresh the availability data in place.
            let (sm, em) = ("/*VIEWNAV*/", "/*ENDVIEWNAV*/");
            if let (Some(s), Some(e)) = (html.find(sm), html.find(em)) {
                let so = s + sm.len();
                let new_html = format!("{}{}{}", &html[..so], view, &html[e..]);
                let _ = std::fs::write(&path, new_html);
            }
            continue;
        }

        let script = format!(
            r#"<script>
(function() {{
  const VIEW = /*VIEWNAV*/{view}/*ENDVIEWNAV*/;
  const ID_RE = /^S(-?\d+)_(-?\d+)_(-?\d+)(?:_(\d+))?$/;
  function parseId(id) {{
    const m = ID_RE.exec(id || '');
    if (!m) return null;
    return {{ n: +m[1], s: +m[2], f: +m[3], idx: m[4] ? +m[4] : 0 }};
  }}

  let menuEl = null;
  function closeMenu() {{
    if (menuEl) {{ menuEl.remove(); menuEl = null; }}
  }}
  document.addEventListener('click', closeMenu);
  document.addEventListener('keydown', (e) => {{ if (e.key === 'Escape') closeMenu(); }});

  function openMenu(x, y, title, items) {{
    closeMenu();
    const m = document.createElement('div');
    m.style.cssText =
      'position:fixed;z-index:10000;min-width:190px;padding:4px 0;' +
      'border-radius:6px;font:13px sans-serif;' +
      'background:var(--background-color, #222);color:var(--text-color, #eee);' +
      'border:1px solid rgba(128,128,128,.5);box-shadow:0 4px 14px rgba(0,0,0,.35);';
    const hd = document.createElement('div');
    hd.textContent = title;
    hd.style.cssText = 'padding:4px 12px 6px;opacity:.6;font-size:11px;' +
      'border-bottom:1px solid rgba(128,128,128,.3);margin-bottom:3px;';
    m.appendChild(hd);
    items.forEach(([label, href]) => {{
      const it = document.createElement('div');
      it.textContent = label;
      it.style.cssText = 'padding:5px 12px;cursor:pointer;';
      it.addEventListener('mouseenter', () => it.style.background = 'rgba(128,128,128,.25)');
      it.addEventListener('mouseleave', () => it.style.background = '');
      it.addEventListener('click', () => {{
        closeMenu();
        // Cross-view jump: drop the saved viewport so the target chart opens
        // at default framing; the #focus hash then pans to the class.
        sessionStorage.removeItem('seqsee_viewport');
        location.href = href;
      }});
      m.appendChild(it);
    }});
    document.body.appendChild(m);
    // Keep the menu on-screen.
    const rect = m.getBoundingClientRect();
    m.style.left = Math.min(x, window.innerWidth - rect.width - 8) + 'px';
    m.style.top = Math.min(y, window.innerHeight - rect.height - 8) + 'px';
    menuEl = m;
  }}

  document.addEventListener('contextmenu', (e) => {{
    const t = e.target.closest && e.target.closest('#nodes-group circle, #nodes-group rect');
    if (!t) return;
    const id = t.id || (t.parentElement && t.parentElement.id) || '';
    const c = parseId(id);
    if (!c) return;
    e.preventDefault();
    const focus = '#focus=' + encodeURIComponent(id);
    const items = [];
    if (VIEW.mode !== 'sphere' && VIEW.spheres.includes(c.n)) {{
      items.push(['Sphere S' + c.n, 'S' + c.n + '_E' + VIEW.r + '.html' + focus]);
    }}
    if (!(VIEW.mode === 'stem' && VIEW.cur === c.s) && VIEW.stems.includes(c.s)) {{
      items.push(['Stem ' + c.s, 'stem' + c.s + '_E' + VIEW.r + '.html' + focus]);
    }}
    if (VIEW.fibers.includes(c.n)) {{
      items.push(['E-source fiber (S' + c.n + ' → S' + (c.n + 1) + ')',
        'fiber' + c.n + '_E' + VIEW.r + '.html' + focus]);
    }}
    if (VIEW.fibers.includes(c.n - 1)) {{
      items.push(['H-source fiber (S' + c.n + ' → S' + (2 * c.n - 1) + ')',
        'fiber' + (c.n - 1) + '_E' + VIEW.r + '.html' + focus]);
    }}
    if (c.n % 2 === 1 && VIEW.fibers.includes((c.n - 1) / 2)) {{
      items.push(['P-source fiber (S' + c.n + ' → S' + ((c.n - 1) / 2) + ')',
        'fiber' + ((c.n - 1) / 2) + '_E' + VIEW.r + '.html' + focus]);
    }}
    if (!items.length) return;
    openMenu(e.clientX, e.clientY, id + '  (n=' + c.n + ', s=' + c.s + ', f=' + c.f + ')', items);
  }});

  // #focus=<nodeId>: highlight the class and pan to it (used by the menu's
  // cross-view links; works in every chart type this script is injected in).
  let focusedEl = null;
  function clearFocus() {{
    if (focusedEl) {{
      focusedEl.setAttribute('stroke', focusedEl.dataset.ehpOrigStroke || 'none');
      focusedEl.setAttribute('stroke-width', focusedEl.dataset.ehpOrigStrokeW || '0');
      focusedEl = null;
    }}
  }}
  function panToEl(el, attempt) {{
    if (window.panZoom) {{
      try {{
        const rect = el.getBoundingClientRect();
        window.panZoom.panBy({{
          x: window.innerWidth / 2 - (rect.x + rect.width / 2),
          y: window.innerHeight / 2 - (rect.y + rect.height / 2),
        }});
      }} catch (err) {{}}
    }} else if (attempt < 25) {{
      setTimeout(() => panToEl(el, attempt + 1), 160);
    }}
  }}
  function focusFromHash() {{
    const m = (location.hash || '').match(/^#focus=(.+)$/);
    if (!m) return;
    const el = document.getElementById(decodeURIComponent(m[1]));
    if (!el) return;
    clearFocus();
    el.dataset.ehpOrigStroke = el.getAttribute('stroke') || 'none';
    el.dataset.ehpOrigStrokeW = el.getAttribute('stroke-width') || '0';
    el.setAttribute('stroke', '#e0245e');
    el.setAttribute('stroke-width', '3');
    focusedEl = el;
    panToEl(el, 0);
  }}
  window.addEventListener('hashchange', focusFromHash);
  if (document.readyState === 'loading') {{
    window.addEventListener('DOMContentLoaded', focusFromHash);
  }} else {{
    focusFromHash();
  }}
}})();
</script>
</body>"#,
        );
        let new_html = html.replace("</body>", &script);
        let _ = std::fs::write(&path, new_html);
    }
}

/// Delete generated chart artifacts (sphere charts, map views, index) from
/// the charts directory. Only files matching our naming patterns are touched.
fn clean_charts_dir(charts_dir: &Path) -> usize {
    let mut removed = 0;
    let Ok(entries) = std::fs::read_dir(charts_dir) else {
        return 0;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let is_ours = name == "index.html"
            || ((name.ends_with(".html") || name.ends_with(".json"))
                && (name.starts_with("map_")
                    || name.starts_with("stem")
                    || name.starts_with("fiber")
                    || (name.starts_with('S') && name.contains("_E"))));
        if is_ours && std::fs::remove_file(entry.path()).is_ok() {
            removed += 1;
        }
    }
    removed
}

/// The python used to run SeqSee scripts, resolved once.
///
/// - `EHP_PYTHON` overrides everything.
/// - The vendored copy (ext/seqsee, detected by its requirements.txt) has no
///   poetry env: plain `python3` is used, with deps from requirements.txt.
/// - External checkouts: the poetry venv's python, probed once — calling it
///   directly skips `poetry run`'s ~0.5-1s startup overhead per invocation.
static SEQSEE_PYTHON: OnceLock<Option<PathBuf>> = OnceLock::new();

fn seqsee_python(seqsee_dir: &Path) -> Option<PathBuf> {
    SEQSEE_PYTHON
        .get_or_init(|| {
            if let Ok(py) = std::env::var("EHP_PYTHON") {
                return Some(PathBuf::from(py));
            }
            if seqsee_dir.join("requirements.txt").exists() {
                // Vendored copy: no poetry env. Prefer python3 from PATH
                // (deps from requirements.txt), but PROBE it first — a
                // bare-bones system python without jinja2 makes every chart
                // subprocess die silently at import time (stderr is nulled).
                let path_py = PathBuf::from("python3");
                if python_has_jinja2(&path_py) {
                    return Some(path_py);
                }
                // PATH python3 lacks deps: fall back to the poetry venv of an
                // external SeqSee checkout, if one exists and passes the probe.
                if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
                    for candidate in [home.join("seqsee/seqsee_new"), home.join("SeqSee")] {
                        if !candidate.is_dir() {
                            continue;
                        }
                        if let Some(py) = poetry_python(&candidate) {
                            if python_has_jinja2(&py) {
                                eprintln!(
                                    "note: PATH python3 lacks SeqSee deps (jinja2); using poetry venv python {} from {}",
                                    py.display(),
                                    candidate.display()
                                );
                                return Some(py);
                            }
                        }
                    }
                }
                eprintln!(
                    "WARNING: no python with SeqSee deps found (python3 on PATH cannot import jinja2, and no usable poetry venv was located). \
Chart generation WILL FAIL silently. Fix with `pip install -r {}/requirements.txt` or set EHP_PYTHON to a python that has the deps.",
                    seqsee_dir.display()
                );
                return Some(path_py);
            }
            poetry_python(seqsee_dir)
        })
        .clone()
}

/// True if `py -c "import jinja2"` succeeds (the canary dep for SeqSee scripts).
fn python_has_jinja2(py: &Path) -> bool {
    std::process::Command::new(py)
        .args(["-c", "import jinja2"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Resolve the poetry venv's python for a directory containing a pyproject.
fn poetry_python(dir: &Path) -> Option<PathBuf> {
    let out = std::process::Command::new("poetry")
        .args(["env", "info", "--executable"])
        .current_dir(dir)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let p = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
    p.exists().then_some(p)
}

/// Command to run a SeqSee python script, using the resolved venv python
/// when available (falling back to `poetry run python`).
fn seqsee_cmd(seqsee_dir: &Path, script: &str) -> std::process::Command {
    let mut c = match seqsee_python(seqsee_dir) {
        Some(py) => {
            let mut c = std::process::Command::new(py);
            c.arg(script);
            c
        }
        None => {
            let mut c = std::process::Command::new("poetry");
            c.args(["run", "python", script]);
            c
        }
    };
    c.current_dir(seqsee_dir);
    c.stdout(std::process::Stdio::null());
    c.stderr(std::process::Stdio::null());
    c
}

/// Run the SeqSee pipeline for a single sphere: CSV -> JSON -> HTML, then inject click script.
fn run_seqsee_pipeline(
    n: i32,
    csv_abs: &Path,
    html_path: &Path,
    seqsee_dir: &Path,
    theme: &str,
    r: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let json_path = html_path.with_extension("json");

    // CSV -> JSON
    let status = seqsee_cmd(seqsee_dir, "jsonmaker.py")
        .arg(csv_abs)
        .arg(&json_path)
        .args(["sphere", &n.to_string()])
        .status()?;
    if !status.success() {
        return Err("jsonmaker.py failed".into());
    }

    // JSON -> HTML
    let status = seqsee_cmd(seqsee_dir, "main.py")
        .arg(&json_path)
        .arg(html_path)
        .arg(theme)
        .status()?;
    if !status.success() {
        return Err("main.py failed".into());
    }

    // Inject click-to-clipboard script
    inject_click_script(html_path, r, theme)?;

    Ok(())
}

/// Theme palettes, loaded once from the shared `themes.json` in the SeqSee
/// repo — the same file main.py reads to emit the per-theme CSS variable
/// blocks, so overlay colors always agree with the chart CSS and adding a
/// theme there is picked up by both sides automatically.
struct ThemeRegistry {
    /// Cycling order shown in the chart UI.
    order: Vec<String>,
    /// theme name → palette role (Catppuccin key names) → hex color.
    palettes: std::collections::HashMap<String, std::collections::HashMap<String, String>>,
}

fn theme_registry() -> &'static ThemeRegistry {
    static REGISTRY: std::sync::OnceLock<ThemeRegistry> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| {
        find_seqsee_dir()
            .and_then(|d| std::fs::read_to_string(d.join("themes.json")).ok())
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(parse_theme_registry)
            .unwrap_or_else(|| {
                eprintln!(
                    "Warning: themes.json not found in the SeqSee dir; \
                     falling back to built-in Catppuccin light/dark"
                );
                fallback_theme_registry()
            })
    })
}

fn parse_theme_registry(v: serde_json::Value) -> Option<ThemeRegistry> {
    let themes = v.get("themes")?.as_object()?;
    let mut palettes = std::collections::HashMap::new();
    for (name, entry) in themes {
        let pal = entry.get("palette")?.as_object()?;
        let map = pal
            .iter()
            .filter_map(|(role, c)| Some((role.clone(), c.as_str()?.to_string())))
            .collect();
        palettes.insert(name.clone(), map);
    }
    let order: Vec<String> = v
        .get("order")?
        .as_array()?
        .iter()
        .filter_map(|s| Some(s.as_str()?.to_string()))
        .filter(|n| palettes.contains_key(n))
        .collect();
    if order.is_empty() {
        return None;
    }
    Some(ThemeRegistry { order, palettes })
}

/// Minimal built-in registry (Catppuccin Latte/Mocha) used only when
/// themes.json is missing, so charts still render sensibly.
fn fallback_theme_registry() -> ThemeRegistry {
    let make = |entries: &[(&str, &str)]| {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<std::collections::HashMap<_, _>>()
    };
    let light = make(&[
        ("base", "#eff1f5"), ("mantle", "#e6e9ef"), ("text", "#4c4f69"),
        ("subtext0", "#6c6f85"), ("surface0", "#ccd0da"), ("surface1", "#bcc0cc"),
        ("surface2", "#acb0be"), ("teal", "#179299"), ("red", "#d20f39"),
        ("green", "#40a02b"), ("blue", "#1e66f5"), ("yellow", "#df8e1d"),
        ("peach", "#fe640b"), ("mauve", "#8839ef"),
    ]);
    let dark_pal = make(&[
        ("base", "#1e1e2e"), ("mantle", "#181825"), ("text", "#cdd6f4"),
        ("subtext0", "#a6adc8"), ("surface0", "#313244"), ("surface1", "#45475a"),
        ("surface2", "#585b70"), ("teal", "#94e2d5"), ("red", "#f38ba8"),
        ("green", "#a6e3a1"), ("blue", "#89b4fa"), ("yellow", "#f9e2af"),
        ("peach", "#fab387"), ("mauve", "#cba6f7"),
    ]);
    ThemeRegistry {
        order: vec!["light".into(), "dark".into()],
        palettes: [("light".to_string(), light), ("dark".to_string(), dark_pal)]
            .into_iter()
            .collect(),
    }
}

/// Color for the `d_r` differential overlay in the given theme, from
/// themes.json. Matches SeqSee's per-page scheme (d2=teal, d3=red, d4=green,
/// d5=blue, d6=yellow, d7=peach, d8=mauve) so the injected overlay agrees
/// with the SeqSee-rendered edges.
fn dr_overlay_color(r: i32, theme: &str) -> String {
    let role = match r {
        2 => "teal",
        3 => "red",
        4 => "green",
        5 => "blue",
        6 => "yellow",
        7 => "peach",
        _ => "mauve",
    };
    let reg = theme_registry();
    reg.palettes
        .get(theme)
        .or_else(|| reg.palettes.get("light"))
        .or_else(|| reg.order.first().and_then(|n| reg.palettes.get(n)))
        .and_then(|pal| pal.get(role))
        .cloned()
        .unwrap_or_else(|| "#888888".to_string())
}

/// Post-process an HTML file to inject the interactive click-to-clipboard script.
fn inject_click_script(
    html_path: &Path,
    page_r: i32,
    theme: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let html = std::fs::read_to_string(html_path)?;
    // One overlay palette per theme (all of themes.json), keyed by theme
    // name. The chart's theme cycler flips data-theme at runtime, so the
    // palette is chosen at draw time, not generation time.
    let reg = theme_registry();
    let palette = |th: &str| -> serde_json::Value {
        (2..=8)
            .map(|r| (r.to_string(), dr_overlay_color(r, th).into()))
            .collect::<serde_json::Map<String, serde_json::Value>>()
            .into()
    };
    let palettes: serde_json::Value = reg
        .order
        .iter()
        .map(|name| (name.clone(), palette(name)))
        .collect::<serde_json::Map<String, serde_json::Value>>()
        .into();
    let initial = if reg.palettes.contains_key(theme) { theme } else { "light" };
    let script = CLICK_SCRIPT
        .replace("{{PAGE_R}}", &page_r.to_string())
        .replace("{{DIFF_PALETTES}}", &palettes.to_string())
        .replace("{{THEME_INITIAL}}", initial);
    let injected = html.replace("</body>", &format!("{}\n</body>", script));
    std::fs::write(html_path, injected)?;
    Ok(())
}

/// Convert deduced differentials to JSON entries for the chart log panel,
/// skipping the mutation itself. Each entry carries the node IDs and the
/// chart file so the log can navigate straight to the differential.
fn deduced_to_json(
    pages: &[PageState],
    deduced: &[DeducedDiff],
    added: &[(i32, DiffVar)],
) -> Vec<serde_json::Value> {
    let mut entries = Vec::new();
    for d in deduced {
        if added.iter().any(|(ar, av)| d.r == *ar && d.var == *av) {
            continue;
        }
        let Some(idx) = page_index(pages, d.r) else { continue };
        let ps = &pages[idx];
        let src_deg = Tridegree::new(d.var.n, d.var.s, d.var.f);
        let tgt_deg = src_deg.diff_target(d.r);
        let src_dim = ps.page.dim_at(src_deg);
        let tgt_dim = ps.page.dim_at(tgt_deg);
        if src_dim == 0 || tgt_dim == 0 {
            continue;
        }
        let src_id = seqsee::gen_name(src_deg.n, src_deg.s, src_deg.f, d.var.col as usize, src_dim);
        let tgt_id = seqsee::gen_name(tgt_deg.n, tgt_deg.s, tgt_deg.f, d.var.row as usize, tgt_dim);
        let chart = if ps.n_values.contains(&d.var.n) {
            serde_json::json!(format!("S{}_E{}.html", d.var.n, d.r))
        } else {
            serde_json::Value::Null
        };
        entries.push(serde_json::json!({
            "r": d.r,
            "n": d.var.n,
            "s": d.var.s,
            "f": d.var.f,
            "row": d.var.row,
            "col": d.var.col,
            "val": d.value,
            "srcId": src_id,
            "tgtId": tgt_id,
            "chart": chart,
        }));
    }
    entries
}

/// Inject (or refresh) the propagation map (`cmd -> deduced differentials`)
/// between markers in every chart HTML — same pattern as `fast_update_diffs`.
fn inject_propagation(
    charts_dir: &Path,
    pages: &[PageState],
    prop_log: &HashMap<String, serde_json::Value>,
) -> usize {
    let obj: serde_json::Map<String, serde_json::Value> = prop_log
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let data_str = serde_json::Value::Object(obj).to_string();

    let mut updated = 0;
    for ps in pages {
        let r = ps.page.r;
        for &n in &ps.n_values {
            let html_path = charts_dir.join(format!("S{}_E{}.html", n, r));
            let Ok(html) = std::fs::read_to_string(&html_path) else { continue };
            let start_marker = "/*PROPDATA*/";
            let end_marker = "/*ENDPROPDATA*/";
            if let (Some(start), Some(end)) = (html.find(start_marker), html.find(end_marker)) {
                let start_offset = start + start_marker.len();
                let new_html = format!("{}{}{}", &html[..start_offset], data_str, &html[end..]);
                if std::fs::write(&html_path, new_html).is_ok() {
                    updated += 1;
                }
            }
        }
    }
    updated
}

/// Push a class vector at tridegree `t` forward through the turned-page
/// quotient chain from page `from_idx` to page `to_idx`. Returns `None` if
/// the class dies (or turned data is missing) along the way.
fn push_forward(
    pages: &[PageState],
    from_idx: usize,
    to_idx: usize,
    t: Tridegree,
    vec: &fp::vector::FpVector,
) -> Option<fp::vector::FpVector> {
    let mut v = vec.clone();
    for ps in &pages[from_idx..to_idx] {
        let tb = ps.turned.as_ref()?.get(&t)?;
        // Guard against turned data that is momentarily stale relative to a
        // rebuilt page (dimensions disagree): skip rather than panic.
        if tb.quotient_map.columns() != v.len() {
            return None;
        }
        let reduced = tb.reduce_against_boundaries(&v);
        v = tb.quotient(&reduced);
        if v.is_zero() {
            return None;
        }
    }
    Some(v)
}

/// Uncertain differentials from *previous* pages, displayed on the current
/// page: for every unknown d_k (k < r) on sphere `n`, push its source and
/// target classes forward through the quotient chain; if both survive, emit
/// a dashed edge between the surviving current-page classes, tagged with k.
fn compute_prior_unknown_edges(
    pages: &[PageState],
    cur_idx: usize,
    sphere_n: i32,
) -> Vec<(String, String, i32)> {
    let cur_page = &pages[cur_idx].page;
    let mut edges: HashSet<(String, String, i32)> = HashSet::new();

    for k in 0..cur_idx {
        let ps = &pages[k];
        let Some(res) = ps.result.as_ref() else { continue };
        let rk = ps.page.r;

        for &var_idx in &res.unknown {
            let v = res.vars[var_idx];
            if v.n != sphere_n {
                continue;
            }
            let src_deg = Tridegree::new(v.n, v.s, v.f);
            let tgt_deg = src_deg.diff_target(rk);

            let src_dim_k = ps.page.dim_at(src_deg);
            let tgt_dim_k = ps.page.dim_at(tgt_deg);
            if v.col as usize >= src_dim_k || v.row as usize >= tgt_dim_k {
                continue;
            }

            let src_vec = ehp_core::gf2::vec_basis(src_dim_k, v.col as usize);
            let tgt_vec = ehp_core::gf2::vec_basis(tgt_dim_k, v.row as usize);
            let (Some(src_cur), Some(tgt_cur)) = (
                push_forward(pages, k, cur_idx, src_deg, &src_vec),
                push_forward(pages, k, cur_idx, tgt_deg, &tgt_vec),
            ) else {
                continue;
            };

            let src_dim = cur_page.dim_at(src_deg);
            let tgt_dim = cur_page.dim_at(tgt_deg);
            for i in ehp_core::gf2::vec_support(&src_cur) {
                for j in ehp_core::gf2::vec_support(&tgt_cur) {
                    edges.insert((
                        seqsee::gen_name(src_deg.n, src_deg.s, src_deg.f, i, src_dim),
                        seqsee::gen_name(tgt_deg.n, tgt_deg.s, tgt_deg.f, j, tgt_dim),
                        rk,
                    ));
                }
            }
        }
    }

    let mut out: Vec<(String, String, i32)> = edges.into_iter().collect();
    out.sort();
    out
}

/// Replace the contents between a marker pair in `html`, returning the new
/// string (None if the markers are absent).
fn replace_marker(html: &str, start_marker: &str, end_marker: &str, data: &str) -> Option<String> {
    let start = html.find(start_marker)?;
    let end = html.find(end_marker)?;
    let start_offset = start + start_marker.len();
    Some(format!("{}{}{}", &html[..start_offset], data, &html[end..]))
}

/// Current dimensions of the degrees on a chart, as the CLASSDIMS JSON
/// (keys `n_s_f`). `filter` selects the chart's degrees.
fn class_dims_json(page: &SATPage, filter: impl Fn(&Tridegree) -> bool) -> String {
    let mut map = serde_json::Map::new();
    for (&t, &d) in &page.dimension {
        if d > 0 && filter(&t) {
            map.insert(format!("{}_{}_{}", t.n, t.s, t.f), serde_json::json!(d));
        }
    }
    serde_json::Value::Object(map).to_string()
}

/// Fast-update chart data without re-running the SeqSee pipeline:
/// - `DIFFDATA`: edges `[src, tgt, determined, r]` — the current page's
///   differentials plus still-unresolved ones from earlier pages;
/// - `CLASSDIMS`: current dimension per degree, so killed classes fade
///   without regenerating the chart (sphere and stem charts alike).
fn fast_update_diffs(charts_dir: &Path, pages: &[PageState], idx: usize) -> usize {
    let ps = &pages[idx];
    let r = ps.page.r;
    let Some(res) = ps.result.as_ref() else { return 0 };

    let mut updated = 0;
    for &n in &ps.n_values {
        let html_path = charts_dir.join(format!("S{}_E{}.html", n, r));
        if !html_path.exists() {
            continue;
        }

        let edges = seqsee::compute_sphere_diff_edges(&ps.page, res, &ps.known_diffs, n);
        let mut data: Vec<serde_json::Value> = edges
            .iter()
            .map(|(s, t, d)| serde_json::json!([s, t, d, r]))
            .collect();
        for (s, t, rk) in compute_prior_unknown_edges(pages, idx, n) {
            data.push(serde_json::json!([s, t, false, rk]));
        }
        let data_str = serde_json::to_string(&data).unwrap_or_else(|_| "[]".to_string());

        let html = match std::fs::read_to_string(&html_path) {
            Ok(h) => h,
            Err(_) => continue,
        };

        let Some(new_html) = replace_marker(&html, "/*DIFFDATA*/", "/*ENDDIFFDATA*/", &data_str)
        else {
            continue;
        };
        let dims = class_dims_json(&ps.page, |t| t.n == n);
        let new_html = replace_marker(&new_html, "/*CLASSDIMS*/", "/*ENDCLASSDIMS*/", &dims)
            .unwrap_or(new_html);
        if std::fs::write(&html_path, new_html).is_ok() {
            updated += 1;
        }
    }

    // Stem charts: refresh their CLASSDIMS too.
    for &k in &ps.stem_values {
        let html_path = charts_dir.join(format!("stem{}_E{}.html", k, r));
        let Ok(html) = std::fs::read_to_string(&html_path) else { continue };
        let dims = class_dims_json(&ps.page, |t| t.s == k && t.n <= t.s + 2);
        if let Some(new_html) = replace_marker(&html, "/*CLASSDIMS*/", "/*ENDCLASSDIMS*/", &dims) {
            if new_html != html && std::fs::write(&html_path, new_html).is_ok() {
                updated += 1;
            }
        }
    }

    // Fiber charts: refresh their CLASSDIMS too (the triple's three spheres).
    for &n in &ps.fiber_values {
        let html_path = charts_dir.join(format!("fiber{}_E{}.html", n, r));
        let Ok(html) = std::fs::read_to_string(&html_path) else { continue };
        let dims = class_dims_json(&ps.page, |t| {
            t.n == n || t.n == n + 1 || t.n == 2 * n + 1
        });
        if let Some(new_html) = replace_marker(&html, "/*CLASSDIMS*/", "/*ENDCLASSDIMS*/", &dims) {
            if new_html != html && std::fs::write(&html_path, new_html).is_ok() {
                updated += 1;
            }
        }
    }
    updated
}

// =============================================================================
// E/H/P map minimap + side-by-side map viewer
// =============================================================================

/// One row of the minimap panel: an E/H/P map on or into a sphere.
struct MapEntry {
    kind: MapKind,
    /// true = map from this sphere, false = map into this sphere.
    outgoing: bool,
    /// The sphere at the other end.
    other_n: i32,
    /// Number of generators (on the map's source sphere) with a nonzero image.
    gen_count: usize,
}

/// Count generators on sphere `src_n` with a nonzero image under `kind`.
fn count_map_gens(page: &SATPage, kind: MapKind, src_n: i32) -> usize {
    let mut count = 0;
    for (&t, &dim) in &page.dimension {
        if t.n != src_n || dim == 0 {
            continue;
        }
        // Charts only show unstable representatives.
        if t.n > t.s + 2 {
            continue;
        }
        if let Some(max_t) = page.max_t {
            if t.s + t.f > max_t {
                continue;
            }
        }
        for col in 0..dim {
            if !seqsee::map_target_names(page, kind, t, col).is_empty() {
                count += 1;
            }
        }
    }
    count
}

/// Node IDs (on the target sphere) in the image of the map from `src_n`.
fn incoming_image_ids(page: &SATPage, kind: MapKind, src_n: i32) -> Vec<String> {
    let mut ids: BTreeSet<String> = BTreeSet::new();
    for (&t, &dim) in &page.dimension {
        if t.n != src_n || dim == 0 || t.n > t.s + 2 {
            continue;
        }
        for col in 0..dim {
            for name in seqsee::map_target_names(page, kind, t, col) {
                ids.insert(name);
            }
        }
    }
    ids.into_iter().collect()
}

/// Sphere-level target of a map (only depends on n).
fn map_target_n(kind: MapKind, n: i32) -> i32 {
    kind.target_degree(Tridegree::new(n, 0, 0)).n
}

/// Is the map defined on sphere n at all?
fn map_domain_n(kind: MapKind, n: i32) -> bool {
    kind.domain_check(Tridegree::new(n, 0, 0))
}

/// Enumerate the E/H/P maps on and into `sphere_n` (with chart data at both
/// ends), counting generators with nonzero images.
fn compute_map_info(page: &SATPage, sphere_n: i32, n_values: &BTreeSet<i32>) -> Vec<MapEntry> {
    let mut entries = Vec::new();

    // Outgoing: E → n+1, H → 2n-1, P → (n-1)/2.
    for kind in MapKind::all() {
        if !map_domain_n(kind, sphere_n) {
            continue;
        }
        let other_n = map_target_n(kind, sphere_n);
        if !n_values.contains(&other_n) {
            continue;
        }
        let gen_count = count_map_gens(page, kind, sphere_n);
        if gen_count > 0 {
            entries.push(MapEntry {
                kind,
                outgoing: true,
                other_n,
                gen_count,
            });
        }
    }

    // Incoming: E from n-1; H from (n+1)/2 when n is odd; P from 2n+1.
    let mut incoming: Vec<(MapKind, i32)> = Vec::new();
    incoming.push((MapKind::E, sphere_n - 1));
    if sphere_n % 2 == 1 {
        incoming.push((MapKind::H, (sphere_n + 1) / 2));
    }
    incoming.push((MapKind::P, 2 * sphere_n + 1));

    for (kind, src_n) in incoming {
        if !map_domain_n(kind, src_n) || map_target_n(kind, src_n) != sphere_n {
            continue;
        }
        if !n_values.contains(&src_n) {
            continue;
        }
        let gen_count = count_map_gens(page, kind, src_n);
        if gen_count > 0 {
            entries.push(MapEntry {
                kind,
                outgoing: false,
                other_n: src_n,
                gen_count,
            });
        }
    }

    entries
}

/// Inject (or refresh) the MAP_INFO data between markers in each sphere's
/// chart HTML — same in-place pattern as `fast_update_diffs`.
fn inject_map_info(
    charts_dir: &Path,
    page: &SATPage,
    n_values: &BTreeSet<i32>,
    r: i32,
) -> usize {
    let mut updated = 0;
    for &n in n_values {
        let html_path = charts_dir.join(format!("S{}_E{}.html", n, r));
        if !html_path.exists() {
            continue;
        }

        let entries = compute_map_info(page, n, n_values);
        let data: Vec<serde_json::Value> = entries
            .iter()
            .map(|e| {
                let mut v = serde_json::json!({
                    "kind": e.kind.name(),
                    "outgoing": e.outgoing,
                    "otherN": e.other_n,
                    "genCount": e.gen_count,
                });
                if !e.outgoing {
                    // Node IDs on this sphere lying in the incoming map's
                    // image — used by Shift+E/H/P image highlighting.
                    v["imageIds"] = serde_json::json!(incoming_image_ids(page, e.kind, e.other_n));
                }
                // If the split-screen view for this map has been generated,
                // record it so the e/h/p keys can navigate straight to it
                // (file:// pages can't probe for file existence themselves).
                let (src, tgt) = if e.outgoing { (n, e.other_n) } else { (e.other_n, n) };
                let view = format!("map_{}_{}_{}_E{}.html", e.kind.name(), src, tgt, r);
                if charts_dir.join(&view).exists() {
                    v["view"] = serde_json::json!(view);
                }
                v
            })
            .collect();
        let data_str = serde_json::to_string(&data).unwrap_or_else(|_| "[]".to_string());

        let html = match std::fs::read_to_string(&html_path) {
            Ok(h) => h,
            Err(_) => continue,
        };
        let start_marker = "/*MAPDATA*/";
        let end_marker = "/*ENDMAPDATA*/";
        if let (Some(start), Some(end)) = (html.find(start_marker), html.find(end_marker)) {
            let start_offset = start + start_marker.len();
            let new_html = format!("{}{}{}", &html[..start_offset], data_str, &html[end..]);
            if std::fs::write(&html_path, new_html).is_ok() {
                updated += 1;
            }
        }
    }
    updated
}

/// Recompute every generated map view's WASD navigation targets, keeping only
/// neighbours whose files exist. Run after `mapview`/`mapview all` so
/// navigation always lands on generated views (and picks up newly generated
/// neighbours of older views).
fn refresh_mapview_nav(charts_dir: &Path, r_range: (i32, i32)) -> usize {
    let Ok(entries) = std::fs::read_dir(charts_dir) else { return 0 };
    let mut updated = 0;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        // map_{kind}_{src}_{tgt}_E{r}.html
        let Some(stem) = name.strip_prefix("map_").and_then(|s| s.strip_suffix(".html")) else {
            continue;
        };
        let parts: Vec<&str> = stem.split('_').collect();
        if parts.len() != 4 {
            continue;
        }
        let kind = match parts[0] {
            "E" => MapKind::E,
            "H" => MapKind::H,
            "P" => MapKind::P,
            _ => continue,
        };
        let (Ok(src), Some(Ok(r))) = (
            parts[1].parse::<i32>(),
            parts[3].strip_prefix('E').map(|p| p.parse::<i32>()),
        ) else {
            continue;
        };

        let step = if kind == MapKind::P { 2 } else { 1 };
        let neighbor = |s: i32, rr: i32| -> serde_json::Value {
            if rr < r_range.0 || rr > r_range.1 || !map_domain_n(kind, s) {
                return serde_json::Value::Null;
            }
            let f = format!("map_{}_{}_{}_E{}.html", kind.name(), s, map_target_n(kind, s), rr);
            if charts_dir.join(&f).exists() {
                serde_json::json!(f)
            } else {
                serde_json::Value::Null
            }
        };
        // Same key semantics as post_process_mapview (unified convention):
        // w/s = source sphere -/+, a/d = page -/+.
        let nav = serde_json::json!({
            "up": neighbor(src - step, r),
            "down": neighbor(src + step, r),
            "left": neighbor(src, r - 1),
            "right": neighbor(src, r + 1),
        });

        let path = entry.path();
        let Ok(html) = std::fs::read_to_string(&path) else { continue };
        let (start_marker, end_marker) = ("/*NAVDATA*/", "/*ENDNAVDATA*/");
        let (Some(start), Some(end)) = (html.find(start_marker), html.find(end_marker)) else {
            continue;
        };
        let start_offset = start + start_marker.len();
        let new_html = format!("{}{}{}", &html[..start_offset], nav, &html[end..]);
        if new_html != html && std::fs::write(&path, new_html).is_ok() {
            updated += 1;
        }
    }
    updated
}

/// Parse a SeqSee node name `S{n}_{s}_{f}[_{idx}]` into (tridegree, idx).
fn parse_gen_name(name: &str) -> Option<(Tridegree, usize)> {
    let rest = name.strip_prefix('S')?;
    let parts: Vec<&str> = rest.split('_').collect();
    if parts.len() != 3 && parts.len() != 4 {
        return None;
    }
    let n: i32 = parts[0].parse().ok()?;
    let s: i32 = parts[1].parse().ok()?;
    let f: i32 = parts[2].parse().ok()?;
    let idx: usize = if parts.len() == 4 {
        parts[3].parse().ok()?
    } else {
        0
    };
    Some((Tridegree::new(n, s, f), idx))
}

/// Read a sphere's chart JSON, add `jmap` fields pointing at the map targets,
/// and write the annotated copy next to it. Returns the annotated path.
fn annotate_json_with_jmap(
    json_path: &Path,
    out_path: &Path,
    page: &SATPage,
    kind: MapKind,
) -> Result<usize, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(json_path)?;
    let mut data: serde_json::Value = serde_json::from_str(&text)?;

    let mut annotated = 0;
    if let Some(nodes) = data.get_mut("nodes").and_then(|v| v.as_object_mut()) {
        for (name, node) in nodes.iter_mut() {
            let Some((t, idx)) = parse_gen_name(name) else {
                continue;
            };
            let targets = seqsee::map_target_names(page, kind, t, idx);
            if !targets.is_empty() {
                if let Some(obj) = node.as_object_mut() {
                    obj.insert("jmap".to_string(), serde_json::json!(targets));
                    annotated += 1;
                }
            }
        }
    }

    std::fs::write(out_path, serde_json::to_string_pretty(&data)?)?;
    Ok(annotated)
}

/// Generate a side-by-side map view (source sphere with `data-jmap`
/// annotations on the left, target sphere on the right) via SeqSee's
/// `--sidebyside` pipeline. Returns the output HTML path.
fn generate_map_sidebyside(
    page: &SATPage,
    kind: MapKind,
    source_n: i32,
    target_n: i32,
    charts_dir: &Path,
    seqsee_dir: &Path,
    theme: &str,
    r: i32,
    r_range: (i32, i32),
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let source_json = charts_dir.join(format!("S{}_E{}.json", source_n, r));
    let target_json = charts_dir.join(format!("S{}_E{}.json", target_n, r));
    if !source_json.exists() {
        return Err(format!("missing {}", source_json.display()).into());
    }
    if !target_json.exists() {
        return Err(format!("missing {}", target_json.display()).into());
    }

    let stem = format!("map_{}_{}_{}_E{}", kind.name(), source_n, target_n, r);
    let annotated_json = charts_dir.join(format!("{}_src.json", stem));
    let out_html = charts_dir.join(format!("{}.html", stem));

    let annotated = annotate_json_with_jmap(&source_json, &annotated_json, page, kind)?;
    if annotated == 0 {
        eprintln!(
            "  note: no generators on S^{} have nonzero {} images",
            source_n,
            kind.name(),
        );
    }

    // Back button returns to the source sphere's single chart.
    let back_url = format!("S{}_E{}.html", source_n, r);

    let status = std::process::Command::new("poetry")
        .args(["run", "python", "main.py", "--sidebyside"])
        .arg(&annotated_json)
        .arg(&target_json)
        .arg(&out_html)
        .arg(theme)
        .arg(&back_url)
        .current_dir(seqsee_dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()?;
    if !status.success() {
        return Err("main.py --sidebyside failed".into());
    }

    post_process_mapview(&out_html, kind, source_n, target_n, r, r_range);

    Ok(out_html)
}

/// Patch a generated side-by-side map view: real panel titles (the template's
/// fallback is SO(0)/S^0) and the WASD navigation script between sibling views.
fn post_process_mapview(
    out_html: &Path,
    kind: MapKind,
    source_n: i32,
    target_n: i32,
    r: i32,
    r_range: (i32, i32),
) {
    let Ok(html) = std::fs::read_to_string(out_html) else { return };
    let patched = html
        .replace(
            r"$\mathrm{E}_{2}(\mathrm{SO}(0))$",
            &format!(r"$\mathrm{{E}}_{{{}}}(S^{{{}}}) \xrightarrow{{{}}}$", r, source_n, kind.name()),
        )
        .replace(
            r"$\mathrm{E}_{2}(S^{0})$",
            &format!(r"$\mathrm{{E}}_{{{}}}(S^{{{}}})$", r, target_n),
        );
    if patched.contains("/*NAVDATA*/") {
        let _ = std::fs::write(out_html, patched);
        return;
    }
    // WASD navigation between sibling map views (unified convention):
    // w/s = source sphere -/+ (stepping by 2 for P, which is only defined
    // on odd spheres), a/d = page -/+.
    let step = if kind == MapKind::P { 2 } else { 1 };
    let neighbor = |src: i32, rr: i32| -> serde_json::Value {
        if rr < r_range.0 || rr > r_range.1 || !map_domain_n(kind, src) {
            return serde_json::Value::Null;
        }
        serde_json::json!(format!(
            "map_{}_{}_{}_E{}.html",
            kind.name(), src, map_target_n(kind, src), rr,
        ))
    };
    let nav = serde_json::json!({
        "up": neighbor(source_n - step, r),
        "down": neighbor(source_n + step, r),
        "left": neighbor(source_n, r - 1),
        "right": neighbor(source_n, r + 1),
    });
    let nav_script = format!(
        r#"<script>
(function() {{
  const NAV = /*NAVDATA*/{nav}/*ENDNAVDATA*/;
  window.addEventListener('keydown', (e) => {{
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'E' || e.key === 'H' || e.key === 'P') {{
      // Shift+E/H/P: toggle the map-image highlight (same as the
      // template's Shift+J jmap image mode).
      e.preventDefault();
      e.stopImmediatePropagation();
      if (typeof toggleJmapImage === 'function') toggleJmapImage();
      return;
    }}
    const dir = {{w: 'up', s: 'down', a: 'left', d: 'right'}}[e.key.toLowerCase()];
    if (!dir) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    const url = NAV[dir];
    if (url) {{
      // Keep the current pan/zoom when stepping between sibling views
      // (template's storeViewport persists it in sessionStorage).
      if (typeof storeViewport === 'function') storeViewport();
      location.href = url;
    }}
  }}, true);
}})();
</script>
</body>"#,
    );
    let patched = patched.replace("</body>", &nav_script);
    let _ = std::fs::write(out_html, patched);
}

/// Pre-generate every split-screen map view for a page (J-map style):
/// annotate the source JSONs in-process, then run the sidebyside generation
/// in one batched python process per CPU. Returns (generated, attempted).
fn pregenerate_mapviews(
    ps: &PageState,
    charts_dir: &Path,
    seqsee_dir: &Path,
    theme: &str,
    r: i32,
    r_range: (i32, i32),
) -> (usize, usize) {
    // Enumerate map-view jobs (same criteria as `mapview all`).
    let mut jobs: Vec<(MapKind, i32, i32)> = Vec::new();
    for &n in &ps.n_values {
        for kind in MapKind::all() {
            if !map_domain_n(kind, n) {
                continue;
            }
            let tgt = map_target_n(kind, n);
            if !ps.n_values.contains(&tgt) {
                continue;
            }
            if count_map_gens(&ps.page, kind, n) == 0 {
                continue;
            }
            jobs.push((kind, n, tgt));
        }
    }

    // Annotate source JSONs and build manifest lines.
    let mut manifest_lines: Vec<String> = Vec::new();
    let mut metas: Vec<(MapKind, i32, i32, PathBuf)> = Vec::new();
    for (kind, n, tgt) in jobs {
        let source_json = charts_dir.join(format!("S{}_E{}.json", n, r));
        let target_json = charts_dir.join(format!("S{}_E{}.json", tgt, r));
        if !source_json.exists() || !target_json.exists() {
            continue;
        }
        let stem = format!("map_{}_{}_{}_E{}", kind.name(), n, tgt, r);
        let annotated = charts_dir.join(format!("{}_src.json", stem));
        let out_html = charts_dir.join(format!("{}.html", stem));
        if annotate_json_with_jmap(&source_json, &annotated, &ps.page, kind).is_err() {
            continue;
        }
        manifest_lines.push(
            serde_json::json!({
                "src": annotated.to_string_lossy(),
                "tgt": target_json.to_string_lossy(),
                "out": out_html.to_string_lossy(),
                "theme": theme,
                "back": format!("S{}_E{}.html", n, r),
            })
            .to_string(),
        );
        metas.push((kind, n, tgt, out_html));
    }
    let total = metas.len();
    if total == 0 {
        return (0, 0);
    }

    // Run manifest chunks in parallel, one python process each.
    let counter = std::sync::atomic::AtomicUsize::new(0);
    let nchunks = rayon::current_num_threads().clamp(1, total);
    let chunk_size = total.div_ceil(nchunks);
    let ok_metas: Vec<(MapKind, i32, i32, PathBuf)> = manifest_lines
        .par_chunks(chunk_size)
        .zip(metas.par_chunks(chunk_size))
        .flat_map(|(lines, ms)| {
            let id = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mpath = charts_dir.join(format!(".mapview_manifest_{}_{}.jsonl", r, id));
            if std::fs::write(&mpath, lines.join("\n")).is_err() {
                return Vec::new();
            }
            let mut cmd = batch_cmd(seqsee_dir);
            cmd.arg("sidebyside").arg(&mpath);
            let out = cmd.output();
            let _ = std::fs::remove_file(&mpath);
            let Ok(out) = out else { return Vec::new() };
            parse_batch_report(&out.stdout, "mapview #", r)
                .into_iter()
                .filter_map(|i| ms.get(i as usize).cloned())
                .collect::<Vec<_>>()
        })
        .collect();

    for (kind, n, tgt, out_html) in &ok_metas {
        post_process_mapview(out_html, *kind, *n, *tgt, r, r_range);
    }
    (ok_metas.len(), total)
}

// =============================================================================
// Index page generation
// =============================================================================

fn generate_index_html(
    charts_dir: &Path,
    chart_files: &[(i32, i32, PathBuf)], // (r, n, path)
    stem_files: &[(i32, i32)],           // (r, k)
    fiber_files: &[(i32, i32)],          // (r, n)
    start_r: i32,
    max_t: i32,
    theme: &str,
    pages: &[PageState],
) -> Result<(), Box<dyn std::error::Error>> {
    let index_path = charts_dir.join("index.html");
    let reg = theme_registry();
    let initial = if reg.palettes.contains_key(theme) { theme } else { "light" };
    let mut index = String::new();
    index.push_str("<!DOCTYPE html>\n");
    index.push_str(&format!("<html data-theme=\"{initial}\">\n<head>\n"));
    index.push_str(&format!(
        "<title>EHP Charts E_{}\u{2013}E_{} (max t={})</title>\n",
        start_r,
        start_r + pages.len() as i32 - 1,
        max_t,
    ));
    index.push_str("<style>\n");
    // Same per-theme CSS variable scheme as the charts (themes.json), so the
    // index follows the t/T-cycled theme stored in localStorage.
    for name in &reg.order {
        let pal = &reg.palettes[name];
        let get = |role: &str, fb: &str| pal.get(role).map(|s| s.as_str()).unwrap_or(fb).to_string();
        let selector = if name == initial {
            format!(":root, :root[data-theme=\"{name}\"]")
        } else {
            format!(":root[data-theme=\"{name}\"]")
        };
        index.push_str(&format!(
            "{selector} {{ --bg-color: {}; --text-color: {}; --accent-color: {}; --muted-color: {}; }}\n",
            get("base", "#fff"),
            get("text", "#333"),
            get("blue", "#0066cc"),
            get("subtext0", "#888"),
        ));
    }
    index.push_str(
        "body { font-family: system-ui, sans-serif; margin: 2em; \
         background: var(--bg-color); color: var(--text-color); }\n",
    );
    index.push_str("a { color: var(--accent-color); }\n");
    index.push_str("ul { list-style: none; padding: 0; }\n");
    index.push_str("li { margin: 0.3em 0; }\n");
    index.push_str("h2 { margin-top: 1.5em; }\n");
    index.push_str("</style>\n</head>\n<body>\n");
    index.push_str(&format!(
        "<script>\n\
         const THEME_NAMES = {};\n\
         let ehpTheme = localStorage.getItem('seqsee-theme') || '{initial}';\n\
         if (!THEME_NAMES.includes(ehpTheme)) ehpTheme = '{initial}';\n\
         document.documentElement.dataset.theme = ehpTheme;\n\
         document.addEventListener('keydown', (e) => {{\n\
           if (e.ctrlKey || e.metaKey || e.altKey) return;\n\
           if (e.key !== 't' && e.key !== 'T') return;\n\
           const step = e.key === 't' ? 1 : THEME_NAMES.length - 1;\n\
           ehpTheme = THEME_NAMES[(THEME_NAMES.indexOf(ehpTheme) + step) % THEME_NAMES.length];\n\
           localStorage.setItem('seqsee-theme', ehpTheme);\n\
           document.documentElement.dataset.theme = ehpTheme;\n\
         }});\n\
         </script>\n",
        serde_json::to_string(&reg.order)?,
    ));
    index.push_str(&format!(
        "<h1>EHP Charts (max t={})</h1>\n",
        max_t,
    ));
    index.push_str(&format!(
        "<p>{} pages (E_{} through E_{}), {} total charts</p>\n",
        pages.len(),
        start_r,
        start_r + pages.len() as i32 - 1,
        chart_files.len(),
    ));

    // Group by page number
    let mut by_r: std::collections::BTreeMap<i32, Vec<&(i32, i32, PathBuf)>> =
        std::collections::BTreeMap::new();
    for entry in chart_files {
        by_r.entry(entry.0).or_default().push(entry);
    }

    for (r, entries) in &by_r {
        index.push_str(&format!("<h2>E_{} page ({} spheres)</h2>\n", r, entries.len()));
        index.push_str("<ul>\n");
        for (_, n, path) in entries.iter() {
            let filename = path.file_name().unwrap().to_string_lossy();
            index.push_str(&format!(
                "  <li><a href=\"{}\">&pi;_*(S^{}) &mdash; E_{}</a></li>\n",
                filename, n, r
            ));
        }
        index.push_str("</ul>\n");
    }

    // Stem view section
    if !stem_files.is_empty() {
        let mut stems_by_r: std::collections::BTreeMap<i32, Vec<i32>> =
            std::collections::BTreeMap::new();
        for &(r, k) in stem_files {
            stems_by_r.entry(r).or_default().push(k);
        }
        index.push_str("<h2>Stem view (one chart per stem, x = n)</h2>\n");
        for (r, ks) in &mut stems_by_r {
            ks.sort_unstable();
            index.push_str(&format!(
                "<h3>E_{} ({} stems)</h3>\n<ul style=\"columns: 6em auto\">\n",
                r,
                ks.len(),
            ));
            for k in ks.iter() {
                index.push_str(&format!(
                    "  <li><a href=\"stem{k}_E{r}.html\">stem {k}</a></li>\n",
                ));
            }
            index.push_str("</ul>\n");
        }
    }

    // Fiber-sequence view section
    if !fiber_files.is_empty() {
        let mut fibers_by_r: std::collections::BTreeMap<i32, Vec<i32>> =
            std::collections::BTreeMap::new();
        for &(r, n) in fiber_files {
            fibers_by_r.entry(r).or_default().push(n);
        }
        index.push_str("<h2>EHP fiber-sequence view (one chart per base sphere)</h2>\n");
        for (r, ns) in &mut fibers_by_r {
            ns.sort_unstable();
            index.push_str(&format!(
                "<h3>E_{} ({} base spheres)</h3>\n<ul style=\"columns: 16em auto\">\n",
                r,
                ns.len(),
            ));
            for &n in ns.iter() {
                index.push_str(&format!(
                    "  <li><a href=\"fiber{n}_E{r}.html\">S<sup>{n}</sup> &rarr; \
                     &Omega;S<sup>{}</sup> &rarr; &Omega;S<sup>{}</sup></a></li>\n",
                    n + 1,
                    2 * n + 1,
                ));
            }
            index.push_str("</ul>\n");
        }
    }

    index.push_str("</body>\n</html>\n");
    std::fs::write(&index_path, index)?;
    eprintln!("Index: {}", index_path.display());
    Ok(())
}

// =============================================================================
// Save known diffs
// =============================================================================

fn save_all_known_diffs(
    pages: &[PageState],
    path: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut file = std::fs::File::create(path)?;
    let mut total = 0;
    for ps in pages {
        let r = ps.page.r;
        let mut entries: Vec<_> = ps.known_diffs.iter().collect();
        entries.sort_by_key(|(dv, _)| (dv.n, dv.s, dv.f, dv.row, dv.col));
        for (dv, val) in entries {
            writeln!(
                file,
                "{},{},{},{},{},{},{}",
                r, dv.n, dv.s, dv.f, dv.row, dv.col,
                if *val { 1 } else { 0 },
            )?;
            total += 1;
        }
    }
    Ok(total)
}

// =============================================================================
// Find SeqSee directory (kept from original)
// =============================================================================

/// The SeqSee copy vendored into the repo (ext/seqsee) — see its README.md.
const VENDORED_SEQSEE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../seqsee");

fn find_seqsee_dir() -> Option<PathBuf> {
    // Explicit override first.
    if let Ok(dir) = std::env::var("EHP_SEQSEE") {
        let p = Path::new(&dir);
        if p.join("main.py").exists() {
            return Some(p.to_path_buf());
        }
        eprintln!("Warning: EHP_SEQSEE={} has no main.py; ignoring", dir);
    }

    // Legacy env var (external checkout root or the seqsee_new dir itself).
    if let Ok(dir) = std::env::var("SEQSEE_DIR") {
        let p = Path::new(&dir).join("seqsee_new");
        if p.join("jsonmaker.py").exists() {
            return Some(p);
        }
        let p = Path::new(&dir);
        if p.join("jsonmaker.py").exists() {
            return Some(p.to_path_buf());
        }
    }

    // Vendored copy in this repo (self-contained default).
    let vendored = Path::new(VENDORED_SEQSEE);
    if vendored.join("main.py").exists() {
        return Some(vendored.to_path_buf());
    }

    // Auto-detect external checkouts (cwd-relative)
    let candidates = [
        "../seqsee/seqsee_new",
        "../../seqsee/seqsee_new",
        "../../SeqSee/seqsee_new",
    ];
    for c in &candidates {
        let p = Path::new(c);
        if p.join("jsonmaker.py").exists() {
            return Some(p.to_path_buf());
        }
    }

    // Check home directory
    if let Ok(home) = std::env::var("HOME") {
        let p = Path::new(&home).join("seqsee/seqsee_new");
        if p.join("jsonmaker.py").exists() {
            return Some(p);
        }
    }

    None
}

// =============================================================================
// JS click-to-clipboard script — injected into each chart HTML before </body>
// =============================================================================
//
// Click two nodes in d_r mode to copy an `add` command to the clipboard.
// The command now includes the page number r: `add <r> <n> <s> <f> <row> <col>`.
// Paste into the REPL to apply.

const CLICK_SCRIPT: &str = r##"
<script>
(function() {
  const PAGE_R = {{PAGE_R}};
  // Per-page differential colors (matches SeqSee's d_r scheme), one palette
  // per theme from themes.json. The chart's theme cycler flips data-theme at
  // runtime, so the palette is chosen at draw time, not generation time.
  const DIFF_PALETTES = {{DIFF_PALETTES}};

  function diffPalette() {
    const t = document.documentElement.dataset.theme
      || localStorage.getItem('seqsee-theme')
      || sessionStorage.getItem('seqsee-theme')
      || '{{THEME_INITIAL}}';
    return DIFF_PALETTES[t] || DIFF_PALETTES['{{THEME_INITIAL}}'] || {};
  }

  let diffClickState = null;

  // Persistent log stored in localStorage
  const LOG_KEY = 'ehp_diff_log';
  function loadLog() {
    try { return JSON.parse(localStorage.getItem(LOG_KEY)) || []; }
    catch { return []; }
  }
  function saveLog(log) {
    localStorage.setItem(LOG_KEY, JSON.stringify(log));
  }

  // =========================================================================
  // UI
  // =========================================================================

  function createUI() {
    // Toolbar
    const toolbar = document.createElement('div');
    toolbar.id = 'ehp-toolbar';
    toolbar.innerHTML = [
      '<button class="ehp-btn" id="btn-log" title="Toggle log (Shift+L)">Log</button>',
      '<button class="ehp-btn" id="btn-maps" title="Toggle maps (Shift+M)">Maps</button>',
    ].join('');
    document.body.appendChild(toolbar);

    // Status bar
    const bar = document.createElement('div');
    bar.id = 'ehp-status';
    bar.textContent = 'Click source node for d_' + PAGE_R;
    document.body.appendChild(bar);

    // Log panel (hidden by default)
    const panel = document.createElement('div');
    panel.id = 'ehp-log';
    panel.style.display = 'none';
    panel.innerHTML = [
      '<div class="ehp-log-header">',
      '  <strong>Differential log</strong>',
      '  <button id="btn-copy-all" title="Copy all commands">Copy all</button>',
      '  <button id="btn-clear-log" title="Clear log">Clear</button>',
      '</div>',
      '<div id="ehp-log-entries"></div>',
    ].join('');
    document.body.appendChild(panel);

    // Maps minimap panel (hidden by default)
    const maps = document.createElement('div');
    maps.id = 'ehp-maps';
    maps.style.display = 'none';
    document.body.appendChild(maps);
    renderMapsPanel();

    // Styles
    const style = document.createElement('style');
    style.textContent = `
      /* Panel chrome follows the chart theme via the CSS variables emitted
         per theme by build_theme_css (themes.json); fallbacks match the old
         light-only styling for charts generated before the refactor. */
      #ehp-toolbar {
        position: fixed; top: 10px; left: 10px; z-index: 100;
        display: flex; gap: 3px; align-items: center;
        background: var(--panel-bg, rgba(255,255,255,0.94));
        color: var(--text-color, #333);
        border: 1px solid var(--grid-color, #bbb); border-radius: 5px;
        padding: 4px 6px; font-family: sans-serif; font-size: 11px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.15);
      }
      .ehp-btn {
        background: var(--button-bg, #f0f0f0); color: var(--button-text, #000);
        border: 1px solid var(--grid-color, #bbb); border-radius: 3px;
        padding: 3px 8px; cursor: pointer; font-family: inherit; font-size: 11px;
      }
      .ehp-btn:hover { background: var(--button-hover, #e0e0e0); }
      .ehp-btn.active {
        background: var(--accent-color, #4a90d9);
        color: var(--bg-color, white);
        border-color: var(--accent-color, #357abd);
      }
      .ehp-sep { border-left: 1px solid var(--grid-color, #ccc); height: 16px; margin: 0 2px; }
      #ehp-status {
        position: fixed; bottom: 0; left: 0; right: 0; z-index: 100;
        background: var(--panel-bg, rgba(255,255,255,0.94));
        border-top: 1px solid var(--grid-color, #ccc); padding: 4px 10px;
        font-family: sans-serif; font-size: 11px; color: var(--text-color, #333);
      }
      .ehp-selected {
        stroke: var(--accent-color, #4a90d9) !important; stroke-width: 3px !important;
      }
      #ehp-log {
        position: fixed; top: 10px; right: 10px; z-index: 100;
        width: 320px; max-height: 60vh; overflow-y: auto;
        background: var(--panel-bg, rgba(255,255,255,0.96));
        color: var(--text-color, #333);
        border: 1px solid var(--grid-color, #bbb); border-radius: 5px;
        padding: 6px 8px; font-family: monospace; font-size: 11px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.15);
      }
      .ehp-log-header {
        display: flex; gap: 6px; align-items: center;
        margin-bottom: 4px; padding-bottom: 4px;
        border-bottom: 1px solid var(--grid-color, #ddd);
      }
      .ehp-log-header strong { flex: 1; font-family: sans-serif; }
      .ehp-log-header button {
        background: var(--button-bg, #f0f0f0); color: var(--button-text, #000);
        border: 1px solid var(--grid-color, #bbb); border-radius: 3px;
        padding: 2px 6px; cursor: pointer; font-size: 10px;
      }
      .ehp-log-entry {
        padding: 2px 0; cursor: pointer;
        border-bottom: 1px solid var(--surface-color, #eee);
      }
      .ehp-log-entry:hover { background: var(--button-bg, #f0f0ff); }
      .ehp-log-entry .cmd { color: var(--text-color, #333); }
      .ehp-log-entry .desc { color: var(--muted-color, #888); font-family: sans-serif; }
      .ehp-copy-btn {
        background: none; border: none; cursor: pointer;
        font-size: 11px; padding: 0 3px; color: var(--muted-color, #666);
      }
      .ehp-copy-btn:hover { color: var(--text-color, #000); }
      .ehp-ded-count { color: var(--accent-color, #4a90d9); font-family: sans-serif; font-size: 10px; }
      .ehp-deduced {
        margin: 3px 0 2px 10px; padding-left: 6px;
        border-left: 2px solid var(--grid-color, #cdd);
      }
      .ehp-ded-entry {
        padding: 1px 0; cursor: pointer; color: var(--text-color, #356);
      }
      .ehp-ded-entry:hover { background: var(--button-bg, #e8f0fe); }
      #ehp-maps {
        position: fixed; top: 46px; left: 10px; z-index: 100;
        width: 230px; max-height: 50vh; overflow-y: auto;
        background: var(--panel-bg, rgba(255,255,255,0.96));
        color: var(--text-color, #333);
        border: 1px solid var(--grid-color, #bbb); border-radius: 5px;
        padding: 6px 8px; font-family: monospace; font-size: 11px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.15);
      }
      .ehp-maps-header {
        font-family: sans-serif; font-weight: bold;
        margin-bottom: 4px; padding-bottom: 4px;
        border-bottom: 1px solid var(--grid-color, #ddd);
      }
      .ehp-maps-section {
        font-family: sans-serif; color: var(--muted-color, #888); margin-top: 4px;
      }
      .ehp-map-entry {
        padding: 2px 0 2px 8px; cursor: pointer;
        border-bottom: 1px solid var(--surface-color, #eee);
      }
      .ehp-map-entry:hover { background: var(--button-bg, #f0f0ff); }
      .ehp-map-entry .gens { color: var(--muted-color, #888); }
    `;
    document.head.appendChild(style);

    // Wire up buttons
    document.getElementById('btn-log').addEventListener('click', toggleLog);
    document.getElementById('btn-maps').addEventListener('click', toggleMaps);
    document.getElementById('btn-copy-all').addEventListener('click', copyAllCommands);
    document.getElementById('btn-clear-log').addEventListener('click', clearLog);

    renderLog();
  }

  function setStatus(text) {
    const el = document.getElementById('ehp-status');
    if (el) el.textContent = text;
  }

  function clearDiffClick() {
    if (diffClickState) {
      diffClickState.el.classList.remove('ehp-selected');
      (diffClickState.targets || []).forEach(t => t.el.classList.remove('ehp-selected'));
      diffClickState = null;
    }
  }

  // =========================================================================
  // Log panel
  // =========================================================================

  function toggleLog() {
    const p = document.getElementById('ehp-log');
    p.style.display = p.style.display === 'none' ? 'block' : 'none';
    document.getElementById('btn-log').classList.toggle('active', p.style.display !== 'none');
  }

  function addToLog(cmd, source, target) {
    const log = loadLog();
    log.push({ cmd, source, target, time: new Date().toLocaleTimeString() });
    saveLog(log);
    renderLog();
  }

  // cmd string -> list of differentials deduced by propagating it
  // (injected/refreshed by the REPL after each add/zero/toggle).
  const PROPAGATION = /*PROPDATA*/{}/*ENDPROPDATA*/;

  function logEntryHtml(cmd, desc) {
    const deduced = PROPAGATION[cmd] || [];
    const badge = deduced.length
      ? ' <span class="ehp-ded-count">\u25b8 ' + deduced.length + ' deduced</span>'
      : '';
    return '<div class="ehp-log-entry" data-cmd="' + cmd + '" title="Click to show deduced differentials">' +
      '<span class="cmd">' + cmd + '</span>' +
      '<button class="ehp-copy-btn" title="Copy command">\u29c9</button>' + badge + '<br>' +
      '<span class="desc">' + desc + '</span>' +
      '<div class="ehp-deduced" style="display:none"></div>' +
      '</div>';
  }

  function renderLog() {
    const el = document.getElementById('ehp-log-entries');
    if (!el) return;
    const log = loadLog();
    // Mutations typed directly into the REPL have propagation data but no
    // browser log entry \u2014 show them too.
    const logCmds = new Set(log.map(e => e.cmd));
    const extras = Object.keys(PROPAGATION).filter(c => !logCmds.has(c)).sort();
    if (log.length === 0 && extras.length === 0) {
      el.innerHTML = '<div style="color:var(--muted-color,#999);font-family:sans-serif">No differentials added yet.</div>';
      return;
    }
    el.innerHTML =
      log.map(e => logEntryHtml(e.cmd, e.source + ' \u2192 ' + e.target)).join('') +
      extras.map(c => logEntryHtml(c, '(typed in REPL)')).join('');
    el.querySelectorAll('.ehp-log-entry').forEach(row => {
      row.querySelector('.ehp-copy-btn').addEventListener('click', (ev) => {
        ev.stopPropagation();
        const cmd = row.dataset.cmd;
        navigator.clipboard.writeText(cmd).then(() => {
          setStatus('Copied: ' + cmd);
        }).catch(() => {});
      });
      row.addEventListener('click', () => toggleDeduced(row));
    });
  }

  function toggleDeduced(row) {
    const box = row.querySelector('.ehp-deduced');
    if (box.style.display !== 'none') {
      box.style.display = 'none';
      return;
    }
    const deduced = PROPAGATION[row.dataset.cmd] || [];
    if (!deduced.length) {
      setStatus('No deduced differentials recorded for this command (reload after the REPL re-solves)');
      return;
    }
    box.innerHTML = deduced.map((d, i) =>
      '<div class="ehp-ded-entry" data-idx="' + i + '" title="Click to show on its chart">' +
      (d.hidden
        ? (d.label || (d.kind + ' hidden value'))
        : 'd' + d.r + '(' + d.n + ',' + d.s + ',' + d.f + ')[' + d.row + ',' + d.col + '] = ' +
          (d.val ? 1 : 0)) +
      (d.chart ? '' : ' <span class="desc">(no chart)</span>') +
      '</div>'
    ).join('');
    box.querySelectorAll('.ehp-ded-entry').forEach(ent => {
      ent.addEventListener('click', (ev) => {
        ev.stopPropagation();
        gotoDiff(deduced[parseInt(ent.dataset.idx)]);
      });
    });
    box.style.display = 'block';
  }

  // Navigate to a deduced differential: switch to its sphere/page chart and
  // highlight + pan to it (via the #diff= fragment handled on load).
  function gotoDiff(d) {
    if (!d.chart) {
      setStatus('No chart for S^' + d.n + ' on E_' + d.r);
      return;
    }
    // Hidden-value entries live on fiber charts, which handle the view-menu
    // #focus= fragment (highlight + pan) but not #diff=.
    const hash = d.hidden
      ? '#focus=' + encodeURIComponent(d.srcId)
      : '#diff=' + encodeURIComponent(d.srcId + ';' + d.tgtId);
    const current = (location.pathname || '').split('/').pop();
    if (current === d.chart) {
      if (location.hash !== hash) {
        location.hash = hash; // hashchange listener highlights
      } else {
        highlightFromHash();
      }
    } else {
      location.href = d.chart + hash;
    }
  }

  function highlightFromHash() {
    const m = (location.hash || '').match(/^#diff=(.+)$/);
    if (!m) return;
    const parts = decodeURIComponent(m[1]).split(';');
    const srcEl = document.getElementById(parts[0]);
    const tgtEl = parts[1] ? document.getElementById(parts[1]) : null;
    if (!srcEl) {
      setStatus('Class ' + parts[0] + ' not found on this chart');
      return;
    }
    document.querySelectorAll('.ehp-selected').forEach(el => el.classList.remove('ehp-selected'));
    srcEl.classList.add('ehp-selected');
    if (tgtEl) tgtEl.classList.add('ehp-selected');
    panWhenReady(srcEl, tgtEl, 0);
    setStatus('Differential: ' + parts[0] + (parts[1] ? ' \u2192 ' + parts[1] : ''));
  }

  // The SeqSee pan-zoom instance is created (and does an initial fit) after
  // page load; poll until it exists, then pan \u2014 and pan once more shortly
  // after, so a late initial fit can't override us.
  function panWhenReady(srcEl, tgtEl, attempt) {
    if (window.panZoom) {
      panToDiff(srcEl, tgtEl);
      setTimeout(() => panToDiff(srcEl, tgtEl), 450);
    } else if (attempt < 25) {
      setTimeout(() => panWhenReady(srcEl, tgtEl, attempt + 1), 160);
    }
  }

  // Zoom to a readable level and pan the SeqSee viewport so the *midpoint*
  // of the differential (between source and target) is centered
  // (screen-pixel deltas, robust to the chart's internal transforms).
  function panToDiff(srcEl, tgtEl) {
    const pz = window.panZoom;
    const svg = document.querySelector('svg');
    if (!pz || !svg) return;
    try {
      // Zoom first (rect positions change with zoom).
      pz.zoom(1.8);
      const sr = srcEl.getBoundingClientRect();
      const tr = (tgtEl || srcEl).getBoundingClientRect();
      const midX = (sr.left + sr.width / 2 + tr.left + tr.width / 2) / 2;
      const midY = (sr.top + sr.height / 2 + tr.top + tr.height / 2) / 2;
      const cr = svg.getBoundingClientRect();
      pz.panBy({
        x: (cr.left + cr.width / 2) - midX,
        y: (cr.top + cr.height / 2) - midY,
      });
    } catch (err) { /* pan is best-effort */ }
  }

  window.addEventListener('hashchange', highlightFromHash);

  function copyAllCommands() {
    // Semicolon-joined so the whole batch pastes as ONE line and the REPL
    // applies it with a single re-solve (multi-line pastes are unreliable
    // in terminals).
    const log = loadLog();
    const text = log.map(e => e.cmd).join('; ');
    navigator.clipboard.writeText(text).then(() => {
      setStatus('Copied ' + log.length + ' commands as one batch — paste into REPL');
    }).catch(() => {
      window.prompt('Copy this batch:', text);
    });
  }

  function clearLog() {
    localStorage.removeItem(LOG_KEY);
    renderLog();
    setStatus('Log cleared');
  }

  // =========================================================================
  // Maps minimap panel
  // =========================================================================

  const MAP_INFO = /*MAPDATA*/[]/*ENDMAPDATA*/;

  // Current sphere, parsed from the chart filename S{n}_E{r}.html.
  function currentSphere() {
    const m = (location.pathname || '').match(/S(\d+)_E\d+\.html/);
    return m ? parseInt(m[1]) : null;
  }

  function toggleMaps() {
    const p = document.getElementById('ehp-maps');
    p.style.display = p.style.display === 'none' ? 'block' : 'none';
    document.getElementById('btn-maps').classList.toggle('active', p.style.display !== 'none');
  }

  function renderMapsPanel() {
    const el = document.getElementById('ehp-maps');
    if (!el) return;
    const n = currentSphere();
    const sub = { 0: '₀', 1: '₁', 2: '₂', 3: '₃', 4: '₄',
                  5: '₅', 6: '₆', 7: '₇', 8: '₈', 9: '₉' };
    const rSub = String(PAGE_R).split('').map(c => sub[c] || c).join('');
    let html = '<div class="ehp-maps-header">Maps (S^' + (n === null ? '?' : n) +
               ' E' + rSub + ')</div>';
    if (!MAP_INFO.length) {
      html += '<div style="color:var(--muted-color,#999);font-family:sans-serif">No maps for this sphere.</div>';
      el.innerHTML = html;
      return;
    }
    const outgoing = MAP_INFO.filter(m => m.outgoing);
    const incoming = MAP_INFO.filter(m => !m.outgoing);
    function rows(list, out) {
      return list.map(m => {
        const srcN = out ? n : m.otherN;
        const arrow = out ? '→' : '←';
        return '<div class="ehp-map-entry" data-kind="' + m.kind + '" data-src="' + srcN + '"' +
               ' title="Click to copy mapview command">' +
               m.kind + ' ' + arrow + ' S^' + m.otherN +
               '  <span class="gens">(' + m.genCount + ' gens)</span></div>';
      }).join('');
    }
    if (outgoing.length) {
      html += '<div class="ehp-maps-section">Out:</div>' + rows(outgoing, true);
    }
    if (incoming.length) {
      html += '<div class="ehp-maps-section">In:</div>' + rows(incoming, false);
    }
    el.innerHTML = html;
    el.querySelectorAll('.ehp-map-entry').forEach(row => {
      row.addEventListener('click', () => {
        const cmd = 'mapview ' + row.dataset.kind + ' ' + row.dataset.src + ' ' + PAGE_R;
        navigator.clipboard.writeText(cmd).then(() => {
          setStatus('Copied: ' + cmd + '  —  paste into REPL');
        }).catch(() => {
          window.prompt('Paste into REPL:', cmd);
        });
      });
    });
  }

  // =========================================================================
  // E/H/P map keys
  //
  // e/h/p opens the split-screen map view for this sphere (navigates to it
  // when already generated, otherwise copies the `mapview` command for the
  // REPL). Shift+E/H/P toggles highlighting of the classes in the incoming
  // map's image. Registered in capture phase to take precedence over
  // SeqSee's built-in e/h/p sphere navigation.
  // =========================================================================

  function openMapView(kind) {
    const n = currentSphere();
    if (n === null) { setStatus('Cannot determine current sphere'); return; }
    const entry = MAP_INFO.find(m => m.outgoing && m.kind === kind);
    if (!entry) {
      setStatus('No ' + kind + ' map out of S^' + n + ' (no nonzero images or no target chart)');
      return;
    }
    if (entry.view) {
      location.href = entry.view;
    } else {
      copyMapCmd('mapview ' + kind + ' ' + n + ' ' + PAGE_R);
    }
  }

  function copyMapCmd(cmd) {
    navigator.clipboard.writeText(cmd).then(() => {
      setStatus('Map view not generated yet — copied "' + cmd +
                '": paste into the REPL (or run `mapview all`), then reload this chart');
    }).catch(() => {
      window.prompt('Run in the REPL to generate the map view:', cmd);
    });
  }

  window.addEventListener('keydown', (ev) => {
    if (ev.target.tagName === 'INPUT' || ev.target.tagName === 'TEXTAREA') return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const k = ev.key;
    if (k === 'e' || k === 'h' || k === 'p') {
      ev.preventDefault();
      ev.stopImmediatePropagation();
      openMapView(k.toUpperCase());
    } else if (k === 'E' || k === 'H' || k === 'P') {
      // Image highlighting lives in the split-screen map view.
      window.__ehpShiftTap = false;  // Shift+E/H/P is not a zero-tap
      ev.preventDefault();
      ev.stopImmediatePropagation();
      setStatus('Image mode is in the split-screen view: press ' + k.toLowerCase() +
                ' to open it, then Shift+' + k + ' (or Shift+J) there.');
    }
  }, true);

  // =========================================================================
  // Node interaction
  // =========================================================================

  function setupNodeInteraction() {
    document.querySelectorAll('#nodes-group circle, #nodes-group rect').forEach(el => {
      el.style.cursor = 'pointer';
      el.addEventListener('click', onNodeClick);
      el.addEventListener('mouseenter', onNodeHover);
      el.addEventListener('mouseleave', onNodeLeave);
    });
  }

  function onNodeHover(e) {
    if (diffClickState) return; // Don't override status when selecting
    const nodeId = e.currentTarget.id;
    if (!nodeId) return;
    const p = parseNodeId(nodeId);
    if (!p) return;
    setStatus('(' + p.n + ', ' + p.s + ', ' + p.f + ')[' + p.idx + ']  \u2014  ' + nodeId);
  }

  function onNodeLeave(e) {
    if (diffClickState) return;
    setStatus('Click source node for d_' + PAGE_R);
  }

  function parseNodeId(nodeId) {
    const m = nodeId.match(/^S(-?\d+)_(-?\d+)_(-?\d+)(?:_(\d+))?$/);
    if (!m) return null;
    return {
      n: parseInt(m[1]),
      s: parseInt(m[2]),
      f: parseInt(m[3]),
      idx: m[4] !== undefined ? parseInt(m[4]) : 0
    };
  }

  function onNodeClick(e) {
    const el = e.currentTarget;
    const nodeId = el.id;
    if (!nodeId) return;
    const p = parseNodeId(nodeId);
    if (!p) return;

    window.__ehpShiftTap = false;  // a click is never part of a bare shift-tap

    if (!diffClickState) {
      diffClickState = { ...p, name: nodeId, el, targets: [] };
      el.classList.add('ehp-selected');
      setStatus('Source: ' + nodeId + ' \u2014 click target for d_' + PAGE_R +
                ' (shift-click to build a sum of targets; tap Shift alone for d_' +
                PAGE_R + ' = 0)');
      return;
    }

    const src = diffClickState;

    if (p.s !== src.s - 1 || p.f !== src.f + PAGE_R) {
      clearDiffClick();
      setStatus('Invalid: d_' + PAGE_R + ' from (s=' + src.s + ',f=' + src.f +
                ') should hit s=' + (src.s - 1) + ',f=' + (src.f + PAGE_R) +
                ', got s=' + p.s + ',f=' + p.f);
      return;
    }

    if (e.shiftKey) {
      // Accumulate a sum of targets; a plain click finishes the differential.
      if (!src.targets.some(t => t.idx === p.idx)) {
        src.targets.push({ idx: p.idx, name: nodeId, el });
        el.classList.add('ehp-selected');
      }
      setStatus('d_' + PAGE_R + '(' + src.name + ') = ' +
                src.targets.map(t => t.name).join(' + ') +
                ' + \u2026  \u2014 shift-click more targets, plain click the last one');
      return;
    }

    // Plain click: finish with all accumulated targets plus this one.
    const targets = src.targets.filter(t => t.idx !== p.idx);
    targets.push({ idx: p.idx, name: nodeId, el });
    clearDiffClick();

    const segs = targets.map(t =>
      'add ' + PAGE_R + ' ' + src.n + ' ' + src.s + ' ' + src.f + ' ' + t.idx + ' ' + src.idx
    );
    const cmd = segs.join('; ');
    const targetDesc = targets.map(t => t.name).join(' + ');
    addToLog(cmd, src.name, targetDesc);
    // Draw a faded preview of the pending differential (becomes solid on regen).
    targets.forEach(t => drawPendingDiff(src.el, t.el));
    navigator.clipboard.writeText(cmd).then(() => {
      setStatus('Copied: ' + cmd + '  \u2014  paste into REPL');
    }).catch(() => {
      setStatus('Command: ' + cmd);
      window.prompt('Paste into REPL:', cmd);
    });

    // Show log panel automatically
    const logPanel = document.getElementById('ehp-log');
    if (logPanel.style.display === 'none') {
      logPanel.style.display = 'block';
      document.getElementById('btn-log').classList.add('active');
    }
  }

  // =========================================================================
  // Keyboard shortcuts
  // =========================================================================

  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'Shift') {
      // Arm a bare shift-tap (source selected + tap Shift = assert d_r = 0).
      // Any click, other key, mouse press, or wheel disarms it, so shift-click
      // target accumulation and Shift+key combos are unaffected.
      if (!e.repeat && diffClickState) window.__ehpShiftTap = true;
      return;
    }
    window.__ehpShiftTap = false;
    if (e.key === 'L') toggleLog();
    else if (e.key === 'M') toggleMaps();
    else if (e.key === 'Escape') { clearDiffClick(); setStatus('Click source node for d_' + PAGE_R); }
  });

  window.addEventListener('keyup', (e) => {
    if (e.key !== 'Shift') return;
    if (!window.__ehpShiftTap) return;
    window.__ehpShiftTap = false;
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    emitZeroForSource();
  });

  window.addEventListener('mousedown', () => { window.__ehpShiftTap = false; }, true);
  window.addEventListener('wheel', () => { window.__ehpShiftTap = false; },
                          { capture: true, passive: true });

  // Current dimension at a degree: live CLASSDIMS if the nav script exposed
  // them (updated in place after every mutation), else count the chart's
  // regen-time nodes.
  function liveDimAt(n, s, f) {
    const dims = window.EHP_CLASS_DIMS;
    if (dims) {
      const key = n + '_' + s + '_' + f;
      return (key in dims) ? dims[key] : 0;
    }
    let count = 0;
    document.querySelectorAll('#nodes-group circle, #nodes-group rect').forEach(el => {
      const p = parseNodeId(el.id || '');
      if (p && p.n === n && p.s === s && p.f === f) count++;
    });
    return count;
  }

  // Bare shift-tap with a source selected: assert d_r(source class) = 0 by
  // emitting one `zero` per target row (a ';'-batch applies with a single
  // cascade in the REPL, same as sum-target adds).
  function emitZeroForSource() {
    if (!diffClickState) return;
    const src = diffClickState;
    if ((src.targets || []).length > 0) {
      setStatus('Shift-tap ignored while building a sum — plain-click the last target, or Escape');
      return;
    }
    const tn = src.n, ts = src.s - 1, tf = src.f + PAGE_R;
    const dim = liveDimAt(tn, ts, tf);
    clearDiffClick();
    if (dim <= 0) {
      setStatus('d_' + PAGE_R + '(' + src.name + ') is already 0 — target (' +
                tn + ',' + ts + ',' + tf + ') has no classes, nothing to assert');
      return;
    }
    const segs = [];
    for (let row = 0; row < dim; row++) {
      segs.push('zero ' + PAGE_R + ' ' + src.n + ' ' + src.s + ' ' + src.f +
                ' ' + row + ' ' + src.idx);
    }
    const cmd = segs.join('; ');
    addToLog(cmd, src.name, '0');
    navigator.clipboard.writeText(cmd).then(() => {
      setStatus('Copied: ' + cmd + '  — paste into REPL (asserts d_' + PAGE_R +
                '(' + src.name + ') = 0)');
    }).catch(() => {
      setStatus('Command: ' + cmd);
      window.prompt('Paste into REPL:', cmd);
    });
    const logPanel = document.getElementById('ehp-log');
    if (logPanel.style.display === 'none') {
      logPanel.style.display = 'block';
      document.getElementById('btn-log').classList.add('active');
    }
  }

  // =========================================================================
  // Differential edge overlay (updated in-place by the REPL, no regen needed)
  // =========================================================================

  const DIFF_EDGES = /*DIFFDATA*/[]/*ENDDIFFDATA*/;

  // Current dimension of each degree on this chart — updated in place by the
  // REPL after adds/undos. Node indices at or beyond the dimension are
  // classes killed since the chart was generated; they fade out. (The
  // surviving quotient basis is re-indexed, so "the last dots fade" can fade
  // the WRONG class. Charts whose basis changed are therefore auto-
  // regenerated after each cascade; this fade is only the fallback for the
  // too-many-charts case and deferred `propagate off` mode.)
  const CLASS_DIMS = /*CLASSDIMS*/null/*ENDCLASSDIMS*/;

  function applyClassDims() {
    if (!CLASS_DIMS) return;
    const dead = new Set();
    document.querySelectorAll('#nodes-group circle, #nodes-group rect').forEach(el => {
      const p = parseNodeId(el.id || '');
      if (!p) return;
      const key = p.n + '_' + p.s + '_' + p.f;
      const dim = (key in CLASS_DIMS) ? CLASS_DIMS[key] : 0;
      const isDead = p.idx >= dim;
      el.style.opacity = isDead ? '0.15' : '';
      if (isDead) dead.add(el.id);
    });
    // Structlines (h0/h1/h2, E, …) into or out of a killed class die with it.
    document.querySelectorAll('#edges-group [data-source]').forEach(el => {
      const s = el.getAttribute('data-source');
      const t = el.getAttribute('data-target');
      el.style.opacity = (dead.has(s) || (t && dead.has(t))) ? '0.1' : '';
    });
  }

  // Draw one differential line between two node elements. Shared by the
  // computed overlay and the click-to-add pending preview.
  function drawDiffLine(svg, srcEl, tgtEl, { width, opacity, dashed, cls, color }) {
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', srcEl.getAttribute('cx'));
    line.setAttribute('y1', srcEl.getAttribute('cy'));
    line.setAttribute('x2', tgtEl.getAttribute('cx'));
    line.setAttribute('y2', tgtEl.getAttribute('cy'));
    line.setAttribute('stroke', color || diffPalette()[PAGE_R] || '#888');
    line.setAttribute('stroke-width', width);
    line.setAttribute('stroke-opacity', opacity);
    line.classList.add(cls);
    if (dashed) line.setAttribute('stroke-dasharray', '6,4');
    const nodesGroup = svg.querySelector('#nodes-group');
    if (nodesGroup && nodesGroup.parentNode) {
      nodesGroup.parentNode.insertBefore(line, nodesGroup);
    } else {
      svg.appendChild(line);
    }
    return line;
  }

  function drawDiffOverlay() {
    const svg = document.querySelector('svg');
    if (!svg) return;

    // Remove previous overlay edges
    svg.querySelectorAll('.diff-overlay').forEach(el => el.remove());

    DIFF_EDGES.forEach(([src, tgt, det, r]) => {
      const srcEl = document.getElementById(src);
      const tgtEl = document.getElementById(tgt);
      if (!srcEl || !tgtEl) return;
      // Unknown diffs dashed, same thickness as determined ones. Each edge
      // keeps its own page's color and styling on every chart it appears on
      // (an unresolved d_3 looks identical on the E_3 and E_4 charts).
      const rr = r === undefined ? PAGE_R : r;
      const pal = diffPalette();
      drawDiffLine(svg, srcEl, tgtEl, {
        width: '2',
        opacity: det ? '1' : '0.7',
        dashed: !det,
        cls: 'diff-overlay',
        color: pal[rr] || pal[PAGE_R],
      });
    });
  }

  // Draw a faded "pending" differential the user just added by clicking, before
  // the charts are regenerated. Same color as a real diff, just faded. These are
  // client-side only and vanish on the next page load / regeneration.
  function drawPendingDiff(srcEl, tgtEl) {
    const svg = document.querySelector('svg');
    if (!svg || !srcEl || !tgtEl) return;
    drawDiffLine(svg, srcEl, tgtEl, {
      width: '2', opacity: '0.4', dashed: false, cls: 'diff-pending',
    });
  }

  // =========================================================================
  // Init
  // =========================================================================

  function init() {
    createUI();
    setupNodeInteraction();
    drawDiffOverlay();
    applyClassDims();
    // Redraw the overlay in the matching palette whenever the theme changes,
    // and report the new theme in the bottom status bar. Every switch path
    // (toolbar button, t/T keys) is a data-theme flip on <html>, so observing
    // that attribute covers them all. THEME_LABELS is a top-level const in
    // the template's own script when present.
    new MutationObserver(() => {
      drawDiffOverlay();
      const t = document.documentElement.dataset.theme;
      const label = (typeof THEME_LABELS !== 'undefined' && THEME_LABELS[t]) || t;
      setStatus('Theme: ' + label);
    }).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    });
    // The template's Sphere/Stem toggle is a placeholder (it only alerts).
    // Repurpose it: navigate to a stem chart of this page.
    const viewBtn = document.getElementById('sphere-toggle');
    if (viewBtn) {
      const fresh = viewBtn.cloneNode(true);
      fresh.removeAttribute('onclick');
      viewBtn.parentNode.replaceChild(fresh, viewBtn);
      fresh.addEventListener('click', () => {
        const k = window.prompt('Open stem view — stem number:');
        if (k !== null && k.trim() !== '' && !isNaN(parseInt(k))) {
          sessionStorage.removeItem('seqsee_viewport'); // cross-view jump
          location.href = 'stem' + parseInt(k) + '_E' + PAGE_R + '.html';
        }
      });
    }
    // "EHP" button: jump to the fiber-sequence view with this sphere as base.
    const ctr = document.getElementById('controls-container');
    if (ctr) {
      const ehpBtn = document.createElement('button');
      ehpBtn.className = 'control-button';
      ehpBtn.textContent = 'EHP';
      ehpBtn.title = 'Open EHP fiber-sequence view for this sphere';
      ehpBtn.addEventListener('click', () => {
        const n = currentSphere();
        if (n === null) { setStatus('Cannot determine current sphere'); return; }
        sessionStorage.removeItem('seqsee_viewport'); // cross-view jump
        location.href = 'fiber' + n + '_E' + PAGE_R + '.html';
      });
      ctr.appendChild(ehpBtn);
    }
    // Highlight a differential linked from another chart's log
    // (panWhenReady waits for the pan-zoom instance).
    if (location.hash.indexOf('#diff=') === 0) {
      highlightFromHash();
    }
  }

  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
</script>
"##;
