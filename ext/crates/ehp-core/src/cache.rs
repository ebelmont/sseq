//! Warm-start cache: the whole startup chain (pages incl. exclusion sets,
//! solve results, known diffs, excluded-Leibniz bookkeeping) is a
//! deterministic function of the input data and the semantics env flags.
//! After a cold startup the per-page state is written to
//! `output/.ehp_cache/<config-hash>/`; a restart with the same hash loads it
//! and skips build + solve + turn entirely.
//!
//! Env: `EHP_CACHE=0` disables; `EHP_CACHE=verify` loads AND recomputes,
//! comparing determined counts / unknown sets / dimensions (trust-building
//! mode; the recomputed state wins). Anything else (default) = use.
//!
//! The config hash covers the CSV file CONTENTS (not mtimes), the
//! outside-diffs directory contents, and every semantics flag
//! (max_t/max_r/start_r, relax, outside parity/skip, d2, solver). A version
//! byte invalidates old caches on format changes.

use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use fp::vector::FpVector;
use hashbrown::HashMap;

use crate::constraints::{DiffVar, ExcludedLeibniz};
use crate::gf2::*;
use crate::io::{
    load_from_binary, read_i32, read_u16, read_u32, read_u64, save_to_binary, write_i32,
    write_u16, write_u32, write_u64,
};
use crate::page::SATPage;
use crate::result::SATResult;
use crate::tridegree::Tridegree;

// v2 (2026-07-20): the default solver changed from `classic` to `uf`, so an
// unset `EHP_SOLVER` now means uf. Old caches hashed unset==classic; bumping
// the version invalidates them rather than silently reusing classic results
// under the new uf default.
// v3 (2026-07-20): pages now carry the lh0 map (new binary section 11) and
// induce it on page turns; invalidate v2 caches that lack it.
// 4: compare_with_python bug-fix consolidation (2026-07-29) — P-map/zero-dim
// naturality gates removed, Leibniz pair enumeration fixed, r-1 max_t
// schedule, turn_page crop, flat product bound, is_cycle guards. All cached
// solves/page chains from version 3 are semantically stale.
const CACHE_VERSION: u32 = 4;

/// One page's cached startup state.
pub struct CachedPage {
    pub page: SATPage,
    pub result: Option<SATResult>,
    pub known_diffs: HashMap<DiffVar, bool>,
    pub excluded_leibniz: ExcludedLeibniz,
}

fn hash_bytes(h: &mut impl std::hash::Hasher, bytes: &[u8]) {
    use std::hash::Hash;
    bytes.hash(h);
}

fn hash_file(h: &mut impl std::hash::Hasher, path: &Path) {
    use std::hash::Hash;
    match std::fs::read(path) {
        Ok(bytes) => {
            path.to_string_lossy().len().hash(h);
            hash_bytes(h, &bytes);
        }
        Err(_) => "MISSING".hash(h),
    }
}

/// Hash everything the startup state depends on. `data_path` is the E2 CSV
/// directory/prefix; env flags are read directly.
pub fn config_hash(data_path: &str, start_r: i32, max_t: i32, max_r: i32) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    CACHE_VERSION.hash(&mut h);
    (start_r, max_t, max_r).hash(&mut h);

    // Data files (contents).
    let dir = Path::new(data_path);
    if dir.is_dir() {
        for name in [
            format!("E{}_rank.csv", start_r),
            format!("E{}_relations.csv", start_r),
            format!("E{}_E.csv", start_r),
            format!("E{}_H.csv", start_r),
            format!("E{}_P.csv", start_r),
            format!("E{}_names.json", start_r),
        ] {
            hash_file(&mut h, &dir.join(name));
        }
    } else {
        for suffix in ["_rank.csv", "_relations.csv", "_E.csv", "_H.csv", "_P.csv"] {
            hash_file(&mut h, Path::new(&format!("{}{}", data_path, suffix)));
        }
    }

    // Outside-diffs directory contents (sorted for determinism).
    if let Ok(dir) = std::env::var("EHP_OUTSIDE_DIFFS") {
        dir.hash(&mut h);
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
            paths.sort();
            for p in paths {
                if p.extension().and_then(|e| e.to_str()) == Some("csv") {
                    hash_file(&mut h, &p);
                }
            }
        }
    }

    // Every semantics flag (value or absence).
    for var in [
        "EHP_RELAX_TARGET_EXCLUDE",
        "EHP_OUTSIDE_PARITY",
        "EHP_OUTSIDE_SKIP",
        "EHP_D2_LINEAR",
        "EHP_SOLVER",
    ] {
        (var, std::env::var(var).ok()).hash(&mut h);
    }

    format!("{:016x}", h.finish())
}

/// `EHP_CACHE` mode: `Off`, `Use` (default), or `Verify`.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum CacheMode {
    Off,
    Use,
    Verify,
}

pub fn cache_mode() -> CacheMode {
    match std::env::var("EHP_CACHE").as_deref() {
        Ok("0") => CacheMode::Off,
        Ok("verify") => CacheMode::Verify,
        _ => CacheMode::Use,
    }
}

/// Warm-start cache directory for a config hash. Besides the serialized
/// pages, the chart REPL archives its generated charts here (`charts/`).
pub fn cache_dir(hash: &str) -> PathBuf {
    PathBuf::from("output").join(".ehp_cache").join(hash)
}

// --- sidecar serialization (everything save_to_binary doesn't carry) -------

fn write_tridegree(buf: &mut Vec<u8>, t: Tridegree) {
    write_i32(buf, t.n);
    write_i32(buf, t.s);
    write_i32(buf, t.f);
}

fn read_tridegree(data: &[u8], pos: &mut usize) -> io::Result<Tridegree> {
    Ok(Tridegree::new(
        read_i32(data, pos)?,
        read_i32(data, pos)?,
        read_i32(data, pos)?,
    ))
}

fn write_diffvar(buf: &mut Vec<u8>, v: &DiffVar) {
    write_i32(buf, v.n);
    write_i32(buf, v.s);
    write_i32(buf, v.f);
    write_u16(buf, v.row);
    write_u16(buf, v.col);
}

fn read_diffvar(data: &[u8], pos: &mut usize) -> io::Result<DiffVar> {
    Ok(DiffVar::new(
        read_i32(data, pos)?,
        read_i32(data, pos)?,
        read_i32(data, pos)?,
        read_u16(data, pos)?,
        read_u16(data, pos)?,
    ))
}

fn write_vec_bits(buf: &mut Vec<u8>, v: &FpVector) {
    write_u32(buf, v.len() as u32);
    let support: Vec<u32> = vec_support(v).map(|i| i as u32).collect();
    write_u32(buf, support.len() as u32);
    for i in support {
        write_u32(buf, i);
    }
}

fn read_vec_bits(data: &[u8], pos: &mut usize) -> io::Result<FpVector> {
    let len = read_u32(data, pos)? as usize;
    let mut v = vec_zero(len);
    let n = read_u32(data, pos)? as usize;
    for _ in 0..n {
        let i = read_u32(data, pos)? as usize;
        vec_set(&mut v, i, true);
    }
    Ok(v)
}

fn write_sidecar(cp: &CachedPage) -> Vec<u8> {
    let mut buf = Vec::new();
    write_u32(&mut buf, CACHE_VERSION);

    // Exclusion sets (deterministic order).
    let mut excl: Vec<Tridegree> = cp.page.exclude_set.iter().copied().collect();
    excl.sort();
    write_u32(&mut buf, excl.len() as u32);
    for t in excl {
        write_tridegree(&mut buf, t);
    }
    let mut tonly: Vec<Tridegree> = cp.page.target_only_exclude.iter().copied().collect();
    tonly.sort();
    write_u32(&mut buf, tonly.len() as u32);
    for t in tonly {
        write_tridegree(&mut buf, t);
    }

    // Known diffs.
    let mut knowns: Vec<(&DiffVar, &bool)> = cp.known_diffs.iter().collect();
    knowns.sort_by_key(|(v, _)| (v.n, v.s, v.f, v.row, v.col));
    write_u32(&mut buf, knowns.len() as u32);
    for (v, &val) in knowns {
        write_diffvar(&mut buf, v);
        buf.push(val as u8);
    }

    // Excluded-Leibniz map.
    let mut exls: Vec<(&Tridegree, &Vec<(Tridegree, Tridegree)>)> =
        cp.excluded_leibniz.iter().collect();
    exls.sort_by_key(|(t, _)| (t.n, t.s, t.f));
    write_u32(&mut buf, exls.len() as u32);
    for (t, pairs) in exls {
        write_tridegree(&mut buf, *t);
        write_u32(&mut buf, pairs.len() as u32);
        for &(a, b) in pairs {
            write_tridegree(&mut buf, a);
            write_tridegree(&mut buf, b);
        }
    }

    // Solve result.
    match &cp.result {
        None => buf.push(0),
        Some(res) => {
            buf.push(1);
            write_u32(&mut buf, res.vars.len() as u32);
            for v in &res.vars {
                write_diffvar(&mut buf, v);
            }
            write_vec_bits(&mut buf, &res.offset);
            let mut unknown: Vec<u32> = res.unknown.iter().map(|&i| i as u32).collect();
            unknown.sort_unstable();
            write_u32(&mut buf, unknown.len() as u32);
            for i in unknown {
                write_u32(&mut buf, i);
            }
            write_u32(&mut buf, res.kernel.rows() as u32);
            write_u32(&mut buf, res.kernel.columns() as u32);
            let words = mat_raw_words(&res.kernel);
            write_u32(&mut buf, words.len() as u32);
            for w in words {
                write_u64(&mut buf, w);
            }
        }
    }
    buf
}

fn read_sidecar(data: &[u8], page: &mut SATPage) -> io::Result<(
    Option<SATResult>,
    HashMap<DiffVar, bool>,
    ExcludedLeibniz,
)> {
    let mut pos = 0usize;
    let ver = read_u32(data, &mut pos)?;
    if ver != CACHE_VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "cache version"));
    }
    let n = read_u32(data, &mut pos)? as usize;
    for _ in 0..n {
        let t = read_tridegree(data, &mut pos)?;
        page.exclude_set.insert(t);
    }
    let n = read_u32(data, &mut pos)? as usize;
    for _ in 0..n {
        let t = read_tridegree(data, &mut pos)?;
        page.target_only_exclude.insert(t);
    }
    let n = read_u32(data, &mut pos)? as usize;
    let mut knowns = HashMap::new();
    for _ in 0..n {
        let v = read_diffvar(data, &mut pos)?;
        let val = data
            .get(pos)
            .copied()
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "known val"))?
            != 0;
        pos += 1;
        knowns.insert(v, val);
    }
    let n = read_u32(data, &mut pos)? as usize;
    let mut exl = ExcludedLeibniz::new();
    for _ in 0..n {
        let t = read_tridegree(data, &mut pos)?;
        let m = read_u32(data, &mut pos)? as usize;
        let mut pairs = Vec::with_capacity(m);
        for _ in 0..m {
            pairs.push((read_tridegree(data, &mut pos)?, read_tridegree(data, &mut pos)?));
        }
        exl.insert(t, pairs);
    }
    let has_result = data
        .get(pos)
        .copied()
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "result flag"))?
        != 0;
    pos += 1;
    let result = if has_result {
        let nv = read_u32(data, &mut pos)? as usize;
        let mut vars = Vec::with_capacity(nv);
        for _ in 0..nv {
            vars.push(read_diffvar(data, &mut pos)?);
        }
        let offset = read_vec_bits(data, &mut pos)?;
        let nu = read_u32(data, &mut pos)? as usize;
        let mut unknown = hashbrown::HashSet::with_capacity(nu);
        for _ in 0..nu {
            unknown.insert(read_u32(data, &mut pos)? as usize);
        }
        let kr = read_u32(data, &mut pos)? as usize;
        let kc = read_u32(data, &mut pos)? as usize;
        let nw = read_u32(data, &mut pos)? as usize;
        let mut words = Vec::with_capacity(nw);
        for _ in 0..nw {
            words.push(read_u64(data, &mut pos)?);
        }
        let kernel = mat_from_raw_words(kr, kc, words);
        let var_index: HashMap<DiffVar, usize> =
            vars.iter().enumerate().map(|(i, v)| (*v, i)).collect();
        Some(SATResult {
            offset,
            unknown,
            kernel,
            vars,
            var_index,
        })
    } else {
        None
    };
    Ok((result, knowns, exl))
}

/// Save the startup chain. Call after a successful cold startup.
pub fn save_startup_cache(hash: &str, pages: &[CachedPage]) -> io::Result<()> {
    save_pages_to_dir(&cache_dir(hash), pages, Some(hash))
}

/// Serialize a solved page chain into an explicit directory (used by both the
/// hash-keyed warm-start cache and the named snapshot system). `label` is
/// recorded in `meta.txt` for provenance; pass the config hash for the cache,
/// or `None` for a snapshot.
pub fn save_pages_to_dir(dir: &Path, pages: &[CachedPage], label: Option<&str>) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    for cp in pages {
        let r = cp.page.r;
        save_to_binary(&cp.page, dir.join(format!("page_E{}.ehp", r)).to_str().unwrap())?;
        let sidecar = write_sidecar(cp);
        let mut f = std::fs::File::create(dir.join(format!("extra_E{}.bin", r)))?;
        f.write_all(&sidecar)?;
    }
    let mut f = std::fs::File::create(dir.join("meta.txt"))?;
    writeln!(
        f,
        "pages: {}\nhash: {}\nversion: {}",
        pages.len(),
        label.unwrap_or("-"),
        CACHE_VERSION,
    )?;
    Ok(())
}

/// Try to load the startup chain for this hash. Returns None if the cache is
/// absent or unreadable (any error = cold startup, never a failure).
pub fn load_startup_cache(hash: &str, start_r: i32, max_r: i32) -> Option<Vec<CachedPage>> {
    load_pages_from_dir(&cache_dir(hash), start_r, max_r)
}

/// Load a solved page chain from an explicit directory (the counterpart to
/// [`save_pages_to_dir`]). Returns None if absent or unreadable — callers
/// treat that as "no cached state", never a hard failure.
pub fn load_pages_from_dir(dir: &Path, start_r: i32, max_r: i32) -> Option<Vec<CachedPage>> {
    if !dir.join("meta.txt").exists() {
        return None;
    }
    let mut out = Vec::new();
    for r in start_r..=max_r {
        let page_path = dir.join(format!("page_E{}.ehp", r));
        let extra_path = dir.join(format!("extra_E{}.bin", r));
        if !page_path.exists() {
            break; // chain legitimately ends early (empty/inconsistent page)
        }
        let mut page = load_from_binary(page_path.to_str().unwrap(), None).ok()?;
        let mut data = Vec::new();
        std::fs::File::open(&extra_path)
            .ok()?
            .read_to_end(&mut data)
            .ok()?;
        let (result, knowns, exl) = read_sidecar(&data, &mut page).ok()?;
        out.push(CachedPage {
            page,
            result,
            known_diffs: knowns,
            excluded_leibniz: exl,
        });
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}
