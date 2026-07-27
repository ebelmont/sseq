//! Interpage assumption propagation ("multi-page propagation").
//!
//! Ports the original Python machinery in `~/EHP_SAT/ehp_sat/interpage.py` and
//! `run_interpage.py`: assume a value for an uncertain differential on page
//! E_r, incrementally update the solution, un-exclude the degrees on E_{r+1}
//! that become certain, recompute their homology/products/maps, activate the
//! constraints that had been skipped there, and repeat up the pages. If any
//! page becomes UNSAT (or d∘d ≠ 0 with determined differentials), the
//! assumption is contradictory — so the opposite value is forced.

use fp::vector::FpVector;
use hashbrown::{HashMap, HashSet};
use log::debug;
use rayon::prelude::*;
use std::sync::Arc;

use crate::constraints::{
    make_basis_single, make_leibniz_constraint_single, make_naturality_constraint_single,
    DiffVar, ExcludedLeibniz,
};
use crate::element::Element;
use crate::gf2::*;
use crate::map::MapKind;
use crate::page::SATPage;
use crate::pageturning::{
    compute_induced_map_single_tb, compute_induced_products_single, make_next_exclude_set,
    TurnContext, TurnedBidegree,
};
use crate::products::ProductKey;
use crate::result::SATResult;
use crate::tridegree::Tridegree;

/// Why an assumption failed.
#[derive(Clone, Debug)]
pub enum TrialError {
    /// The constraint system on page E_r became unsatisfiable.
    Unsat { r: i32 },
    /// d_r ∘ d_r ≠ 0 with both differentials fully determined.
    D2 { r: i32, degree: Tridegree },
    /// The assumed differential is not a variable on the starting page.
    UnknownVar(DiffVar),
}

impl std::fmt::Display for TrialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrialError::Unsat { r } => write!(f, "E_{} system UNSAT", r),
            TrialError::D2 { r, degree } => write!(
                f,
                "d_{}∘d_{} != 0 at ({}, {}, {})",
                r, r, degree.n, degree.s, degree.f
            ),
            TrialError::UnknownVar(v) => write!(
                f,
                "d({},{},{})[{},{}] is not a variable",
                v.n, v.s, v.f, v.row, v.col
            ),
        }
    }
}

/// A differential newly determined during propagation.
#[derive(Clone, Debug)]
pub struct LearnedDiff {
    pub r: i32,
    pub var: DiffVar,
    pub value: bool,
}

/// Per-page inputs to [`try_diffs`]. Consecutive pages E_r, E_{r+1}, ...
///
/// Note trials never consult startup turned data: every turned degree they
/// need is recomputed from the trial's updated solution so that all quotient
/// coordinates are consistent.
pub struct InterpagePage<'a> {
    pub page: &'a SATPage,
    pub result: &'a SATResult,
    /// Leibniz pairs skipped because a degree was excluded, keyed by that
    /// degree (from `ConstraintSystem::excluded_leibniz`).
    pub excluded_leibniz: &'a ExcludedLeibniz,
}

fn stable_rep(t: Tridegree) -> Tridegree {
    if t.n > t.s + 2 {
        Tridegree::new(t.s + 2, t.s, t.f)
    } else {
        t
    }
}

// =============================================================================
// Incremental solution update (ports interpage.py::update_sat_result)
// =============================================================================

/// Update a solved system to reflect `new_vars` (appended after the existing
/// variables) and `new_constraints` (variable indices in the extended space).
///
/// Rather than re-solving from scratch, this intersects the existing solution
/// space `offset + span(kernel)` with the new constraints:
/// augment the kernel with unit vectors for the new variables, solve
/// `B·(offset + Kᵀy) = b` for `y`, and map back. Returns `None` if the
/// combined system is unsatisfiable; otherwise the newly determined
/// differentials and the updated [`SATResult`].
pub fn update_sat_result(
    dsat: &SATResult,
    new_constraints: &[(Vec<usize>, bool)],
    new_vars: &[DiffVar],
) -> Option<(Vec<(DiffVar, bool)>, SATResult)> {
    let old_n = dsat.vars.len();
    let k_new = new_vars.len();
    let total_n = old_n + k_new;

    // Augmented kernel: old kernel rows padded with zeros, plus a unit vector
    // for each new (initially fully free) variable.
    let mut kaug: Vec<FpVector> = Vec::with_capacity(dsat.kernel.rows() + k_new);
    for i in 0..dsat.kernel.rows() {
        let row = mat_get_row(&dsat.kernel, i);
        kaug.push(vec_concat(&row, &vec_zero(k_new)));
    }
    for j in 0..k_new {
        kaug.push(vec_basis(total_n, old_n + j));
    }

    // Offset extended with zeros for the new variables.
    let offset_aug = vec_concat(&dsat.offset, &vec_zero(k_new));

    // Constraint rows B over the extended variable space.
    let b_rows: Vec<FpVector> = new_constraints
        .iter()
        .map(|(indices, _)| {
            let mut row = vec_zero(total_n);
            for &i in indices {
                vec_flip(&mut row, i);
            }
            row
        })
        .collect();

    // M = B·Kaugᵀ ;  rhs = b + B·offset_aug
    let m_rows: Vec<FpVector> = b_rows
        .iter()
        .map(|brow| {
            let mut row = vec_zero(kaug.len());
            for (j, krow) in kaug.iter().enumerate() {
                if vec_dot(brow, krow) {
                    vec_set(&mut row, j, true);
                }
            }
            row
        })
        .collect();
    let mut rhs = vec_zero(new_constraints.len());
    for (i, ((_, val), brow)) in new_constraints.iter().zip(&b_rows).enumerate() {
        if *val != vec_dot(brow, &offset_aug) {
            vec_set(&mut rhs, i, true);
        }
    }

    let m = mat_from_rows(m_rows, kaug.len());
    let g = gauss_solve(&m, &rhs);
    if !g.consistent {
        return None;
    }

    // New particular solution: offset + Kᵀ·y0
    let y0 = g.solution.unwrap();
    let mut new_offset = offset_aug;
    for j in vec_support(&y0) {
        vec_xor_assign(&mut new_offset, &kaug[j]);
    }

    // New kernel: rows u·Kaug for each kernel basis vector u of M.
    let mut new_kernel_rows: Vec<FpVector> = Vec::with_capacity(g.kernel.len());
    for u in &g.kernel {
        let mut acc = vec_zero(total_n);
        for j in vec_support(u) {
            vec_xor_assign(&mut acc, &kaug[j]);
        }
        if !acc.is_zero() {
            new_kernel_rows.push(acc);
        }
    }

    // A variable is free iff some kernel row touches it.
    let mut free = vec![false; total_n];
    for row in &new_kernel_rows {
        for idx in vec_support(row) {
            free[idx] = true;
        }
    }

    // Candidates for (re-)determination: previously unknown vars + all new vars.
    let mut to_check: Vec<usize> = dsat.unknown.iter().copied().collect();
    to_check.extend(old_n..total_n);
    to_check.sort_unstable();

    let all_vars = |idx: usize| -> DiffVar {
        if idx < old_n {
            dsat.vars[idx]
        } else {
            new_vars[idx - old_n]
        }
    };

    let mut learned = Vec::new();
    let mut new_unknown: HashSet<usize> = HashSet::new();
    for &idx in &to_check {
        if free[idx] {
            new_unknown.insert(idx);
        } else {
            let value = vec_get(&new_offset, idx);
            learned.push((all_vars(idx), value));
        }
    }

    // Assemble the updated SATResult.
    let mut vars = dsat.vars.clone();
    vars.extend_from_slice(new_vars);
    let mut var_index = dsat.var_index.clone();
    for (j, v) in new_vars.iter().enumerate() {
        var_index.insert(*v, old_n + j);
    }
    let kernel = if new_kernel_rows.is_empty() {
        mat_zero(0, total_n)
    } else {
        mat_from_rows(new_kernel_rows, total_n)
    };

    let result = SATResult {
        offset: new_offset,
        unknown: new_unknown,
        kernel,
        vars,
        var_index,
    };

    Some((learned, result))
}

// =============================================================================
// Sweep-shared turned-degree cache
// =============================================================================

/// Pre-computed [`TurnContext::get_tb`] results for a *fixed* (page, result)
/// pair — the stock first page and its base solution.
///
/// During a sweep, every trial re-turns degrees whose bounding differentials
/// the trial did not touch; for those degrees `get_tb` is a pure function of
/// the base page/result and therefore identical across all trials. This cache
/// turns every page degree once (in parallel) when the sweep starts and is
/// read-only — and therefore lock-free — while the trials run.
///
/// Callers must only consult the cache for degrees whose inputs are untouched
/// by the trial: `get_tb(t)` reads the determined differential entries at
/// `rep(t)` and `rep(t_in)` (plus fixed page dimensions), so a trial must
/// bypass the cache whenever either rep is in its dirty set — the degrees of
/// the assumed and newly learned variables (see [`try_diffs_with_cache`]).
pub struct TbCache {
    map: HashMap<Tridegree, Option<Arc<TurnedBidegree>>>,
}

impl TbCache {
    pub fn new(page: &SATPage, result: &SATResult) -> Self {
        let ctx = TurnContext::new(page, result);
        let keys: Vec<Tridegree> = page.page.keys().copied().collect();
        let entries: Vec<(Tridegree, Option<Arc<TurnedBidegree>>)> = keys
            .par_iter()
            .filter_map(|&t| {
                if page.dim_at(t) == 0 {
                    return None;
                }
                match ctx.get_tb(t) {
                    Ok(tb) => Some((t, tb.map(Arc::new))),
                    // Base-level d∘d ≠ 0 can't normally happen (startup page
                    // turning would have failed) — leave the degree absent so
                    // any trial that needs it recomputes fresh and surfaces
                    // the contradiction through the normal path.
                    Err(_) => None,
                }
            })
            .collect();
        TbCache {
            map: entries.into_iter().collect(),
        }
    }
}

/// A [`TbCache`] plus the trial's dirty degree set: turned data may be read
/// from the shared cache only at degrees whose differential inputs (their own
/// rep and their incoming differential's rep) are not dirty.
pub struct SharedTurn<'c> {
    pub cache: &'c TbCache,
    /// Stable-rep degrees of the trial's assumed + learned variables.
    pub dirty: &'c HashSet<Tridegree>,
}

impl SharedTurn<'_> {
    /// Cached turned data at `t`, or `None` if `t`'s inputs are dirty (or the
    /// degree is not cached) and it must be re-turned from the trial's
    /// updated result.
    fn get_clean(&self, t: Tridegree, r: i32) -> Option<Option<Arc<TurnedBidegree>>> {
        let t_in = Tridegree::new(t.n, t.s + 1, t.f - r);
        if self.dirty.contains(&stable_rep(t)) || self.dirty.contains(&stable_rep(t_in)) {
            return None;
        }
        self.cache.map.get(&t).cloned()
    }
}

// =============================================================================
// Per-page trial-invariant indices
// =============================================================================

/// Indices over a *stock* target page that [`build_overlay_page`] needs every
/// trial: partner lists for product-candidate generation, and inverted
/// indices from a stable-rep degree to the page keys / naturality squares /
/// product blocks it participates in. All of it depends only on the stock
/// page, so a sweep computes it once and every trial reuses it, applying its
/// small dimension patch (`LocalTurn::new_dims`) as an explicit delta.
pub struct PageIndex {
    /// Page keys grouped by `n` (product partner lookup: y with y.n = x.n+x.s).
    by_n: HashMap<i32, Vec<Tridegree>>,
    /// Page keys grouped by `n + s` (product partner lookup: x with x.n+x.s = y.n).
    by_ns: HashMap<i32, Vec<Tridegree>>,
    /// Stable rep -> page keys with that rep (incl. past-stable copies).
    keys_by_rep: HashMap<Tridegree, Vec<Tridegree>>,
    /// Stable rep -> naturality squares (kind, src, tgt) with rep(src) or
    /// rep(tgt) equal to it.
    map_squares_by_rep: HashMap<Tridegree, Vec<(MapKind, Tridegree, Tridegree)>>,
    /// Stable rep -> product blocks (d1, d2) where the rep is rep(d1), rep(d2)
    /// or rep(d1·d2).
    blocks_by_rep: HashMap<Tridegree, Vec<(Tridegree, Tridegree)>>,
}

impl PageIndex {
    pub fn new(page: &SATPage) -> Self {
        let mut by_n: HashMap<i32, Vec<Tridegree>> = HashMap::new();
        let mut by_ns: HashMap<i32, Vec<Tridegree>> = HashMap::new();
        let mut keys_by_rep: HashMap<Tridegree, Vec<Tridegree>> = HashMap::new();
        for &t in page.page.keys() {
            by_n.entry(t.n).or_default().push(t);
            by_ns.entry(t.n + t.s).or_default().push(t);
            keys_by_rep.entry(stable_rep(t)).or_default().push(t);
        }

        let mut map_squares_by_rep: HashMap<Tridegree, Vec<(MapKind, Tridegree, Tridegree)>> =
            HashMap::new();
        for kind in MapKind::all() {
            for &t in page.page.keys() {
                if !kind.domain_check(t) {
                    continue;
                }
                let tgt = kind.target_degree(t);
                let (r1, r2) = (stable_rep(t), stable_rep(tgt));
                map_squares_by_rep.entry(r1).or_default().push((kind, t, tgt));
                if r2 != r1 {
                    map_squares_by_rep.entry(r2).or_default().push((kind, t, tgt));
                }
            }
        }

        let mut blocks_by_rep: HashMap<Tridegree, Vec<(Tridegree, Tridegree)>> = HashMap::new();
        for (&(d1, d2), _) in page.products.iter_blocks() {
            let prod = Tridegree::new(d1.n, d1.s + d2.s, d1.f + d2.f);
            let mut reps = [stable_rep(d1), stable_rep(d2), stable_rep(prod)];
            reps.sort();
            let mut prev = None;
            for rep in reps {
                if Some(rep) != prev {
                    blocks_by_rep.entry(rep).or_default().push((d1, d2));
                    prev = Some(rep);
                }
            }
        }

        PageIndex {
            by_n,
            by_ns,
            keys_by_rep,
            map_squares_by_rep,
            blocks_by_rep,
        }
    }
}

/// Index over a *stock* source page for [`turn_page_local`]: past-stable
/// copies of a stable-edge degree, keyed by `(s, f)`.
pub struct SourceIndex {
    stable_by_sf: HashMap<(i32, i32), Vec<Tridegree>>,
}

impl SourceIndex {
    pub fn new(page: &SATPage) -> Self {
        let mut stable_by_sf: HashMap<(i32, i32), Vec<Tridegree>> = HashMap::new();
        for &t in page.page.keys() {
            if t.n > t.s + 2 {
                stable_by_sf.entry((t.s, t.f)).or_default().push(t);
            }
        }
        SourceIndex { stable_by_sf }
    }
}

/// Everything a sweep shares across its trials: the base-result turned-degree
/// cache for the first page step, and the stock-page indices for every step.
pub struct SweepCache {
    /// Turned-degree cache for `(pages[0].page, pages[0].result)`.
    pub tb: TbCache,
    /// [`SourceIndex`] for `pages[0].page` (first-step `turn_page_local`).
    pub source_index: SourceIndex,
    /// [`PageIndex`] for each target page `pages[1..]`.
    pub target_indexes: Vec<PageIndex>,
}

impl SweepCache {
    pub fn new(pages: &[InterpagePage]) -> Self {
        let first = &pages[0];
        SweepCache {
            tb: TbCache::new(first.page, first.result),
            source_index: SourceIndex::new(first.page),
            target_indexes: pages[1..].iter().map(|p| PageIndex::new(p.page)).collect(),
        }
    }
}

// =============================================================================
// Local page turning at un-excluded degrees (ports interpage.py::turn_page_local)
// =============================================================================

/// Result of re-turning the source page at the degrees that became certain.
pub struct LocalTurn {
    /// Degrees excluded on the base target page but certain under the updated
    /// result (stable-representative form).
    pub unexclude: Vec<Tridegree>,
    /// The updated exclude set for the target page.
    pub new_exclude: HashSet<Tridegree>,
    /// The updated target-only subset of `new_exclude` (see
    /// [`SATPage::target_only_exclude`]).
    pub new_target_only: HashSet<Tridegree>,
    /// Corrected target-page dimensions at un-excluded degrees (including
    /// past-stable copies).
    pub new_dims: HashMap<Tridegree, usize>,
    /// Recomputed turned data at those degrees.
    pub tbs: HashMap<Tridegree, Arc<TurnedBidegree>>,
    /// New differential variables to add on the target page.
    pub new_vars: Vec<DiffVar>,
}

/// Re-turn the source page at the degrees whose bounding differentials became
/// determined under `d_result`, giving corrected target-page data.
///
/// `new_vars` is filled in later by [`collect_new_vars`] (it needs the overlay
/// page's dimensions and exclusions).
pub fn turn_page_local(
    source_page: &SATPage,
    target_page: &SATPage,
    d_result: &SATResult,
    source_index: Option<&SourceIndex>,
) -> Result<LocalTurn, TrialError> {
    let r = source_page.r;
    let (new_exclude, new_target_only) = make_next_exclude_set(source_page, d_result);
    let mut unexclude: Vec<Tridegree> = target_page
        .exclude_set
        .difference(&new_exclude)
        .copied()
        .collect();
    unexclude.sort();

    let ctx = TurnContext::new(source_page, d_result);
    let mut new_dims = HashMap::new();
    let mut tbs = HashMap::new();

    let turn_at = |t: Tridegree,
                       new_dims: &mut HashMap<Tridegree, usize>,
                       tbs: &mut HashMap<Tridegree, Arc<TurnedBidegree>>|
     -> Result<(), TrialError> {
        let tb = ctx.get_tb(t).map_err(|e| TrialError::D2 {
            r,
            degree: e.degree,
        })?;
        new_dims.insert(t, tb.as_ref().map_or(0, |tb| tb.dim()));
        if let Some(tb) = tb {
            tbs.insert(t, Arc::new(tb));
        }
        Ok(())
    };

    for &t in &unexclude {
        turn_at(t, &mut new_dims, &mut tbs)?;
        // Past-stable degrees are not in the exclude list but their homology
        // still has to be corrected (same (s, f), larger n).
        if t.n == t.s + 2 {
            let stable_copies: Vec<Tridegree> = match source_index {
                Some(idx) => idx
                    .stable_by_sf
                    .get(&(t.s, t.f))
                    .map(|v| v.clone())
                    .unwrap_or_default(),
                None => source_page
                    .page
                    .keys()
                    .filter(|x| x.n > x.s + 2 && x.s == t.s && x.f == t.f)
                    .copied()
                    .collect(),
            };
            for x in stable_copies {
                turn_at(x, &mut new_dims, &mut tbs)?;
            }
        }
    }

    debug!(
        "turn_page_local: un-excluding {} degrees on E_{}",
        new_dims.len(),
        target_page.r
    );

    Ok(LocalTurn {
        unexclude,
        new_exclude,
        new_target_only,
        new_dims,
        tbs,
        new_vars: Vec::new(),
    })
}

/// Enumerate the new differential variables made possible by the un-excluded
/// degrees. Every un-excluded degree can be the source or the target of a
/// d_{r+1} (plus the stable neighbour for degrees at the stable edge).
pub fn collect_new_vars(overlay: &SATPage, existing: &SATResult, unexclude: &[Tridegree]) -> Vec<DiffVar> {
    let r1 = overlay.r;
    let mut seen: HashSet<DiffVar> = HashSet::new();
    let mut new_vars = Vec::new();
    for &t in unexclude {
        let mut candidates = vec![
            t,
            Tridegree::new(t.n, t.s + 1, t.f - r1),
        ];
        if t.n == t.s + 2 {
            candidates.push(Tridegree::new(t.n + 1, t.s + 1, t.f - r1));
        }
        for c in candidates.drain(..) {
            for v in make_basis_single(overlay, c) {
                if !existing.var_index.contains_key(&v) && seen.insert(v) {
                    new_vars.push(v);
                }
            }
        }
    }
    new_vars.sort();
    new_vars
}

// =============================================================================
// Overlay page (ports overlay_ss.py::OverlaySATPage, materialized eagerly)
// =============================================================================

/// Build a corrected copy of the target page: patched dimensions and
/// exclusions at un-excluded degrees, with products and maps recomputed
/// wherever a recomputed degree participates.
///
/// `d_result` is the (updated) solve result for `source_page`. Every turned
/// degree the recompute needs is derived from it — startup turned data is
/// never consulted, because it is expressed in the *stock* page's quotient
/// coordinates, which the trial's newly determined differentials invalidate
/// (the original recomputes every needed TurnedBidegree from
/// `prev_d_result` via `tb_cache` for the same reason). Returns `Err` only
/// for a genuine d∘d ≠ 0 contradiction found while re-turning.
pub fn build_overlay_page(
    source_page: &SATPage,
    target_page: &SATPage,
    lt: &LocalTurn,
    d_result: &SATResult,
    shared: Option<&SharedTurn>,
    index: Option<&PageIndex>,
) -> Result<SATPage, TrialError> {
    // Cheap since ProductTable/MapTable blocks are Arc-shared (the deep-copy
    // version of this clone was ~50s per page at max_t=50) and pairs/names
    // are skipped (nothing on this path reads them).
    let mut overlay = target_page.overlay_clone();
    overlay.exclude_set = lt.new_exclude.clone();
    overlay.target_only_exclude = lt.new_target_only.clone();

    // Apply the dimension patch, tracking how the key set differs from the
    // stock page so the stock-page indices below can be used with a delta.
    let mut added: Vec<Tridegree> = Vec::new();
    let mut removed: HashSet<Tridegree> = HashSet::new();
    for (&t, &d) in &lt.new_dims {
        if d > 0 {
            if !target_page.page.contains_key(&t) {
                added.push(t);
            }
            overlay.dimension.insert(t, d);
            overlay
                .page
                .insert(t, (0..d).map(|i| Element::basis(t, d, i)).collect());
        } else {
            if target_page.page.contains_key(&t) {
                removed.insert(t);
            }
            overlay.dimension.remove(&t);
            overlay.page.remove(&t);
        }
    }

    // Trial-invariant indices over the stock target page: reused across the
    // whole sweep when provided, otherwise built for this call (same cost as
    // the full-page scans this replaces).
    let owned_index;
    let idx = match index {
        Some(i) => i,
        None => {
            owned_index = PageIndex::new(target_page);
            &owned_index
        }
    };

    // Degrees whose data must be recomputed (unexclude list, rep form).
    let recompute: HashSet<Tridegree> = lt.unexclude.iter().copied().collect();
    let is_recomputable = |t: Tridegree| recompute.contains(&stable_rep(t));

    // ---- products ----
    // Drop every stale block a recomputed degree participates in, then
    // recompute the corresponding triples. (The overlay's blocks are the
    // stock page's at this point, so the stock index is exact.)
    let mut seen_blocks: HashSet<(Tridegree, Tridegree)> = HashSet::new();
    for rep in &recompute {
        if let Some(blocks) = idx.blocks_by_rep.get(rep) {
            for &(d1, d2) in blocks {
                if seen_blocks.insert((d1, d2)) {
                    overlay.products.remove_block(d1, d2);
                }
            }
        }
    }

    // Live page keys at a given n (resp. n+s): the stock list minus removed
    // keys, plus patch-added keys. Iterated in place — building filtered
    // copies per lookup dominated the whole sweep at large max_t.
    let partners_n = |n: i32| {
        idx.by_n
            .get(&n)
            .into_iter()
            .flatten()
            .copied()
            .filter(|t| !removed.contains(t))
            .chain(added.iter().copied().filter(move |t| t.n == n))
    };
    let partners_ns = |ns: i32| {
        idx.by_ns
            .get(&ns)
            .into_iter()
            .flatten()
            .copied()
            .filter(|t| !removed.contains(t))
            .chain(added.iter().copied().filter(move |t| t.n + t.s == ns))
    };

    // Recomputed degrees in actual (non-rep) form, stable copies included.
    let mut recompute_actual: Vec<Tridegree> = Vec::new();
    for rep in &recompute {
        if let Some(keys) = idx.keys_by_rep.get(rep) {
            recompute_actual.extend(keys.iter().copied().filter(|t| !removed.contains(t)));
        }
    }
    recompute_actual.extend(added.iter().copied().filter(|&t| is_recomputable(t)));

    // Generate candidate triples directly from the recomputed degrees: each
    // can appear as x, as y (y.n = x.n + x.s), or as the product
    // xy = (x.n, x.s + y.s, x.f + y.f). The filters below are identical to
    // the full enumeration in compute_induced_products, so the recomputed
    // set is the same.
    let mut candidates: Vec<(Tridegree, Tridegree)> = Vec::new();
    let mut seen_pairs: HashSet<(Tridegree, Tridegree)> = HashSet::new();
    for &d in &recompute_actual {
        // d as x: y ranges over degrees at sphere x.n + x.s.
        for y in partners_n(d.n + d.s) {
            if seen_pairs.insert((d, y)) {
                candidates.push((d, y));
            }
        }
        // d as y: x ranges over degrees with x.n + x.s = y.n.
        for x in partners_ns(d.n) {
            if seen_pairs.insert((x, d)) {
                candidates.push((x, d));
            }
        }
        // d as xy: x shares d's sphere; y is determined by the degree sum.
        for x in partners_n(d.n) {
            let y = Tridegree::new(x.n + x.s, d.s - x.s, d.f - x.f);
            if overlay.page.contains_key(&y) && seen_pairs.insert((x, y)) {
                candidates.push((x, y));
            }
        }
    }

    // Map recomputation targets (also determine which turned degrees we need):
    // squares whose source or target rep is recomputed, from the stock index
    // plus the patch-added keys.
    let mut map_affected: Vec<(MapKind, Tridegree, Tridegree)> = Vec::new();
    let mut seen_squares: HashSet<(MapKind, Tridegree)> = HashSet::new();
    for rep in &recompute {
        if let Some(squares) = idx.map_squares_by_rep.get(rep) {
            for &(kind, t, tgt) in squares {
                if !removed.contains(&t) && seen_squares.insert((kind, t)) {
                    map_affected.push((kind, t, tgt));
                }
            }
        }
    }
    for &t in &added {
        for kind in MapKind::all() {
            if !kind.domain_check(t) {
                continue;
            }
            let tgt = kind.target_degree(t);
            if (is_recomputable(t) || is_recomputable(tgt)) && seen_squares.insert((kind, t)) {
                map_affected.push((kind, t, tgt));
            }
        }
    }

    // Apply the cheap product filters (identical to the ones the full
    // enumeration in compute_induced_products uses) *before* collecting the
    // turned degrees the surviving triples need, so rejected triples never
    // cost a page turn. No n-based bound here (matches Python and the fixed
    // compute_induced_products): max_t bounds t = s + f only, never n.
    candidates.retain(|&(x, y)| {
        if x.s == 0 && x.f == 0 {
            return false;
        }
        let xy = Tridegree::new(x.n, x.s + y.s, x.f + y.f);
        if !(is_recomputable(x) || is_recomputable(y) || is_recomputable(xy)) {
            return false;
        }
        let shifted = Tridegree::new(y.n - 1, y.s, y.f);
        if !overlay.page.contains_key(&shifted) {
            return false;
        }
        [x, y, xy]
            .iter()
            .all(|s| overlay.is_in_computed_polygon_source(*s))
    });

    // Turn every degree the recompute touches, freshly from the updated
    // result, so all quotient coordinates are consistent by construction.
    // `lt.tbs` (computed with the same context in turn_page_local) serves as
    // a cache. Reusing startup turned data here mixed stock-page coordinates
    // with overlay vectors — the source of the `mat_mul_vec` sweep panics,
    // and silently wrong when the dimensions happened to agree.
    //
    // When a sweep-shared cache is available, degrees whose differential
    // inputs the trial did not touch are read from it instead of re-turned:
    // for those degrees the trial's updated result agrees with the base
    // result, so the turned data is identical (see [`TbCache`]).
    let mut needed: HashSet<Tridegree> = HashSet::new();
    for &(x, y) in &candidates {
        needed.insert(x);
        needed.insert(y);
        needed.insert(Tridegree::new(x.n, x.s + y.s, x.f + y.f));
    }
    for &(_, src, tgt) in &map_affected {
        needed.insert(src);
        needed.insert(tgt);
    }
    let ctx = TurnContext::new(source_page, d_result);
    let mut tbs: HashMap<Tridegree, Arc<TurnedBidegree>> = HashMap::new();
    for &t in &needed {
        if source_page.dim_at(t) == 0 {
            continue;
        }
        if let Some(tb) = lt.tbs.get(&t) {
            tbs.insert(t, tb.clone());
            continue;
        }
        if let Some(shared) = shared {
            if let Some(cached) = shared.get_clean(t, source_page.r) {
                if let Some(tb) = cached {
                    tbs.insert(t, tb);
                }
                continue;
            }
        }
        if let Some(tb) = ctx.get_tb(t).map_err(|e| TrialError::D2 {
            r: source_page.r,
            degree: e.degree,
        })? {
            tbs.insert(t, Arc::new(tb));
        }
    }
    let tb_at = |t: Tridegree| tbs.get(&t);

    for (x, y) in candidates {
        let xy = Tridegree::new(x.n, x.s + y.s, x.f + y.f);
        let (Some(tb_x), Some(tb_y), Some(tb_xy)) = (tb_at(x), tb_at(y), tb_at(xy)) else {
            continue;
        };
        let dim1 = overlay.dim_at(x);
        let dim2 = overlay.dim_at(y);
        for (ex, ey, exy) in compute_induced_products_single(tb_x, tb_y, tb_xy, source_page) {
            let xi: Vec<usize> = vec_support(&ex.vec).collect();
            let yi: Vec<usize> = vec_support(&ey.vec).collect();
            if xi.len() == 1 && yi.len() == 1 {
                let key = ProductKey::new(x, xi[0] as u16, y, yi[0] as u16);
                overlay.products.insert(key, exy.vec.clone(), dim1, dim2);
            }
        }
    }

    // ---- maps ----
    for (kind, src, tgt) in map_affected {
        let entries = match (tb_at(src), tb_at(tgt)) {
            (Some(tb_src), Some(tb_tgt)) => {
                compute_induced_map_single_tb(src, kind, source_page, tb_src, tb_tgt)
            }
            _ => Vec::new(),
        };
        let src_dim = overlay.dim_at(src);
        let tgt_dim = overlay.dim_at(tgt);
        let map_table = overlay.maps.get_mut(&kind).unwrap();
        if src_dim == 0 || tgt_dim == 0 {
            map_table.matrices.remove(&src);
            continue;
        }
        let mut mat = mat_zero(src_dim, tgt_dim);
        let mut any = false;
        for (e_src, e_tgt) in &entries {
            let si: Vec<usize> = vec_support(&e_src.vec).collect();
            if si.len() == 1 && si[0] < src_dim && e_tgt.vec.len() == tgt_dim {
                mat_set_row(&mut mat, si[0], &e_tgt.vec);
                any = true;
            }
        }
        if any {
            map_table.set_matrix(src, mat);
        } else {
            map_table.matrices.remove(&src);
        }
    }

    Ok(overlay)
}

// =============================================================================
// New constraints for un-excluded degrees (ports interpage.py::make_new_constraints)
// =============================================================================

/// Generate the naturality and re-activated Leibniz constraints related to the
/// un-excluded degrees, with variable indices in the extended space.
pub fn make_new_constraints(
    overlay: &mut SATPage,
    source_max_t: Option<i32>,
    source_r: i32,
    unexclude: &[Tridegree],
    excluded_leibniz: &ExcludedLeibniz,
    ext_var_index: &HashMap<DiffVar, usize>,
    stale_blocks: &HashSet<Tridegree>,
) -> Vec<(Vec<usize>, bool)> {
    // The original restricts constraint generation to the region that is
    // trustworthy after turning: cutoff = source.max_t - (r - 1).
    let saved_max_t = overlay.max_t;
    overlay.max_t = source_max_t.map(|t| t - (source_r - 1));

    let r1 = overlay.r;
    let mut cons: Vec<Vec<usize>> = Vec::new();

    // `stale_blocks` are variable blocks that already exist in the base
    // system but whose degree's coordinates changed in this trial (possible
    // only with the target-only exclusion relaxation, where variables exist
    // at excluded degrees). New constraints must not reference them — the
    // base variables are expressed in the stock (over-kept) basis, while the
    // overlay's data at those degrees is in the recomputed basis; mixing the
    // two would silently corrupt the system. Empty when the relaxation is
    // off, so this filter is a no-op for the strict behavior.
    let block_stale = |u: Tridegree| stale_blocks.contains(&stable_rep(u));

    for &t in unexclude {
        for kind in MapKind::all() {
            let deg = t;
            let diff_deg = Tridegree::new(t.n, t.s + 1, t.f - r1);
            // Same convention as the naturality generator: the P square is
            // anchored at its target, all others at their source.
            let source_deg = match kind {
                MapKind::P => kind.target_degree(t),
                _ => kind.source_degree(t),
            };
            let source_diff_deg =
                Tridegree::new(source_deg.n, source_deg.s + 1, source_deg.f - r1);

            for d in [deg, diff_deg, source_deg, source_diff_deg] {
                // The square anchored at `d` references the variable blocks
                // at its source and target degrees (same derivation as
                // make_naturality_constraint_single).
                let (sq_src, sq_tgt) = match kind {
                    MapKind::P => (Tridegree::new(2 * d.n + 1, d.s - d.n + 1, d.f - 2), d),
                    _ => (d, kind.target_degree(d)),
                };
                if block_stale(sq_src) || block_stale(sq_tgt) {
                    continue;
                }
                cons.extend(make_naturality_constraint_single(
                    overlay,
                    d,
                    kind,
                    ext_var_index,
                ));
            }
        }
    }

    for &t in unexclude {
        if let Some(pairs) = excluded_leibniz.get(&t) {
            for &(deg1, deg2) in pairs {
                // The pair references the blocks at prod_deg, deg1 and deg2.
                let prod = Tridegree::new(deg1.n, deg1.s + deg2.s, deg1.f + deg2.f);
                if block_stale(prod) || block_stale(deg1) || block_stale(deg2) {
                    continue;
                }
                cons.extend(make_leibniz_constraint_single(
                    overlay,
                    deg1,
                    deg2,
                    ext_var_index,
                    None,
                ));
            }
        }
    }

    overlay.max_t = saved_max_t;

    // Deduplicate (all homogeneous).
    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    let mut out = Vec::new();
    for mut c in cons {
        c.sort_unstable();
        if seen.insert(c.clone()) {
            out.push((c, false));
        }
    }
    out
}

// =============================================================================
// Assumption propagation driver (ports run_interpage.py::try_diffs_from_list)
// =============================================================================

/// Assume values for differentials on `pages[0]` and propagate the
/// consequences through the remaining pages. Returns every newly determined
/// differential, or the contradiction that the assumption produced.
pub fn try_diffs(
    pages: &[InterpagePage],
    assumptions: &[(DiffVar, bool)],
) -> Result<Vec<LearnedDiff>, TrialError> {
    try_diffs_with_cache(pages, assumptions, None)
}

/// [`try_diffs`] with an optional sweep-shared [`SweepCache`] built from
/// `pages`. The turned-degree cache is consulted only for the first page step
/// (whose source page and base result are trial-invariant) and only at
/// degrees outside the trial's dirty set — the degrees of the assumed and
/// learned variables, which are exactly where the trial's updated result can
/// differ from the base result. The stock-page indices are used on every
/// step (they depend only on the fixed target pages).
pub fn try_diffs_with_cache(
    pages: &[InterpagePage],
    assumptions: &[(DiffVar, bool)],
    cache: Option<&SweepCache>,
) -> Result<Vec<LearnedDiff>, TrialError> {
    assert!(!pages.is_empty());
    let first = &pages[0];
    let r0 = first.page.r;

    let mut cons = Vec::with_capacity(assumptions.len());
    for &(var, val) in assumptions {
        let idx = *first
            .result
            .var_index
            .get(&var)
            .ok_or(TrialError::UnknownVar(var))?;
        cons.push((vec![idx], val));
    }

    let (learned0, mut dsat) =
        update_sat_result(first.result, &cons, &[]).ok_or(TrialError::Unsat { r: r0 })?;
    let mut learned_all: Vec<LearnedDiff> = learned0
        .into_iter()
        .map(|(var, value)| LearnedDiff { r: r0, var, value })
        .collect();

    // The trial's updated first-page result differs from the base result
    // exactly at the assumed + learned variables; their degrees form the
    // dirty set outside which the shared cache is valid.
    let dirty: HashSet<Tridegree> = assumptions
        .iter()
        .map(|(v, _)| stable_rep(Tridegree::new(v.n, v.s, v.f)))
        .chain(
            learned_all
                .iter()
                .map(|l| stable_rep(Tridegree::new(l.var.n, l.var.s, l.var.f))),
        )
        .collect();
    let shared = cache.map(|c| SharedTurn {
        cache: &c.tb,
        dirty: &dirty,
    });

    let mut prev_overlay: Option<SATPage> = None;

    for i in 1..pages.len() {
        let target = &pages[i];
        let source_page: &SATPage = prev_overlay.as_ref().unwrap_or(first.page);
        // The turned-degree cache and source index are built from the stock
        // first page (+ base result); they are only valid while that page is
        // the turning source (the first step). The target-page index depends
        // only on the stock target page, which every step uses.
        let first_step = prev_overlay.is_none();
        let step_shared = if first_step { shared.as_ref() } else { None };
        let step_source_index = if first_step {
            cache.map(|c| &c.source_index)
        } else {
            None
        };
        let step_index = cache.and_then(|c| c.target_indexes.get(i - 1));

        let mut lt = turn_page_local(source_page, target.page, &dsat, step_source_index)?;
        if lt.unexclude.is_empty() {
            // No degree became certain: the target page's system gains no new
            // constraints, so nothing can change on this or any higher page.
            break;
        }
        let mut overlay =
            build_overlay_page(source_page, target.page, &lt, &dsat, step_shared, step_index)?;

        lt.new_vars = collect_new_vars(&overlay, target.result, &lt.unexclude);

        let old_n = target.result.vars.len();
        let mut ext_index = target.result.var_index.clone();
        for (j, v) in lt.new_vars.iter().enumerate() {
            ext_index.insert(*v, old_n + j);
        }

        // Pre-existing variable blocks whose coordinates this trial changed
        // (target-only-relaxed degrees whose dimension shifted): equal dims
        // guarantee an identical recomputed basis (kernels only shrink and
        // boundaries only grow as entries become determined, so equal
        // homology dimension means identical spans, and the echelon bases
        // are canonical); a changed dim means the base variables are in a
        // different basis than the overlay. New constraints must skip them.
        let r1 = target.page.r;
        let changed: HashSet<Tridegree> = lt
            .new_dims
            .iter()
            .filter(|(&t, &d)| d != target.page.dim_at(t))
            .map(|(&t, _)| stable_rep(t))
            .collect();
        let mut stale_blocks: HashSet<Tridegree> = HashSet::new();
        for &c in &changed {
            for u in [c, Tridegree::new(c.n, c.s + 1, c.f - r1)] {
                let urep = stable_rep(u);
                if target
                    .result
                    .var_index
                    .contains_key(&DiffVar::new(urep.n, urep.s, urep.f, 0, 0))
                {
                    stale_blocks.insert(urep);
                }
            }
        }

        let new_cons = make_new_constraints(
            &mut overlay,
            source_page.max_t,
            source_page.r,
            &lt.unexclude,
            target.excluded_leibniz,
            &ext_index,
            &stale_blocks,
        );
        debug!(
            "try_diffs: E_{}: {} un-excluded degrees, {} new vars, {} new constraints",
            target.page.r,
            lt.unexclude.len(),
            lt.new_vars.len(),
            new_cons.len()
        );

        let (learned, next_dsat) = update_sat_result(target.result, &new_cons, &lt.new_vars)
            .ok_or(TrialError::Unsat {
                r: target.page.r,
            })?;
        learned_all.extend(learned.into_iter().map(|(var, value)| LearnedDiff {
            r: target.page.r,
            var,
            value,
        }));

        dsat = next_dsat;
        prev_overlay = Some(overlay);
    }

    Ok(learned_all)
}

// =============================================================================
// Trial-and-error sweep (ports run_interpage.py::trial_error_loop)
// =============================================================================

/// Outcome of testing both values of one unknown differential.
#[derive(Clone, Debug)]
pub struct SweepFinding {
    pub var: DiffVar,
    /// `Some(v)`: assuming `v` led to a contradiction, so the value is forced
    /// to `!v`. Reported per contradicting value; if both values contradict,
    /// two findings are produced (the base system itself is inconsistent!).
    pub contradicted_value: bool,
    pub error: TrialError,
}

/// For every unknown differential on `pages[0]` (optionally restricted to a
/// stem range), assume each value in {0, 1} and propagate. Contradictions mean
/// the opposite value is forced. Runs trials in parallel.
pub fn trial_error_sweep(
    pages: &[InterpagePage],
    min_stem: i32,
    max_stem: i32,
    progress: impl Fn(usize, usize) + Sync,
) -> Vec<SweepFinding> {
    let first = &pages[0];
    let mut unknown_vars: Vec<DiffVar> = first
        .result
        .unknown
        .iter()
        .map(|&idx| first.result.vars[idx])
        .filter(|v| v.s >= min_stem && v.s < max_stem)
        .collect();
    unknown_vars.sort();

    let jobs: Vec<(DiffVar, bool)> = unknown_vars
        .iter()
        .flat_map(|&v| [(v, false), (v, true)])
        .collect();
    let total = jobs.len();
    let done = std::sync::atomic::AtomicUsize::new(0);

    // Turned data at degrees a trial doesn't touch is identical across all
    // trials — compute it once and share it (the bulk of a trial's work) —
    // and precompute the stock-page indices every trial's overlay build needs.
    let cache = SweepCache::new(pages);

    jobs.par_iter()
        .filter_map(|&(var, value)| {
            let result = try_diffs_with_cache(pages, &[(var, value)], Some(&cache));
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            progress(n, total);
            match result {
                Err(e @ (TrialError::Unsat { .. } | TrialError::D2 { .. })) => {
                    Some(SweepFinding {
                        var,
                        contradicted_value: value,
                        error: e,
                    })
                }
                _ => None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::ConstraintSystem;
    use crate::solver::solve;

    fn dv(i: u16) -> DiffVar {
        // Distinct fake variables; the degrees are irrelevant to the algebra.
        DiffVar::new(2, 1 + i as i32, 1, 0, 0)
    }

    /// Solve { x0 ⊕ x1 = 0 } over vars {x0, x1, x2}.
    fn base_result() -> SATResult {
        let vars = vec![dv(0), dv(1), dv(2)];
        let mut sys = ConstraintSystem::new(vars);
        sys.add_constraint_indices(&[0, 1], false);
        solve(&sys).expect("consistent")
    }

    #[test]
    fn solver_marks_correlated_pivots_unknown() {
        // x0 = x1 with x1 free: BOTH are undetermined, as is untouched x2.
        let res = base_result();
        assert_eq!(res.unknown.len(), 3, "x0 is correlated, not determined");
    }

    #[test]
    fn update_determines_correlated_vars() {
        let res = base_result();
        // Assume x1 = 1 → x0 = 1 follows; x2 stays unknown.
        let (learned, upd) =
            update_sat_result(&res, &[(vec![1], true)], &[]).expect("consistent");
        let learned_map: HashMap<DiffVar, bool> =
            learned.iter().map(|&(v, b)| (v, b)).collect();
        assert_eq!(learned_map.get(&dv(0)), Some(&true));
        assert_eq!(learned_map.get(&dv(1)), Some(&true));
        assert!(!learned_map.contains_key(&dv(2)));
        assert!(upd.unknown.contains(&2));
        assert_eq!(upd.unknown.len(), 1);
        assert!(vec_get(&upd.offset, 0));
        assert!(vec_get(&upd.offset, 1));
    }

    #[test]
    fn update_detects_unsat() {
        let res = base_result();
        // x0 = 0 and x1 = 1 contradict x0 ⊕ x1 = 0.
        let out = update_sat_result(&res, &[(vec![0], false), (vec![1], true)], &[]);
        assert!(out.is_none());
    }

    #[test]
    fn update_with_new_vars() {
        let res = base_result();
        // Add x3 with x3 ⊕ x2 = 0: both stay unknown but correlated.
        let (learned, upd) =
            update_sat_result(&res, &[(vec![2, 3], false)], &[dv(3)]).expect("consistent");
        assert!(learned.is_empty());
        assert_eq!(upd.vars.len(), 4);
        assert!(upd.unknown.contains(&2) && upd.unknown.contains(&3));

        // Now pin x2 = 1 → x3 = 1 follows.
        let (learned2, upd2) =
            update_sat_result(&upd, &[(vec![2], true)], &[]).expect("consistent");
        let learned_map: HashMap<DiffVar, bool> =
            learned2.iter().map(|&(v, b)| (v, b)).collect();
        assert_eq!(learned_map.get(&dv(2)), Some(&true));
        assert_eq!(learned_map.get(&dv(3)), Some(&true));
        assert_eq!(upd2.unknown.len(), 2, "x0, x1 still correlated-unknown");
    }
}
