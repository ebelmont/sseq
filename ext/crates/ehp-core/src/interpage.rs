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
use crate::page::{OverlayBases, SATPage};
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

    let m = mat_from_rows(&m_rows, kaug.len());
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
        mat_from_rows(&new_kernel_rows, total_n)
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
    /// Arc'd table snapshots for each target page `pages[1..]` — the base
    /// layer every trial's thin overlay reads through. One full table clone
    /// per page per sweep instead of per trial page step.
    pub target_bases: Vec<OverlayBases>,
}

impl SweepCache {
    pub fn new(pages: &[InterpagePage]) -> Self {
        let first = &pages[0];
        SweepCache {
            tb: TbCache::new(first.page, first.result),
            source_index: SourceIndex::new(first.page),
            target_indexes: pages[1..].iter().map(|p| PageIndex::new(p.page)).collect(),
            target_bases: pages[1..].iter().map(|p| OverlayBases::new(p.page)).collect(),
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
    bases: Option<&OverlayBases>,
) -> Result<SATPage, TrialError> {
    // THIN overlay: a layered view over the stock tables — O(patch) to build
    // and drop, vs the old eager `overlay_clone` (~2M-entry HashMap clone
    // plus the matching teardown per page step, the measured 76–91% of trial
    // CPU at t=100). Sweeps share `bases` across all trials; the uncached
    // path builds one here (same cost as the eager clone it replaces).
    let owned_bases;
    let bases = match bases {
        Some(b) => b,
        None => {
            owned_bases = OverlayBases::new(target_page);
            &owned_bases
        }
    };
    let overlay = target_page.thin_overlay(bases);
    build_overlay_page_from(overlay, source_page, target_page, lt, d_result, shared, index)
}

/// The eager-clone form of [`build_overlay_page`] — the pre-thin-overlay
/// behavior, kept as the ground truth for `EHP_TRIAL_VERIFY`.
pub fn build_overlay_page_eager(
    source_page: &SATPage,
    target_page: &SATPage,
    lt: &LocalTurn,
    d_result: &SATResult,
    shared: Option<&SharedTurn>,
    index: Option<&PageIndex>,
) -> Result<SATPage, TrialError> {
    build_overlay_page_from(
        target_page.overlay_clone(),
        source_page,
        target_page,
        lt,
        d_result,
        shared,
        index,
    )
}

fn build_overlay_page_from(
    mut overlay: SATPage,
    source_page: &SATPage,
    target_page: &SATPage,
    lt: &LocalTurn,
    d_result: &SATResult,
    shared: Option<&SharedTurn>,
    index: Option<&PageIndex>,
) -> Result<SATPage, TrialError> {
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
    // compute_induced_products): max_t bounds t = s + f only, never n; the
    // product bound is the flat source_page.max_t - 1 (see the schedule
    // comment on compute_induced_products), and there is no shifted-degree
    // or polygon filter — the full enumeration dropped both, and a stricter
    // pre-filter here would make overlay recomputes silently miss products
    // a fresh page build now includes.
    let product_max_t = source_page.max_t.map(|t| t - 1);
    candidates.retain(|&(x, y)| {
        if x.s == 0 && x.f == 0 {
            return false;
        }
        let xy = Tridegree::new(x.n, x.s + y.s, x.f + y.f);
        if !(is_recomputable(x) || is_recomputable(y) || is_recomputable(xy)) {
            return false;
        }
        [x, y, xy]
            .iter()
            .all(|s| product_max_t.is_none_or(|mt| s.s + s.f <= mt))
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
            map_table.remove_matrix(src);
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
            map_table.remove_matrix(src);
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
// Thin-overlay verification (EHP_TRIAL_VERIFY)
// =============================================================================

/// `EHP_TRIAL_VERIFY=1`: per page step, build BOTH the thin and the eager
/// overlay and cross-check them (see [`verify_overlay_step`]).
fn trial_verify_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("EHP_TRIAL_VERIFY").is_ok_and(|v| !v.is_empty() && v != "0")
    })
}

/// Panic (with per-entry diffs) unless `thin` and `full` are pointwise
/// identical as pages AND produce identical `collect_new_vars` /
/// `make_new_constraints` outputs. Constraint lists are compared as SETS —
/// the two pages' internal hash layouts differ, and generator output order
/// (never the content) can depend on them.
#[allow(clippy::too_many_arguments)]
fn verify_overlay_step(
    thin: &SATPage,
    full: &SATPage,
    lt: &LocalTurn,
    target: &InterpagePage,
    source_page: &SATPage,
    ext_index: &HashMap<DiffVar, usize>,
    stale_blocks: &HashSet<Tridegree>,
    thin_cons: &[(Vec<usize>, bool)],
) {
    let mut errs: Vec<String> = Vec::new();
    let d = |t: Tridegree| format!("({},{},{})", t.n, t.s, t.f);

    for (&t, &dim) in &full.dimension {
        if thin.dim_at(t) != dim {
            errs.push(format!("dim{}: thin {} vs full {}", d(t), thin.dim_at(t), dim));
        }
    }
    for (&t, &dim) in &thin.dimension {
        if full.dim_at(t) != dim {
            errs.push(format!("dim{}: thin {} vs full {}", d(t), dim, full.dim_at(t)));
        }
    }
    for t in full.page.keys() {
        if !thin.page.contains_key(t) {
            errs.push(format!("page key {} missing in thin", d(*t)));
        }
    }
    for t in thin.page.keys() {
        if !full.page.contains_key(t) {
            errs.push(format!("page key {} extra in thin", d(*t)));
        }
    }
    if thin.exclude_set != full.exclude_set {
        errs.push("exclude_set differs".to_string());
    }
    if thin.target_only_exclude != full.target_only_exclude {
        errs.push("target_only_exclude differs".to_string());
    }

    for (&(d1, d2), pb) in full.products.iter_blocks() {
        match thin.products.block(d1, d2) {
            Some(pa) if pa == pb => {}
            Some(_) => errs.push(format!("product block {}x{} differs", d(d1), d(d2))),
            None => errs.push(format!("product block {}x{} missing in thin", d(d1), d(d2))),
        }
    }
    for (&(d1, d2), _) in thin.products.iter_blocks() {
        if full.products.block(d1, d2).is_none() {
            errs.push(format!("product block {}x{} extra in thin", d(d1), d(d2)));
        }
    }

    for (kind, fmt) in &full.maps {
        let Some(tmt) = thin.maps.get(kind) else {
            errs.push(format!("{:?} map table missing in thin", kind));
            continue;
        };
        for (t, arc) in fmt.iter() {
            match tmt.matrix_at(*t) {
                Some(m) if m == arc.as_ref() => {}
                Some(_) => errs.push(format!("{:?} map at {} differs", kind, d(*t))),
                None => errs.push(format!("{:?} map at {} missing in thin", kind, d(*t))),
            }
        }
        for (t, _) in tmt.iter() {
            if !fmt.contains(*t) {
                errs.push(format!("{:?} map at {} extra in thin", kind, d(*t)));
            }
        }
    }

    // Derived outputs from the eager page.
    let full_vars = collect_new_vars(full, target.result, &lt.unexclude);
    if full_vars != lt.new_vars {
        errs.push(format!(
            "collect_new_vars differs: thin {} vs full {} vars",
            lt.new_vars.len(),
            full_vars.len()
        ));
    }
    let mut full_m = full.clone();
    let full_cons = make_new_constraints(
        &mut full_m,
        source_page.max_t,
        source_page.r,
        &lt.unexclude,
        target.excluded_leibniz,
        ext_index,
        stale_blocks,
    );
    let as_set = |cons: &[(Vec<usize>, bool)]| -> HashSet<(Vec<usize>, bool)> {
        cons.iter().cloned().collect()
    };
    let (ts, fs) = (as_set(thin_cons), as_set(&full_cons));
    for c in fs.difference(&ts) {
        errs.push(format!("constraint only in full: {:?}", c));
    }
    for c in ts.difference(&fs) {
        errs.push(format!("constraint only in thin: {:?}", c));
    }

    if !errs.is_empty() {
        for e in &errs {
            eprintln!("[EHP_TRIAL_VERIFY] {}", e);
        }
        panic!(
            "EHP_TRIAL_VERIFY: thin overlay diverges from eager overlay on E_{} step ({} diffs)",
            thin.r,
            errs.len()
        );
    }
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

/// Everything a consistent trial determined, plus the basis bookkeeping a
/// consumer needs to compare trials ACROSS worlds: `dim_changed` holds the
/// stable reps of every degree whose dimension this trial's overlays changed
/// (union over page steps). A learned value at degrees OUTSIDE this set is a
/// statement in the stock partial-page basis (equal dims ⇒ identical
/// canonical echelon bases — see the stale-blocks comment in the trial body),
/// so it is comparable between the two worlds of a switch.
pub struct TrialRun {
    pub learned: Vec<LearnedDiff>,
    pub dim_changed: HashSet<Tridegree>,
    /// Re-turned dimensions per `(page r, degree)` — the world's corrected
    /// dims wherever a page step recomputed them (`LocalTurn::new_dims`).
    /// Consumed by the zero-map consensus (a dim of 0 = the source classes
    /// are dead in this world, so its differential is vacuously zero).
    pub new_dims: HashMap<(i32, Tridegree), usize>,
    /// The trial's final updated solve result per page reached: `(r, result)`
    /// for the swept page and every stepped page. Pages beyond the last
    /// entry were never stepped — their state is the base result. Consumed
    /// by the possibility-set consensus (block projections of the final
    /// solution spaces); moves, never clones, so it is essentially free.
    pub final_results: Vec<(i32, SATResult)>,
}

/// Per-stage trial profiling: lock-free atomic sums over every
/// [`try_diffs_with_cache`] call since the last report. Overhead is a handful
/// of `Instant::now()` calls per trial — negligible against trial cost — so
/// collection is always on; only reporting is caller-gated. The
/// assumed-1 / learned-1 splits price the "zero-stratum" optimization: a
/// trial whose assumed AND learned values are all 0 changes no turned data
/// (unknowns already turn as 0), so its turn/overlay stage time is in
/// principle removable.
pub mod trial_stats {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub static TRIALS: AtomicU64 = AtomicU64::new(0);
    pub static ASSUMED_ONE: AtomicU64 = AtomicU64::new(0);
    pub static LEARNED_ONE: AtomicU64 = AtomicU64::new(0);
    /// Assumed all-0 AND learned only 0s: the trials whose turn/overlay work
    /// is provably a no-op (nothing turning-effective changed).
    pub static ZERO_STRATUM: AtomicU64 = AtomicU64::new(0);
    pub static CONTRADICTED: AtomicU64 = AtomicU64::new(0);
    pub static PAGE_STEPS: AtomicU64 = AtomicU64::new(0);
    pub static NS_FIRST_SOLVE: AtomicU64 = AtomicU64::new(0);
    pub static NS_TURN: AtomicU64 = AtomicU64::new(0);
    pub static NS_OVERLAY: AtomicU64 = AtomicU64::new(0);
    pub static NS_CONSTRAINTS: AtomicU64 = AtomicU64::new(0);
    pub static NS_SOLVE: AtomicU64 = AtomicU64::new(0);
    /// Dropping the per-step overlay pages (~2M Arc'd product blocks each):
    /// suspected owner of the time the other stages don't account for.
    pub static NS_DROP: AtomicU64 = AtomicU64::new(0);
    pub static NS_TOTAL: AtomicU64 = AtomicU64::new(0);

    pub fn add(counter: &AtomicU64, ns: u128) {
        counter.fetch_add(ns as u64, Relaxed);
    }

    /// Human-readable report of everything since the last call, then reset.
    /// `None` if no trials ran.
    pub fn report_and_reset() -> Option<String> {
        let trials = TRIALS.swap(0, Relaxed);
        if trials == 0 {
            return None;
        }
        let s = |c: &AtomicU64| c.swap(0, Relaxed) as f64 / 1e9;
        let n = |c: &AtomicU64| c.swap(0, Relaxed);
        Some(format!(
            "{} trials ({} assumed-1, {} learned-1, {} zero-stratum, {} contradicted), \
             {} page steps; \
             cpu-side: first-solve {:.1}s, turn {:.1}s, overlay {:.1}s, constraints {:.1}s, \
             re-solve {:.1}s, teardown {:.1}s, total {:.1}s",
            trials,
            n(&ASSUMED_ONE),
            n(&LEARNED_ONE),
            n(&ZERO_STRATUM),
            n(&CONTRADICTED),
            n(&PAGE_STEPS),
            s(&NS_FIRST_SOLVE),
            s(&NS_TURN),
            s(&NS_OVERLAY),
            s(&NS_CONSTRAINTS),
            s(&NS_SOLVE),
            s(&NS_DROP),
            s(&NS_TOTAL),
        ))
    }
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
    try_diffs_full(pages, assumptions, cache).map(|run| run.learned)
}

/// [`try_diffs_with_cache`] returning the full [`TrialRun`].
pub fn try_diffs_full(
    pages: &[InterpagePage],
    assumptions: &[(DiffVar, bool)],
    cache: Option<&SweepCache>,
) -> Result<TrialRun, TrialError> {
    use std::time::Instant;
    let t_trial = Instant::now();
    trial_stats::TRIALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if assumptions.iter().any(|&(_, v)| v) {
        trial_stats::ASSUMED_ONE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let out = try_diffs_with_cache_inner(pages, assumptions, cache);
    match &out {
        Ok(run) => {
            // Learned-1 means a value-1 determination BEYOND the assumptions
            // themselves (the assumed var reappears in its own learned list):
            // together with assumed-1 this splits off the "zero stratum" —
            // trials that determine only zeros and therefore change no
            // turned data.
            let r0 = pages[0].page.r;
            let learned_one = run.learned.iter().any(|l| {
                l.value && !(l.r == r0 && assumptions.iter().any(|&(v, _)| v == l.var))
            });
            if learned_one {
                trial_stats::LEARNED_ONE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if !learned_one && assumptions.iter().all(|&(_, v)| !v) {
                trial_stats::ZERO_STRATUM.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        Err(_) => {
            trial_stats::CONTRADICTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    trial_stats::add(&trial_stats::NS_TOTAL, t_trial.elapsed().as_nanos());
    out
}

fn try_diffs_with_cache_inner(
    pages: &[InterpagePage],
    assumptions: &[(DiffVar, bool)],
    cache: Option<&SweepCache>,
) -> Result<TrialRun, TrialError> {
    use std::time::Instant;
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

    let t_s = Instant::now();
    let first_solved = update_sat_result(first.result, &cons, &[]);
    trial_stats::add(&trial_stats::NS_FIRST_SOLVE, t_s.elapsed().as_nanos());
    let (learned0, mut dsat) = first_solved.ok_or(TrialError::Unsat { r: r0 })?;
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
    let mut dim_changed_all: HashSet<Tridegree> = HashSet::new();
    // Final per-page results: `dsat` moves in here whenever the loop replaces
    // it with the next page's result (and once at the end).
    let mut finals: Vec<(i32, SATResult)> = Vec::new();
    let mut new_dims_all: HashMap<(i32, Tridegree), usize> = HashMap::new();
    let mut dsat_r = r0;

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
        let step_bases = cache.and_then(|c| c.target_bases.get(i - 1));

        trial_stats::PAGE_STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let t_turn = Instant::now();
        let mut lt = turn_page_local(source_page, target.page, &dsat, step_source_index)?;
        trial_stats::add(&trial_stats::NS_TURN, t_turn.elapsed().as_nanos());
        if lt.unexclude.is_empty() {
            // No degree became certain: the target page's system gains no new
            // constraints, so nothing can change on this or any higher page.
            break;
        }
        let t_ov = Instant::now();
        let mut overlay = build_overlay_page(
            source_page,
            target.page,
            &lt,
            &dsat,
            step_shared,
            step_index,
            step_bases,
        )?;
        trial_stats::add(&trial_stats::NS_OVERLAY, t_ov.elapsed().as_nanos());

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
        dim_changed_all.extend(changed.iter().copied());
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

        let t_cons = Instant::now();
        let new_cons = make_new_constraints(
            &mut overlay,
            source_page.max_t,
            source_page.r,
            &lt.unexclude,
            target.excluded_leibniz,
            &ext_index,
            &stale_blocks,
        );
        trial_stats::add(&trial_stats::NS_CONSTRAINTS, t_cons.elapsed().as_nanos());
        debug!(
            "try_diffs: E_{}: {} un-excluded degrees, {} new vars, {} new constraints",
            target.page.r,
            lt.unexclude.len(),
            lt.new_vars.len(),
            new_cons.len()
        );

        // EHP_TRIAL_VERIFY=1: rebuild this page step's overlay the old eager
        // way and require the thin overlay to be POINTWISE identical (dims,
        // page keys, exclusions, every visible product block and map matrix,
        // both directions) and to yield identical new_vars/new_cons. This is
        // the guard against the thin overlay's one failure mode — a missed
        // base block silently reading as zero. Panics with per-entry diffs.
        if trial_verify_enabled() {
            let full = build_overlay_page_eager(
                source_page,
                target.page,
                &lt,
                &dsat,
                step_shared,
                step_index,
            )?;
            verify_overlay_step(&overlay, &full, &lt, target, source_page, &ext_index, &stale_blocks, &new_cons);
        }

        let t_solve = Instant::now();
        let solved = update_sat_result(target.result, &new_cons, &lt.new_vars);
        trial_stats::add(&trial_stats::NS_SOLVE, t_solve.elapsed().as_nanos());
        let (learned, next_dsat) = solved.ok_or(TrialError::Unsat {
            r: target.page.r,
        })?;
        learned_all.extend(learned.into_iter().map(|(var, value)| LearnedDiff {
            r: target.page.r,
            var,
            value,
        }));

        new_dims_all.extend(lt.new_dims.iter().map(|(&t, &d)| ((target.page.r, t), d)));
        finals.push((dsat_r, std::mem::replace(&mut dsat, next_dsat)));
        dsat_r = target.page.r;
        let t_drop = Instant::now();
        drop(std::mem::replace(&mut prev_overlay, Some(overlay)));
        trial_stats::add(&trial_stats::NS_DROP, t_drop.elapsed().as_nanos());
    }
    finals.push((dsat_r, dsat));

    let t_drop = Instant::now();
    drop(prev_overlay);
    trial_stats::add(&trial_stats::NS_DROP, t_drop.elapsed().as_nanos());
    Ok(TrialRun {
        learned: learned_all,
        dim_changed: dim_changed_all,
        new_dims: new_dims_all,
        final_results: finals,
    })
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
    trial_error_sweep_full(pages, min_stem, max_stem, progress).contradictions
}

/// A value determined UNCONDITIONALLY by case analysis on one unknown
/// switch: BOTH values of `via` are consistent, and each world independently
/// forces `var = value` on page `r`. Since `via` must be 0 or 1, the value
/// holds regardless — the classic "same in every possible world" argument.
/// Only emitted when neither world changed the dimensions at the learned
/// var's source/target degrees (equal dims ⇒ identical canonical bases, so
/// the two worlds' statements are about the same variable).
#[derive(Clone, Debug)]
pub struct ConsensusFinding {
    pub via: DiffVar,
    pub via_r: i32,
    pub r: i32,
    pub var: DiffVar,
    pub value: bool,
}

/// Possibility set of one differential block, after case analysis over the
/// sweep's switches (the addendum's "possibility-set consensus", tiers 1–2).
///
/// The true d_r matrix at `degree` must lie in `P_w0(u) ∪ P_w1(u)` for EVERY
/// unknown switch u (each world's projected solution-space coset), so the
/// running intersection of those unions — further intersected with the BASE
/// solution's projection — is a sound upper bound on the possible matrices.
/// Per-entry consensus only sees "same singleton in both worlds"; this
/// carries the CORRELATIONS (whole-block cosets), so it can rule out
/// matrices no single entry rules out, and different switches can rule out
/// different candidates.
#[derive(Clone, Debug)]
pub struct BlockPossibilities {
    pub r: i32,
    /// Stable-rep source degree of the block.
    pub degree: Tridegree,
    /// The block's variables, sorted; bit `i` of a mask is `vars[i]`'s value.
    pub vars: Vec<DiffVar>,
    /// Sorted masks not ruled out. Empty = the base system is inconsistent.
    pub possible: Vec<u64>,
    /// Size of the base solution's projection (what "no new information"
    /// looks like); `possible.len() < base_count` means something was ruled
    /// out.
    pub base_count: usize,
    /// Number of switches whose union contributed to the intersection.
    pub via_count: usize,
}

impl BlockPossibilities {
    /// Tier-1 extraction: entries constant across every surviving mask are
    /// forced. Returns `(var, value)` pairs.
    pub fn forced_entries(&self) -> Vec<(DiffVar, bool)> {
        if self.possible.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for (i, &v) in self.vars.iter().enumerate() {
            let first = self.possible[0] >> i & 1;
            if self.possible.iter().all(|m| m >> i & 1 == first) {
                out.push((v, first == 1));
            }
        }
        out
    }
}

/// `EHP_POSSIBILITY=0` disables the possibility-set consensus computation.
fn possibility_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("EHP_POSSIBILITY").map_or(true, |v| v != "0"))
}

/// Block size caps: blocks with more variables, or restricted-kernel rank
/// beyond this, are skipped (the sets get too large to be useful anyway).
const POSS_MAX_BITS: usize = 16;
const POSS_MAX_RANK: usize = 12;

/// Project a solution space `offset + span(kernel)` onto the variable block
/// at (stable-rep) `deg`: the coset enumerated as bitmasks. `None` if the
/// degree has no variables, the block exceeds the caps, or (conservatively)
/// anything else prevents an exact small enumeration.
fn project_block(result: &SATResult, deg: Tridegree) -> Option<(Vec<DiffVar>, Vec<u64>)> {
    let mut block: Vec<(usize, DiffVar)> = result
        .vars
        .iter()
        .enumerate()
        .filter(|(_, v)| Tridegree::new(v.n, v.s, v.f) == deg)
        .map(|(i, v)| (i, *v))
        .collect();
    if block.is_empty() || block.len() > POSS_MAX_BITS {
        return None;
    }
    block.sort_by_key(|&(_, v)| v);

    let mut off: u64 = 0;
    for (bit, &(idx, _)) in block.iter().enumerate() {
        if vec_get(&result.offset, idx) {
            off |= 1 << bit;
        }
    }

    // Restrict kernel rows to the block and build an independent XOR basis
    // (bit-indexed: slot `b` holds the basis vector with leading bit `b`).
    let mut by_bit = [0u64; POSS_MAX_BITS];
    let mut rank = 0usize;
    for i in 0..result.kernel.rows() {
        let mut m: u64 = 0;
        for (bit, &(idx, _)) in block.iter().enumerate() {
            if mat_get(&result.kernel, i, idx) {
                m |= 1 << bit;
            }
        }
        while m != 0 {
            let lead = 63 - m.leading_zeros() as usize;
            if by_bit[lead] == 0 {
                by_bit[lead] = m;
                rank += 1;
                if rank > POSS_MAX_RANK {
                    return None;
                }
                break;
            }
            m ^= by_bit[lead];
        }
    }
    let basis: Vec<u64> = by_bit.iter().copied().filter(|&b| b != 0).collect();

    let k = basis.len();
    let mut pts: Vec<u64> = Vec::with_capacity(1 << k);
    for mask in 0u32..(1u32 << k) {
        let mut p = off;
        for (j, &b) in basis.iter().enumerate() {
            if mask >> j & 1 == 1 {
                p ^= b;
            }
        }
        pts.push(p);
    }
    pts.sort_unstable();
    pts.dedup();
    let vars = block.into_iter().map(|(_, v)| v).collect();
    Some((vars, pts))
}

/// Sorted-vec intersection.
fn intersect_sorted(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// A d_r source block that is provably ZERO OR VACUOUS in BOTH worlds of a
/// switch: every entry individually determined 0 (block projection = {0}),
/// or the source classes dead (re-turned dim 0). In no possible world does a
/// differential from `degree` hit its target — so the target is not killed
/// by it, unconditionally. Recording the stock-basis block as 0 is sound
/// even though the SOURCE basis differs between the worlds (a value the
/// per-entry consensus must refuse): zeros never quotient anything, and the
/// only downstream effect is un-excluding the TARGET, whose dims are
/// guarded unchanged in both worlds. This is the resolution for
/// self-obstructed ghosts — e.g. an uncertain d4 whose only obstruction is
/// the class's own uncertain d3 (Leibniz forces d4 = 0 if the class
/// survives; the class is dead if it doesn't).
#[derive(Clone, Debug)]
pub struct ZeroMapFinding {
    pub via: DiffVar,
    pub via_r: i32,
    pub r: i32,
    /// Stable-rep source degree whose d_r block is zero in every world.
    pub degree: Tridegree,
}

/// `EHP_ZERO_CONSENSUS=0` disables the zero-map consensus channel.
fn zero_consensus_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("EHP_ZERO_CONSENSUS").map_or(true, |v| v != "0"))
}

/// Is the d_r map at (stable-rep) `deg` provably the ZERO map in the world
/// described by `result` (`dim_override` = the world's re-turned source dim,
/// where it changed)? Zero means: no source classes at all, or every block
/// entry individually determined 0. Free or ABSENT variables are NOT zero —
/// unknown ≠ zero is the whole point of the exclusion machinery.
fn map_provably_zero(
    dim_override: Option<usize>,
    result: &SATResult,
    deg: Tridegree,
) -> bool {
    if dim_override == Some(0) {
        return true;
    }
    match project_block(result, deg) {
        Some((_, pts)) => pts.len() == 1 && pts[0] == 0,
        None => false,
    }
}

/// Contradiction forcings plus both-worlds consensus determinations.
pub struct SweepOutcome {
    pub contradictions: Vec<SweepFinding>,
    pub consensus: Vec<ConsensusFinding>,
    /// Possibility-set consensus results (only blocks where something was
    /// ruled out relative to the base projection).
    pub possibilities: Vec<BlockPossibilities>,
    /// Zero-map consensus results ("target not hit in any world").
    pub zero_maps: Vec<ZeroMapFinding>,
}

/// [`trial_error_sweep`] that ALSO harvests the both-worlds consensus:
/// when both values of an unknown are consistent, the learned sets of the
/// two worlds are intersected instead of discarded — any var forced to the
/// SAME value in both worlds is determined unconditionally (subject to the
/// dim-guard on [`ConsensusFinding`]). This is what removes uncertainties
/// that exist only because a degree sits in the exclude list of an earlier
/// unknown: e.g. an "uncertain" d4 that Leibniz forces to 0 whether the
/// obstructing d3 is 0 or 1.
pub fn trial_error_sweep_full(
    pages: &[InterpagePage],
    min_stem: i32,
    max_stem: i32,
    progress: impl Fn(usize, usize) + Sync,
) -> SweepOutcome {
    let first = &pages[0];
    let mut unknown_vars: Vec<DiffVar> = first
        .result
        .unknown
        .iter()
        .map(|&idx| first.result.vars[idx])
        .filter(|v| v.s >= min_stem && v.s < max_stem)
        .collect();
    unknown_vars.sort();
    trial_error_sweep_vars(pages, &unknown_vars, progress)
}

/// [`trial_error_sweep_full`] over an EXPLICIT list of unknown vars on
/// `pages[0]` — the influence-based pass skipping in `interpage try` sweeps
/// only the vars whose influence cone intersects what previous passes
/// changed; everything else provably repeats its previous outcome.
pub fn trial_error_sweep_vars(
    pages: &[InterpagePage],
    unknown_vars: &[DiffVar],
    progress: impl Fn(usize, usize) + Sync,
) -> SweepOutcome {
    let first = &pages[0];
    let r0 = first.page.r;
    let total = 2 * unknown_vars.len();
    let done = std::sync::atomic::AtomicUsize::new(0);

    // Turned data at degrees a trial doesn't touch is identical across all
    // trials — compute it once and share it (the bulk of a trial's work) —
    // and precompute the stock-page indices every trial's overlay build needs.
    let cache = SweepCache::new(pages);

    // One job per var, running both values, so the two worlds' learned sets
    // can be intersected. Ordering matches the old flat sweep: vars
    // ascending, the value-0 finding before the value-1 finding.
    type PerSwitchPoss = Vec<(i32, Tridegree, Vec<DiffVar>, Vec<u64>)>;
    type PerVarOut = (
        Vec<SweepFinding>,
        Vec<ConsensusFinding>,
        PerSwitchPoss,
        Vec<ZeroMapFinding>,
    );
    let per_var: Vec<PerVarOut> = unknown_vars
        .par_iter()
        .map(|&var| {
            let tick = || {
                let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                progress(n, total);
            };
            let run0 = try_diffs_full(pages, &[(var, false)], Some(&cache));
            tick();
            let run1 = try_diffs_full(pages, &[(var, true)], Some(&cache));
            tick();

            let mut findings = Vec::new();
            let mut consensus = Vec::new();
            let mut poss: PerSwitchPoss = Vec::new();
            let mut zeros: Vec<ZeroMapFinding> = Vec::new();
            match (run0, run1) {
                (Err(e), Ok(_)) => findings.push(SweepFinding {
                    var,
                    contradicted_value: false,
                    error: e,
                }),
                (Ok(_), Err(e)) => findings.push(SweepFinding {
                    var,
                    contradicted_value: true,
                    error: e,
                }),
                (Err(e0), Err(e1)) => {
                    // Both values contradict: the base system itself is
                    // inconsistent — two findings, like the old flat sweep.
                    findings.push(SweepFinding { var, contradicted_value: false, error: e0 });
                    findings.push(SweepFinding { var, contradicted_value: true, error: e1 });
                }
                (Ok(w0), Ok(w1)) => {
                    // Both worlds consistent: intersect their learned sets.
                    let learned1: HashMap<(i32, DiffVar), bool> = w1
                        .learned
                        .iter()
                        .map(|l| ((l.r, l.var), l.value))
                        .collect();
                    for l in &w0.learned {
                        // The switch itself trivially differs between worlds.
                        if l.r == r0 && l.var == var {
                            continue;
                        }
                        if learned1.get(&(l.r, l.var)) != Some(&l.value) {
                            continue;
                        }
                        // Basis guard: the learned var's degrees must have
                        // stock dimensions in BOTH worlds.
                        let src = stable_rep(Tridegree::new(l.var.n, l.var.s, l.var.f));
                        let tgt = stable_rep(src.diff_target(l.r));
                        if [src, tgt].iter().any(|t| {
                            w0.dim_changed.contains(t) || w1.dim_changed.contains(t)
                        }) {
                            continue;
                        }
                        consensus.push(ConsensusFinding {
                            via: var,
                            via_r: r0,
                            r: l.r,
                            var: l.var,
                            value: l.value,
                        });
                    }

                    // Possibility sets: union the two worlds' block cosets
                    // at every degree either world touched. A degree only
                    // one world stepped past still projects from that
                    // world's final result (untouched ⇒ base projection —
                    // still a correct P_w). Degrees whose basis either
                    // world changed are skipped (same guard as the value
                    // consensus); blocks whose var lists differ between the
                    // worlds (world-specific new vars) are skipped too.
                    let mut touched: Vec<(i32, Tridegree)> = w0
                        .learned
                        .iter()
                        .chain(w1.learned.iter())
                        .map(|l| (l.r, stable_rep(Tridegree::new(l.var.n, l.var.s, l.var.f))))
                        .collect();
                    touched.sort();
                    touched.dedup();

                    // Zero-map consensus: candidates are every degree a
                    // world learned about OR re-turned (new_dims) — the
                    // latter catches blocks that are dead in one world and
                    // zero in the other without anything being learned
                    // (e.g. both worlds kill the class). One uniform test,
                    // no per-case logic.
                    if zero_consensus_enabled() {
                        let mut zm_candidates = touched.clone();
                        for run in [&w0, &w1] {
                            for &(lr, t) in run.new_dims.keys() {
                                if t.n <= t.s + 2 {
                                    zm_candidates.push((lr, t));
                                }
                            }
                        }
                        zm_candidates.sort();
                        zm_candidates.dedup();
                        for &(lr, deg) in &zm_candidates {
                            let tgt = stable_rep(deg.diff_target(lr));
                            let Some(base_page) = pages.iter().find(|p| p.page.r == lr)
                            else {
                                continue;
                            };
                            let base_res = base_page.result;
                            // No information if the base block is already
                            // fully determined zero, or no diff is possible
                            // at all (source or target empty in the base).
                            if base_page.page.dim_at(deg) == 0
                                || base_page.page.dim_at(tgt) == 0
                                || map_provably_zero(None, base_res, deg)
                            {
                                continue;
                            }
                            // Per-world zero test: source dead, TARGET dead
                            // (vacuous — nothing to hit; covers targets
                            // excluded by their OWN unknown diff, killed in
                            // one world), or every entry determined 0. No
                            // target-dim guard: zeros never quotient, and
                            // the target's other exclusion causes survive
                            // the recompute, so a target-dead world cannot
                            // be corrupted by the recorded stock-basis 0s.
                            let zero_in = |run: &TrialRun| -> bool {
                                if run.new_dims.get(&(lr, deg)) == Some(&0)
                                    || run.new_dims.get(&(lr, tgt)) == Some(&0)
                                {
                                    return true;
                                }
                                let res = run
                                    .final_results
                                    .iter()
                                    .find(|(rr, _)| *rr == lr)
                                    .map(|(_, r)| r)
                                    .unwrap_or(base_res);
                                map_provably_zero(run.new_dims.get(&(lr, deg)).copied(), res, deg)
                            };
                            if zero_in(&w0) && zero_in(&w1) {
                                zeros.push(ZeroMapFinding {
                                    via: var,
                                    via_r: r0,
                                    r: lr,
                                    degree: deg,
                                });
                            }
                        }
                    }

                    if possibility_enabled() {
                        for &(lr, deg) in &touched {
                            let tgt = stable_rep(deg.diff_target(lr));
                            if [deg, tgt].iter().any(|t| {
                                w0.dim_changed.contains(t) || w1.dim_changed.contains(t)
                            }) {
                                continue;
                            }
                            let proj = |run: &TrialRun| -> Option<(Vec<DiffVar>, Vec<u64>)> {
                                let res = run
                                    .final_results
                                    .iter()
                                    .find(|(rr, _)| *rr == lr)
                                    .map(|(_, res)| res)
                                    .or_else(|| {
                                        pages
                                            .iter()
                                            .find(|p| p.page.r == lr)
                                            .map(|p| p.result)
                                    })?;
                                project_block(res, deg)
                            };
                            let (Some((v0, p0)), Some((v1, p1))) = (proj(&w0), proj(&w1))
                            else {
                                continue;
                            };
                            if v0 != v1 {
                                continue;
                            }
                            let mut union = p0;
                            union.extend(p1);
                            union.sort_unstable();
                            union.dedup();
                            poss.push((lr, deg, v0, union));
                        }
                    }
                }
            }
            (findings, consensus, poss, zeros)
        })
        .collect();

    let mut contradictions = Vec::new();
    let mut consensus = Vec::new();
    // Running intersection of the per-switch unions, per (r, degree), then
    // against the base projection. Blocks whose var lists disagree across
    // switches are dropped (basis drift between passes of the same sweep
    // cannot happen, but new-var sets can differ — conservative).
    let mut poss_map: HashMap<(i32, Tridegree), BlockPossibilities> = HashMap::new();
    let mut poss_dropped: HashSet<(i32, Tridegree)> = HashSet::new();
    let mut zero_maps: Vec<ZeroMapFinding> = Vec::new();
    let mut zero_seen: HashSet<(i32, Tridegree)> = HashSet::new();
    for (f, c, ps, zs) in per_var {
        contradictions.extend(f);
        consensus.extend(c);
        for z in zs {
            if zero_seen.insert((z.r, z.degree)) {
                zero_maps.push(z);
            }
        }
        for (lr, deg, vars, union) in ps {
            let key = (lr, deg);
            if poss_dropped.contains(&key) {
                continue;
            }
            match poss_map.entry(key) {
                hashbrown::hash_map::Entry::Occupied(mut e) => {
                    let bp = e.get_mut();
                    if bp.vars != vars {
                        e.remove();
                        poss_dropped.insert(key);
                        continue;
                    }
                    bp.possible = intersect_sorted(&bp.possible, &union);
                    bp.via_count += 1;
                }
                hashbrown::hash_map::Entry::Vacant(slot) => {
                    // Seed with the BASE projection intersected in — the
                    // possibility set may never exceed what the base system
                    // already allows.
                    let Some(base_res) =
                        pages.iter().find(|p| p.page.r == lr).map(|p| p.result)
                    else {
                        continue;
                    };
                    let Some((bvars, bset)) = project_block(base_res, deg) else {
                        continue;
                    };
                    if bvars != vars {
                        poss_dropped.insert(key);
                        continue;
                    }
                    let possible = intersect_sorted(&bset, &union);
                    slot.insert(BlockPossibilities {
                        r: lr,
                        degree: deg,
                        vars,
                        possible,
                        base_count: bset.len(),
                        via_count: 1,
                    });
                }
            }
        }
    }
    let mut possibilities: Vec<BlockPossibilities> = poss_map
        .into_values()
        .filter(|bp| bp.possible.len() < bp.base_count)
        .collect();
    possibilities.sort_by_key(|bp| (bp.r, bp.degree));
    zero_maps.sort_by_key(|z| (z.r, z.degree));
    SweepOutcome { contradictions, consensus, possibilities, zero_maps }
}

// =============================================================================
// Kernel degree components (influence-cone support for pass skipping)
// =============================================================================

/// Connected components of the CURRENT solution space's kernel, at degree
/// granularity: two (stable-rep) degrees are joined iff some kernel row
/// touches variables at both (transitively). A trial that pins one variable
/// can determine values only within that variable's component — that is what
/// bounds its influence on its own page.
///
/// Components must be computed from the result CURRENT at decision time:
/// applying a forced value shrinks the kernel to a subspace whose rows can
/// COMBINE previously separate rows, so components computed on a stale
/// result can be finer than the true ones (an unsound under-approximation).
pub struct DegreeComponents {
    comp: HashMap<Tridegree, usize>,
    members: Vec<Vec<Tridegree>>,
}

impl DegreeComponents {
    pub fn new(result: &SATResult) -> Self {
        // Union-find over stable-rep degrees, joined per kernel row.
        let mut parent: HashMap<Tridegree, Tridegree> = HashMap::new();
        fn find(parent: &mut HashMap<Tridegree, Tridegree>, t: Tridegree) -> Tridegree {
            let p = *parent.entry(t).or_insert(t);
            if p == t {
                return t;
            }
            let root = find(parent, p);
            parent.insert(t, root);
            root
        }
        for i in 0..result.kernel.rows() {
            let row = mat_get_row(&result.kernel, i);
            let mut first: Option<Tridegree> = None;
            for idx in vec_support(&row) {
                let v = result.vars[idx];
                let t = stable_rep(Tridegree::new(v.n, v.s, v.f));
                match first {
                    None => {
                        first = Some(find(&mut parent, t));
                    }
                    Some(root) => {
                        let r2 = find(&mut parent, t);
                        if r2 != root {
                            parent.insert(r2, root);
                        }
                    }
                }
            }
        }
        let keys: Vec<Tridegree> = parent.keys().copied().collect();
        let mut comp: HashMap<Tridegree, usize> = HashMap::new();
        let mut members: Vec<Vec<Tridegree>> = Vec::new();
        let mut root_id: HashMap<Tridegree, usize> = HashMap::new();
        for t in keys {
            let root = find(&mut parent, t);
            let id = *root_id.entry(root).or_insert_with(|| {
                members.push(Vec::new());
                members.len() - 1
            });
            comp.insert(t, id);
            members[id].push(t);
        }
        DegreeComponents { comp, members }
    }

    /// Component id of the (stable-rep) degree, if it touches any kernel row.
    /// Vars in the same component have identical influence cones — memoize on
    /// this.
    pub fn id_of(&self, t: Tridegree) -> Option<usize> {
        self.comp.get(&t).copied()
    }

    /// Component members of the (stable-rep) degree, or an empty slice if the
    /// degree touches no kernel row (fully determined there).
    pub fn members_of(&self, t: Tridegree) -> &[Tridegree] {
        match self.comp.get(&t) {
            Some(&id) => &self.members[id],
            None => &[],
        }
    }

    /// All components intersecting `set` (rep form), unioned.
    pub fn closure(&self, set: &HashSet<Tridegree>) -> HashSet<Tridegree> {
        let mut out = set.clone();
        let mut seen_comps: HashSet<usize> = HashSet::new();
        for t in set {
            if let Some(&id) = self.comp.get(t) {
                if seen_comps.insert(id) {
                    out.extend(self.members[id].iter().copied());
                }
            }
        }
        out
    }
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

    /// Zero-map decision: vacuous (dim 0) and all-determined-zero blocks
    /// count as zero; free or absent vars do not (unknown ≠ zero).
    #[test]
    fn zero_map_decision() {
        let deg = Tridegree::new(2, 5, 1);
        let bv = |col: u16| DiffVar::new(2, 5, 1, 0, col);
        // All entries pinned to 0 → zero map.
        let mut sys = ConstraintSystem::new(vec![bv(0), bv(1)]);
        sys.add_constraint_indices(&[0], false);
        sys.add_constraint_indices(&[1], false);
        let res = solve(&sys).expect("consistent");
        assert!(map_provably_zero(None, &res, deg));
        // One entry free → NOT zero.
        let mut sys2 = ConstraintSystem::new(vec![bv(0), bv(1)]);
        sys2.add_constraint_indices(&[0], false);
        let res2 = solve(&sys2).expect("consistent");
        assert!(!map_provably_zero(None, &res2, deg));
        // No vars at the degree → NOT zero (absent ≠ zero)...
        let sys3 = ConstraintSystem::new(vec![dv(9)]);
        let res3 = solve(&sys3).expect("consistent");
        assert!(!map_provably_zero(None, &res3, deg));
        // ...unless the world's re-turned dim is 0 (source dead) → vacuous.
        assert!(map_provably_zero(Some(0), &res3, deg));
    }

    /// project_block enumerates exactly the solution coset restricted to a
    /// degree's block, and forced_entries extracts the constant bits.
    #[test]
    fn possibility_projection_and_forced_entries() {
        // Three vars at ONE degree; constraint x0 ⊕ x1 = 1, x2 = 1.
        let deg = Tridegree::new(2, 5, 1);
        let bv = |col: u16| DiffVar::new(2, 5, 1, 0, col);
        let vars = vec![bv(0), bv(1), bv(2)];
        let mut sys = ConstraintSystem::new(vars.clone());
        sys.add_constraint_indices(&[0, 1], true);
        sys.add_constraint_indices(&[2], true);
        let res = solve(&sys).expect("consistent");

        let (bvars, pts) = project_block(&res, deg).expect("projectable");
        assert_eq!(bvars, vars);
        // x0 ≠ x1, x2 = 1 → masks {bit2 | bit0} and {bit2 | bit1}.
        assert_eq!(pts, vec![0b101, 0b110]);

        let bp = BlockPossibilities {
            r: 2,
            degree: deg,
            vars: bvars,
            possible: pts,
            base_count: 8,
            via_count: 1,
        };
        assert_eq!(bp.forced_entries(), vec![(bv(2), true)]);

        // Intersecting with a set that pins x0 leaves a singleton whose
        // remaining entries all become forced.
        let inter = intersect_sorted(&bp.possible, &[0b100, 0b101]);
        assert_eq!(inter, vec![0b101]);
        let bp2 = BlockPossibilities { possible: inter, ..bp };
        assert_eq!(
            bp2.forced_entries(),
            vec![(bv(0), true), (bv(1), false), (bv(2), true)]
        );
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
