use fp::matrix::Matrix;
use fp::vector::FpVector;
use crate::gf2::*;
use hashbrown::HashMap;
use log::info;
use rayon::prelude::*;

use crate::element::Element;
use crate::map::MapKind;
use crate::page::{MapTable, SATPage};
use crate::products::{ProductKey, ProductTable};
use crate::result::SATResult;
use crate::tridegree::Tridegree;

/// Result of page turning at a single tridegree.
#[derive(Clone, Debug)]
pub struct TurnedBidegree {
    /// Tridegree.
    pub degree: Tridegree,
    /// Basis elements for the new page (in the quotient H = Z/B).
    pub basis: Vec<Element>,
    /// Quotient map matrix: maps old-page vectors to new-page vectors.
    /// Rows of the matrix form a basis for H; given an old vector v,
    /// quotient_map * v gives the new-page coordinates.
    pub quotient_map: Matrix,
    /// Lift map: given new-page coordinates, produce an old-page representative.
    /// This is a right inverse of quotient_map restricted to Z.
    pub lift_map: Matrix,
    /// Boundary subspace basis (for reducing vectors modulo boundaries).
    pub boundary_basis: Vec<FpVector>,
}

impl TurnedBidegree {
    /// Map an old-page vector to the new-page quotient.
    pub fn quotient(&self, v: &FpVector) -> FpVector {
        mat_mul_vec(&self.quotient_map, v)
    }

    /// Lift a new-page vector to an old-page representative.
    pub fn lift(&self, v: &FpVector) -> FpVector {
        mat_mul_vec(&self.lift_map, v)
    }

    /// Reduce an old-page vector modulo boundaries.
    pub fn reduce_against_boundaries(&self, v: &FpVector) -> FpVector {
        let mut reduced = v.clone();
        for b in &self.boundary_basis {
            if let Some(pivot) = vec_first_set(b) {
                if vec_get(&reduced, pivot) {
                    reduced += b;
                }
            }
        }
        reduced
    }

    /// Is `v` actually a cycle (in the domain `quotient_map` was built for,
    /// i.e. in the kernel Z of the outgoing differential)?
    ///
    /// `quotient_map` is a genuine linear map defined on the *whole*
    /// old-page space (it reads off h-basis pivot coordinates), not
    /// restricted to Z. Feeding it a vector that isn't actually a cycle
    /// silently returns a basis-dependent phantom value instead of
    /// correctly reporting "no image" -- Python's `compute_induced_map_single`
    /// / `compute_induced_products_single` guard exactly this
    /// (`if map_reduced in tb_target.quotient_map.domain(): ...`).
    /// `lift` is a right inverse of `quotient` restricted to Z, so a
    /// lift∘quotient round trip is idempotent exactly on Z and moves `v`
    /// for anything outside it -- reproduces Python's domain check without
    /// needing to store the raw kernel basis. See
    /// notes/UNCERTAIN_DEGREE_HANDLING_TODO.md for the full history (this
    /// exact check was previously implemented for the maps call site only,
    /// then deliberately left uncommitted; re-added here plus at the
    /// products call site after confirming a real, currently-trusted
    /// mismatch traced to exactly this gap, 2026-07-27).
    pub fn is_cycle(&self, v: &FpVector) -> bool {
        self.lift(&self.quotient(v)) == *v
    }

    /// Dimension of the new page at this tridegree.
    pub fn dim(&self) -> usize {
        self.basis.len()
    }

    /// Zero element in the quotient.
    pub fn zero(&self) -> Element {
        Element::zero(self.degree, self.basis.len())
    }
}

/// Compute page turning at a single tridegree.
///
/// Given the outgoing differential d_out: (n,s,f) -> (n,s-1,f+r)
/// and incoming differential d_in: (n,s+1,f-r) -> (n,s,f),
/// compute H = ker(d_out) / im(d_in).
pub fn turn_page_single(
    degree: Tridegree,
    d_out: &Matrix,
    d_in: &Matrix,
    old_dim: usize,
) -> Option<TurnedBidegree> {
    // d_out has dimensions: tgt_dim × src_dim (maps source to target)
    // d_in has dimensions: src_dim × prev_dim (maps previous to source)

    // Z = right kernel of d_out (cycles)
    // Note: d_out * v = 0 => v is a cycle
    let z_basis = gauss_right_kernel(d_out);

    // B = column space of d_in (boundaries)
    // The columns of d_in span the boundary subspace. `gauss_image` already
    // returns the column space of its argument, so pass d_in directly — an
    // extra transpose here would compute the row space (wrong subspace, and in
    // the source dimension), which silently drops incoming differentials.
    let b_basis = gauss_image(d_in);

    if z_basis.is_empty() {
        return None;
    }

    // Compute H = Z/B using relative row echelon
    let h_basis = gauss_relative_row_echelon(&b_basis, &z_basis);

    if h_basis.is_empty() {
        return None;
    }

    let h_dim = h_basis.len();

    // Build quotient map: projects old-page vectors to H coordinates
    // The quotient map takes a vector in GF(2)^old_dim and returns
    // its coordinates in the h_dim-dimensional quotient space.
    //
    // We build it as follows:
    // - h_basis[i] are representatives in GF(2)^old_dim
    // - To project v, we reduce v against B, then express it in h_basis
    //
    // Build the projection matrix: h_dim rows × old_dim cols
    // First, put B ∪ H in echelon form and build the projection
    let mut all_basis = b_basis.clone();
    all_basis.extend(h_basis.iter().cloned());

    // Put boundary basis in echelon form for reduction
    let b_echelon = if b_basis.is_empty() {
        Vec::new()
    } else {
        let mut bmat = mat_from_rows(&b_basis, old_dim);
        let (rank, _pivot_cols) = mat_echelon_form(&mut bmat);
        (0..rank)
            .map(|i| mat_get_row(&bmat, i).to_owned())
            .filter(|r| !r.is_zero())
            .collect()
    };

    // Build echelon form of H basis (already should be in echelon from relative_row_echelon)
    let h_pivots: Vec<usize> = h_basis
        .iter()
        .filter_map(|b| vec_first_set(b))
        .collect();

    // Quotient map: for each old-dim basis vector, reduce mod B, then
    // express in terms of h_basis
    let mut q_rows = Vec::with_capacity(h_dim);
    for i in 0..h_dim {
        let mut row = vec_zero(old_dim);
        // The i-th output coordinate: check if h_basis[i]'s pivot is set
        if let Some(pivot) = h_pivots.get(i) {
            vec_set(&mut row, *pivot, true);
        }
        q_rows.push(row);
    }
    let quotient_map = mat_from_rows(&q_rows, old_dim);

    // Lift map: h_dim cols → old_dim rows
    // Maps standard basis vectors of H to representatives in old space
    let mut l_rows = Vec::with_capacity(old_dim);
    for i in 0..old_dim {
        let mut row = vec_zero(h_dim);
        for (j, hb) in h_basis.iter().enumerate() {
            if vec_get(hb, i) {
                vec_set(&mut row, j, true);
            }
        }
        l_rows.push(row);
    }
    let lift_map = mat_from_rows(&l_rows, h_dim);

    let basis: Vec<Element> = (0..h_dim)
        .map(|i| Element::basis(degree, h_dim, i))
        .collect();

    Some(TurnedBidegree {
        degree,
        basis,
        quotient_map,
        lift_map,
        boundary_basis: b_echelon,
    })
}

/// A genuine contradiction discovered while turning: d_r ∘ d_r ≠ 0 with both
/// differentials fully determined (ports the original's `d2Exception`).
#[derive(Clone, Debug)]
pub struct D2Error {
    pub degree: Tridegree,
    pub r: i32,
}

impl std::fmt::Display for D2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "d_{} ∘ d_{} != 0 at ({}, {}, {})",
            self.r, self.r, self.degree.n, self.degree.s, self.degree.f
        )
    }
}

impl std::error::Error for D2Error {}

/// Context for turning a page with uncertainty-aware semantics:
///
/// - Differential matrices are used **partially**: determined entries are
///   kept (they are real kills/boundaries — quotienting by them is correct,
///   and this is what makes user-asserted differentials at partially-unknown
///   degrees take effect), while unknown entries are treated as **zero**
///   (classes that *might* die stay alive). Degrees with any unknown entry
///   are still excluded from constraint generation on the next page, so
///   their possibly-incomplete homology never feeds naturality/Leibniz.
///   (The original Python zeroed the whole matrix at such degrees; that
///   conservatism guarded against wrongly-"determined" values, which came
///   from bugs that are now fixed — the `gauss_image` transpose, the solver
///   marking correlated pivots as determined, and exclusion-set gaps.)
/// - If a degree has no differential variables at all (excluded, or a zero
///   source/target), the differential is treated as zero rather than
///   dropping the degree.
/// - If both bounding differentials are fully determined but d∘d ≠ 0, that is
///   a genuine contradiction ([`D2Error`]); the check is skipped while either
///   degree still has unknown entries.
pub struct TurnContext<'a> {
    pub page: &'a SATPage,
    pub result: &'a SATResult,
    /// Stable-representative tridegrees that have at least one unknown d_r entry.
    unknown_reps: hashbrown::HashSet<Tridegree>,
}

/// Map a tridegree to its stable representative (variables and exclusions are
/// keyed by `n = min(n, s+2)`).
fn stable_rep(t: Tridegree) -> Tridegree {
    if t.n > t.s + 2 {
        Tridegree::new(t.s + 2, t.s, t.f)
    } else {
        t
    }
}

impl<'a> TurnContext<'a> {
    pub fn new(page: &'a SATPage, result: &'a SATResult) -> Self {
        let mut unknown_reps = hashbrown::HashSet::new();
        for &idx in &result.unknown {
            let v = &result.vars[idx];
            unknown_reps.insert(Tridegree::new(v.n, v.s, v.f));
        }
        TurnContext {
            page,
            result,
            unknown_reps,
        }
    }

    /// Does the differential at this degree have any unknown entry?
    pub fn is_unknown(&self, t: Tridegree) -> bool {
        self.unknown_reps.contains(&stable_rep(t))
    }

    /// The determined differential matrix at `t` (unknown entries as 0), or
    /// `None` if the degree has no differential variables at all.
    fn var_matrix(&self, t: Tridegree) -> Option<Matrix> {
        let rep = stable_rep(t);
        let first = crate::constraints::DiffVar::new(rep.n, rep.s, rep.f, 0, 0);
        if !self.result.var_index.contains_key(&first) {
            return None;
        }
        let src_dim = self.page.dim_at(rep);
        let tgt_dim = self.page.dim_at(rep.diff_target(self.page.r));
        let mut mat = mat_zero(tgt_dim, src_dim);
        for row in 0..tgt_dim {
            for col in 0..src_dim {
                let var =
                    crate::constraints::DiffVar::new(rep.n, rep.s, rep.f, row as u16, col as u16);
                if let Some(&idx) = self.result.var_index.get(&var) {
                    if !self.result.unknown.contains(&idx) && vec_get(&self.result.offset, idx) {
                        mat_set(&mut mat, row, col, true);
                    }
                }
            }
        }
        Some(mat)
    }

    /// Turn a single tridegree, applying the uncertainty semantics above.
    pub fn get_tb(&self, t: Tridegree) -> Result<Option<TurnedBidegree>, D2Error> {
        let r = self.page.r;
        let t_in = Tridegree::new(t.n, t.s + 1, t.f - r);
        let old_dim = self.page.dim_at(t);

        let d1 = self.var_matrix(t);
        let d2 = self.var_matrix(t_in);

        if let (Some(d1m), Some(d2m)) = (&d1, &d2) {
            if d1m.rows() > 0
                && d1m.columns() == d2m.rows()
                && !mat_is_zero(&mat_mul(d1m, d2m))
                && !self.is_unknown(t)
                && !self.is_unknown(t_in)
            {
                return Err(D2Error { degree: t, r });
            }
        }

        // Partial matrices: determined entries kept, unknown entries zero
        // (var_matrix already zeroes unknowns).
        let d_out = d1.unwrap_or_else(|| mat_zero(self.page.dim_at(t.diff_target(r)), old_dim));
        let d_in = d2.unwrap_or_else(|| mat_zero(old_dim, self.page.dim_at(t_in)));

        Ok(turn_page_single(t, &d_out, &d_in, old_dim))
    }
}

/// Turn the entire page: compute homology at every tridegree.
///
/// Returns `Err` if a fully-determined d∘d ≠ 0 is found (a genuine contradiction).
pub fn turn_page(
    page: &SATPage,
    sat_result: &SATResult,
    _new_r: i32,
    _max_t: Option<i32>,
) -> Result<HashMap<Tridegree, TurnedBidegree>, D2Error> {
    let ctx = TurnContext::new(page, sat_result);

    // Crop to the OLD page's own max_t before turning. `page.page` can
    // (legitimately) extend past `page.max_t` -- the page that built it
    // was itself built "generously" with no cap on its dimension table,
    // matching the Python reference's own next_page (SATPage.next_page
    // uses a flat max_t-1 there, deliberately not cropping the dimension
    // table, so a later reload can serve a wider range of downstream
    // max_t targets from the same saved data). Python enforces the real
    // cap the ONE time it matters: whenever that saved page is reloaded
    // as input for building the NEXT page, load_spectral_sequence filters
    // `if s+f > tot: continue` with tot = this page's own (recursively
    // decremented) max_t. Rust has no separate reload step, so this is
    // where that crop has to happen instead -- otherwise an over-extended
    // page (e.g. E3's dimension table legitimately reaching s+f=80, one
    // past its own max_t=79) propagates its over-extension into every
    // later page uncontrolled, producing real rust-vs-python rank
    // mismatches at exactly the old over-extension boundary (confirmed
    // 2026-07-26: 4372 E4_rank.csv mismatches, ALL at s+f=80, none
    // elsewhere). `_max_t` (the NEW page's cutoff) is intentionally not
    // used for this -- that would crop too early/differently from Python
    // (see build_next_page's own max_t decrement comment for why the two
    // schedules aren't interchangeable).
    let tridegrees: Vec<Tridegree> = page
        .page
        .keys()
        .copied()
        .filter(|t| page.max_t.is_none_or(|mt| t.s + t.f <= mt))
        .collect();

    info!("Turning page: {} tridegrees", tridegrees.len());

    let results: Result<Vec<(Tridegree, Option<TurnedBidegree>)>, D2Error> = tridegrees
        .par_iter()
        .map(|&t| {
            let old_dim = page.dim_at(t);
            if old_dim == 0 {
                return Ok((t, None));
            }
            Ok((t, ctx.get_tb(t)?))
        })
        .collect();
    let results = results?;

    let mut turned_page = HashMap::new();
    for (t, tb) in results {
        if let Some(tb) = tb {
            if !page.page.get(&t).map_or(true, |v| v.is_empty()) {
                turned_page.insert(t, tb);
            }
        }
    }

    info!(
        "Page turned: {} tridegrees with nonzero homology",
        turned_page.len()
    );
    Ok(turned_page)
}

/// Compute induced map on the next page for a single tridegree.
/// What [`patch_page_from_turned`] actually changed.
#[derive(Default)]
pub struct PatchOutcome {
    /// Every degree whose next-page content changed: dims/basis, an induced
    /// map matrix at that source (or its target), or a product block whose
    /// key or product degree is the degree. This is the "dirty" input for
    /// patching the page AFTER the next one (content changes here alter its
    /// induction inputs at exactly these degrees).
    pub changed: hashbrown::HashSet<Tridegree>,
    /// Subset bookkeeping: degrees whose dimension changed (drives chart
    /// regen + the cascade's re-turn carry, exactly like the full-rebuild
    /// path's dims scan).
    pub dims_changed: hashbrown::HashSet<Tridegree>,
}

/// Incrementally patch `next` — a page previously built by
/// [`build_page_from_turned`] from this same `old_page` — after `turned`
/// changed ONLY at `dirty` degrees (and/or `old_page`'s own content changed
/// only at `dirty` degrees). Byte-equivalent to a full rebuild by
/// construction:
/// - dims/basis are reset from the dirty turned entries (removed when the
///   basis vanished),
/// - induced-map matrices are recomputed at every source degree whose source
///   or target is dirty, with the exact presence rule of the full rebuild
///   (a matrix exists iff ≥1 singleton-support entry and both dims > 0 —
///   presence matters: an ABSENT stable-range E matrix means IDENTITY),
/// - induced-product blocks are recomputed for every triple (x, y, xy)
///   touching a dirty degree — the same enumeration filters as
///   [`compute_induced_products`] (unit rule, turned membership, flat
///   `old.max_t - 1` bound) — and stale blocks touching a dirty degree that
///   are no longer valid triples are removed.
///
/// The only intentional difference from a full rebuild is storage sharing
/// (no `dedup_shared` pass on patched entries) — content-equal, more Arcs.
/// `EHP_CASCADE_VERIFY=1` in the REPL cross-checks patched pages against
/// full rebuilds.
pub fn patch_page_from_turned(
    old_page: &SATPage,
    turned: &HashMap<Tridegree, TurnedBidegree>,
    dirty: &hashbrown::HashSet<Tridegree>,
    next: &mut SATPage,
) -> PatchOutcome {
    let mut out = PatchOutcome::default();
    if dirty.is_empty() {
        return out;
    }

    // 1. Dimensions + basis at dirty degrees (nonempty turned entry = live).
    for &t in dirty {
        match turned.get(&t).filter(|tb| !tb.basis.is_empty()) {
            Some(tb) => {
                if next.page.get(&t) != Some(&tb.basis) {
                    if next.dim_at(t) != tb.basis.len() {
                        out.dims_changed.insert(t);
                    }
                    out.changed.insert(t);
                    next.dimension.insert(t, tb.basis.len());
                    next.page.insert(t, tb.basis.clone());
                }
            }
            None => {
                if next.dimension.remove(&t).is_some() {
                    out.dims_changed.insert(t);
                    out.changed.insert(t);
                }
                next.page.remove(&t);
            }
        }
    }
    next.compute_max_values();

    // 2. Induced maps at influenced source degrees (source or target dirty).
    for kind in MapKind::all_with_lh0() {
        let mut srcs: hashbrown::HashSet<Tridegree> = hashbrown::HashSet::new();
        for &d in dirty {
            for s in [d, kind.source_degree(d)] {
                // Visit every domain-eligible influenced source — including
                // ones that just DIED on the old page (their recompute yields
                // no entries, which REMOVES the stale matrix; filtering them
                // out here left dead matrices behind — and an absent vs
                // stale E matrix is the identity-default distinction).
                if kind.domain_check(s) {
                    srcs.insert(s);
                }
            }
        }
        if srcs.is_empty() {
            continue;
        }
        let updates: Vec<(Tridegree, Option<crate::products::ProductMatrix>)> = srcs
            .into_iter()
            .map(|src| {
                let entries = compute_induced_map_single(src, kind, old_page, turned);
                // Full-rebuild presence rule: by_src key iff ≥1 singleton-
                // support entry; then matrix built (possibly zero rows) iff
                // both dims > 0.
                let rows: Vec<(usize, FpVector)> = entries
                    .iter()
                    .filter_map(|(x, tgt)| {
                        let idx: Vec<usize> = vec_support(&x.vec).collect();
                        (idx.len() == 1).then(|| (idx[0], tgt.vec.clone()))
                    })
                    .collect();
                let src_dim = next.dim_at(src);
                let tgt_dim = next.dim_at(kind.target_degree(src));
                let new_block = if !rows.is_empty() && src_dim > 0 && tgt_dim > 0 {
                    let mut mat = mat_zero(src_dim, tgt_dim);
                    for (row_idx, vec) in &rows {
                        if *row_idx < src_dim && vec.len() == tgt_dim {
                            mat_set_row(&mut mat, *row_idx, vec);
                        }
                    }
                    Some(crate::products::ProductMatrix::from_matrix(
                        src_dim as u16,
                        1,
                        tgt_dim as u16,
                        &mat,
                    ))
                } else {
                    None
                };
                (src, new_block)
            })
            .collect();
        let table = next.maps.entry(kind).or_insert_with(|| MapTable::new(kind));
        for (src, new_block) in updates {
            let changed = match (&new_block, table.matrix_at(src)) {
                (Some(nb), Some(ob)) => nb != ob,
                (Some(_), None) | (None, Some(_)) => true,
                (None, None) => false,
            };
            if changed {
                out.changed.insert(src);
                out.changed.insert(kind.target_degree(src));
                match new_block {
                    Some(nb) => table.set_block(src, nb),
                    None => {
                        table.matrices.remove(&src);
                    }
                }
            }
        }
    }

    // 3. Induced product blocks for triples touching a dirty degree.
    let mut by_n: HashMap<i32, Vec<Tridegree>> = HashMap::new();
    let mut by_ns: HashMap<i32, Vec<Tridegree>> = HashMap::new();
    for &t in next.page.keys() {
        by_n.entry(t.n).or_default().push(t);
        by_ns.entry(t.n + t.s).or_default().push(t);
    }
    let product_max_t = old_page.max_t.map(|t| t - 1);
    let valid_triple = |x: Tridegree, y: Tridegree| -> Option<(Tridegree, Tridegree, Tridegree)> {
        if y.n != x.n + x.s {
            return None;
        }
        if x.s == 0 && x.f == 0 && !old_page.products.has_block(x, y) {
            return None; // unit rule: only data-seeded identity products
        }
        let xy = Tridegree::new(x.n, x.s + y.s, x.f + y.f);
        if !turned.contains_key(&x) || !turned.contains_key(&y) || !turned.contains_key(&xy) {
            return None;
        }
        if ![x, y, xy]
            .iter()
            .all(|s| product_max_t.is_none_or(|mt| s.s + s.f <= mt))
        {
            return None;
        }
        Some((x, y, xy))
    };
    let mut candidates: hashbrown::HashMap<(Tridegree, Tridegree), Tridegree> =
        hashbrown::HashMap::new();
    for &d in dirty {
        // d as x.
        if let Some(ys) = by_n.get(&(d.n + d.s)) {
            for &y in ys {
                if let Some((x, y, xy)) = valid_triple(d, y) {
                    candidates.insert((x, y), xy);
                }
            }
        }
        // d as y.
        if let Some(xs) = by_ns.get(&d.n) {
            for &x in xs {
                if let Some((x, y, xy)) = valid_triple(x, d) {
                    candidates.insert((x, y), xy);
                }
            }
        }
        // d as xy.
        if let Some(xs) = by_n.get(&d.n) {
            for &x in xs {
                let y = Tridegree::new(x.n + x.s, d.s - x.s, d.f - x.f);
                if next.page.contains_key(&y) {
                    if let Some((x, y, xy)) = valid_triple(x, y) {
                        candidates.insert((x, y), xy);
                    }
                }
            }
        }
    }

    // Stale blocks: touching a dirty degree but no longer a valid triple.
    let stale: Vec<(Tridegree, Tridegree)> = next
        .products
        .iter_blocks()
        .map(|(&k, _)| k)
        .filter(|&(x, y)| {
            let xy = Tridegree::new(x.n, x.s + y.s, x.f + y.f);
            (dirty.contains(&x) || dirty.contains(&y) || dirty.contains(&xy))
                && !candidates.contains_key(&(x, y))
        })
        .collect();
    for (x, y) in stale {
        next.products.remove_block(x, y);
        out.changed.extend([x, y, Tridegree::new(x.n, x.s + y.s, x.f + y.f)]);
    }

    // Recompute candidate blocks (parallel like the full rebuild).
    let cand_vec: Vec<((Tridegree, Tridegree), Tridegree)> =
        candidates.iter().map(|(&k, &xy)| (k, xy)).collect();
    let recomputed: Vec<((Tridegree, Tridegree), crate::products::ProductMatrix)> = cand_vec
        .par_iter()
        .map(|&((x, y), xy)| {
            let (tb_x, tb_y, tb_xy) = (&turned[&x], &turned[&y], &turned[&xy]);
            let dim1 = tb_x.basis.len();
            let dim2 = tb_y.basis.len();
            let tgt_dim = tb_xy.basis.len();
            let mut block = crate::products::ProductMatrix::zero(dim1, dim2, tgt_dim);
            for (xe, ye, res) in compute_induced_products_single(tb_x, tb_y, tb_xy, old_page) {
                let xi: Vec<usize> = vec_support(&xe.vec).collect();
                let yi: Vec<usize> = vec_support(&ye.vec).collect();
                if xi.len() == 1 && yi.len() == 1 && res.vec.len() == tgt_dim {
                    block.set_row(xi[0] * dim2 + yi[0], &res.vec);
                }
            }
            ((x, y), block)
        })
        .collect();
    for ((x, y), block) in recomputed {
        let unchanged = next.products.block(x, y).is_some_and(|ob| *ob == block);
        if !unchanged {
            out.changed.extend([x, y, Tridegree::new(x.n, x.s + y.s, x.f + y.f)]);
            next.products.insert_block(x, y, block);
        }
    }

    out
}

pub fn compute_induced_map_single(
    src_degree: Tridegree,
    map_kind: MapKind,
    page: &SATPage,
    turned_page: &HashMap<Tridegree, TurnedBidegree>,
) -> Vec<(Element, Element)> {
    let target_degree = map_kind.target_degree(src_degree);

    let tb_src = match turned_page.get(&src_degree) {
        Some(tb) => tb,
        None => return Vec::new(),
    };
    let tb_tgt = match turned_page.get(&target_degree) {
        Some(tb) => tb,
        None => return Vec::new(),
    };

    compute_induced_map_single_tb(src_degree, map_kind, page, tb_src, tb_tgt)
}

/// Compute induced map on the next page given the turned data directly.
pub fn compute_induced_map_single_tb(
    src_degree: Tridegree,
    map_kind: MapKind,
    page: &SATPage,
    tb_src: &TurnedBidegree,
    tb_tgt: &TurnedBidegree,
) -> Vec<(Element, Element)> {
    let target_degree = map_kind.target_degree(src_degree);

    let map_mat = page.map_matrix(map_kind, src_degree);

    if map_mat.columns() == 0 {
        return Vec::new();
    }

    let mut results = Vec::new();

    for x in &tb_src.basis {
        // Lift x back to old-page representative
        let x_pre = tb_src.lift(&x.vec);

        // Apply the map (Python uses v * mat where mat is src_dim × tgt_dim)
        let map_vec = mat_vec_mul(&map_mat, &x_pre);

        if map_vec.is_zero() {
            continue;
        }

        // Reduce modulo boundaries in target
        let map_reduced = tb_tgt.reduce_against_boundaries(&map_vec);

        if map_reduced.is_zero() {
            continue;
        }

        // Not actually a d_r-cycle at the target -- Python's domain check
        // (`if map_reduced in tb_target.quotient_map.domain(): ...`) skips
        // here; quotient_map has no domain restriction of its own and would
        // silently return a basis-dependent phantom value instead. See
        // notes/UNCERTAIN_DEGREE_HANDLING_TODO.md and TurnedBidegree::is_cycle.
        if !tb_tgt.is_cycle(&map_reduced) {
            continue;
        }

        // Project to quotient
        let projected = tb_tgt.quotient(&map_reduced);

        if !projected.is_zero() {
            let target_elem = Element::new(target_degree, projected);
            results.push((x.clone(), target_elem));
        }
    }

    results
}

/// Compute all induced maps for the next page.
pub fn compute_induced_maps(
    page: &SATPage,
    turned_page: &HashMap<Tridegree, TurnedBidegree>,
) -> HashMap<MapKind, Vec<(Element, Element)>> {
    let mut all_maps = HashMap::new();

    // Induce E/H/P AND lh0 the same way (lift → apply → reduce mod
    // boundaries → project). lh0 lands in next.maps like the EHP maps but is
    // only read at the display/export sites (all_with_lh0), never by the
    // solver/fiber/interpage code (which iterate the EHP-only all()).
    for kind in MapKind::all_with_lh0() {
        let domain: Vec<Tridegree> = page.map_domain(kind);

        let map_entries: Vec<(Element, Element)> = domain
            .par_iter()
            .flat_map(|&t| {
                if !turned_page.contains_key(&t) {
                    return Vec::new();
                }
                let target = kind.target_degree(t);
                if !turned_page.contains_key(&target) {
                    return Vec::new();
                }
                compute_induced_map_single(t, kind, page, turned_page)
            })
            .collect();

        info!("{} map: {} induced entries", kind.name(), map_entries.len());
        all_maps.insert(kind, map_entries);
    }

    all_maps
}

/// Compute induced products at a single triple of tridegrees.
pub fn compute_induced_products_single(
    tb_x: &TurnedBidegree,
    tb_y: &TurnedBidegree,
    tb_xy: &TurnedBidegree,
    page: &SATPage,
) -> Vec<(Element, Element, Element)> {
    let mut results = Vec::new();

    for x in &tb_x.basis {
        let x_pre = tb_x.lift(&x.vec);

        for y in &tb_y.basis {
            let y_pre = tb_y.lift(&y.vec);

            // Compute product in old page
            let xy_pre = page.products.multiply(
                tb_x.degree,
                &x_pre,
                tb_y.degree,
                &y_pre,
                page.dim_at(tb_xy.degree),
            );

            if xy_pre.is_zero() || tb_xy.basis.is_empty() {
                results.push((x.clone(), y.clone(), tb_xy.zero()));
            } else {
                let xy_reduced = tb_xy.reduce_against_boundaries(&xy_pre);
                if !tb_xy.is_cycle(&xy_reduced) {
                    // Not actually a d_r-cycle -- Python's `tb_xy.quotient(...)`
                    // raises here (caught, entry left absent = zero downstream);
                    // Rust's quotient_map has no domain restriction and would
                    // silently return a basis-dependent phantom value instead.
                    results.push((x.clone(), y.clone(), tb_xy.zero()));
                    continue;
                }
                let xy_proj = tb_xy.quotient(&xy_reduced);
                let result = Element::new(tb_xy.degree, xy_proj);
                results.push((x.clone(), y.clone(), result));
            }
        }
    }

    results
}

/// Compute all induced products for the next page.
pub fn compute_induced_products(
    page: &SATPage,
    next_page: &SATPage,
    turned_page: &HashMap<Tridegree, TurnedBidegree>,
) -> ProductTable {
    let mut new_products = ProductTable::new();

    // Build index by n for the next page
    let mut by_n: HashMap<i32, Vec<Tridegree>> = HashMap::new();
    for &t in next_page.page.keys() {
        by_n.entry(t.n).or_default().push(t);
    }

    // Products are bound by a FLAT decrement of the OLD page's own max_t
    // (page.max_t - 1), NOT next_page.max_t (the recursively-decremented
    // cutoff that governs next_page's own later d_r solve constraint
    // generation -- a different, unrelated purpose/value). Confirmed
    // against the Python reference (2026-07-26): products are computed
    // during SATPage.next_page's one-time "build" phase, where the target
    // page's max_t field is set via the flat `self.max_t - 1` (sat_ss.py),
    // and `_induced_product_helper`'s `target_page.in_bounds(...)` check
    // uses THAT flat value, not the recursive schedule a later, separate
    // run.py invocation computes when reloading this page as input to
    // solve ITS OWN d_r. Using next_page.max_t here (the recursive value)
    // wrongly excluded real products at exactly next_page.max_t+1 through
    // page.max_t-1 (confirmed: 11430 real E4 product mismatches, all at
    // product-degree page.max_t-1, i.e. one past next_page.max_t=77 but
    // within the flat bound=78).
    let product_max_t = page.max_t.map(|t| t - 1);

    let mut product_triples = Vec::new();

    for &x in next_page.page.keys() {
        // No n-based bound here (matches the Python original, which has
        // none): max_t bounds t = s + f only, never n — a previous version
        // of this code compared x.n against max_t directly, wrongly
        // excluding high-n/low-t product triples (e.g. many s=0, high-n
        // stable-range products).
        //
        // The unit class (s=0, f=0) is normally skipped (Python parity: it
        // never participates in Leibniz pairs), but the E2 data seeds
        // h_i-on-identity product blocks that the chart h_i columns and the
        // fiber view read (hi_target_names looks up (unit, h_i) products).
        // Skipping the unit outright dropped exactly those blocks on every
        // turn, so unit h_i-towers vanished from E3+ charts. Keep unit
        // triples for precisely the (x, y) pairs whose product block exists
        // on the old page — that propagates the seeded identity products
        // (quotienting h_i's image like any other product) and nothing else.
        let x_is_unit = x.s == 0 && x.f == 0;

        let y_n = x.n + x.s;
        if let Some(y_list) = by_n.get(&y_n) {
            for &y in y_list {
                if x_is_unit && !page.products.has_block(x, y) {
                    continue;
                }
                let xy = Tridegree::new(x.n, x.s + y.s, x.f + y.f);

                // Check all three are in turned_page
                if !turned_page.contains_key(&x)
                    || !turned_page.contains_key(&y)
                    || !turned_page.contains_key(&xy)
                {
                    continue;
                }

                let sources = [x, y, xy];
                if !sources
                    .iter()
                    .all(|s| product_max_t.is_none_or(|mt| s.s + s.f <= mt))
                {
                    continue;
                }

                product_triples.push((x, y, xy));
            }
        }
    }

    info!(
        "Computing induced products for {} triples",
        product_triples.len()
    );

    let results: Vec<Vec<(Element, Element, Element)>> = product_triples
        .par_iter()
        .map(|(x, y, xy)| {
            let tb_x = &turned_page[x];
            let tb_y = &turned_page[y];
            let tb_xy = &turned_page[xy];
            compute_induced_products_single(tb_x, tb_y, tb_xy, page)
        })
        .collect();

    for batch in results {
        for (x, y, result) in batch {
            // For basis elements, use the index from the support
            let x_indices: Vec<usize> = vec_support(&x.vec).collect();
            let y_indices: Vec<usize> = vec_support(&y.vec).collect();
            if x_indices.len() == 1 && y_indices.len() == 1 {
                let dim1 = next_page.dim_at(x.degree);
                let dim2 = next_page.dim_at(y.degree);
                let key =
                    ProductKey::new(x.degree, x_indices[0] as u16, y.degree, y_indices[0] as u16);
                new_products.insert(key, result.vec.clone(), dim1, dim2);
            }
        }
    }

    new_products
}

/// Build a new SATPage from pre-computed homology data (turned page).
///
/// Given the turned page data (homology at each tridegree), build the full
/// next-page SATPage with dimensions, induced maps, induced products, and pairs.
/// This is the second half of [`build_next_page`] and can be called independently
/// with a patched turned HashMap for incremental updates.
pub fn build_page_from_turned(
    old_page: &SATPage,
    turned: &HashMap<Tridegree, TurnedBidegree>,
) -> SATPage {
    let new_r = old_page.r + 1;
    // Matches the Python reference's `run.py` per-page cutoff schedule
    // (`max_t -= i - 2` for `i` in `3..=r`, which nets to a decrement of
    // `old_page.r - 1` per turn — 80→79→77→74… for r=2,3,4,5 — not a flat
    // `-1`). A flat `-1` only coincides with this on the very first turn
    // (r=2, decrement=1); every later turn would leave Rust's cutoff too
    // permissive relative to Python's, letting near-the-old-boundary
    // degrees (whose true differential is genuinely unknown, just
    // untracked) leak into real constraint generation instead of being
    // excluded the way Python's shrunk window excludes them.
    let new_max_t = old_page.max_t.map(|t| t - (old_page.r - 1));

    let mut next = SATPage::new(new_r);
    next.max_t = new_max_t;

    // Set dimensions and basis elements
    for (t, tb) in turned {
        if !tb.basis.is_empty() {
            next.dimension.insert(*t, tb.basis.len());
            next.page.insert(*t, tb.basis.clone());
        }
    }

    next.compute_max_values();

    // Compute induced maps
    let induced_maps = compute_induced_maps(old_page, turned);
    for (kind, entries) in induced_maps {
        // Group by source tridegree first
        let mut by_src: HashMap<Tridegree, Vec<(usize, FpVector)>> = HashMap::new();
        for (src, tgt) in &entries {
            let src_indices: Vec<usize> = vec_support(&src.vec).collect();
            if src_indices.len() == 1 {
                by_src
                    .entry(src.degree)
                    .or_default()
                    .push((src_indices[0], tgt.vec.clone()));
            }
        }

        // Build matrices and insert into map table
        let mut matrices: Vec<(Tridegree, Matrix)> = Vec::new();
        for (t, row_data) in by_src {
            let src_dim = next.dim_at(t);
            let tgt_deg = kind.target_degree(t);
            let tgt_dim = next.dim_at(tgt_deg);
            if src_dim == 0 || tgt_dim == 0 {
                continue;
            }
            let mut mat = mat_zero(src_dim, tgt_dim);
            for (row_idx, vec) in &row_data {
                if *row_idx < src_dim && vec.len() == tgt_dim {
                    mat_set_row(&mut mat, *row_idx, vec);
                }
            }
            matrices.push((t, mat));
        }

        let map_table = next.maps.entry(kind).or_insert_with(|| MapTable::new(kind));
        for (t, mat) in matrices {
            map_table.set_matrix(t, mat);
        }
        map_table.dedup_shared();
    }

    // Compute induced products
    let mut induced_products = compute_induced_products(old_page, &next, turned);
    induced_products.dedup_shared_blocks();
    next.products = induced_products;

    next
}

/// Build the exclude set for the next page (E_{r+1}) from unknown differentials
/// on the current page. Ports the original `_make_exclude_set` (sat_ss.py).
///
/// When a differential `d_r` at `(n,s,f)` is undetermined, the homology at both
/// its source `(n,s,f)` and its target `(n,s-1,f+r)` is uncertain (we don't know
/// which classes are cycles or boundaries). Both are excluded from constraint
/// generation on E_{r+1}, so we never emit naturality/Leibniz constraints that
/// treat an uncertain class as a clean survivor. Prior exclusions are carried
/// forward along incoming differentials.
///
/// Only unstable degrees (`n <= s+2`) are stored; stable degrees are handled by
/// `SATPage::is_excluded` mapping to their `(s+2, s, f)` representative.
///
/// Returns `(exclude, target_only)`. `target_only ⊆ exclude` classifies the
/// degrees excluded *solely* as the target of an unknown incoming
/// differential: they are not the source of any unknown d_r themselves, and
/// they are not carried-forward prior exclusions (nor targets of those —
/// conservative, since a carried-forward degree's outgoing differential is
/// untrustworthy for reasons this page cannot see). Under partial turning
/// such a degree's basis consists of real classes that are at worst
/// over-kept (the computed space surjects onto the true page), so its
/// *outgoing* differential is still representable and Leibniz constraints
/// whose only excluded participant is such a degree in the constrained-source
/// (product) position are sound. That relaxation is controlled by the
/// `EHP_RELAX_TARGET_EXCLUDE` kill-switch (default ON; set to `0` to restore
/// the strict behavior where every excluded degree is fully inert) — see
/// [`crate::constraints::relax_target_exclude`].
pub fn make_next_exclude_set(
    page: &SATPage,
    sat_result: &SATResult,
) -> (
    hashbrown::HashSet<Tridegree>,
    hashbrown::HashSet<Tridegree>,
) {
    let r = page.r;

    // Degrees where d_r is unknown (source of an undetermined differential), plus
    // the previous page's exclusions carried forward along their incoming d_r.
    // `hard_uncertain` (carried-forward entries) never spawn target-only
    // classifications; `unknown_sources` do (for their targets).
    let mut unknown_sources: Vec<Tridegree> = Vec::new();
    let mut seen = hashbrown::HashSet::new();
    for &idx in &sat_result.unknown {
        let v = &sat_result.vars[idx];
        let key = Tridegree::new(v.n, v.s, v.f);
        if seen.insert(key) {
            unknown_sources.push(key);
        }
    }
    let mut carried: Vec<Tridegree> = Vec::new();
    for &deg in &page.exclude_set {
        carried.push(deg);
        carried.push(Tridegree::new(deg.n, deg.s + 1, deg.f - r));
    }

    let mut exclude = hashbrown::HashSet::new();
    let mut hard = hashbrown::HashSet::new();
    let mut targets_of_unknown = hashbrown::HashSet::new();
    let insert_pair = |deg: Tridegree,
                           from_unknown: bool,
                           exclude: &mut hashbrown::HashSet<Tridegree>,
                           hard: &mut hashbrown::HashSet<Tridegree>,
                           targets_of_unknown: &mut hashbrown::HashSet<Tridegree>| {
        if deg.n > deg.s + 2 {
            return; // stable: is_excluded() maps to the (s+2, s, f) representative
        }
        exclude.insert(deg);
        hard.insert(deg);
        // The target lives at s-1, so it can be stable even when the source is
        // not (n = s+2 gives n > (s-1)+2). Store its stable representative so
        // `SATPage::is_excluded` lookups agree with what we insert.
        let tgt = stable_rep(Tridegree::new(deg.n, deg.s - 1, deg.f + r));
        exclude.insert(tgt);
        if from_unknown {
            targets_of_unknown.insert(tgt);
        } else {
            hard.insert(tgt);
        }
    };
    for deg in unknown_sources {
        insert_pair(deg, true, &mut exclude, &mut hard, &mut targets_of_unknown);
    }
    for deg in carried {
        insert_pair(deg, false, &mut exclude, &mut hard, &mut targets_of_unknown);
    }

    let target_only: hashbrown::HashSet<Tridegree> = targets_of_unknown
        .difference(&hard)
        .copied()
        .collect();
    (exclude, target_only)
}

/// Build the next page (E_{r+1}) from the current page.
///
/// Returns `Err` if a fully-determined d∘d ≠ 0 is found (a genuine contradiction).
pub fn build_next_page(
    page: &SATPage,
    sat_result: &SATResult,
) -> Result<(SATPage, HashMap<Tridegree, TurnedBidegree>), D2Error> {
    let new_r = page.r + 1;
    // Same r-1 decrement schedule as build_page_from_turned (see the comment
    // there); the two sites must stay in sync.
    let new_max_t = page.max_t.map(|t| t - (page.r - 1));

    // Turn the page (compute homology)
    let turned = turn_page(page, sat_result, new_r, new_max_t)?;

    let mut next = build_page_from_turned(page, &turned);

    // Exclude degrees whose homology is uncertain because a bounding differential
    // was undetermined, so their (possibly wrong) products don't generate
    // spurious constraints on the next page.
    let (exclude, target_only) = make_next_exclude_set(page, sat_result);
    next.exclude_set = exclude;
    next.target_only_exclude = target_only;

    Ok((next, turned))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{ConstraintSystem, DiffVar};
    use crate::solver::solve;

    /// Solve an unconstrained system over the given vars: all unknown.
    fn all_unknown_result(vars: Vec<DiffVar>) -> SATResult {
        let sys = ConstraintSystem::new(vars);
        solve(&sys).expect("empty system is consistent")
    }

    #[test]
    fn target_only_classification() {
        let page = SATPage::new(2);
        // One unknown d_2 at (3,3,2); its target is (3,2,4).
        let res = all_unknown_result(vec![DiffVar::new(3, 3, 2, 0, 0)]);
        let (exclude, target_only) = make_next_exclude_set(&page, &res);
        let src = Tridegree::new(3, 3, 2);
        let tgt = Tridegree::new(3, 2, 4);
        assert!(exclude.contains(&src) && exclude.contains(&tgt));
        assert!(!target_only.contains(&src), "unknown source is hard");
        assert!(target_only.contains(&tgt), "pure target is target-only");
    }

    #[test]
    fn target_that_is_also_source_is_hard() {
        let page = SATPage::new(2);
        // (3,2,4) is the target of the unknown at (3,3,2) AND itself the
        // source of another unknown: it must not be target-only.
        let res = all_unknown_result(vec![
            DiffVar::new(3, 3, 2, 0, 0),
            DiffVar::new(3, 2, 4, 0, 0),
        ]);
        let (exclude, target_only) = make_next_exclude_set(&page, &res);
        let mid = Tridegree::new(3, 2, 4);
        assert!(exclude.contains(&mid));
        assert!(!target_only.contains(&mid));
        // The chain's far end (3,1,6) is a pure target.
        assert!(target_only.contains(&Tridegree::new(3, 1, 6)));
    }

    #[test]
    fn carried_forward_targets_are_hard() {
        let mut page = SATPage::new(2);
        page.exclude_set.insert(Tridegree::new(2, 5, 3));
        // One unrelated unknown far away, so the solve is non-degenerate.
        let res = all_unknown_result(vec![DiffVar::new(9, 9, 9, 0, 0)]);
        let (exclude, target_only) = make_next_exclude_set(&page, &res);
        // Carried-forward degree, its diff-source, and its target are all
        // excluded, none of them target-only (conservative).
        assert!(exclude.contains(&Tridegree::new(2, 5, 3)));
        assert!(exclude.contains(&Tridegree::new(2, 4, 5)));
        assert!(!target_only.contains(&Tridegree::new(2, 5, 3)));
        assert!(!target_only.contains(&Tridegree::new(2, 4, 5)));
        // The unrelated unknown's target is the only target-only entry.
        assert_eq!(target_only.len(), 1);
        assert!(target_only.contains(&Tridegree::new(9, 8, 11)));
    }
}
