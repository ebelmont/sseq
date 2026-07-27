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
        let mut bmat = mat_from_rows(b_basis.clone(), old_dim);
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
    let quotient_map = mat_from_rows(q_rows, old_dim);

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
    let lift_map = mat_from_rows(l_rows, h_dim);

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

    let tridegrees: Vec<Tridegree> = page.page.keys().copied().collect();

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

    for kind in MapKind::all() {
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

    let max_t = next_page.max_t.unwrap_or(i32::MAX);

    let mut product_triples = Vec::new();

    for &x in next_page.page.keys() {
        if x.n > max_t {
            continue;
        }
        if x.s == 0 && x.f == 0 {
            continue;
        }

        let y_n = x.n + x.s;
        if let Some(y_list) = by_n.get(&y_n) {
            for &y in y_list {
                let xy = Tridegree::new(x.n, x.s + y.s, x.f + y.f);

                // Check all three are in turned_page
                if !turned_page.contains_key(&x)
                    || !turned_page.contains_key(&y)
                    || !turned_page.contains_key(&xy)
                {
                    continue;
                }

                let shifted = Tridegree::new(y.n - 1, y.s, y.f);
                if !next_page.page.contains_key(&shifted) {
                    continue;
                }

                let sources = [x, y, xy];
                if !sources
                    .iter()
                    .all(|s| next_page.is_in_computed_polygon_source(*s))
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
    let new_max_t = old_page.max_t.map(|t| t - 1);

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
    }

    // Compute induced products
    let induced_products = compute_induced_products(old_page, &next, turned);
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
    let mut insert_pair = |deg: Tridegree,
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
    let new_max_t = page.max_t.map(|t| t - 1);

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
