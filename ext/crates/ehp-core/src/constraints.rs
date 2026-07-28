use fp::matrix::Matrix;
use fp::vector::FpVector;
use hashbrown::HashMap;
use log::debug;

use crate::gf2::*;
use crate::map::MapKind;
use crate::page::SATPage;
use crate::tridegree::Tridegree;

/// A variable in the GF(2) linear system.
/// Represents entry (row, col) of the differential matrix at tridegree (n, s, f).
/// n is capped at s+2 (stable range collapse).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DiffVar {
    pub n: i32,
    pub s: i32,
    pub f: i32,
    pub row: u16, // target (d_r image) coordinate
    pub col: u16, // source coordinate
}

impl DiffVar {
    pub fn new(n: i32, s: i32, f: i32, row: u16, col: u16) -> Self {
        DiffVar { n, s, f, row, col }
    }

    pub fn tridegree(&self) -> Tridegree {
        Tridegree::new(self.n, self.s, self.f)
    }
}

/// Leibniz pairs that were skipped because a participating degree was excluded,
/// keyed by the excluded degree (stable-representative form). Used by the
/// interpage machinery to re-activate exactly these constraints when a degree
/// becomes certain (ports the original's `excluded_Yconstraints`).
pub type ExcludedLeibniz = HashMap<Tridegree, Vec<(Tridegree, Tridegree)>>;

/// The constraint system: a list of XOR equations over GF(2).
///
/// Each constraint is a list of variable indices whose XOR equals a RHS value.
/// Stored as (row of coefficient matrix, rhs bit).
pub struct ConstraintSystem {
    /// Total number of variables.
    pub num_vars: usize,
    /// The variable list: maps global index -> DiffVar.
    pub vars: Vec<DiffVar>,
    /// Reverse map: DiffVar -> global index.
    pub var_index: HashMap<DiffVar, usize>,
    /// Constraint rows: each is the sorted, deduplicated list of variable
    /// indices with a nonzero (1) coefficient. Sparse by construction --
    /// individual constraints (Leibniz/naturality relations) only ever touch
    /// a handful of variables, so storing a dense `num_vars`-bit row per
    /// constraint here would cost `num_constraints * num_vars` bits (measured
    /// 16GB+ at EHP_MAX_T=100 on just page E_2 before the fix).
    pub rows: Vec<Vec<usize>>,
    /// RHS values.
    pub rhs: Vec<bool>,
    /// Leibniz pairs skipped due to excluded degrees (see [`ExcludedLeibniz`]).
    pub excluded_leibniz: ExcludedLeibniz,
}

/// Build the sorted, deduplicated sparse row (indices with coefficient 1) for
/// a constraint given as a list of variable indices, replicating the
/// GF(2)-XOR/toggle semantics of repeatedly flipping bits: an index appearing
/// an even number of times cancels out to 0, odd number of times leaves 1.
fn sparse_row_from_indices(indices: &[usize]) -> Vec<usize> {
    let mut parity: HashMap<usize, bool> = HashMap::new();
    for &i in indices {
        let e = parity.entry(i).or_insert(false);
        *e = !*e;
    }
    let mut row: Vec<usize> = parity
        .into_iter()
        .filter_map(|(i, set)| set.then_some(i))
        .collect();
    row.sort_unstable();
    row
}

impl ConstraintSystem {
    pub fn new(vars: Vec<DiffVar>) -> Self {
        let num_vars = vars.len();
        let var_index: HashMap<DiffVar, usize> =
            vars.iter().enumerate().map(|(i, v)| (*v, i)).collect();
        ConstraintSystem {
            num_vars,
            vars,
            var_index,
            rows: Vec::new(),
            rhs: Vec::new(),
            excluded_leibniz: ExcludedLeibniz::new(),
        }
    }

    /// Add a constraint: XOR of variables at given indices equals rhs_val.
    pub fn add_constraint_indices(&mut self, indices: &[usize], rhs_val: bool) {
        self.rows.push(sparse_row_from_indices(indices));
        self.rhs.push(rhs_val);
    }

    /// Add a constraint from a list of DiffVars.
    /// Returns false if any variable is not in the system (constraint skipped).
    pub fn add_constraint_vars(&mut self, vars: &[DiffVar], rhs_val: bool) -> bool {
        let mut indices = Vec::with_capacity(vars.len());
        for v in vars {
            match self.var_index.get(v) {
                Some(&i) => indices.push(i),
                None => return false,
            }
        }
        self.add_constraint_indices(&indices, rhs_val);
        true
    }

    /// Number of constraints.
    pub fn num_constraints(&self) -> usize {
        self.rows.len()
    }

}

/// Opt-in switch for the target-only exclusion relaxation (env
/// `EHP_RELAX_TARGET_EXCLUDE`, read once). **Default OFF** (strict: excluded
/// degrees never get differential variables and every Leibniz pair touching
/// an excluded degree is skipped) — the relaxation was the enabling change
/// for the t=80 outside-diffs E4 UNSAT (2026-07-07 certificate,
/// notes/CHANGES_2026-07-07.md §11–12c), and its validation gauntlet (§8a)
/// was never completed, so the user chose strict-by-default. Set to `1` to
/// enable: a degree excluded *only* as the target of an unknown incoming
/// differential (see [`crate::page::SATPage::target_only_exclude`]) still
/// gets its outgoing d_r variables, and Leibniz pairs may treat it as the
/// constrained product degree (guarded by the e_d_deg2 exclusion check in
/// `make_leibniz_constraint_single`).
pub fn relax_target_exclude() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("EHP_RELAX_TARGET_EXCLUDE").map_or(false, |v| v == "1")
    })
}

/// Enumerate the differential variables for a single source tridegree
/// (empty unless both the source and the diff target are unexcluded, in
/// bounds, and have nonzero dimension).
///
/// With [`relax_target_exclude`] ON, a source degree whose exclusion is
/// target-only still gets variables: its basis is real (at worst over-kept),
/// so its outgoing differential is representable. The diff target must still
/// be unexcluded (its coordinates must be trustworthy for the matrix rows).
pub fn make_basis_single(page: &SATPage, t: Tridegree) -> Vec<DiffVar> {
    let r = page.r;
    let mut vars = Vec::new();

    if t.n > t.s + 2 {
        return vars;
    }
    let diff_target = t.diff_target(r);

    let src_hard_excluded = page.is_excluded(t)
        && !(relax_target_exclude() && page.is_excluded_target_only(t));
    if src_hard_excluded || page.is_excluded(diff_target) {
        return vars;
    }
    if !page.is_in_computed_polygon_source(t) || !page.is_in_computed_polygon_source(diff_target) {
        return vars;
    }

    let src_dim = page.dim_at(t);
    let tgt_dim = page.dim_at(diff_target);

    if src_dim == 0 || tgt_dim == 0 {
        return vars;
    }

    for row in 0..tgt_dim {
        for col in 0..src_dim {
            vars.push(DiffVar::new(t.n, t.s, t.f, row as u16, col as u16));
        }
    }
    vars
}

/// Enumerate all differential variables for the page.
///
/// Each variable represents entry d_r[row, col] at tridegree (n, s, f),
/// where the differential goes from (n, s, f) to (n, s-1, f+r).
/// n is capped at s+2 (stable range).
pub fn make_basis(page: &SATPage, cutoff: i32) -> Vec<DiffVar> {
    let mut vars = Vec::new();

    let mut keys: Vec<Tridegree> = page.dimension.keys().copied().collect();
    keys.sort();

    for t in keys {
        if t.s + t.f > cutoff {
            continue;
        }
        vars.extend(make_basis_single(page, t));
    }

    vars
}

/// Symbolic matrix-vector product: computes which variables appear in each entry
/// of `A * D[src_tridegree]`, where D is the unknown differential matrix.
///
/// Returns a list of lists: for each entry of the result, the list of variable
/// indices whose XOR equals that entry. Returns `None` if a referenced
/// differential is not a variable in `var_index` — silently dropping the term
/// would corrupt the constraint (the original raises here), so the caller must
/// skip the whole constraint set.
fn matrix_mult_left(
    a: &Matrix,
    src_n: i32,
    src_s: i32,
    src_f: i32,
    src_dim: usize,
    var_index: &HashMap<DiffVar, usize>,
    col: Option<usize>,
) -> Option<Vec<Vec<usize>>> {
    let a_rows = a.rows();
    let n_var = src_n.min(src_s + 2);

    let col_range: Vec<usize> = match col {
        Some(c) => vec![c],
        None => (0..src_dim).collect(),
    };

    let mut result = Vec::new();
    for i in 0..a_rows {
        for &j in &col_range {
            let mut indices = Vec::new();
            for k in a.row(i).iter_nonzero().map(|(k, _)| k) {
                let var = DiffVar::new(n_var, src_s, src_f, k as u16, j as u16);
                indices.push(*var_index.get(&var)?);
            }
            result.push(indices);
        }
    }
    Some(result)
}

/// Symbolic matrix-vector product: computes which variables appear in each entry
/// of `D[tgt_tridegree] * A`, where D is the unknown differential matrix.
/// `None` on a missing variable, like [`matrix_mult_left`].
fn matrix_mult_right(
    a: &Matrix,
    tgt_n: i32,
    tgt_s: i32,
    tgt_f: i32,
    _r: i32,
    tgt_dim: usize,
    _src_dim: usize,
    var_index: &HashMap<DiffVar, usize>,
) -> Option<Vec<Vec<usize>>> {
    let a_cols = a.columns();
    let a_rows = a.rows(); // = B_cols in Python = source dimension of D
    let n_var = tgt_n.min(tgt_s + 2);
    // D[tgt] maps (tgt_n, tgt_s, tgt_f) -> (tgt_n, tgt_s-1, tgt_f+r)
    // D has dimensions: tgt_dim rows × a_rows cols
    let d_rows = tgt_dim;

    let mut result = Vec::new();
    for i in 0..d_rows {
        for j in 0..a_cols {
            let mut indices = Vec::new();
            for t in 0..a_rows {
                if mat_get(&a, t, j) {
                    let var = DiffVar::new(n_var, tgt_s, tgt_f, i as u16, t as u16);
                    indices.push(*var_index.get(&var)?);
                }
            }
            result.push(indices);
        }
    }
    Some(result)
}

/// Compute the symmetric difference of two lists of variable indices.
/// This encodes the XOR equation: lhs[i] XOR rhs[i] = 0.
fn equals_rel(lhs: &[Vec<usize>], rhs: &[Vec<usize>]) -> Vec<Vec<usize>> {
    assert_eq!(lhs.len(), rhs.len());
    let mut result = Vec::new();
    for (l, r) in lhs.iter().zip(rhs.iter()) {
        if l.is_empty() && r.is_empty() {
            continue;
        }
        let mut combined: Vec<usize> = Vec::new();
        // Symmetric difference
        let mut l_set: hashbrown::HashSet<usize> = l.iter().copied().collect();
        let mut r_set: hashbrown::HashSet<usize> = r.iter().copied().collect();
        for &v in l {
            if !r_set.remove(&v) {
                combined.push(v);
            }
        }
        for &v in r {
            if !l_set.remove(&v) {
                // Only add if not already removed by symmetric diff
                if !l.contains(&v) {
                    combined.push(v);
                }
            }
        }
        // Proper symmetric difference
        let mut l_sorted = l.clone();
        let mut r_sorted = r.clone();
        l_sorted.sort();
        r_sorted.sort();
        let mut sym_diff = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < l_sorted.len() && j < r_sorted.len() {
            if l_sorted[i] < r_sorted[j] {
                sym_diff.push(l_sorted[i]);
                i += 1;
            } else if l_sorted[i] > r_sorted[j] {
                sym_diff.push(r_sorted[j]);
                j += 1;
            } else {
                i += 1;
                j += 1;
            }
        }
        while i < l_sorted.len() {
            sym_diff.push(l_sorted[i]);
            i += 1;
        }
        while j < r_sorted.len() {
            sym_diff.push(r_sorted[j]);
            j += 1;
        }

        if !sym_diff.is_empty() {
            result.push(sym_diff);
        }
    }
    result
}

/// Symmetric difference, preserving empty entries for alignment.
fn equals_rel_preserve(lhs: &[Vec<usize>], rhs: &[Vec<usize>]) -> Vec<Vec<usize>> {
    assert_eq!(lhs.len(), rhs.len());
    let mut result = Vec::with_capacity(lhs.len());
    for (l, r) in lhs.iter().zip(rhs.iter()) {
        let mut l_sorted = l.clone();
        let mut r_sorted = r.clone();
        l_sorted.sort();
        r_sorted.sort();
        let mut sym_diff = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < l_sorted.len() && j < r_sorted.len() {
            if l_sorted[i] < r_sorted[j] {
                sym_diff.push(l_sorted[i]);
                i += 1;
            } else if l_sorted[i] > r_sorted[j] {
                sym_diff.push(r_sorted[j]);
                j += 1;
            } else {
                i += 1;
                j += 1;
            }
        }
        while i < l_sorted.len() {
            sym_diff.push(l_sorted[i]);
            i += 1;
        }
        while j < r_sorted.len() {
            sym_diff.push(r_sorted[j]);
            j += 1;
        }
        result.push(sym_diff);
    }
    result
}

/// Generate naturality constraints for a single map at a single tridegree.
///
/// The naturality constraint is: phi_target * d_src = d_tgt * phi_source
/// where phi is the map (E, H, or P).
pub fn make_naturality_constraint_single(
    page: &SATPage,
    t: Tridegree,
    map_kind: MapKind,
    var_index: &HashMap<DiffVar, usize>,
) -> Vec<Vec<usize>> {
    let r = page.r;

    // No domain_check gate here: the Python reference's STANDARD_MAPS
    // domain_check lambdas (e.g. P's `n >= 5 && n % 2 == 1`) are never
    // actually wired up at runtime — SATPage.initialize_maps() constructs
    // each Map(name, n_transform, s_transform, f_transform) without passing
    // domain_check, so every map's check silently defaults to `lambda: True`
    // (see Map.__init__ in lib.py). Confirmed by direct repro against
    // ext/data/E2: Python emits real P constraints at e.g. t=(18,56,19),
    // which MapKind::P::domain_check's formula would (correctly, per the
    // *documented* math) reject. The real domain restrictions Python
    // actually enforces are the explicit checks below (E's stable-range
    // skip, P's `f - 2 < 0 || s - n < 0`), not this dead lambda.

    // For E map: skip stable range (naturality is trivial when n >= s+2)
    if map_kind == MapKind::E && t.n >= t.s + 2 {
        return Vec::new();
    }

    // Determine source and target tridegrees for the naturality square
    let (src, diff_src, tgt) = match map_kind {
        MapKind::P => {
            // P is "inverted": source of the naturality square is at (2n+1, s-n+1, f-2)
            let src = Tridegree::new(2 * t.n + 1, t.s - t.n + 1, t.f - 2);
            let diff_src = Tridegree::new(2 * t.n + 1, t.s - t.n, t.f - 2 + r);
            if t.f - 2 < 0 || t.s - t.n < 0 {
                return Vec::new();
            }
            (src, diff_src, t)
        }
        _ => {
            let tgt = map_kind.target_degree(t);
            let diff_src = t.diff_target(r);
            (t, diff_src, tgt)
        }
    };

    let diff_tgt = tgt.diff_target(r);

    // Check bounds
    // The square is only trustworthy r-1 inside the cutoff (the original's
    // `src_s + src_f + r - 1 <= max_t` check, applied to both columns).
    if let Some(max_t) = page.max_t {
        if src.s + src.f + r - 1 > max_t || tgt.s + tgt.f + r - 1 > max_t {
            return Vec::new();
        }
    }

    let all_trideg = [src, diff_src, tgt, diff_tgt];
    for &td in &all_trideg {
        if page.is_excluded(td) || !page.is_in_computed_polygon_source(td) || !page.is_in_computed_polygon(td) {
            return Vec::new();
        }
    }

    let src_dim = page.dim_at(src);
    let diff_src_dim = page.dim_at(diff_src);
    let tgt_dim = page.dim_at(tgt);
    let diff_tgt_dim = page.dim_at(diff_tgt);

    // No early return on a zero dimension here (the original has none either):
    // when e.g. tgt_dim == 0, the RHS (D[tgt] * phi_source) is trivially zero
    // while the LHS (phi_target * D[src]) can still be nonzero, forcing those
    // src-differential entries to zero — a real constraint, not a no-op. The
    // matrix_mult_left/right helpers below already degrade correctly to empty
    // output on a zero dimension, so skipping here just discarded constraints.

    // Get map matrices (transposed, per Python convention). Missing data means
    // the zero map (or stable-E identity) — the square still constrains.
    let phi_source = page.map_matrix_ref(map_kind, src).to_matrix_transposed();
    let phi_target = page.map_matrix_ref(map_kind, diff_src).to_matrix_transposed();

    // Check dimensions
    if phi_source.rows() != tgt_dim || phi_source.columns() != src_dim {
        return Vec::new();
    }
    if phi_target.rows() != diff_tgt_dim || phi_target.columns() != diff_src_dim {
        return Vec::new();
    }

    // LHS: phi_target * D[src] ; RHS: D[tgt] * phi_source. A missing variable
    // means the square references a differential outside the system — skip it
    // entirely rather than emit a corrupted constraint (the original raises).
    let mult = || -> Option<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
        let lhs = matrix_mult_left(
            &phi_target,
            src.n,
            src.s,
            src.f,
            src_dim,
            var_index,
            None,
        )?;
        let rhs = matrix_mult_right(
            &phi_source,
            tgt.n,
            tgt.s,
            tgt.f,
            r,
            diff_tgt_dim,
            tgt_dim,
            var_index,
        )?;
        Some((lhs, rhs))
    };
    let Some((lhs, rhs)) = mult() else {
        debug!(
            "naturality {} square at ({},{},{}) references a non-variable differential — skipped",
            map_kind.name(), t.n, t.s, t.f
        );
        return Vec::new();
    };

    equals_rel(&lhs, &rhs)
}

/// Generate all naturality constraints for a map.
pub fn make_naturality_constraints(
    page: &SATPage,
    cutoff: i32,
    map_kind: MapKind,
    var_index: &HashMap<DiffVar, usize>,
) -> Vec<Vec<usize>> {
    let mut constraints = Vec::new();

    // The original iterates n in [2, s+2]: stable copies (n > s+2) share the
    // representative's variables, and map data is only stored at the
    // representative — with missing-data-means-zero semantics, generating
    // their squares directly would emit wrong constraints.
    for s in 0..=cutoff {
        for f in 0..=(cutoff - s) {
            for n in 2..=(s + 2) {
                let t = Tridegree::new(n, s, f);
                let new = make_naturality_constraint_single(page, t, map_kind, var_index);
                constraints.extend(new);
            }
        }
    }

    // Deduplicate
    deduplicate_constraints(constraints)
}

/// Generate Leibniz constraints for a single pair of degrees.
///
/// Encodes: d(x * E(y)) = x * d(E(y)) + d(x) * y
///
/// If `excluded_out` is provided and the pair is skipped because a
/// participating degree is excluded, the pair is recorded under each excluded
/// degree (stable-representative form) so it can be re-activated later.
pub fn make_leibniz_constraint_single(
    page: &SATPage,
    deg1: Tridegree,
    deg2: Tridegree,
    var_index: &HashMap<DiffVar, usize>,
    mut excluded_out: Option<&mut ExcludedLeibniz>,
) -> Vec<Vec<usize>> {
    let r = page.r;

    if !page.is_in_computed_polygon_source(deg2) {
        return Vec::new();
    }
    if page.dim_at(deg2) == 0 {
        return Vec::new();
    }

    let e_deg2 = Tridegree::new(deg2.n + 1, deg2.s, deg2.f);
    let prod_deg = Tridegree::new(deg1.n, deg1.s + deg2.s, deg1.f + deg2.f);
    let prod_dr_deg = prod_deg.diff_target(r);
    let d_deg1 = deg1.diff_target(r);
    let d_deg2 = deg2.diff_target(r);

    let all_trideg = [deg1, deg2, e_deg2, d_deg2, d_deg1, prod_deg, prod_dr_deg];

    if !page.is_in_computed_polygon_source(prod_deg)
        || !page.is_in_computed_polygon_source(prod_dr_deg)
    {
        return Vec::new();
    }
    if page.dim_at(prod_dr_deg) == 0 {
        return Vec::new();
    }

    // The degree of d(E(deg2)) — consulted by the RHS1 Ytilde columns and by
    // the E-naturality substitution d(E y) = E(d y), but historically absent
    // from `all_trideg` in BOTH pipelines (latent in the original because
    // every pair reaching it also has an excluded prod_deg and dies there in
    // strict mode). The relax carve-out must therefore check it explicitly:
    // the t=80 outside-diffs E4 UNSAT certificate's contradicting row was a
    // relaxed pair whose e_d_deg2 was excluded (unknown d3-out) — its data
    // there is not trustworthy. Only the carve-out consults this (strict
    // mode stays byte-identical).
    // The 8th consulted degree: e_d_deg2 = where d(E(y)) lives. The RHS1
    // Ytilde columns are products into its basis and the d(Ey) = E(dy)
    // substitution contracts through its coordinates — but it was
    // historically ABSENT from all_trideg in BOTH pipelines (d_deg2 is
    // checked; unstably its E-shift is a different degree). That leak let
    // pairs consult over-kept coordinates / silently-zeroed product blocks
    // at an excluded degree even in strict mode — the mechanism behind the
    // t=80 outside-diffs E4 UNSAT certificate (notes/CHANGES_2026-07-07.md
    // §12b) and consistent with the strict-mode d4(3,32,9) UNSAT-on-assert
    // (§8b). Checked unconditionally since 2026-07-08 (user decision: every
    // consulted degree with prior-page uncertainty must skip the pair).
    let e_d_deg2 = e_deg2.diff_target(r);

    let relax = relax_target_exclude();
    let mut excluded_any = false;
    let record = |td: Tridegree, excluded_out: &mut Option<&mut ExcludedLeibniz>| {
        if let Some(map) = excluded_out.as_deref_mut() {
            let rep = if td.n > td.s + 2 {
                Tridegree::new(td.s + 2, td.s, td.f)
            } else {
                td
            };
            map.entry(rep).or_default().push((deg1, deg2));
        }
    };
    for (i, &td) in all_trideg.iter().enumerate() {
        if page.is_excluded(td) {
            // Position 5 = prod_deg, the source of the differential this pair
            // constrains. A target-only exclusion there is safe: the degree's
            // basis is real but possibly over-kept, and forcing the (lifted)
            // differential on an over-kept class is at worst gauge-fixing a
            // phantom — never wrong about surviving classes — provided every
            // OTHER participating degree is clean (checked here and in the
            // unconditional e_d_deg2 check below). Such pairs are emitted, so
            // they are deliberately NOT recorded in `excluded_out` (nothing
            // to re-activate, and replaying them after an un-exclusion could
            // re-emit in changed coordinates).
            if relax && i == 5 && page.is_excluded_target_only(td) {
                continue;
            }
            excluded_any = true;
            record(td, &mut excluded_out);
        }
    }
    if page.is_excluded(e_d_deg2) {
        excluded_any = true;
        record(e_d_deg2, &mut excluded_out);
    }
    if excluded_any {
        return Vec::new();
    }

    let dim1 = page.dim_at(deg1);
    let dim2 = page.dim_at(deg2);
    let prod_dr_dim = page.dim_at(prod_dr_deg);

    if dim1 == 0 || dim2 == 0 {
        return Vec::new();
    }

    // Product-support fast path: the only product blocks this relation can
    // consult are (deg1, e_deg2) for the LHS, (deg1, d(E(deg2))) for RHS1,
    // and (d_deg1, deg2) for RHS2. `ProductTable::multiply` returns zero for
    // an absent block, so with all three missing every term below is zero
    // and `equals_rel` emits nothing (the `skip()` paths only trigger on
    // nonzero products). Three hash lookups replace the whole (i1, i2) loop;
    // most surviving pairs at large max_t have no product data at all.
    // (Placed AFTER the exclusion recording above so `excluded_out`
    // bookkeeping is unchanged; `e_d_deg2` is defined before the exclusion
    // loop.)
    if !page.products.has_block(deg1, e_deg2)
        && !page.products.has_block(deg1, e_d_deg2)
        && !page.products.has_block(d_deg1, deg2)
    {
        return Vec::new();
    }

    let mut all_constraints = Vec::new();

    // Get E map matrix at deg2 (zero / stable-identity when no data — the
    // relation still constrains the other terms). Borrowed/virtual form:
    // materializing the default per pair dominated t=80 startup.
    let e_matrix = page.map_matrix_ref(MapKind::E, deg2);

    for i2 in 0..dim2 {
        let elt2_vec = vec_basis(dim2, i2);

        // Compute E(elt2)
        let e_elt2_vec = e_matrix.apply_vec(&elt2_vec);

        for i1 in 0..dim1 {
            // Compute product elt1 * E(elt2)
            let prod_vec = page.products.multiply(
                deg1,
                &vec_basis(dim1, i1),
                e_deg2,
                &e_elt2_vec,
                page.dim_at(prod_deg),
            );

            // LHS: d_r(elt1 * E(elt2))
            // Python: prod_mat = column matrix (prod_dim x 1)
            // matrix_mult_right computes D[prod_deg] * prod_mat
            // A missing variable anywhere in the relation means it references
            // a differential outside the system; skip the whole pair rather
            // than emit a corrupted constraint (the original raises here).
            let skip = || {
                debug!(
                    "leibniz pair ({},{},{}) x ({},{},{}) references a non-variable differential — skipped",
                    deg1.n, deg1.s, deg1.f, deg2.n, deg2.s, deg2.f
                );
                Vec::new()
            };

            let lhs = if prod_vec.is_zero() {
                vec![vec![]; prod_dr_dim]
            } else {
                let prod_col = mat_col_matrix(&prod_vec);
                match matrix_mult_right(
                    &prod_col,
                    prod_deg.n,
                    prod_deg.s,
                    prod_deg.f,
                    r,
                    prod_dr_dim,
                    page.dim_at(prod_deg),
                    var_index,
                ) {
                    Some(v) => v,
                    None => return skip(),
                }
            };

            // RHS1: Ytilde(elt1, d_r(E(elt2)))
            // Build Ytilde matrix: columns are products of elt1 with basis elements of target
            let target_deg = Tridegree::new(deg2.n + 1, deg2.s - 1, deg2.f + r);
            let tgt_dim = page.dim_at(target_deg);

            let rhs1 = if deg2.s == 0 || tgt_dim == 0 {
                vec![vec![]; prod_dr_dim]
            } else {
                // Build Ytilde matrix
                let mut ytil_cols: Vec<FpVector> = Vec::with_capacity(tgt_dim);
                for j in 0..tgt_dim {
                    let basis_j = vec_basis(tgt_dim, j);
                    let col_prod = page.products.multiply(
                        deg1,
                        &vec_basis(dim1, i1),
                        target_deg,
                        &basis_j,
                        prod_dr_dim,
                    );
                    ytil_cols.push(col_prod);
                }

                // All-zero products ⇒ Ytilde is the zero matrix ⇒ same result
                // as the mat_is_zero branch below — skip WITHOUT building the
                // matrix (the build was ~85% of a t=80 startup profile; most
                // pairs have all-zero products).
                if ytil_cols.iter().all(|c| c.is_zero()) {
                    vec![vec![]; prod_dr_dim]
                } else {
                // Build Ytil matrix (prod_dr_dim rows × tgt_dim cols)
                let mut ytil_rows = Vec::with_capacity(prod_dr_dim);
                for row in 0..prod_dr_dim {
                    let mut r_vec = vec_zero(tgt_dim);
                    for (col, ytcol) in ytil_cols.iter().enumerate() {
                        if ytcol.len() > row && vec_get(ytcol, row) {
                            vec_set(&mut r_vec, col, true);
                        }
                    }
                    ytil_rows.push(r_vec);
                }
                let ytil = mat_from_rows(&ytil_rows, tgt_dim);

                if mat_is_zero(&ytil) {
                    vec![vec![]; prod_dr_dim]
                } else {
                    // Get E matrix at d_deg2 (transposed); zero when no data —
                    // the constraint lhs = rhs2 must still be emitted.
                    let e_mat_at_d =
                        page.map_matrix_ref(MapKind::E, d_deg2).to_matrix_transposed();

                    // Ytil * E_mat gives the combined matrix
                    let combined = mat_mul(&ytil, &e_mat_at_d);

                    match matrix_mult_left(
                        &combined,
                        deg2.n,
                        deg2.s,
                        deg2.f,
                        dim2,
                        var_index,
                        Some(i2),
                    ) {
                        Some(v) => v,
                        None => return skip(),
                    }
                }
                }
            };

            // RHS2: Ytilde(d_r(elt1), elt2)
            let target_deg2 = d_deg1;
            let tgt_dim2 = page.dim_at(target_deg2);

            let rhs2 = if deg1.s == 0 || tgt_dim2 == 0 {
                vec![vec![]; prod_dr_dim]
            } else {
                let mut ytil2_cols: Vec<FpVector> = Vec::with_capacity(tgt_dim2);
                for j in 0..tgt_dim2 {
                    let basis_j = vec_basis(tgt_dim2, j);
                    let col_prod = page.products.multiply(
                        target_deg2,
                        &basis_j,
                        deg2,
                        &vec_basis(dim2, i2),
                        prod_dr_dim,
                    );
                    ytil2_cols.push(col_prod);
                }

                // Same all-zero early-out as rhs1: zero Ytilde ⇒ identical
                // result to the mat_is_zero branch, without the matrix build.
                if ytil2_cols.iter().all(|c| c.is_zero()) {
                    vec![vec![]; prod_dr_dim]
                } else {
                let mut ytil2_rows = Vec::with_capacity(prod_dr_dim);
                for row in 0..prod_dr_dim {
                    let mut r_vec = vec_zero(tgt_dim2);
                    for (col, ytcol) in ytil2_cols.iter().enumerate() {
                        if ytcol.len() > row && vec_get(ytcol, row) {
                            vec_set(&mut r_vec, col, true);
                        }
                    }
                    ytil2_rows.push(r_vec);
                }
                let ytil2 = mat_from_rows(&ytil2_rows, tgt_dim2);

                if mat_is_zero(&ytil2) {
                    vec![vec![]; prod_dr_dim]
                } else {
                    match matrix_mult_left(
                        &ytil2,
                        deg1.n,
                        deg1.s,
                        deg1.f,
                        dim1,
                        var_index,
                        Some(i1),
                    ) {
                        Some(v) => v,
                        None => return skip(),
                    }
                }
                }
            };

            // Combine: lhs = rhs1 + rhs2 (XOR)
            let rhs = equals_rel_preserve(&rhs1, &rhs2);
            let new_rels = equals_rel(&lhs, &rhs);
            all_constraints.extend(new_rels);
        }
    }

    all_constraints
}

/// Generate all Leibniz constraints, plus the map of pairs skipped because a
/// degree was excluded (keyed by the excluded degree).
pub fn make_leibniz_constraints(
    page: &SATPage,
    cutoff: i32,
    var_index: &HashMap<DiffVar, usize>,
) -> (Vec<Vec<usize>>, ExcludedLeibniz) {
    let _r = page.r;
    let mut constraints = Vec::new();
    let mut excluded = ExcludedLeibniz::new();

    // Index degrees by n for efficient lookup
    let mut degrees_by_n: HashMap<i32, Vec<(i32, i32)>> = HashMap::new();
    for (&t, &d) in &page.dimension {
        if t.s + t.f <= cutoff && d > 0 && t.f >= 1 {
            degrees_by_n.entry(t.n).or_default().push((t.s, t.f));
        }
    }

    // max_t is a bound on t = s + f only — never on n/th (user decision,
    // 2026-07-25): for a fixed t, every relevant n is assumed present in the
    // data, so no n-derived cutoff (neither the stem cutoff nor a
    // data-derived max n) belongs here at all. th1 therefore ranges over
    // every n that actually has source degrees (no range bound needed); th2
    // needs no explicit bound either, since deg2's own t = s2 + f2 is
    // automatically <= cutoff by construction (s3 + f3 <= cutoff, s1 + f1 >=
    // 0) — page.dim_at(deg2) == 0 / is_in_computed_polygon* inside
    // make_leibniz_constraint_single already reject any degree the data
    // doesn't actually have.
    for s3 in 0..=cutoff {
        for f3 in 1..=(cutoff - s3) {
            for (&th1, source_degrees) in &degrees_by_n {
                for &(s1, f1) in source_degrees {
                    let th2 = th1 + s1 - 1;
                    let f2 = f3 - f1;
                    let s2 = s3 - s1;

                    // Python's _leibniz_helper also skips s2 < 1 — this was
                    // missing here, letting degenerate deg2 (s <= 0) pairs
                    // through that Python never considers.
                    if s2 < 1 {
                        continue;
                    }

                    let deg1 = Tridegree::new(th1, s1, f1);
                    let deg2 = Tridegree::new(th2, s2, f2);

                    let new = make_leibniz_constraint_single(
                        page,
                        deg1,
                        deg2,
                        var_index,
                        Some(&mut excluded),
                    );
                    constraints.extend(new);
                }
            }
        }
    }

    debug!(
        "Generated {} Leibniz constraints ({} excluded degrees recorded)",
        constraints.len(),
        excluded.len()
    );
    (deduplicate_constraints(constraints), excluded)
}

/// Generate constraints for known differential values.
pub fn make_known_constraints(
    known: &HashMap<DiffVar, bool>,
    var_index: &HashMap<DiffVar, usize>,
) -> Vec<(Vec<usize>, bool)> {
    let mut constraints = Vec::new();
    for (var, &val) in known {
        if let Some(&idx) = var_index.get(var) {
            constraints.push((vec![idx], val));
        } else {
            log::warn!(
                "known diff d({},{},{})[{},{}] = {} has no variable (zero dims or out of range) — NOT enforced",
                var.n, var.s, var.f, var.row, var.col, val as u8,
            );
        }
    }
    constraints
}

/// Extend the variable list so every user-asserted (known) differential is
/// representable, even where automatic variable creation skipped the degree.
///
/// `make_basis` skips degrees that are excluded (uncertainty from the previous
/// page) or outside the polygon/cutoff — right for *derived* constraints, but
/// it made explicitly asserted differentials silently unenforceable on E₃ and
/// higher (E₂ has no exclusions). A known diff is ground truth the user takes
/// responsibility for, so we add the full variable block for its degree pair;
/// the pinned entry becomes a constraint while the remaining entries stay
/// free (honestly unknown). Naturality/Leibniz at excluded degrees remain
/// disabled, and turning still zeroes the matrix while any entry is unknown.
fn add_known_diff_vars(page: &SATPage, known: &HashMap<DiffVar, bool>, vars: &mut Vec<DiffVar>) {
    let r = page.r;
    let mut have: hashbrown::HashSet<(i32, i32, i32)> =
        vars.iter().map(|v| (v.n, v.s, v.f)).collect();

    let mut degrees: Vec<Tridegree> = known
        .keys()
        .map(|dv| {
            let n_var = dv.n.min(dv.s + 2);
            Tridegree::new(n_var, dv.s, dv.f)
        })
        .collect();
    degrees.sort();
    degrees.dedup();

    for t in degrees {
        if have.contains(&(t.n, t.s, t.f)) {
            continue;
        }
        let src_dim = page.dim_at(t);
        let tgt_dim = page.dim_at(t.diff_target(r));
        if src_dim == 0 || tgt_dim == 0 {
            // No classes to connect — make_known_constraints will warn.
            continue;
        }
        debug!(
            "adding variable block at ({},{},{}) for user-asserted d_{} (degree was excluded or out of bounds)",
            t.n, t.s, t.f, r,
        );
        have.insert((t.n, t.s, t.f));
        for row in 0..tgt_dim {
            for col in 0..src_dim {
                vars.push(DiffVar::new(t.n, t.s, t.f, row as u16, col as u16));
            }
        }
    }
}

/// Opt-out switch for the d²=0 one-leg linearization (env `EHP_D2_LINEAR`,
/// read once; set to `0` to disable). See [`make_d2_linear_rows`].
pub fn d2_linearize_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("EHP_D2_LINEAR").map_or(true, |v| v != "0"))
}

/// d²=0, linearized against the current solve: for composable degrees
/// A → B → C on this page (`B = A.diff_target(r)`, `C = B.diff_target(r)`),
/// the relation `Σⱼ d(B)[k,j]·d(A)[j,i] = 0` is quadratic in general, but
/// the moment one leg is determined it becomes LINEAR in the other:
///
/// - column `i` of `d(A)` fully determined with support `S` ⇒ for every `k`:
///   `Σ_{j∈S} d(B)[k,j] = 0`;
/// - row `k` of `d(B)` fully determined with support `T` ⇒ for every `i`:
///   `Σ_{j∈T} d(A)[j,i] = 0`.
///
/// These rows are true consequences of d∘d = 0 and the (trustworthy,
/// clean-degree) determined values, so adding them is sound. Emitted only
/// when A, B and C are all unexcluded, in the computed polygon, and
/// nonzero-dimensional — the uncertainty-tracking contract: never build a
/// constraint through a degree with prior-page uncertainty.
///
/// Returns sparse rows `(variable indices, rhs)` ready for
/// [`crate::interpage::update_sat_result`]. Rows whose XOR support is empty
/// (all-zero determined leg) are skipped — they are `0 = 0`.
pub fn make_d2_linear_rows(
    page: &SATPage,
    res: &crate::result::SATResult,
) -> Vec<(Vec<usize>, bool)> {
    let r = page.r;
    let entry = |t: Tridegree, row: u16, col: u16| -> Option<bool> {
        let n_var = t.n.min(t.s + 2);
        let idx = *res.var_index.get(&DiffVar::new(n_var, t.s, t.f, row, col))?;
        if res.unknown.contains(&idx) {
            None
        } else {
            Some(res.offset.entry(idx) != 0)
        }
    };
    let var_of = |t: Tridegree, row: u16, col: u16| -> Option<usize> {
        let n_var = t.n.min(t.s + 2);
        res.var_index
            .get(&DiffVar::new(n_var, t.s, t.f, row, col))
            .copied()
    };

    let mut rows: Vec<(Vec<usize>, bool)> = Vec::new();
    let mut degrees: Vec<Tridegree> = page
        .dimension
        .iter()
        .filter(|(_, &d)| d > 0)
        .map(|(&t, _)| t)
        .collect();
    degrees.sort();

    for &a_deg in &degrees {
        // Stable copies share the representative's differential — generating
        // their rows would duplicate the rep's (in identical variables).
        if a_deg.n > a_deg.s + 2 {
            continue;
        }
        let b_deg = a_deg.diff_target(r);
        let c_deg = b_deg.diff_target(r);
        let (a, b, c) = (page.dim_at(a_deg), page.dim_at(b_deg), page.dim_at(c_deg));
        if a == 0 || b == 0 || c == 0 {
            continue;
        }
        if page.is_excluded(a_deg) || page.is_excluded(b_deg) || page.is_excluded(c_deg) {
            continue;
        }
        if !page.is_in_computed_polygon_source(a_deg)
            || !page.is_in_computed_polygon_source(b_deg)
            || !page.is_in_computed_polygon(c_deg)
        {
            continue;
        }

        // Direction 1: a fully determined column of d(A) constrains d(B).
        for i in 0..a as u16 {
            let support: Option<Vec<u16>> = (0..b as u16)
                .map(|j| entry(a_deg, j, i).map(|v| (j, v)))
                .collect::<Option<Vec<_>>>()
                .map(|col| col.into_iter().filter(|&(_, v)| v).map(|(j, _)| j).collect());
            let Some(s) = support else { continue };
            if s.is_empty() {
                continue;
            }
            for k in 0..c as u16 {
                let idxs: Option<Vec<usize>> =
                    s.iter().map(|&j| var_of(b_deg, k, j)).collect();
                if let Some(mut idxs) = idxs {
                    idxs.sort_unstable();
                    rows.push((idxs, false));
                }
            }
        }

        // Direction 2: a fully determined row of d(B) constrains d(A).
        for k in 0..c as u16 {
            let support: Option<Vec<u16>> = (0..b as u16)
                .map(|j| entry(b_deg, k, j).map(|v| (j, v)))
                .collect::<Option<Vec<_>>>()
                .map(|row| row.into_iter().filter(|&(_, v)| v).map(|(j, _)| j).collect());
            let Some(t) = support else { continue };
            if t.is_empty() {
                continue;
            }
            for i in 0..a as u16 {
                let idxs: Option<Vec<usize>> =
                    t.iter().map(|&j| var_of(a_deg, j, i)).collect();
                if let Some(mut idxs) = idxs {
                    idxs.sort_unstable();
                    rows.push((idxs, false));
                }
            }
        }
    }
    rows
}

/// Why an outside-knowledge-base differential row was not enforced — see
/// [`prune_outside_diffs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutsideDropReason {
    /// Source or target degree is hard-excluded (unknown lower-page
    /// differentials): Rust's basis there is a PARTIAL quotient, so recorded
    /// `[row, col]` coordinates (taken in the recording pipeline's fully
    /// quotiented basis) may index a different class.
    Excluded,
    /// Source or target degree is target-only excluded: the basis is real but
    /// possibly over-kept, so coordinates are equally unreliable — and under
    /// `EHP_RELAX_TARGET_EXCLUDE` these variables participate in Leibniz
    /// constraints, where one mis-indexed pin makes the page UNSAT.
    TargetOnlyExcluded,
    /// s+f is within the per-page cutoff margin of the data edge — the
    /// original pipeline creates no variables there (page max_t decrements
    /// per turn, minus an extra r−1), because the E_r basis near the edge is
    /// computed from incomplete differential data.
    BeyondMargin,
    /// Source or target outside the computed polygon.
    OutsidePolygon,
    /// Zero-dimensional source or target, or row/col beyond the current dims.
    NoVariable,
}

impl std::fmt::Display for OutsideDropReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            OutsideDropReason::Excluded => "degree excluded (partial-quotient basis)",
            OutsideDropReason::TargetOnlyExcluded => {
                "degree target-only excluded (over-kept basis)"
            }
            OutsideDropReason::BeyondMargin => "within cutoff margin of the data edge",
            OutsideDropReason::OutsidePolygon => "outside computed polygon",
            OutsideDropReason::NoVariable => "zero dims or row/col out of range",
        };
        f.write_str(s)
    }
}

/// Python-parity gate for the outside-diffs knowledge base: remove rows whose
/// `[row, col]` coordinates cannot be trusted on THIS page and return them
/// with the reason.
///
/// Rationale: outside rows were recorded against the reference (Python)
/// pipeline's fully quotiented E_r bases. The reference pipeline enforces a
/// row only if its variable already exists (`make_known_constraints` skips
/// otherwise); variables exist only away from exclusions and a per-page
/// cutoff margin. Rust instead force-creates variable blocks for every known
/// diff (`add_known_diff_vars`) — right for interactive user asserts made
/// against the CURRENT chart, but wrong for bulk-imported rows: at excluded /
/// over-kept / edge degrees Rust's basis differs from the recording basis, so
/// a mathematically correct row can pin the wrong matrix entry, and (under
/// the relaxed target exclusion) collide with sound Leibniz constraints —
/// observed as E4 UNSAT at EHP_MAX_T=80 with verified-correct data.
///
/// The margin uses the reference rule (page max_t decrements per turn, then
/// r−1 more): rows with s+f > max_t − (2r − 3) are dropped. Set
/// `EHP_OUTSIDE_PARITY=0` to disable pruning entirely (restores
/// force-enforcement of every row).
pub fn prune_outside_diffs(
    page: &SATPage,
    diffs: &mut HashMap<DiffVar, bool>,
) -> Vec<(DiffVar, bool, OutsideDropReason)> {
    let r = page.r;
    let margin_cutoff = page.max_t.map(|m| m - (2 * r - 3));
    let mut dropped = Vec::new();
    diffs.retain(|dv, val| {
        let src = Tridegree::new(dv.n.min(dv.s + 2), dv.s, dv.f);
        let tgt = src.diff_target(r);
        let reason = if page.is_excluded(src) || page.is_excluded(tgt) {
            Some(OutsideDropReason::Excluded)
        } else if page.is_excluded_target_only(src) || page.is_excluded_target_only(tgt) {
            Some(OutsideDropReason::TargetOnlyExcluded)
        } else if margin_cutoff.is_some_and(|c| src.s + src.f > c) {
            Some(OutsideDropReason::BeyondMargin)
        } else if !page.is_in_computed_polygon_source(src) || !page.is_in_computed_polygon(tgt)
        {
            Some(OutsideDropReason::OutsidePolygon)
        } else if (dv.row as usize) >= page.dim_at(tgt) || (dv.col as usize) >= page.dim_at(src)
        {
            Some(OutsideDropReason::NoVariable)
        } else {
            None
        };
        match reason {
            Some(rsn) => {
                dropped.push((*dv, *val, rsn));
                false
            }
            None => true,
        }
    });
    dropped.sort_by_key(|(dv, _, _)| (dv.s + dv.f, dv.n, dv.s, dv.f, dv.row, dv.col));
    dropped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tridegree::Tridegree;

    /// User-asserted differentials must be representable even at
    /// uncertainty-excluded degrees (this is what makes click-add work on
    /// E₃ and higher, where exclude sets are nonempty).
    #[test]
    fn known_diff_var_created_at_excluded_degree() {
        let mut page = SATPage::new(2);
        let src = Tridegree::new(2, 1, 1);
        let tgt = src.diff_target(2);
        page.dimension.insert(src, 2);
        page.dimension.insert(tgt, 1);
        page.exclude_set.insert(src);

        // Without a known diff, the excluded degree gets no variables.
        let sys = build_constraint_system(&page, 10, &HashMap::new());
        assert_eq!(sys.num_vars, 0);

        // A user-asserted value forces the full 1×2 block plus its constraint.
        let mut known = HashMap::new();
        known.insert(DiffVar::new(2, 1, 1, 0, 1), true);
        let sys = build_constraint_system(&page, 10, &known);
        assert_eq!(sys.num_vars, 2);
        assert!(sys.var_index.contains_key(&DiffVar::new(2, 1, 1, 0, 1)));
        assert_eq!(sys.num_constraints(), 1);
    }

    /// A source degree excluded only as the target of an unknown incoming
    /// differential still gets its d_r variables (over-kept basis is real);
    /// hard exclusions and excluded diff targets still block.
    #[test]
    fn target_only_excluded_source_keeps_variables() {
        if !relax_target_exclude() {
            return; // running with EHP_RELAX_TARGET_EXCLUDE=0
        }
        let mut page = SATPage::new(2);
        let src = Tridegree::new(2, 1, 1);
        let tgt = src.diff_target(2);
        page.dimension.insert(src, 2);
        page.dimension.insert(tgt, 1);

        // Hard exclusion: no vars.
        page.exclude_set.insert(src);
        assert!(make_basis_single(&page, src).is_empty());

        // Target-only exclusion: the full 1×2 block exists.
        page.target_only_exclude.insert(src);
        assert_eq!(make_basis_single(&page, src).len(), 2);

        // An excluded diff target still blocks regardless.
        page.exclude_set.insert(tgt);
        assert!(make_basis_single(&page, src).is_empty());
    }
}

/// Deduplicate constraints by sorting and removing duplicates.
fn deduplicate_constraints(constraints: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let mut sorted: Vec<Vec<usize>> = constraints
        .into_iter()
        .map(|mut c| {
            c.sort();
            c
        })
        .collect();
    sorted.sort();
    sorted.dedup();
    sorted
}

/// Build the complete constraint system for a page.
pub fn build_constraint_system(
    page: &SATPage,
    cutoff: i32,
    known_diffs: &HashMap<DiffVar, bool>,
) -> ConstraintSystem {
    let mut vars = make_basis(page, cutoff);
    add_known_diff_vars(page, known_diffs, &mut vars);
    let var_index: HashMap<DiffVar, usize> =
        vars.iter().enumerate().map(|(i, v)| (*v, i)).collect();

    debug!("Variable count: {}", vars.len());

    let mut system = ConstraintSystem::new(vars);

    // Naturality constraints for E, H, P
    for map_kind in MapKind::all() {
        let nat_constraints =
            make_naturality_constraints(page, cutoff, map_kind, &var_index);
        debug!(
            "{} naturality constraints: {}",
            map_kind.name(),
            nat_constraints.len()
        );
        for c in nat_constraints {
            system.add_constraint_indices(&c, false);
        }
    }

    // Leibniz constraints
    let (leibniz_constraints, excluded_leibniz) = make_leibniz_constraints(page, cutoff, &var_index);
    debug!("Leibniz constraints: {}", leibniz_constraints.len());
    for c in leibniz_constraints {
        system.add_constraint_indices(&c, false);
    }
    system.excluded_leibniz = excluded_leibniz;

    // Known constraints
    let known = make_known_constraints(known_diffs, &var_index);
    debug!("Known constraints: {}", known.len());
    for (indices, val) in known {
        system.add_constraint_indices(&indices, val);
    }

    debug!("Total constraints: {}", system.num_constraints());
    system
}
