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
    /// Constraint rows (each is an FpVector of length num_vars).
    pub rows: Vec<FpVector>,
    /// RHS values.
    pub rhs: Vec<bool>,
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
        }
    }

    /// Add a constraint: XOR of variables at given indices equals rhs_val.
    pub fn add_constraint_indices(&mut self, indices: &[usize], rhs_val: bool) {
        let mut row = vec_zero(self.num_vars);
        for &i in indices {
            vec_flip(&mut row, i);
        }
        self.rows.push(row);
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

    /// Build the coefficient matrix A and RHS vector b.
    pub fn to_matrix(&self) -> (Matrix, FpVector) {
        let nrows = self.rows.len();
        let a = mat_from_rows(self.rows.clone(), self.num_vars);
        let mut b = vec_zero(nrows);
        for (i, &rhs) in self.rhs.iter().enumerate() {
            if rhs {
                vec_set(&mut b, i, true);
            }
        }
        (a, b)
    }
}

/// Enumerate all differential variables for the page.
///
/// Each variable represents entry d_r[row, col] at tridegree (n, s, f),
/// where the differential goes from (n, s, f) to (n, s-1, f+r).
/// n is capped at s+2 (stable range).
pub fn make_basis(page: &SATPage, cutoff: i32) -> Vec<DiffVar> {
    let r = page.r;
    let mut vars = Vec::new();

    let mut keys: Vec<Tridegree> = page.dimension.keys().copied().collect();
    keys.sort();

    for t in keys {
        if t.s + t.f > cutoff {
            continue;
        }
        if t.n > t.s + 2 {
            continue;
        }
        let diff_target = t.diff_target(r);

        if page.is_excluded(t) || page.is_excluded(diff_target) {
            continue;
        }
        if !page.is_in_computed_polygon_source(t)
            || !page.is_in_computed_polygon_source(diff_target)
        {
            continue;
        }

        let src_dim = page.dim_at(t);
        let tgt_dim = page.dim_at(diff_target);

        if src_dim == 0 || tgt_dim == 0 {
            continue;
        }

        for row in 0..tgt_dim {
            for col in 0..src_dim {
                vars.push(DiffVar::new(t.n, t.s, t.f, row as u16, col as u16));
            }
        }
    }

    vars
}

/// Symbolic matrix-vector product: computes which variables appear in each entry
/// of `A * D[src_tridegree]`, where D is the unknown differential matrix.
///
/// Returns a list of lists: for each entry of the result, the list of variable
/// indices whose XOR equals that entry.
fn matrix_mult_left(
    a: &Matrix,
    src_n: i32,
    src_s: i32,
    src_f: i32,
    src_dim: usize,
    var_index: &HashMap<DiffVar, usize>,
    col: Option<usize>,
) -> Vec<Vec<usize>> {
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
                if let Some(&idx) = var_index.get(&var) {
                    indices.push(idx);
                }
            }
            result.push(indices);
        }
    }
    result
}

/// Symbolic matrix-vector product: computes which variables appear in each entry
/// of `D[tgt_tridegree] * A`, where D is the unknown differential matrix.
fn matrix_mult_right(
    a: &Matrix,
    tgt_n: i32,
    tgt_s: i32,
    tgt_f: i32,
    _r: i32,
    tgt_dim: usize,
    _src_dim: usize,
    var_index: &HashMap<DiffVar, usize>,
) -> Vec<Vec<usize>> {
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
                    if let Some(&idx) = var_index.get(&var) {
                        indices.push(idx);
                    }
                }
            }
            result.push(indices);
        }
    }
    result
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

    if !map_kind.domain_check(t) {
        return Vec::new();
    }

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

    if src_dim == 0 || diff_src_dim == 0 || tgt_dim == 0 || diff_tgt_dim == 0 {
        return Vec::new();
    }

    // Get map matrices (transposed, per Python convention)
    let phi_source = match page.map_matrix(map_kind, src) {
        Some(m) => mat_transpose(&m),
        None => return Vec::new(),
    };
    let phi_target = match page.map_matrix(map_kind, diff_src) {
        Some(m) => mat_transpose(&m),
        None => return Vec::new(),
    };

    // Check dimensions
    if phi_source.rows() != tgt_dim || phi_source.columns() != src_dim {
        return Vec::new();
    }
    if phi_target.rows() != diff_tgt_dim || phi_target.columns() != diff_src_dim {
        return Vec::new();
    }

    // LHS: phi_target * D[src]
    let lhs = matrix_mult_left(
        &phi_target,
        src.n,
        src.s,
        src.f,
        src_dim,
        var_index,
        None,
    );

    // RHS: D[tgt] * phi_source
    let rhs = matrix_mult_right(
        &phi_source,
        tgt.n,
        tgt.s,
        tgt.f,
        r,
        diff_tgt_dim,
        tgt_dim,
        var_index,
    );

    equals_rel(&lhs, &rhs)
}

/// Generate all naturality constraints for a map.
pub fn make_naturality_constraints(
    page: &SATPage,
    cutoff: i32,
    map_kind: MapKind,
    max_n: i32,
    var_index: &HashMap<DiffVar, usize>,
) -> Vec<Vec<usize>> {
    let mut constraints = Vec::new();

    for s in 0..=cutoff {
        for f in 0..=(cutoff - s) {
            for n in 2..=max_n {
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
pub fn make_leibniz_constraint_single(
    page: &SATPage,
    deg1: Tridegree,
    deg2: Tridegree,
    var_index: &HashMap<DiffVar, usize>,
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

    for &td in &all_trideg {
        if page.is_excluded(td) {
            return Vec::new();
        }
    }

    let dim1 = page.dim_at(deg1);
    let dim2 = page.dim_at(deg2);
    let prod_dr_dim = page.dim_at(prod_dr_deg);

    if dim1 == 0 || dim2 == 0 {
        return Vec::new();
    }

    let mut all_constraints = Vec::new();

    // Get E map matrix at deg2
    let e_matrix = page.map_matrix(MapKind::E, deg2);

    for i2 in 0..dim2 {
        let elt2_vec = vec_basis(dim2, i2);

        // Compute E(elt2)
        let e_elt2_vec = match &e_matrix {
            Some(mat) => mat_vec_mul(&mat, &elt2_vec),
            None => continue,
        };

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
            let lhs = if prod_vec.is_zero() {
                vec![vec![]; prod_dr_dim]
            } else {
                let prod_col = mat_col_matrix(&prod_vec);
                matrix_mult_right(
                    &prod_col,
                    prod_deg.n,
                    prod_deg.s,
                    prod_deg.f,
                    r,
                    prod_dr_dim,
                    page.dim_at(prod_deg),
                    var_index,
                )
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
                let ytil = mat_from_rows(ytil_rows, tgt_dim);

                if mat_is_zero(&ytil) {
                    vec![vec![]; prod_dr_dim]
                } else {
                    // Get E matrix at d_deg2 (transposed)
                    let e_mat_at_d = match page.map_matrix(MapKind::E, d_deg2) {
                        Some(m) => mat_transpose(&m),
                        None => {
                            continue;
                        }
                    };

                    // Ytil * E_mat gives the combined matrix
                    let combined = mat_mul(&ytil, &e_mat_at_d);

                    matrix_mult_left(
                        &combined,
                        deg2.n,
                        deg2.s,
                        deg2.f,
                        dim2,
                        var_index,
                        Some(i2),
                    )
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
                let ytil2 = mat_from_rows(ytil2_rows, tgt_dim2);

                if mat_is_zero(&ytil2) {
                    vec![vec![]; prod_dr_dim]
                } else {
                    matrix_mult_left(
                        &ytil2,
                        deg1.n,
                        deg1.s,
                        deg1.f,
                        dim1,
                        var_index,
                        Some(i1),
                    )
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

/// Generate all Leibniz constraints.
pub fn make_leibniz_constraints(
    page: &SATPage,
    cutoff: i32,
    var_index: &HashMap<DiffVar, usize>,
) -> Vec<Vec<usize>> {
    let _r = page.r;
    let mut constraints = Vec::new();

    // Index degrees by n for efficient lookup
    let mut degrees_by_n: HashMap<i32, Vec<(i32, i32)>> = HashMap::new();
    for (&t, &d) in &page.dimension {
        if t.s + t.f <= cutoff && d > 0 && t.f >= 1 {
            degrees_by_n.entry(t.n).or_default().push((t.s, t.f));
        }
    }

    for s3 in 0..=cutoff {
        for f3 in 1..=(cutoff - s3) {
            for th1 in 2..=cutoff {
                let source_degrees = match degrees_by_n.get(&th1) {
                    Some(v) => v,
                    None => continue,
                };
                for &(s1, f1) in source_degrees {
                    let th2 = th1 + s1 - 1;
                    let f2 = f3 - f1;
                    let s2 = s3 - s1;

                    if th2 > cutoff {
                        continue;
                    }

                    let deg1 = Tridegree::new(th1, s1, f1);
                    let deg2 = Tridegree::new(th2, s2, f2);

                    let new = make_leibniz_constraint_single(page, deg1, deg2, var_index);
                    constraints.extend(new);
                }
            }
        }
    }

    debug!("Generated {} Leibniz constraints", constraints.len());
    deduplicate_constraints(constraints)
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
        }
    }
    constraints
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
    let vars = make_basis(page, cutoff);
    let var_index: HashMap<DiffVar, usize> =
        vars.iter().enumerate().map(|(i, v)| (*v, i)).collect();

    debug!("Variable count: {}", vars.len());

    let mut system = ConstraintSystem::new(vars);
    let max_n = page.max_n.unwrap_or(cutoff);

    // Naturality constraints for E, H, P
    for map_kind in MapKind::all() {
        let nat_constraints =
            make_naturality_constraints(page, cutoff, map_kind, max_n, &var_index);
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
    let leibniz_constraints = make_leibniz_constraints(page, cutoff, &var_index);
    debug!("Leibniz constraints: {}", leibniz_constraints.len());
    for c in leibniz_constraints {
        system.add_constraint_indices(&c, false);
    }

    // Known constraints
    let known = make_known_constraints(known_diffs, &var_index);
    debug!("Known constraints: {}", known.len());
    for (indices, val) in known {
        system.add_constraint_indices(&indices, val);
    }

    debug!("Total constraints: {}", system.num_constraints());
    system
}
