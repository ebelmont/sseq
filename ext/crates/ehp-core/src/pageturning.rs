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
    // The columns of d_in span the boundary subspace
    let b_basis = gauss_image(&mat_transpose(d_in));

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

/// Turn the entire page: compute homology at every tridegree.
pub fn turn_page(
    page: &SATPage,
    sat_result: &SATResult,
    _new_r: i32,
    _max_t: Option<i32>,
) -> HashMap<Tridegree, TurnedBidegree> {
    let r = page.r;

    let tridegrees: Vec<Tridegree> = page.page.keys().copied().collect();

    info!("Turning page: {} tridegrees", tridegrees.len());

    let results: Vec<(Tridegree, Option<TurnedBidegree>)> = tridegrees
        .par_iter()
        .filter_map(|&t| {
            let old_dim = page.dim_at(t);
            if old_dim == 0 {
                return None;
            }

            // Get outgoing differential: d_r at (n, s, f)
            let d_out = sat_result.diff_matrix(t, &page.dimension)?;

            // Get incoming differential: d_r at (n, s+1, f-r)
            let d_in_src = Tridegree::new(t.n, t.s + 1, t.f - r);
            let d_in = sat_result.diff_matrix(d_in_src, &page.dimension)
                .unwrap_or_else(|| mat_zero(old_dim, page.dim_at(d_in_src)));

            let tb = turn_page_single(t, &d_out, &d_in, old_dim);
            Some((t, tb))
        })
        .collect();

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
    turned_page
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

    let map_mat = match page.map_matrix(map_kind, src_degree) {
        Some(m) => m,
        None => return Vec::new(),
    };

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

    let max_s = next_page.max_s.unwrap_or(i32::MAX);

    let mut product_triples = Vec::new();

    for &x in next_page.page.keys() {
        if x.n > max_s {
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

/// Build the next page (E_{r+1}) from the current page.
pub fn build_next_page(
    page: &SATPage,
    sat_result: &SATResult,
) -> (SATPage, HashMap<Tridegree, TurnedBidegree>) {
    let new_r = page.r + 1;
    let new_max_t = page.max_t.map(|t| t - 1);

    // Turn the page (compute homology)
    let turned = turn_page(page, sat_result, new_r, new_max_t);

    // Build new page
    let mut next = SATPage::new(new_r);
    next.max_t = new_max_t;

    // Set dimensions and basis elements
    for (t, tb) in &turned {
        if !tb.basis.is_empty() {
            next.dimension.insert(*t, tb.basis.len());
            next.page.insert(*t, tb.basis.clone());
        }
    }

    next.compute_max_values();

    // Compute induced maps
    let induced_maps = compute_induced_maps(page, &turned);
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
    let induced_products = compute_induced_products(page, &next, &turned);
    next.products = induced_products;

    // Build pairs
    next.build_pairs();

    (next, turned)
}
