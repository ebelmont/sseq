use fp::matrix::Matrix;
use hashbrown::HashMap;

use crate::gf2::*;
use crate::tridegree::Tridegree;

/// Wraps an AffineSubspace to represent a matrix-shaped affine subspace.
///
/// The ambient space is GF(2)^(nrows*ncols), representing nrows×ncols matrices
/// over GF(2). The subspace constrains which differential matrices are possible.
#[derive(Clone, Debug)]
pub struct AffineMatrixSubspace {
    pub nrows: usize,
    pub ncols: usize,
    inner: AffineSubspace,
}

impl AffineMatrixSubspace {
    /// Create a new unconstrained matrix subspace (full ambient space).
    pub fn new(nrows: usize, ncols: usize) -> Self {
        let dim = nrows * ncols;
        AffineMatrixSubspace {
            nrows,
            ncols,
            inner: AffineSubspace::full(dim),
        }
    }

    /// Create a zero matrix subspace (single point at origin).
    pub fn zero(nrows: usize, ncols: usize) -> Self {
        let dim = nrows * ncols;
        AffineMatrixSubspace {
            nrows,
            ncols,
            inner: AffineSubspace::origin(dim),
        }
    }

    /// Create from a known matrix (single point).
    pub fn from_matrix(mat: &Matrix) -> Self {
        AffineMatrixSubspace {
            nrows: mat.rows(),
            ncols: mat.columns(),
            inner: AffineSubspace::point(mat_vectorize(mat)),
        }
    }

    /// Create from an existing AffineSubspace.
    pub fn from_inner(nrows: usize, ncols: usize, inner: AffineSubspace) -> Self {
        debug_assert_eq!(inner.ambient_dim(), nrows * ncols);
        AffineMatrixSubspace {
            nrows,
            ncols,
            inner,
        }
    }

    /// Dimension of the affine subspace.
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Whether the differential is fully determined (dimension 0).
    pub fn is_forced(&self) -> bool {
        self.nrows == 0 || self.ncols == 0 || self.inner.is_point()
    }

    /// Whether this is a zero differential that is fully determined.
    pub fn is_zero_cycle(&self) -> bool {
        self.is_forced() && self.inner.offset().is_zero()
    }

    /// Get the particular (offset) matrix.
    pub fn offset_matrix(&self) -> Matrix {
        mat_unvectorize(self.inner.offset(), self.nrows, self.ncols)
    }

    /// Get the basis vectors as matrices.
    pub fn basis_matrices(&self) -> Vec<Matrix> {
        self.inner
            .basis()
            .iter()
            .map(|b| mat_unvectorize(b, self.nrows, self.ncols))
            .collect()
    }

    /// Access the inner affine subspace.
    pub fn inner(&self) -> &AffineSubspace {
        &self.inner
    }

    /// Intersect with another AffineMatrixSubspace.
    pub fn intersect(&self, other: &AffineMatrixSubspace) -> Option<AffineMatrixSubspace> {
        assert_eq!(self.nrows, other.nrows);
        assert_eq!(self.ncols, other.ncols);
        self.inner.intersection(&other.inner).map(|inner| {
            AffineMatrixSubspace {
                nrows: self.nrows,
                ncols: self.ncols,
                inner,
            }
        })
    }

    /// Restrict by intersecting with another affine subspace.
    /// Returns None if intersection is empty.
    pub fn restrict(&mut self, restriction: &AffineSubspace) -> bool {
        match self.inner.intersection(restriction) {
            Some(new_inner) => {
                self.inner = new_inner;
                true
            }
            None => false,
        }
    }

    /// Set to a specific matrix (point subspace).
    pub fn set_matrix(&mut self, mat: &Matrix) {
        assert_eq!(mat.rows(), self.nrows);
        assert_eq!(mat.columns(), self.ncols);
        self.inner = AffineSubspace::point(mat_vectorize(mat));
    }

    /// Left multiply: given a fixed matrix L (k×nrows), compute L * self.
    /// Maps {offset + span(basis)} to {L*offset + span(L*basis_i)}.
    pub fn mul_left(&self, l: &Matrix) -> AffineMatrixSubspace {
        assert_eq!(l.columns(), self.nrows);
        let new_nrows = l.rows();
        let new_ncols = self.ncols;

        let offset_mat = self.offset_matrix();
        let new_offset = mat_mul(l, &offset_mat);

        let mut new_basis = Vec::new();
        for b in self.inner.basis() {
            let b_mat = mat_unvectorize(b, self.nrows, self.ncols);
            let prod = mat_mul(l, &b_mat);
            let v = mat_vectorize(&prod);
            if !v.is_zero() {
                new_basis.push(v);
            }
        }

        AffineMatrixSubspace {
            nrows: new_nrows,
            ncols: new_ncols,
            inner: AffineSubspace::new(mat_vectorize(&new_offset), new_basis),
        }
    }

    /// Right multiply: given a fixed matrix R (ncols×k), compute self * R.
    pub fn mul_right(&self, r: &Matrix) -> AffineMatrixSubspace {
        assert_eq!(self.ncols, r.rows());
        let new_nrows = self.nrows;
        let new_ncols = r.columns();

        let offset_mat = self.offset_matrix();
        let new_offset = mat_mul(&offset_mat, r);

        let mut new_basis = Vec::new();
        for b in self.inner.basis() {
            let b_mat = mat_unvectorize(b, self.nrows, self.ncols);
            let prod = mat_mul(&b_mat, r);
            let v = mat_vectorize(&prod);
            if !v.is_zero() {
                new_basis.push(v);
            }
        }

        AffineMatrixSubspace {
            nrows: new_nrows,
            ncols: new_ncols,
            inner: AffineSubspace::new(mat_vectorize(&new_offset), new_basis),
        }
    }

    /// Left quotient: find X such that X * L ∈ self,
    /// where L is a fixed matrix (nrows(L)=self.nrows, ncols(L)=k).
    ///
    /// Returns the affine subspace of X matrices (size self.nrows × L.nrows()).
    pub fn left_quotient(&self, l: &Matrix) -> Option<AffineMatrixSubspace> {
        // We want: X * L in self
        // vectorize(X * L) = vectorize(self.offset) + sum a_i vectorize(basis_i)
        // Build the linear map: X -> vectorize(X * L)
        let x_nrows = self.nrows;
        let x_ncols = l.rows();
        let x_dim = x_nrows * x_ncols;
        let target_dim = self.nrows * self.ncols;

        if x_dim == 0 || target_dim == 0 {
            return Some(AffineMatrixSubspace::zero(x_nrows, x_ncols));
        }

        // Build the map matrix: x_dim columns -> target_dim rows
        let mut map_mat = mat_zero(target_dim, x_dim);
        for idx in 0..x_dim {
            // Standard basis vector e_idx in X-space
            let x_basis = mat_unvectorize(&vec_basis(x_dim, idx), x_nrows, x_ncols);
            let prod = mat_mul(&x_basis, l);
            let prod_vec = mat_vectorize(&prod);
            for bit in vec_support(&prod_vec) {
                mat_set(&mut map_mat, bit, idx, true);
            }
        }

        // Solve: map_mat * x_vec in self (affine subspace)
        self.inner.preimage(&map_mat).map(|pre| {
            AffineMatrixSubspace::from_inner(x_nrows, x_ncols, pre)
        })
    }

    /// Right quotient: find X such that L * X ∈ self,
    /// where L is a fixed matrix (nrows(L)=k, ncols(L)=self.ncols).
    pub fn right_quotient(&self, l: &Matrix) -> Option<AffineMatrixSubspace> {
        let x_nrows = l.columns();
        let x_ncols = self.ncols;
        let x_dim = x_nrows * x_ncols;
        let target_dim = self.nrows * self.ncols;

        if x_dim == 0 || target_dim == 0 {
            return Some(AffineMatrixSubspace::zero(x_nrows, x_ncols));
        }

        let mut map_mat = mat_zero(target_dim, x_dim);
        for idx in 0..x_dim {
            let x_basis = mat_unvectorize(&vec_basis(x_dim, idx), x_nrows, x_ncols);
            let prod = mat_mul(l, &x_basis);
            let prod_vec = mat_vectorize(&prod);
            for bit in vec_support(&prod_vec) {
                mat_set(&mut map_mat, bit, idx, true);
            }
        }

        self.inner.preimage(&map_mat).map(|pre| {
            AffineMatrixSubspace::from_inner(x_nrows, x_ncols, pre)
        })
    }

    /// Left tensor: self ⊗ I_k.
    /// If self represents matrices A (nrows×ncols), then
    /// self ⊗ I_k represents block-diagonal matrices (nrows*k × ncols*k).
    pub fn tensor(&self, k: usize) -> AffineMatrixSubspace {
        let id = mat_identity(k);
        let offset_mat = mat_tensor(&self.offset_matrix(), &id);
        let new_nrows = self.nrows * k;
        let new_ncols = self.ncols * k;

        let mut new_basis = Vec::new();
        for b in self.inner.basis() {
            let b_mat = mat_unvectorize(b, self.nrows, self.ncols);
            let v = mat_vectorize(&mat_tensor(&b_mat, &id));
            if !v.is_zero() {
                new_basis.push(v);
            }
        }

        AffineMatrixSubspace {
            nrows: new_nrows,
            ncols: new_ncols,
            inner: AffineSubspace::new(mat_vectorize(&offset_mat), new_basis),
        }
    }

    /// Right tensor: I_k ⊗ self.
    pub fn rtensor(&self, k: usize) -> AffineMatrixSubspace {
        let id = mat_identity(k);
        let offset_mat = mat_tensor(&id, &self.offset_matrix());
        let new_nrows = k * self.nrows;
        let new_ncols = k * self.ncols;

        let mut new_basis = Vec::new();
        for b in self.inner.basis() {
            let b_mat = mat_unvectorize(b, self.nrows, self.ncols);
            let v = mat_vectorize(&mat_tensor(&id, &b_mat));
            if !v.is_zero() {
                new_basis.push(v);
            }
        }

        AffineMatrixSubspace {
            nrows: new_nrows,
            ncols: new_ncols,
            inner: AffineSubspace::new(mat_vectorize(&offset_mat), new_basis),
        }
    }
}

/// Collection of differential matrix subspaces indexed by source tridegree.
pub struct DifferentialStore {
    pub r: i32,
    store: HashMap<Tridegree, AffineMatrixSubspace>,
    dimensions: HashMap<Tridegree, usize>,
}

impl DifferentialStore {
    pub fn new(r: i32, dimensions: HashMap<Tridegree, usize>) -> Self {
        DifferentialStore {
            r,
            store: HashMap::new(),
            dimensions,
        }
    }

    /// Get dimension at a tridegree.
    pub fn dimension(&self, t: Tridegree) -> usize {
        self.dimensions.get(&t).copied().unwrap_or(0)
    }

    /// Get or create the differential subspace at a source tridegree.
    pub fn get_or_create(&mut self, t: Tridegree) -> &mut AffineMatrixSubspace {
        let r = self.r;
        let src_dim = self.dimension(t);
        let tgt = t.diff_target(r);
        let tgt_dim = self.dimension(tgt);
        self.store
            .entry(t)
            .or_insert_with(|| {
                if src_dim == 0 || tgt_dim == 0 {
                    AffineMatrixSubspace::zero(tgt_dim, src_dim)
                } else {
                    AffineMatrixSubspace::new(tgt_dim, src_dim)
                }
            })
    }

    /// Get the differential subspace at a source tridegree (if it exists).
    pub fn get(&self, t: &Tridegree) -> Option<&AffineMatrixSubspace> {
        self.store.get(t)
    }

    /// Get the offset (particular) differential matrix at a tridegree.
    /// Returns None if no differential is stored there.
    pub fn get_matrix(&self, t: Tridegree) -> Option<Matrix> {
        self.store.get(&t).map(|ams| ams.offset_matrix())
    }

    /// Set the differential to a specific matrix.
    pub fn set_matrix(&mut self, t: Tridegree, mat: &Matrix) {
        let ams = self.get_or_create(t);
        ams.set_matrix(mat);
    }

    /// Iterate over all stored tridegrees.
    pub fn tridegrees(&self) -> impl Iterator<Item = &Tridegree> {
        self.store.keys()
    }

    /// Iterate over all stored entries.
    pub fn iter(&self) -> impl Iterator<Item = (&Tridegree, &AffineMatrixSubspace)> {
        self.store.iter()
    }

    pub fn dimensions_map(&self) -> &HashMap<Tridegree, usize> {
        &self.dimensions
    }
}
