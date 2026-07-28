//! GF(2) linear algebra helpers built on top of the `fp` crate.
//!
//! Provides convenience functions and types that gf2-linalg had but fp doesn't,
//! including transpose, submatrix extraction, matrix construction helpers,
//! Gaussian elimination wrappers, and an AffineSubspace type.

use fp::matrix::Matrix;
use fp::vector::FpVector;
use fp::prime::ValidPrime;

/// The prime p = 2 used throughout this crate.
pub fn prime() -> ValidPrime {
    ValidPrime::new(2)
}

// =============================================================================
// FpVector helpers
// =============================================================================

/// Create a zero FpVector of given length over GF(2).
pub fn vec_zero(len: usize) -> FpVector {
    FpVector::new(prime(), len)
}

/// Create a basis vector e_i in GF(2)^len.
pub fn vec_basis(len: usize, i: usize) -> FpVector {
    let mut v = vec_zero(len);
    v.set_entry(i, 1);
    v
}

/// Create an FpVector from a slice of 0/1 u8 values.
pub fn vec_from_bits(bits: &[u8]) -> FpVector {
    let mut v = vec_zero(bits.len());
    for (i, &b) in bits.iter().enumerate() {
        if b != 0 {
            v.set_entry(i, 1);
        }
    }
    v
}

/// Get bit at position i (as bool).
#[inline]
pub fn vec_get(v: &FpVector, i: usize) -> bool {
    v.entry(i) != 0
}

/// Set bit at position i from a bool.
#[inline]
pub fn vec_set(v: &mut FpVector, i: usize, val: bool) {
    v.set_entry(i, if val { 1 } else { 0 });
}

/// Flip (toggle) bit at position i.
#[inline]
pub fn vec_flip(v: &mut FpVector, i: usize) {
    let cur = v.entry(i);
    v.set_entry(i, 1 - cur);
}

/// XOR-add other into self (GF(2) addition in place).
#[inline]
pub fn vec_xor_assign(dst: &mut FpVector, src: &FpVector) {
    *dst += src;
}

/// Return self XOR other (new vector).
pub fn vec_xor(a: &FpVector, b: &FpVector) -> FpVector {
    let mut result = a.clone();
    result += b;
    result
}

/// Iterator over indices of set bits (support) in an FpVector.
pub fn vec_support(v: &FpVector) -> impl Iterator<Item = usize> + '_ {
    v.iter_nonzero().map(|(i, _)| i)
}

/// GF(2) dot product: AND then popcount mod 2.
pub fn vec_dot(a: &FpVector, b: &FpVector) -> bool {
    debug_assert_eq!(a.len(), b.len());
    let mut count = 0u32;
    for (i, val) in a.iter_nonzero() {
        if val != 0 && b.entry(i) != 0 {
            count += 1;
        }
    }
    count % 2 == 1
}

/// Number of 1-bits (Hamming weight).
pub fn vec_popcount(v: &FpVector) -> usize {
    v.iter_nonzero().count()
}

/// Index of the first set bit, or None if zero.
pub fn vec_first_set(v: &FpVector) -> Option<usize> {
    v.iter_nonzero().next().map(|(i, _)| i)
}

/// Check if vector is all zeros.
#[inline]
pub fn vec_is_zero(v: &FpVector) -> bool {
    v.is_zero()
}

/// Extract a sub-range [start..start+len) from an FpVector.
pub fn vec_slice(v: &FpVector, start: usize, len: usize) -> FpVector {
    assert!(start + len <= v.len());
    let mut result = vec_zero(len);
    for i in 0..len {
        result.set_entry(i, v.entry(start + i));
    }
    result
}

/// Concatenate two FpVectors.
pub fn vec_concat(a: &FpVector, b: &FpVector) -> FpVector {
    let new_len = a.len() + b.len();
    let mut result = vec_zero(new_len);
    for (i, val) in a.iter().enumerate() {
        if val != 0 {
            result.set_entry(i, val);
        }
    }
    for (i, val) in b.iter().enumerate() {
        if val != 0 {
            result.set_entry(a.len() + i, val);
        }
    }
    result
}

// =============================================================================
// Matrix helpers
// =============================================================================

/// Create a zero matrix over GF(2).
pub fn mat_zero(nrows: usize, ncols: usize) -> Matrix {
    Matrix::new(prime(), nrows, ncols)
}

/// Create an identity matrix over GF(2).
pub fn mat_identity(n: usize) -> Matrix {
    Matrix::identity(prime(), n)
}

/// Create a matrix from a Vec of FpVector rows.
pub fn mat_from_rows(rows: &[FpVector], ncols: usize) -> Matrix {
    let nrows = rows.len();
    let mut mat = mat_zero(nrows, ncols);
    for (i, row) in rows.iter().enumerate() {
        mat.row_mut(i).assign(row.as_slice());
    }
    mat
}

/// Create a matrix from a flat list of 0/1 values, row-major.
pub fn mat_from_flat(nrows: usize, ncols: usize, data: &[u8]) -> Matrix {
    assert_eq!(data.len(), nrows * ncols);
    let mut mat = mat_zero(nrows, ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            if data[i * ncols + j] != 0 {
                mat.row_mut(i).set_entry(j, 1);
            }
        }
    }
    mat
}

/// Create a single-column matrix from an FpVector.
pub fn mat_col_matrix(v: &FpVector) -> Matrix {
    let nrows = v.len();
    let mut mat = mat_zero(nrows, 1);
    for (i, val) in v.iter_nonzero() {
        if val != 0 {
            mat.row_mut(i).set_entry(0, 1);
        }
    }
    mat
}

/// Get element at (row, col) as bool.
#[inline]
pub fn mat_get(m: &Matrix, row: usize, col: usize) -> bool {
    m.row(row).entry(col) != 0
}

/// Set element at (row, col) from bool.
#[inline]
pub fn mat_set(m: &mut Matrix, row: usize, col: usize, val: bool) {
    m.row_mut(row).set_entry(col, if val { 1 } else { 0 });
}

/// Set a row of the matrix to a given FpVector.
pub fn mat_set_row(m: &mut Matrix, i: usize, row: &FpVector) {
    m.row_mut(i).assign(row.as_slice());
}

/// Get row i as an owned FpVector.
pub fn mat_get_row(m: &Matrix, i: usize) -> FpVector {
    m.row(i).to_owned()
}

/// Check if the matrix is all zeros.
pub fn mat_is_zero(m: &Matrix) -> bool {
    m.is_zero()
}

/// Transpose a matrix.
pub fn mat_transpose(m: &Matrix) -> Matrix {
    let nrows = m.rows();
    let ncols = m.columns();
    let mut result = mat_zero(ncols, nrows);
    for i in 0..nrows {
        for (j, val) in m.row(i).iter_nonzero() {
            if val != 0 {
                result.row_mut(j).set_entry(i, 1);
            }
        }
    }
    result
}

/// Matrix multiplication over GF(2): a * b.
pub fn mat_mul(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.columns(), b.rows(),
        "dimension mismatch: {}x{} * {}x{}",
        a.rows(), a.columns(), b.rows(), b.columns()
    );
    let bt = mat_transpose(b);
    let mut result = mat_zero(a.rows(), b.columns());
    for i in 0..a.rows() {
        for j in 0..b.columns() {
            // Dot product of row i of a and row j of bt (= col j of b)
            let a_row = a.row(i);
            let bt_row = bt.row(j);
            let mut count = 0u32;
            // Iterate over nonzero entries of one and check the other
            for (k, val) in a_row.iter_nonzero() {
                if val != 0 && bt_row.entry(k) != 0 {
                    count += 1;
                }
            }
            if count % 2 == 1 {
                result.row_mut(i).set_entry(j, 1);
            }
        }
    }
    result
}

/// Matrix-vector multiply: m * v.
pub fn mat_mul_vec(m: &Matrix, v: &FpVector) -> FpVector {
    assert_eq!(m.columns(), v.len());
    let mut result = vec_zero(m.rows());
    for i in 0..m.rows() {
        let row = m.row(i);
        let mut count = 0u32;
        for (k, val) in row.iter_nonzero() {
            if val != 0 && v.entry(k) != 0 {
                count += 1;
            }
        }
        if count % 2 == 1 {
            result.set_entry(i, 1);
        }
    }
    result
}

/// Vector-matrix multiply: v^T * m (left multiplication).
pub fn mat_vec_mul(m: &Matrix, v: &FpVector) -> FpVector {
    assert_eq!(m.rows(), v.len());
    let mut result = vec_zero(m.columns());
    for (i, val) in v.iter_nonzero() {
        if val != 0 {
            let row_vec = m.row(i).to_owned();
            result.add(&row_vec, 1);
        }
    }
    result
}

/// Kronecker/tensor product: a ⊗ b.
pub fn mat_tensor(a: &Matrix, b: &Matrix) -> Matrix {
    let new_rows = a.rows() * b.rows();
    let new_cols = a.columns() * b.columns();
    let mut result = mat_zero(new_rows, new_cols);
    for i in 0..a.rows() {
        for (j, a_val) in a.row(i).iter_nonzero() {
            if a_val != 0 {
                for k in 0..b.rows() {
                    for (l, b_val) in b.row(k).iter_nonzero() {
                        if b_val != 0 {
                            result.row_mut(i * b.rows() + k)
                                .set_entry(j * b.columns() + l, 1);
                        }
                    }
                }
            }
        }
    }
    result
}

/// Flatten matrix to a single FpVector (row-major vectorization).
pub fn mat_vectorize(m: &Matrix) -> FpVector {
    let nrows = m.rows();
    let ncols = m.columns();
    if nrows == 0 || ncols == 0 {
        return vec_zero(0);
    }
    let total = nrows * ncols;
    let mut result = vec_zero(total);
    for i in 0..nrows {
        for (j, val) in m.row(i).iter_nonzero() {
            if val != 0 {
                result.set_entry(i * ncols + j, 1);
            }
        }
    }
    result
}

/// Reshape an FpVector into a matrix.
pub fn mat_unvectorize(v: &FpVector, nrows: usize, ncols: usize) -> Matrix {
    assert_eq!(v.len(), nrows * ncols);
    let mut mat = mat_zero(nrows, ncols);
    for (idx, val) in v.iter_nonzero() {
        if val != 0 {
            let i = idx / ncols;
            let j = idx % ncols;
            mat.row_mut(i).set_entry(j, 1);
        }
    }
    mat
}

/// Augment: [a | b] horizontally.
pub fn mat_augment(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.rows(), b.rows());
    let ncols = a.columns() + b.columns();
    let mut result = mat_zero(a.rows(), ncols);
    for i in 0..a.rows() {
        for (j, val) in a.row(i).iter_nonzero() {
            if val != 0 {
                result.row_mut(i).set_entry(j, 1);
            }
        }
        for (j, val) in b.row(i).iter_nonzero() {
            if val != 0 {
                result.row_mut(i).set_entry(a.columns() + j, 1);
            }
        }
    }
    result
}

/// Extract a contiguous range of rows [start..start+count).
pub fn mat_submatrix_rows(m: &Matrix, start: usize, count: usize) -> Matrix {
    assert!(start + count <= m.rows());
    let mut result = mat_zero(count, m.columns());
    for k in 0..count {
        result.row_mut(k).assign(m.row(start + k));
    }
    result
}

/// Extract rows at strided positions: start, start+stride, start+2*stride, ...
pub fn mat_submatrix_rows_strided(m: &Matrix, start: usize, stride: usize, count: usize) -> Matrix {
    assert!(stride > 0);
    let mut result = mat_zero(count, m.columns());
    for k in 0..count {
        let idx = start + k * stride;
        assert!(idx < m.rows());
        result.row_mut(k).assign(m.row(idx));
    }
    result
}

/// Get column j as an FpVector.
pub fn mat_col(m: &Matrix, j: usize) -> FpVector {
    let mut v = vec_zero(m.rows());
    for i in 0..m.rows() {
        if m.row(i).entry(j) != 0 {
            v.set_entry(i, 1);
        }
    }
    v
}

/// Row echelon form in place. Returns the rank and pivot column indices.
pub fn mat_echelon_form(m: &mut Matrix) -> (usize, Vec<usize>) {
    let nrows = m.rows();
    let ncols = m.columns();
    let mut pivot_row = 0;
    let mut pivot_cols = Vec::new();

    for col in 0..ncols {
        // Find pivot in this column
        let mut found = None;
        for row in pivot_row..nrows {
            if m.row(row).entry(col) != 0 {
                found = Some(row);
                break;
            }
        }
        let Some(prow) = found else { continue };

        m.swap_rows(pivot_row, prow);

        // Eliminate all other rows at this column
        for row in 0..nrows {
            if row != pivot_row && m.row(row).entry(col) != 0 {
                // row += pivot_row (over GF(2), adding is same as subtracting)
                m.safe_row_op(row, pivot_row, 1);
            }
        }

        pivot_cols.push(col);
        pivot_row += 1;
    }
    (pivot_row, pivot_cols)
}

/// Serialize matrix to a flat buffer of u64 words (row-major, bitpacked).
/// Compatible with the EHPB binary format.
pub fn mat_raw_words(m: &Matrix) -> Vec<u64> {
    let ncols = m.columns();
    let words_per_row = (ncols + 63) / 64;
    let mut out = Vec::with_capacity(m.rows() * words_per_row);
    let mut buf = vec![0u8; words_per_row * 8];
    for i in 0..m.rows() {
        // Use to_bytes to extract the raw limb data
        buf.fill(0);
        let _cursor = std::io::Cursor::new(&mut buf[..]);
        // Write the row's data
        let row = m.row(i);
        // Manual extraction: iterate over entries and pack into u64 words
        let mut words = vec![0u64; words_per_row];
        for (j, val) in row.iter_nonzero() {
            if val != 0 {
                words[j / 64] |= 1u64 << (j % 64);
            }
        }
        for &w in &words {
            out.push(w);
        }
    }
    out
}

/// Construct matrix from a flat buffer of u64 words (row-major, bitpacked).
pub fn mat_from_raw_words(nrows: usize, ncols: usize, data: Vec<u64>) -> Matrix {
    let words_per_row = (ncols + 63) / 64;
    assert_eq!(data.len(), nrows * words_per_row,
        "from_raw_words: expected {} words, got {}",
        nrows * words_per_row, data.len()
    );
    let mut mat = mat_zero(nrows, ncols);
    for i in 0..nrows {
        let start = i * words_per_row;
        for w_idx in 0..words_per_row {
            let word = data[start + w_idx];
            let base = w_idx * 64;
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let col = base + bit;
                if col < ncols {
                    mat.row_mut(i).set_entry(col, 1);
                }
            }
        }
    }
    mat
}

// =============================================================================
// Gaussian elimination / solver
// =============================================================================

/// Result of Gaussian elimination on an augmented system [A | b].
#[derive(Debug, Clone)]
pub struct GaussResult {
    /// Particular solution (if consistent).
    pub solution: Option<FpVector>,
    /// Kernel basis vectors (null space of A).
    pub kernel: Vec<FpVector>,
    /// Pivot column indices.
    pub pivot_cols: Vec<usize>,
    /// Free column indices.
    pub free_cols: Vec<usize>,
    /// Rank of A.
    pub rank: usize,
    /// Whether the system is consistent.
    pub consistent: bool,
}

/// Solve the system Ax = b over GF(2).
pub fn gauss_solve(a: &Matrix, b: &FpVector) -> GaussResult {
    assert_eq!(a.rows(), b.len());
    let nrows = a.rows();
    let ncols = a.columns();

    // Build augmented matrix [A | b]
    let b_col = mat_col_matrix(b);
    let mut aug = mat_augment(a, &b_col);

    // Row reduce
    let (rank, pivot_cols) = mat_echelon_form(&mut aug);

    // Check consistency: any row [0 ... 0 | 1] means inconsistent
    for i in 0..nrows {
        let row = aug.row(i);
        let all_zero_in_a = (0..ncols).all(|j| row.entry(j) == 0);
        if all_zero_in_a && row.entry(ncols) != 0 {
            return GaussResult {
                solution: None,
                kernel: Vec::new(),
                pivot_cols,
                free_cols: Vec::new(),
                rank,
                consistent: false,
            };
        }
    }

    // Identify free columns
    let pivot_set: std::collections::HashSet<usize> = pivot_cols.iter().copied().collect();
    let free_cols: Vec<usize> = (0..ncols).filter(|c| !pivot_set.contains(c)).collect();

    // Back-substitution for particular solution
    let mut solution = vec_zero(ncols);
    for i in 0..rank {
        if aug.row(i).entry(ncols) != 0 {
            solution.set_entry(pivot_cols[i], 1);
        }
    }

    // Compute kernel basis
    let mut kernel = Vec::with_capacity(free_cols.len());
    for &fc in &free_cols {
        let mut kv = vec_zero(ncols);
        kv.set_entry(fc, 1);
        for i in 0..rank {
            if aug.row(i).entry(fc) != 0 {
                kv.set_entry(pivot_cols[i], 1);
            }
        }
        kernel.push(kv);
    }

    GaussResult {
        solution: Some(solution),
        kernel,
        pivot_cols,
        free_cols,
        rank,
        consistent: true,
    }
}

/// Solve the sparse system Ax = b over GF(2), where `rows[i]` is the sorted
/// list of nonzero-coefficient column indices for constraint `i` (matching
/// `ConstraintSystem.rows`'s representation) and `rhs[i]` is that row's target
/// bit. Returns the same `GaussResult` shape as [`gauss_solve`] (this is the
/// sparse-storage counterpart used when the dense `num_constraints x ncols`
/// matrix would be too large to materialize -- see `ConstraintSystem`'s doc
/// comment).
///
/// Uses the standard augmented-nullspace trick instead of hand-rolled RREF:
/// build `M = [A | b]` (b as one extra column, index `ncols`), and take
/// `M.nullspace()` (via the `sparse-bin-mat` crate). Nullspace vectors `(x,
/// y)` satisfy `A*x = b*y` (since `M*(x,y) = A*x + b*y = 0` over GF(2), i.e.
/// `A*x = b*y`): vectors with `y = 1` are exactly the particular solutions to
/// `Ax = b`, and vectors with `y = 0` are exactly `ker(A)`. Consistency:
/// solvable iff at least one nullspace basis vector has `y = 1` (the
/// nullspace's `y`-projection is linear, so if every basis vector has `y =
/// 0`, so does every vector in their span).
///
/// `pivot_cols`/`free_cols` are left empty -- not computed by this path, and
/// (per audit) not read by any current caller of `gauss_solve`/this function.
pub fn sparse_gauss_solve(rows: &[Vec<usize>], rhs: &[bool], ncols: usize) -> GaussResult {
    assert_eq!(rows.len(), rhs.len());

    let aug_rows: Vec<Vec<usize>> = rows
        .iter()
        .zip(rhs)
        .map(|(r, &b)| {
            let mut row = r.clone();
            if b {
                // Every index in r is < ncols (a valid variable index), so
                // appending ncols keeps the row sorted.
                row.push(ncols);
            }
            row
        })
        .collect();

    let aug = sparse_bin_mat::SparseBinMat::new(ncols + 1, aug_rows);
    let ns = aug.nullspace();

    let mut y0_rows: Vec<Vec<usize>> = Vec::new();
    let mut y1_rows: Vec<Vec<usize>> = Vec::new();
    for i in 0..ns.number_of_rows() {
        let Some(row) = ns.row(i) else { continue };
        let positions = row.as_slice();
        if positions.last() == Some(&ncols) {
            y1_rows.push(positions[..positions.len() - 1].to_vec());
        } else {
            y0_rows.push(positions.to_vec());
        }
    }

    if y1_rows.is_empty() {
        return GaussResult {
            solution: None,
            kernel: Vec::new(),
            pivot_cols: Vec::new(),
            free_cols: Vec::new(),
            rank: aug.rank(),
            consistent: false,
        };
    }

    let particular = y1_rows.remove(0);
    let mut kernel_sparse = y0_rows;
    for row in y1_rows {
        kernel_sparse.push(xor_sorted(&row, &particular));
    }

    let solution = sparse_positions_to_fpvector(&particular, ncols);
    let kernel: Vec<FpVector> = kernel_sparse
        .iter()
        .map(|positions| sparse_positions_to_fpvector(positions, ncols))
        .collect();

    GaussResult {
        solution: Some(solution),
        kernel,
        pivot_cols: Vec::new(),
        free_cols: Vec::new(),
        rank: aug.rank(),
        consistent: true,
    }
}

fn sparse_positions_to_fpvector(positions: &[usize], len: usize) -> FpVector {
    let mut v = vec_zero(len);
    for &p in positions {
        v.set_entry(p, 1);
    }
    v
}

/// Symmetric difference of two sorted, deduplicated index lists (GF(2) XOR
/// of the sparse vectors they represent).
fn xor_sorted(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            result.push(a[i]);
            i += 1;
        } else {
            result.push(b[j]);
            j += 1;
        }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

/// Right kernel of A: vectors x such that Ax = 0.
pub fn gauss_right_kernel(a: &Matrix) -> Vec<FpVector> {
    let b = vec_zero(a.rows());
    gauss_solve(a, &b).kernel
}

/// Image (column space) of A as a list of basis vectors.
pub fn gauss_image(a: &Matrix) -> Vec<FpVector> {
    let mut t = mat_transpose(a);
    let (rank, _) = mat_echelon_form(&mut t);
    (0..rank)
        .map(|i| mat_get_row(&t, i))
        .filter(|r| !r.is_zero())
        .collect()
}

/// Relative row echelon: compute a complement of B in Z.
pub fn gauss_relative_row_echelon(b_basis: &[FpVector], z_basis: &[FpVector]) -> Vec<FpVector> {
    if z_basis.is_empty() {
        return Vec::new();
    }
    let dim = z_basis[0].len();

    // Put B in echelon form
    let mut b_mat = if b_basis.is_empty() {
        mat_zero(0, dim)
    } else {
        mat_from_rows(b_basis, dim)
    };
    let (b_rank, b_pivots) = mat_echelon_form(&mut b_mat);

    let mut result = Vec::new();

    for zv in z_basis {
        let mut reduced = zv.clone();
        for i in 0..b_rank {
            let pivot = b_pivots[i];
            if reduced.entry(pivot) != 0 {
                reduced += &mat_get_row(&b_mat, i);
            }
        }
        if !reduced.is_zero() {
            result.push(reduced);
        }
    }

    // Put result in echelon form
    if !result.is_empty() {
        let ncols = result[0].len();
        let mut rmat = mat_from_rows(&result, ncols);
        let (rank, _) = mat_echelon_form(&mut rmat);
        result = (0..rank).map(|i| mat_get_row(&rmat, i)).collect();
    }

    result
}

// =============================================================================
// AffineSubspace
// =============================================================================

/// An affine subspace of GF(2)^n: { offset + span(basis) }.
///
/// The basis is stored in reduced row echelon form.
#[derive(Clone, Debug)]
pub struct AffineSubspace {
    /// A particular point in the subspace.
    offset: FpVector,
    /// Basis vectors for the linear part (in echelon form).
    basis: Vec<FpVector>,
    /// Ambient dimension.
    ambient_dim: usize,
}

impl AffineSubspace {
    /// Create from an offset and spanning vectors.
    pub fn new(offset: FpVector, spanning: Vec<FpVector>) -> Self {
        let ambient_dim = offset.len();
        let (basis, reduced_offset) = Self::echelonize_and_reduce(spanning, offset, ambient_dim);
        AffineSubspace { offset: reduced_offset, basis, ambient_dim }
    }

    /// Single-point subspace.
    pub fn point(v: FpVector) -> Self {
        let ambient_dim = v.len();
        AffineSubspace { offset: v, basis: Vec::new(), ambient_dim }
    }

    /// Full ambient space GF(2)^n.
    pub fn full(n: usize) -> Self {
        let basis = (0..n).map(|i| vec_basis(n, i)).collect();
        AffineSubspace { offset: vec_zero(n), basis, ambient_dim: n }
    }

    /// Single point at origin.
    pub fn origin(n: usize) -> Self {
        Self::point(vec_zero(n))
    }

    pub fn offset(&self) -> &FpVector { &self.offset }
    pub fn basis(&self) -> &[FpVector] { &self.basis }
    pub fn ambient_dim(&self) -> usize { self.ambient_dim }
    pub fn dim(&self) -> usize { self.basis.len() }
    pub fn is_point(&self) -> bool { self.basis.is_empty() }

    pub fn is_zero_point(&self) -> bool {
        self.basis.is_empty() && self.offset.is_zero()
    }

    /// Check if a vector is contained in this affine subspace.
    pub fn contains(&self, v: &FpVector) -> bool {
        assert_eq!(v.len(), self.ambient_dim);
        let diff = vec_xor(v, &self.offset);
        self.is_in_linear_part(&diff)
    }

    fn is_in_linear_part(&self, v: &FpVector) -> bool {
        let mut reduced = v.clone();
        for b in &self.basis {
            if let Some((pivot, _)) = b.iter_nonzero().next() {
                if reduced.entry(pivot) != 0 {
                    reduced += b;
                }
            }
        }
        reduced.is_zero()
    }

    /// Intersect two affine subspaces.
    pub fn intersection(&self, other: &AffineSubspace) -> Option<AffineSubspace> {
        assert_eq!(self.ambient_dim, other.ambient_dim);
        let n = self.ambient_dim;
        let d1 = self.basis.len();
        let d2 = other.basis.len();
        let nvars = d1 + d2;

        if nvars == 0 {
            if self.offset == other.offset {
                return Some(AffineSubspace::point(self.offset.clone()));
            } else {
                return None;
            }
        }

        let diff = vec_xor(&self.offset, &other.offset);

        let mut mat = mat_zero(n, nvars);
        for (j, bv) in self.basis.iter().enumerate() {
            for (bit, val) in bv.iter_nonzero() {
                if val != 0 {
                    mat_set(&mut mat, bit, j, true);
                }
            }
        }
        for (j, bv) in other.basis.iter().enumerate() {
            for (bit, val) in bv.iter_nonzero() {
                if val != 0 {
                    mat_set(&mut mat, bit, d1 + j, true);
                }
            }
        }

        let result = gauss_solve(&mat, &diff);
        if !result.consistent {
            return None;
        }

        let sol = result.solution.unwrap();

        let mut int_offset = self.offset.clone();
        for i in 0..d1 {
            if sol.entry(i) != 0 {
                int_offset += &self.basis[i];
            }
        }

        let mut int_basis = Vec::new();
        for kv in &result.kernel {
            let mut v = vec_zero(n);
            for i in 0..d1 {
                if kv.entry(i) != 0 {
                    v += &self.basis[i];
                }
            }
            if !v.is_zero() {
                int_basis.push(v);
            }
        }

        Some(AffineSubspace::new(int_offset, int_basis))
    }

    /// Compute the linear image under a matrix M.
    pub fn linear_image(&self, m: &Matrix) -> AffineSubspace {
        let new_offset = mat_mul_vec(m, &self.offset);
        let new_basis: Vec<FpVector> = self.basis.iter()
            .map(|b| mat_mul_vec(m, b))
            .collect();
        AffineSubspace::new(new_offset, new_basis)
    }

    /// Compute the preimage under a linear map M.
    pub fn preimage(&self, m: &Matrix) -> Option<AffineSubspace> {
        let nrows = m.rows();
        let ncols = m.columns();
        assert_eq!(nrows, self.ambient_dim);

        let d = self.basis.len();
        let nvars = ncols + d;

        let mut mat = mat_zero(nrows, nvars);
        // First ncols columns: M
        for i in 0..nrows {
            for (j, val) in m.row(i).iter_nonzero() {
                if val != 0 {
                    mat_set(&mut mat, i, j, true);
                }
            }
        }
        // Next d columns: basis vectors
        for (j, bv) in self.basis.iter().enumerate() {
            for (bit, val) in bv.iter_nonzero() {
                if val != 0 {
                    mat_set(&mut mat, bit, ncols + j, true);
                }
            }
        }

        let result = gauss_solve(&mat, &self.offset);
        if !result.consistent {
            return None;
        }

        let sol = result.solution.unwrap();
        let pre_offset = vec_slice(&sol, 0, ncols);

        let mut pre_basis = Vec::new();
        for kv in &result.kernel {
            let x_part = vec_slice(kv, 0, ncols);
            if !x_part.is_zero() {
                pre_basis.push(x_part);
            }
        }

        Some(AffineSubspace::new(pre_offset, pre_basis))
    }

    fn echelonize_and_reduce(
        spanning: Vec<FpVector>,
        offset: FpVector,
        ambient_dim: usize,
    ) -> (Vec<FpVector>, FpVector) {
        if spanning.is_empty() {
            return (Vec::new(), offset);
        }

        let mut mat = mat_from_rows(&spanning, ambient_dim);
        let (rank, _) = mat_echelon_form(&mut mat);

        let basis: Vec<FpVector> = (0..rank)
            .map(|i| mat_get_row(&mat, i))
            .filter(|r| !r.is_zero())
            .collect();

        let mut reduced = offset;
        for b in &basis {
            if let Some((pivot, _)) = b.iter_nonzero().next() {
                if reduced.entry(pivot) != 0 {
                    reduced += b;
                }
            }
        }

        (basis, reduced)
    }
}
