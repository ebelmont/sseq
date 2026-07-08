use std::sync::Arc;

use fp::matrix::Matrix;
use fp::vector::FpVector;
use hashbrown::HashMap;

use crate::gf2::*;
use crate::tridegree::Tridegree;

/// Key for the product table: (degree1, basis_index1, degree2, basis_index2).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ProductKey {
    pub deg1: Tridegree,
    pub idx1: u16,
    pub deg2: Tridegree,
    pub idx2: u16,
}

impl ProductKey {
    pub fn new(deg1: Tridegree, idx1: u16, deg2: Tridegree, idx2: u16) -> Self {
        ProductKey {
            deg1,
            idx1,
            deg2,
            idx2,
        }
    }
}

/// A block matrix storing all products for a pair of tridegrees.
///
/// Row `i*dim2 + j` = product of `basis_i@deg1` × `basis_j@deg2`,
/// expressed in the target tridegree with `tgt_dim` coordinates.
#[derive(Clone)]
pub struct ProductMatrix {
    pub dim1: u16,
    pub dim2: u16,
    pub tgt_dim: u16,
    pub matrix: Matrix, // (dim1*dim2) rows × tgt_dim cols
}

/// The product table stores basis element × basis element → result vector,
/// organized as block matrices indexed by `(Tridegree, Tridegree)` pairs.
///
/// Blocks live behind `Arc` so cloning a table is refcount bumps rather than
/// deep-copying every block's (padded, tile-aligned) fp storage — the
/// interpage overlay clones the whole page per trial, which used to cost
/// ~50s/page. Mutation copy-on-writes the single block touched.
#[derive(Clone)]
pub struct ProductTable {
    matrices: HashMap<(Tridegree, Tridegree), Arc<ProductMatrix>>,
}

impl ProductTable {
    pub fn new() -> Self {
        ProductTable {
            matrices: HashMap::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        ProductTable {
            matrices: HashMap::with_capacity(cap),
        }
    }

    /// Insert a single basis×basis product result.
    ///
    /// `dim1` and `dim2` are the dimensions of the source tridegrees,
    /// needed to allocate/size the block matrix on first insert.
    pub fn insert(&mut self, key: ProductKey, value: FpVector, dim1: usize, dim2: usize) {
        let block_key = (key.deg1, key.deg2);
        let tgt_dim = value.len();

        let pm = Arc::make_mut(self.matrices.entry(block_key).or_insert_with(|| {
            Arc::new(ProductMatrix {
                dim1: dim1 as u16,
                dim2: dim2 as u16,
                tgt_dim: tgt_dim as u16,
                matrix: mat_zero(dim1 * dim2, tgt_dim),
            })
        }));

        let row_idx = key.idx1 as usize * pm.dim2 as usize + key.idx2 as usize;
        if row_idx < pm.matrix.rows() && tgt_dim == pm.tgt_dim as usize {
            mat_set_row(&mut pm.matrix, row_idx, &value);
        }
    }

    /// Insert an entire block matrix for a degree pair.
    pub fn insert_block(
        &mut self,
        deg1: Tridegree,
        deg2: Tridegree,
        pm: ProductMatrix,
    ) {
        self.matrices.insert((deg1, deg2), Arc::new(pm));
    }

    /// Remove the block matrix for a degree pair (if present).
    pub fn remove_block(&mut self, deg1: Tridegree, deg2: Tridegree) {
        self.matrices.remove(&(deg1, deg2));
    }

    /// Get the block matrix for a degree pair.
    pub fn block(&self, deg1: Tridegree, deg2: Tridegree) -> Option<&ProductMatrix> {
        self.matrices.get(&(deg1, deg2)).map(|a| a.as_ref())
    }

    /// Look up a basis × basis product.
    pub fn get(&self, key: &ProductKey) -> Option<FpVector> {
        let block_key = (key.deg1, key.deg2);
        if let Some(pm) = self.matrices.get(&block_key) {
            let row_idx = key.idx1 as usize * pm.dim2 as usize + key.idx2 as usize;
            if row_idx < pm.matrix.rows() {
                let row = pm.matrix.row(row_idx).to_owned();
                return Some(row);
            }
        }
        None
    }

    /// Multiply two elements given by coefficient vectors.
    /// Expands bilinearly using the stored block matrix.
    ///
    /// Single hash lookup per call instead of O(support1 × support2).
    /// Whether any product block is stored for this degree pair. `multiply`
    /// returns zero for absent blocks, so `!has_block(a, b)` guarantees every
    /// `multiply(a, _, b, _, _)` is zero — used to fail fast in constraint
    /// generation.
    pub fn has_block(&self, deg1: Tridegree, deg2: Tridegree) -> bool {
        self.matrices.contains_key(&(deg1, deg2))
    }

    pub fn multiply(
        &self,
        deg1: Tridegree,
        vec1: &FpVector,
        deg2: Tridegree,
        vec2: &FpVector,
        result_dim: usize,
    ) -> FpVector {
        let mut result = vec_zero(result_dim);

        let block_key = (deg1, deg2);
        let pm = match self.matrices.get(&block_key) {
            Some(pm) => pm,
            None => return result,
        };

        if pm.tgt_dim as usize != result_dim {
            return result;
        }

        for i in vec_support(vec1) {
            for j in vec_support(vec2) {
                let row_idx = i * pm.dim2 as usize + j;
                if row_idx < pm.matrix.rows() {
                    result += &pm.matrix.row(row_idx).to_owned();
                }
            }
        }
        result
    }

    /// Build a matrix M where M * v gives the product of (element with vec v at deg1)
    /// with a fixed basis element idx2 at deg2.
    ///
    /// Returns result_dim × dim1 matrix.
    pub fn product_matrix_left(
        &self,
        deg1: Tridegree,
        dim1: usize,
        deg2: Tridegree,
        idx2: u16,
        result_dim: usize,
    ) -> Matrix {
        let block_key = (deg1, deg2);
        if let Some(pm) = self.matrices.get(&block_key) {
            if pm.tgt_dim as usize == result_dim && (idx2 as usize) < pm.dim2 as usize {
                // Extract rows: for each j in 0..dim1, row index is j*dim2 + idx2
                // This is a strided extraction: start=idx2, stride=dim2, count=dim1
                let sub = mat_submatrix_rows_strided(
                    &pm.matrix,
                    idx2 as usize,
                    pm.dim2 as usize,
                    dim1,
                );
                // sub is dim1 × tgt_dim, we need result_dim × dim1 (transpose)
                return mat_transpose(&sub);
            }
        }
        mat_zero(result_dim, dim1)
    }

    /// Build a matrix whose columns represent products of a fixed element idx1 at deg1
    /// with each basis element of deg2.
    ///
    /// Returns result_dim × dim2 matrix.
    pub fn product_matrix_right(
        &self,
        deg1: Tridegree,
        idx1: u16,
        deg2: Tridegree,
        dim2: usize,
        result_dim: usize,
    ) -> Matrix {
        let block_key = (deg1, deg2);
        if let Some(pm) = self.matrices.get(&block_key) {
            if pm.tgt_dim as usize == result_dim && (idx1 as usize) < pm.dim1 as usize {
                // Extract contiguous rows: start = idx1*dim2, count = dim2
                let start = idx1 as usize * pm.dim2 as usize;
                let sub = mat_submatrix_rows(&pm.matrix, start, dim2);
                // sub is dim2 × tgt_dim, we need result_dim × dim2 (transpose)
                return mat_transpose(&sub);
            }
        }
        mat_zero(result_dim, dim2)
    }

    /// Number of individual product entries (sum of dim1*dim2 across all blocks).
    pub fn len(&self) -> usize {
        self.matrices
            .values()
            .map(|pm| pm.dim1 as usize * pm.dim2 as usize)
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.matrices.is_empty()
    }

    /// Iterate over individual product entries as `(ProductKey, FpVector)`.
    /// Filters out zero products for compatibility with CSV save.
    pub fn iter(&self) -> impl Iterator<Item = (ProductKey, FpVector)> + '_ {
        self.matrices.iter().flat_map(|(&(deg1, deg2), pm)| {
            let dim2 = pm.dim2 as usize;
            (0..pm.matrix.rows()).filter_map(move |row_idx| {
                let row = pm.matrix.row(row_idx);
                if row.is_zero() {
                    return None;
                }
                let idx1 = (row_idx / dim2) as u16;
                let idx2 = (row_idx % dim2) as u16;
                let key = ProductKey::new(deg1, idx1, deg2, idx2);
                Some((key, row.to_owned()))
            })
        })
    }

    /// Iterate over block matrices: yields `(&(Tridegree, Tridegree), &ProductMatrix)`.
    pub fn iter_blocks(&self) -> impl Iterator<Item = (&(Tridegree, Tridegree), &ProductMatrix)> {
        self.matrices.iter().map(|(k, v)| (k, v.as_ref()))
    }

    /// Number of block matrices.
    pub fn num_blocks(&self) -> usize {
        self.matrices.len()
    }
}

impl Default for ProductTable {
    fn default() -> Self {
        Self::new()
    }
}
