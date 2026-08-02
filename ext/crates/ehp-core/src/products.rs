use std::sync::Arc;

use fp::matrix::Matrix;
use fp::vector::FpVector;
use hashbrown::{HashMap, HashSet};

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
///
/// Storage is COMPACT bit-packed rows (`words_per_row` u64s per row, bit `c`
/// of a row = column `c`) instead of an `fp::Matrix`: with ~2M mostly-1×1×1
/// blocks at full data, per-block fp overhead (aligned buffer, stride
/// padding, pivots vec — two allocations each) dominated both the E2 load
/// time (~99% of the load profile was `Matrix::new` inside inserts) and the
/// table's RAM by an order of magnitude. One plain `Box<[u64]>` per block
/// carries exactly the data bits.
#[derive(Clone, PartialEq, Eq)]
pub struct ProductMatrix {
    pub dim1: u16,
    pub dim2: u16,
    pub tgt_dim: u16,
    data: Box<[u64]>,
}

impl ProductMatrix {
    #[inline]
    fn words_per_row(tgt_dim: usize) -> usize {
        tgt_dim.div_ceil(64)
    }

    /// All-zero block of the given dimensions.
    pub fn zero(dim1: usize, dim2: usize, tgt_dim: usize) -> Self {
        let words = dim1 * dim2 * Self::words_per_row(tgt_dim);
        ProductMatrix {
            dim1: dim1 as u16,
            dim2: dim2 as u16,
            tgt_dim: tgt_dim as u16,
            data: vec![0u64; words].into_boxed_slice(),
        }
    }

    /// Number of rows (`dim1 * dim2`).
    #[inline]
    pub fn rows(&self) -> usize {
        self.dim1 as usize * self.dim2 as usize
    }

    #[inline]
    fn wpr(&self) -> usize {
        Self::words_per_row(self.tgt_dim as usize)
    }

    #[inline]
    fn row_words(&self, r: usize) -> &[u64] {
        let w = self.wpr();
        &self.data[r * w..(r + 1) * w]
    }

    /// Row as an `FpVector` of length `tgt_dim`.
    pub fn row_vec(&self, r: usize) -> FpVector {
        let mut v = vec_zero(self.tgt_dim as usize);
        for (wi, &word) in self.row_words(r).iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                v.set_entry(wi * 64 + b, 1);
                bits &= bits - 1;
            }
        }
        v
    }

    /// Whether a row is entirely zero.
    #[inline]
    pub fn row_is_zero(&self, r: usize) -> bool {
        self.row_words(r).iter().all(|&w| w == 0)
    }

    /// Overwrite row `r` from a vector of length `tgt_dim`.
    pub fn set_row(&mut self, r: usize, v: &FpVector) {
        let w = self.wpr();
        let words = &mut self.data[r * w..(r + 1) * w];
        words.fill(0);
        for j in vec_support(v) {
            words[j / 64] |= 1u64 << (j % 64);
        }
    }

    /// XOR row `r` into a word accumulator of length `words_per_row`.
    #[inline]
    fn xor_row_into(&self, r: usize, acc: &mut [u64]) {
        for (a, &w) in acc.iter_mut().zip(self.row_words(r)) {
            *a ^= w;
        }
    }

    /// Materialize as an `fp::Matrix` (rows × tgt_dim) — diagnostics and the
    /// legacy binary format only; not used on hot paths.
    pub fn to_matrix(&self) -> Matrix {
        let mut m = mat_zero(self.rows(), self.tgt_dim as usize);
        for r in 0..self.rows() {
            if !self.row_is_zero(r) {
                mat_set_row(&mut m, r, &self.row_vec(r));
            }
        }
        m
    }

    /// Raw bit-packed words (rows × words_per_row, row-major) — the compact
    /// binary format writes these directly.
    pub fn raw_words(&self) -> &[u64] {
        &self.data
    }

    /// Rebuild from raw words (the compact binary format's read path).
    pub fn from_raw_parts(dim1: u16, dim2: u16, tgt_dim: u16, data: Vec<u64>) -> Option<Self> {
        let expected =
            dim1 as usize * dim2 as usize * Self::words_per_row(tgt_dim as usize);
        if data.len() != expected {
            return None;
        }
        Some(ProductMatrix {
            dim1,
            dim2,
            tgt_dim,
            data: data.into_boxed_slice(),
        })
    }

    /// Build from an `fp::Matrix` (rows = dim1*dim2, cols = tgt_dim) — the
    /// legacy binary loader's path.
    pub fn from_matrix(dim1: u16, dim2: u16, tgt_dim: u16, m: &Matrix) -> Self {
        let mut pm = Self::zero(dim1 as usize, dim2 as usize, tgt_dim as usize);
        let nrows = pm.rows().min(m.rows());
        for r in 0..nrows {
            let row = m.row(r).to_owned();
            pm.set_row(r, &row);
        }
        pm
    }
}

/// The product table stores basis element × basis element → result vector,
/// organized as block matrices indexed by `(Tridegree, Tridegree)` pairs.
///
/// Blocks live behind `Arc` so cloning a table is refcount bumps rather than
/// deep-copying every block — the interpage overlay clones the whole page
/// per trial, which used to cost ~50s/page. Mutation copy-on-writes the
/// single block touched.
#[derive(Clone)]
pub struct ProductTable {
    matrices: HashMap<(Tridegree, Tridegree), Arc<ProductMatrix>>,
    /// Layered-overlay support (interpage trial overlays): reads fall through
    /// to `base` for keys not in `matrices` and not in `tombstones`, so an
    /// overlay table holds only its patch instead of a clone of the whole
    /// base map (~2M Arc bumps per trial page step — the measured majority of
    /// trial CPU). A base table must itself be flat (`base.base == None`).
    ///
    /// The fallthrough preserves the absent-block-is-zero semantics of
    /// `multiply`/`has_block` exactly: a key is "present" iff the patched
    /// view has it, which is what an eager clone-plus-patch would contain.
    base: Option<Arc<ProductTable>>,
    /// Keys removed relative to `base` (removal must shadow, since `multiply`
    /// treats absent blocks as zero).
    tombstones: HashSet<(Tridegree, Tridegree)>,
}

impl ProductTable {
    pub fn new() -> Self {
        ProductTable {
            matrices: HashMap::new(),
            base: None,
            tombstones: HashSet::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        ProductTable {
            matrices: HashMap::with_capacity(cap),
            base: None,
            tombstones: HashSet::new(),
        }
    }

    /// A layered view over `base`: initially identical to it, mutations land
    /// in this table's own patch/tombstones. O(1) to build and to drop.
    pub fn overlay_of(base: Arc<ProductTable>) -> Self {
        assert!(
            base.base.is_none(),
            "ProductTable overlays do not stack (base must be flat)"
        );
        ProductTable {
            matrices: HashMap::new(),
            base: Some(base),
            tombstones: HashSet::new(),
        }
    }

    /// The block Arc visible at `k` through the layered view.
    #[inline]
    fn arc_at(&self, k: &(Tridegree, Tridegree)) -> Option<&Arc<ProductMatrix>> {
        if let Some(a) = self.matrices.get(k) {
            return Some(a);
        }
        if self.tombstones.contains(k) {
            return None;
        }
        self.base.as_ref().and_then(|b| b.matrices.get(k))
    }

    /// Base entries still visible through the patch (for merged iteration).
    fn base_visible(
        &self,
    ) -> impl Iterator<Item = (&(Tridegree, Tridegree), &Arc<ProductMatrix>)> {
        self.base
            .as_deref()
            .into_iter()
            .flat_map(|b| b.matrices.iter())
            .filter(|(k, _)| !self.matrices.contains_key(*k) && !self.tombstones.contains(*k))
    }

    /// Insert a single basis×basis product result.
    ///
    /// `dim1` and `dim2` are the dimensions of the source tridegrees,
    /// needed to allocate/size the block matrix on first insert.
    pub fn insert(&mut self, key: ProductKey, value: FpVector, dim1: usize, dim2: usize) {
        let block_key = (key.deg1, key.deg2);
        let tgt_dim = value.len();

        // A layered table must seed a first write to a base-visible block
        // from the BASE block (what an eager clone would have mutated), not
        // from zero. Tombstoned or absent keys start from zero as before.
        if !self.matrices.contains_key(&block_key) {
            let seed = if self.tombstones.remove(&block_key) {
                None
            } else {
                self.base
                    .as_ref()
                    .and_then(|b| b.matrices.get(&block_key))
                    .cloned()
            };
            self.matrices.insert(
                block_key,
                seed.unwrap_or_else(|| Arc::new(ProductMatrix::zero(dim1, dim2, tgt_dim))),
            );
        }
        let pm = Arc::make_mut(self.matrices.get_mut(&block_key).unwrap());

        let row_idx = key.idx1 as usize * pm.dim2 as usize + key.idx2 as usize;
        if row_idx < pm.rows() && tgt_dim == pm.tgt_dim as usize {
            pm.set_row(row_idx, &value);
        }
    }

    /// Insert an entire block matrix for a degree pair.
    pub fn insert_block(&mut self, deg1: Tridegree, deg2: Tridegree, pm: ProductMatrix) {
        self.tombstones.remove(&(deg1, deg2));
        self.matrices.insert((deg1, deg2), Arc::new(pm));
    }

    /// Remove the block matrix for a degree pair (if present).
    pub fn remove_block(&mut self, deg1: Tridegree, deg2: Tridegree) {
        let k = (deg1, deg2);
        self.matrices.remove(&k);
        if self.base.as_ref().is_some_and(|b| b.matrices.contains_key(&k)) {
            self.tombstones.insert(k);
        }
    }

    /// Get the block matrix for a degree pair.
    pub fn block(&self, deg1: Tridegree, deg2: Tridegree) -> Option<&ProductMatrix> {
        self.arc_at(&(deg1, deg2)).map(|a| a.as_ref())
    }

    /// Look up a basis × basis product.
    pub fn get(&self, key: &ProductKey) -> Option<FpVector> {
        let block_key = (key.deg1, key.deg2);
        if let Some(pm) = self.arc_at(&block_key) {
            let row_idx = key.idx1 as usize * pm.dim2 as usize + key.idx2 as usize;
            if row_idx < pm.rows() {
                return Some(pm.row_vec(row_idx));
            }
        }
        None
    }

    /// Whether any product block is stored for this degree pair. `multiply`
    /// returns zero for absent blocks, so `!has_block(a, b)` guarantees every
    /// `multiply(a, _, b, _, _)` is zero — used to fail fast in constraint
    /// generation.
    pub fn has_block(&self, deg1: Tridegree, deg2: Tridegree) -> bool {
        self.arc_at(&(deg1, deg2)).is_some()
    }

    /// Multiply two elements given by coefficient vectors.
    /// Expands bilinearly using the stored block matrix.
    ///
    /// Single hash lookup per call instead of O(support1 × support2); the
    /// bilinear XOR accumulates in words and converts to an `FpVector` once.
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
        let pm = match self.arc_at(&block_key) {
            Some(pm) => pm,
            None => return result,
        };

        if pm.tgt_dim as usize != result_dim {
            return result;
        }

        let wpr = pm.wpr();
        let mut acc_stack = [0u64; 4];
        let mut acc_heap;
        let acc: &mut [u64] = if wpr <= 4 {
            &mut acc_stack[..wpr]
        } else {
            acc_heap = vec![0u64; wpr];
            &mut acc_heap
        };

        let nrows = pm.rows();
        for i in vec_support(vec1) {
            for j in vec_support(vec2) {
                let row_idx = i * pm.dim2 as usize + j;
                if row_idx < nrows {
                    pm.xor_row_into(row_idx, acc);
                }
            }
        }
        for (wi, &word) in acc.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                result.set_entry(wi * 64 + b, 1);
                bits &= bits - 1;
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
        if let Some(pm) = self.arc_at(&block_key) {
            if pm.tgt_dim as usize == result_dim && (idx2 as usize) < pm.dim2 as usize {
                // Strided row extraction (start=idx2, stride=dim2, count=dim1)
                // into a dim1 × tgt_dim matrix, then transpose.
                let mut sub = mat_zero(dim1, result_dim);
                for j in 0..dim1 {
                    let r = j * pm.dim2 as usize + idx2 as usize;
                    if r < pm.rows() && !pm.row_is_zero(r) {
                        mat_set_row(&mut sub, j, &pm.row_vec(r));
                    }
                }
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
        if let Some(pm) = self.arc_at(&block_key) {
            if pm.tgt_dim as usize == result_dim && (idx1 as usize) < pm.dim1 as usize {
                // Contiguous rows starting at idx1*dim2 into a dim2 × tgt_dim
                // matrix, then transpose.
                let start = idx1 as usize * pm.dim2 as usize;
                let mut sub = mat_zero(dim2, result_dim);
                for j in 0..dim2 {
                    let r = start + j;
                    if r < pm.rows() && !pm.row_is_zero(r) {
                        mat_set_row(&mut sub, j, &pm.row_vec(r));
                    }
                }
                return mat_transpose(&sub);
            }
        }
        mat_zero(result_dim, dim2)
    }

    /// Number of individual product entries (sum of dim1*dim2 across all blocks).
    pub fn len(&self) -> usize {
        self.matrices
            .values()
            .chain(self.base_visible().map(|(_, v)| v))
            .map(|pm| pm.dim1 as usize * pm.dim2 as usize)
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.matrices.is_empty() && self.base_visible().next().is_none()
    }

    /// Iterate over individual product entries as `(ProductKey, FpVector)`.
    /// Filters out zero products for compatibility with CSV save.
    pub fn iter(&self) -> impl Iterator<Item = (ProductKey, FpVector)> + '_ {
        self.matrices
            .iter()
            .chain(self.base_visible())
            .flat_map(|(&(deg1, deg2), pm)| {
                let dim2 = pm.dim2 as usize;
                (0..pm.rows()).filter_map(move |row_idx| {
                    if pm.row_is_zero(row_idx) {
                        return None;
                    }
                    let idx1 = (row_idx / dim2) as u16;
                    let idx2 = (row_idx % dim2) as u16;
                    let key = ProductKey::new(deg1, idx1, deg2, idx2);
                    Some((key, pm.row_vec(row_idx)))
                })
            })
    }

    /// Iterate over block matrices: yields `(&(Tridegree, Tridegree), &ProductMatrix)`.
    pub fn iter_blocks(&self) -> impl Iterator<Item = (&(Tridegree, Tridegree), &ProductMatrix)> {
        self.matrices
            .iter()
            .chain(self.base_visible())
            .map(|(k, v)| (k, v.as_ref()))
    }

    /// Number of block matrices.
    pub fn num_blocks(&self) -> usize {
        self.matrices.len() + self.base_visible().count()
    }

    /// Share storage between blocks with IDENTICAL content (dims + bits):
    /// duplicate blocks point at one `Arc`. Lookup behavior is bit-for-bit
    /// unchanged (keys and presence untouched — this is NOT the orbit-fold
    /// dedup, which was falsified in vivo: folding LOOKUPS changes which
    /// blocks are "present" asymmetrically inside a Leibniz relation and
    /// emits truncated relations; see CHANGES §19c). ~94% of E2 CSV rows are
    /// suspension copies, so roughly half the block STORAGE collapses.
    /// Copy-on-write still works: mutating a shared block clones it via
    /// `Arc::make_mut`.
    pub fn dedup_shared_blocks(&mut self) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        debug_assert!(self.base.is_none(), "dedup runs on flat (load-time) tables only");
        let mut by_content: HashMap<u64, Vec<Arc<ProductMatrix>>> = HashMap::new();
        let mut shared = 0usize;
        for arc in self.matrices.values_mut() {
            let mut h = DefaultHasher::new();
            (arc.dim1, arc.dim2, arc.tgt_dim).hash(&mut h);
            arc.data.hash(&mut h);
            let bucket = by_content.entry(h.finish()).or_default();
            if let Some(existing) = bucket.iter().find(|e| e.as_ref() == arc.as_ref()) {
                *arc = Arc::clone(existing);
                shared += 1;
            } else {
                bucket.push(Arc::clone(arc));
            }
        }
        shared
    }
}

impl Default for ProductTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The compact block must behave exactly like the fp::Matrix it replaced.
    #[test]
    fn compact_block_roundtrip_and_multiply() {
        let d1 = Tridegree::new(2, 3, 1);
        let d2 = Tridegree::new(5, 2, 2);
        let mut table = ProductTable::new();

        // 2×3 block into a 70-dim target (forces 2 words per row).
        let tgt = 70usize;
        let mk = |bits: &[usize]| {
            let mut v = vec_zero(tgt);
            for &b in bits {
                v.set_entry(b, 1);
            }
            v
        };
        table.insert(ProductKey::new(d1, 0, d2, 0), mk(&[0, 69]), 2, 3);
        table.insert(ProductKey::new(d1, 0, d2, 2), mk(&[5]), 2, 3);
        table.insert(ProductKey::new(d1, 1, d2, 1), mk(&[5, 63, 64]), 2, 3);

        // get() returns exactly what was inserted; absent rows are zero.
        assert_eq!(table.get(&ProductKey::new(d1, 0, d2, 0)), Some(mk(&[0, 69])));
        assert_eq!(table.get(&ProductKey::new(d1, 1, d2, 0)), Some(mk(&[])));

        // Bilinear multiply: (e0+e1) × (e0+e1+e2) = rows 00 + 02 + 11 XORed.
        let mut v1 = vec_zero(2);
        v1.set_entry(0, 1);
        v1.set_entry(1, 1);
        let mut v2 = vec_zero(3);
        v2.set_entry(0, 1);
        v2.set_entry(1, 1);
        v2.set_entry(2, 1);
        let prod = table.multiply(d1, &v1, d2, &v2, tgt);
        assert_eq!(prod, mk(&[0, 69, 5, 5, 63, 64][..].iter().fold(
            std::collections::HashSet::new(),
            |mut s, &b| { if !s.remove(&b) { s.insert(b); } s },
        ).into_iter().collect::<Vec<_>>().as_slice()));

        // to_matrix/from_matrix round-trip.
        let pm = table.block(d1, d2).unwrap();
        let m = pm.to_matrix();
        let back = ProductMatrix::from_matrix(pm.dim1, pm.dim2, pm.tgt_dim, &m);
        for r in 0..pm.rows() {
            assert_eq!(pm.row_vec(r), back.row_vec(r));
        }

        // Wrong result_dim → zero (legacy semantics).
        assert!(table.multiply(d1, &v1, d2, &v2, tgt + 1).is_zero());
        // Absent block → zero.
        assert!(table.multiply(d2, &v2, d1, &v1, tgt).is_zero());

        // iter() yields only the three nonzero rows.
        assert_eq!(table.iter().count(), 3);
        assert_eq!(table.len(), 6);
    }
}
