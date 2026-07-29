use fp::matrix::Matrix;
use fp::vector::FpVector;
use hashbrown::HashMap;

use crate::element::Element;
use crate::gf2::*;
use crate::map::MapKind;
use crate::products::ProductTable;
use crate::tridegree::Tridegree;

/// Map table: stores the matrix of a map (E, H, or P) at each source tridegree.
/// The matrix has dimensions src_dim × tgt_dim (maps source basis vectors to
/// target) and is stored as a compact bit-block (`ProductMatrix` with
/// dim1 = src_dim, dim2 = 1): ~200k map entries across the cached pages made
/// per-entry fp matrices the dominant cache-load cost, exactly like products.
///
/// Blocks are behind `Arc` so cloning a table (the interpage overlay clones
/// the whole page per trial) shares storage instead of deep-copying it.
#[derive(Clone)]
pub struct MapTable {
    pub kind: MapKind,
    /// Bit-block of the map at each source tridegree.
    /// Convention: rows = source basis vectors, columns = target coordinates.
    /// So applying the map to element v gives v * matrix.
    pub matrices: HashMap<Tridegree, std::sync::Arc<crate::products::ProductMatrix>>,
}

impl MapTable {
    pub fn new(kind: MapKind) -> Self {
        MapTable {
            kind,
            matrices: HashMap::new(),
        }
    }

    /// Apply the map to an element.
    pub fn apply(&self, elem: &Element, target_dim: usize) -> Element {
        let target_deg = self.kind.target_degree(elem.degree);
        if let Some(b) = self.matrices.get(&elem.degree) {
            let mut result = vec_zero(b.tgt_dim as usize);
            for i in vec_support(&elem.vec) {
                if i < b.rows() {
                    result += &b.row_vec(i);
                }
            }
            Element::new(target_deg, result)
        } else {
            Element::zero(target_deg, target_dim)
        }
    }

    /// Get the block at a tridegree.
    pub fn matrix_at(&self, t: Tridegree) -> Option<&crate::products::ProductMatrix> {
        self.matrices.get(&t).map(|a| a.as_ref())
    }

    /// Set the matrix at a tridegree (fp form — converted to the compact
    /// block; rows = source dim, cols = target dim).
    pub fn set_matrix(&mut self, t: Tridegree, mat: Matrix) {
        let block = crate::products::ProductMatrix::from_matrix(
            mat.rows() as u16,
            1,
            mat.columns() as u16,
            &mat,
        );
        self.matrices.insert(t, std::sync::Arc::new(block));
    }

    /// Set a pre-built compact block (the binary V2 read path).
    pub fn set_block(&mut self, t: Tridegree, block: crate::products::ProductMatrix) {
        self.matrices.insert(t, std::sync::Arc::new(block));
    }

    /// Share storage between entries with IDENTICAL blocks (same pattern as
    /// `ProductTable::dedup_shared_blocks`): stable-range map matrices are
    /// overwhelmingly duplicates. Lookup behavior unchanged; entries are
    /// read-only after construction (mutation replaces the Arc).
    pub fn dedup_shared(&mut self) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut by_content: HashMap<u64, Vec<std::sync::Arc<crate::products::ProductMatrix>>> =
            HashMap::new();
        let mut shared = 0usize;
        for arc in self.matrices.values_mut() {
            let mut h = DefaultHasher::new();
            (arc.dim1, arc.dim2, arc.tgt_dim).hash(&mut h);
            arc.raw_words().hash(&mut h);
            let bucket = by_content.entry(h.finish()).or_default();
            if let Some(existing) = bucket.iter().find(|e| e.as_ref() == arc.as_ref()) {
                *arc = std::sync::Arc::clone(existing);
                shared += 1;
            } else {
                bucket.push(std::sync::Arc::clone(arc));
            }
        }
        shared
    }
}

/// Borrowed / virtual result of a map-matrix lookup — see
/// [`SATPage::map_matrix_ref`]. `Stored` borrows the real matrix;
/// `Identity(n)` and `Zero(src_dim, tgt_dim)` are the un-materialized
/// defaults (rows = source basis, columns = target, `v * M` convention).
pub enum MapMatrixRef<'a> {
    Stored(&'a crate::products::ProductMatrix),
    Identity(usize),
    Zero(usize, usize),
}

impl MapMatrixRef<'_> {
    /// Materialize — exactly what [`SATPage::map_matrix`] used to return.
    pub fn to_matrix(&self) -> Matrix {
        match *self {
            MapMatrixRef::Stored(b) => b.to_matrix(),
            MapMatrixRef::Identity(n) => mat_identity(n),
            MapMatrixRef::Zero(src, tgt) => mat_zero(src, tgt),
        }
    }

    /// Materialize the TRANSPOSE without building the un-transposed default
    /// first (columns = source basis, i.e. the Python `.T` convention used by
    /// the naturality/Leibniz generators).
    pub fn to_matrix_transposed(&self) -> Matrix {
        match *self {
            MapMatrixRef::Stored(b) => {
                let (rows, cols) = (b.rows(), b.tgt_dim as usize);
                let mut m = mat_zero(cols, rows);
                for r in 0..rows {
                    let row = b.row_vec(r);
                    for c in vec_support(&row) {
                        m.row_mut(c).set_entry(r, 1);
                    }
                }
                m
            }
            MapMatrixRef::Identity(n) => mat_identity(n),
            MapMatrixRef::Zero(src, tgt) => mat_zero(tgt, src),
        }
    }

    /// Apply to a source-coordinate vector (`v * M`), matching the old
    /// `mat_vec_mul(&self.to_matrix(), v)` (including its rows == len
    /// assertion) without materializing defaults.
    pub fn apply_vec(&self, v: &FpVector) -> FpVector {
        match *self {
            MapMatrixRef::Stored(b) => {
                assert_eq!(b.rows(), v.len());
                let mut result = vec_zero(b.tgt_dim as usize);
                for i in vec_support(v) {
                    result += &b.row_vec(i);
                }
                result
            }
            MapMatrixRef::Identity(n) => {
                assert_eq!(n, v.len());
                v.clone()
            }
            MapMatrixRef::Zero(src, tgt) => {
                assert_eq!(src, v.len());
                vec_zero(tgt)
            }
        }
    }
}

/// One page E_r of the spectral sequence.
#[derive(Clone)]
pub struct SATPage {
    pub r: i32,
    /// Dimension at each tridegree.
    pub dimension: HashMap<Tridegree, usize>,
    /// Basis elements at each tridegree.
    pub page: HashMap<Tridegree, Vec<Element>>,
    /// Product table.
    pub products: ProductTable,
    /// E, H, P map tables.
    pub maps: HashMap<MapKind, MapTable>,
    /// Element names (for display).
    pub names: HashMap<String, String>,
    /// Multiplication pairs: for each (n,s,f) with s>0 or f>0,
    /// list of y-degrees (n2,s2,f2) such that we have products in E_{n+s}.
    ///
    /// NOTE: no longer populated anywhere — nothing in the pipeline reads it
    /// (constraint generation enumerates pairs itself), and building it cost
    /// ~13% of startup. Call [`SATPage::build_pairs`] explicitly if a future
    /// consumer needs it.
    pub pairs: HashMap<Tridegree, Vec<Tridegree>>,
    /// Maximum values for bounds checking.
    pub max_n: Option<i32>,
    pub max_s: Option<i32>,
    pub max_f: Option<i32>,
    pub max_t: Option<i32>,
    /// Excluded tridegrees.
    pub exclude_set: hashbrown::HashSet<Tridegree>,
    /// Subset of `exclude_set`: degrees excluded *only* as the target of an
    /// unknown incoming differential (no unknown outgoing d_r of their own,
    /// not carried forward from a prior page). Under partial turning their
    /// basis is real but possibly over-kept (it surjects onto the true page),
    /// so selected constraints there stay sound — see
    /// [`crate::pageturning::make_next_exclude_set`] and the
    /// `EHP_RELAX_TARGET_EXCLUDE` kill-switch. Not serialized (`exclude_set`
    /// isn't either; both are recomputed when pages are built).
    pub target_only_exclude: hashbrown::HashSet<Tridegree>,
}

impl SATPage {
    pub fn new(r: i32) -> Self {
        let mut maps = HashMap::new();
        for kind in MapKind::all() {
            maps.insert(kind, MapTable::new(kind));
        }
        SATPage {
            r,
            dimension: HashMap::new(),
            page: HashMap::new(),
            products: ProductTable::new(),
            maps,
            names: HashMap::new(),
            pairs: HashMap::new(),
            max_n: None,
            max_s: None,
            max_f: None,
            max_t: None,
            exclude_set: hashbrown::HashSet::new(),
            target_only_exclude: hashbrown::HashSet::new(),
        }
    }

    /// Clone for use as a trial overlay page: identical to `clone()` except
    /// `pairs` and `names` are left empty — nothing on the interpage
    /// propagation path reads them (`pairs` is only consumed by the startup
    /// pair enumeration and `names` by chart export), and `pairs` is the
    /// single largest structure on the page, so skipping them makes the
    /// per-trial clone far cheaper.
    pub fn overlay_clone(&self) -> SATPage {
        SATPage {
            r: self.r,
            dimension: self.dimension.clone(),
            page: self.page.clone(),
            products: self.products.clone(),
            maps: self.maps.clone(),
            names: HashMap::new(),
            pairs: HashMap::new(),
            max_n: self.max_n,
            max_s: self.max_s,
            max_f: self.max_f,
            max_t: self.max_t,
            exclude_set: self.exclude_set.clone(),
            target_only_exclude: self.target_only_exclude.clone(),
        }
    }

    /// Get dimension at a tridegree (0 if not present).
    pub fn dim_at(&self, t: Tridegree) -> usize {
        self.dimension.get(&t).copied().unwrap_or(0)
    }

    /// Get zero element at a tridegree.
    pub fn zero(&self, t: Tridegree) -> Element {
        Element::zero(t, self.dim_at(t))
    }

    /// Check if there are elements at a tridegree.
    pub fn has_elements(&self, t: Tridegree) -> bool {
        self.page.contains_key(&t)
    }

    /// All tridegrees with elements, sorted.
    pub fn all_tridegrees(&self) -> Vec<Tridegree> {
        let mut tris: Vec<Tridegree> = self.page.keys().copied().collect();
        tris.sort_by_key(|t| (t.total(), t.n, t.s, t.f));
        tris
    }

    /// Get the map matrix at a source tridegree (rows = source basis, columns
    /// = target coordinates), with the original `Map.matrix` semantics: a map
    /// with no stored matrix is the ZERO map of the right dimensions — except
    /// E in the stable range (n > s+1), which is the IDENTITY. Skipping
    /// no-data degrees instead of defaulting silently drops naturality/Leibniz
    /// constraints of the form `d·φ = 0` (the original always constrains).
    pub fn map_matrix(&self, kind: MapKind, t: Tridegree) -> Matrix {
        self.map_matrix_ref(kind, t).to_matrix()
    }

    /// Allocation-free form of [`SATPage::map_matrix`]: most lookups hit the
    /// zero/identity DEFAULT branch, and materializing a padded tile-aligned
    /// `Matrix` per call dominated constraint generation at large max_t
    /// (posix_memalign churn was ~65% of a t=80 startup profile). Callers
    /// match on the default cases instead of multiplying by a materialized
    /// zero/identity; `to_matrix()` reproduces the old behavior exactly.
    pub fn map_matrix_ref(&self, kind: MapKind, t: Tridegree) -> MapMatrixRef<'_> {
        if let Some(m) = self.maps.get(&kind).and_then(|mt| mt.matrix_at(t)) {
            return MapMatrixRef::Stored(m);
        }
        let src_dim = self.dim_at(t);
        let tgt_dim = self.dim_at(kind.target_degree(t));
        if kind == MapKind::E && t.n > t.s + 1 && src_dim == tgt_dim {
            MapMatrixRef::Identity(src_dim)
        } else {
            MapMatrixRef::Zero(src_dim, tgt_dim)
        }
    }

    /// Check polygon bounds.
    pub fn is_in_computed_polygon(&self, t: Tridegree) -> bool {
        self.polygon_check(t, false)
    }

    pub fn is_in_computed_polygon_source(&self, t: Tridegree) -> bool {
        self.polygon_check(t, true)
    }

    fn polygon_check(&self, t: Tridegree, source: bool) -> bool {
        if t.n < -1 || t.s < -1 || t.f < -1 {
            return false;
        }
        let max_f = match self.max_f {
            Some(v) => v,
            None => return true,
        };
        let max_s = match self.max_s {
            Some(v) => v,
            None => return true,
        };
        let max_t = match self.max_t {
            Some(v) => v,
            None => return true,
        };
        if t.f > max_f || t.s > max_s || t.s + t.f > max_t {
            return false;
        }
        if let Some(max_n) = self.max_n {
            let bound = if source { 2 * max_n - 3 } else { max_n };
            if t.n > bound {
                return false;
            }
        }
        true
    }

    pub fn is_excluded(&self, t: Tridegree) -> bool {
        if t.n > t.s + 2 {
            self.exclude_set.contains(&Tridegree::new(t.s + 2, t.s, t.f))
        } else {
            self.exclude_set.contains(&t)
        }
    }

    /// Is the degree's exclusion of the "target-only" kind (see
    /// [`SATPage::target_only_exclude`])? Folds to the stable representative
    /// like [`SATPage::is_excluded`].
    pub fn is_excluded_target_only(&self, t: Tridegree) -> bool {
        if t.n > t.s + 2 {
            self.target_only_exclude
                .contains(&Tridegree::new(t.s + 2, t.s, t.f))
        } else {
            self.target_only_exclude.contains(&t)
        }
    }

    pub fn is_computable(&self, t: Tridegree) -> bool {
        self.polygon_check(t, false)
    }

    /// Compute max_n, max_s, max_f, max_t from dimensions.
    pub fn compute_max_values(&mut self) {
        let mut mn = 0i32;
        let mut ms = 0i32;
        let mut mf = 0i32;
        let mut mt = 0i32;
        for (&t, &d) in &self.dimension {
            if d > 0 {
                mn = mn.max(t.n);
                ms = ms.max(t.s);
                mf = mf.max(t.f);
                mt = mt.max(t.s + t.f);
            }
        }
        if self.max_n.is_none() {
            self.max_n = Some(mn - 2);
        }
        if self.max_s.is_none() {
            self.max_s = Some(ms - 2);
        }
        if self.max_f.is_none() {
            self.max_f = Some(mf - 2);
        }
        if self.max_t.is_none() {
            self.max_t = Some(mt - 2);
        }
    }

    /// Build multiplication pairs (which tridegrees multiply together).
    pub fn build_pairs(&mut self) {
        self.pairs.clear();

        let mut by_n: HashMap<i32, Vec<Tridegree>> = HashMap::new();
        for &t in self.page.keys() {
            by_n.entry(t.n).or_default().push(t);
        }

        let mut keys: Vec<Tridegree> = self.page.keys().copied().collect();
        keys.sort();

        for x in &keys {
            // No n-based bound here (matches Python and compute_induced_products):
            // max_t bounds t = s + f only, never n.
            if x.s == 0 && x.f == 0 {
                continue;
            }

            let y_n = x.n + x.s;
            let mut bucket = Vec::new();

            if let Some(y_list) = by_n.get(&y_n) {
                for &y in y_list {
                    // Check (y.n - 1, y.s, y.f) is in page
                    let shifted = Tridegree::new(y.n - 1, y.s, y.f);
                    if !self.page.contains_key(&shifted) {
                        continue;
                    }
                    let xy = Tridegree::new(x.n, x.s + y.s, x.f + y.f);
                    let sources = [*x, y, xy];
                    if sources
                        .iter()
                        .all(|s| self.is_in_computed_polygon_source(*s))
                    {
                        bucket.push(y);
                    }
                }
            }

            bucket.sort();
            if !bucket.is_empty() {
                self.pairs.insert(*x, bucket);
            }
        }
    }

    /// Get the domain tridegrees for a given map.
    pub fn map_domain(&self, kind: MapKind) -> Vec<Tridegree> {
        let mut domain: Vec<Tridegree> = self
            .page
            .keys()
            .copied()
            .filter(|t| kind.domain_check(*t))
            .collect();
        domain.sort();
        domain
    }
}
