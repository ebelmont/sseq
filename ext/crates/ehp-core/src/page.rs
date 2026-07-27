use fp::matrix::Matrix;
use fp::vector::FpVector;
use hashbrown::HashMap;

use crate::element::Element;
use crate::gf2::*;
use crate::map::MapKind;
use crate::products::ProductTable;
use crate::tridegree::Tridegree;

/// Map table: stores the matrix of a map (E, H, or P) at each source tridegree.
/// The matrix has dimensions src_dim × tgt_dim (maps source basis vectors to target).
///
/// Matrices are behind `Arc` so cloning a table (the interpage overlay clones
/// the whole page per trial) shares the fp storage instead of deep-copying it.
#[derive(Clone)]
pub struct MapTable {
    pub kind: MapKind,
    /// Matrix of the map at each source tridegree.
    /// Convention: rows = source basis vectors, columns = target coordinates.
    /// So applying the map to element v gives v * matrix.
    pub matrices: HashMap<Tridegree, std::sync::Arc<Matrix>>,
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
        if let Some(mat) = self.matrices.get(&elem.degree) {
            let result = mat_vec_mul(mat.as_ref(), &elem.vec);
            Element::new(target_deg, result)
        } else {
            Element::zero(target_deg, target_dim)
        }
    }

    /// Get the matrix at a tridegree.
    pub fn matrix_at(&self, t: Tridegree) -> Option<&Matrix> {
        self.matrices.get(&t).map(|a| a.as_ref())
    }

    /// Set the matrix at a tridegree.
    pub fn set_matrix(&mut self, t: Tridegree, mat: Matrix) {
        self.matrices.insert(t, std::sync::Arc::new(mat));
    }
}

/// Borrowed / virtual result of a map-matrix lookup — see
/// [`SATPage::map_matrix_ref`]. `Stored` borrows the real matrix;
/// `Identity(n)` and `Zero(src_dim, tgt_dim)` are the un-materialized
/// defaults (rows = source basis, columns = target, `v * M` convention).
pub enum MapMatrixRef<'a> {
    Stored(&'a Matrix),
    Identity(usize),
    Zero(usize, usize),
}

impl MapMatrixRef<'_> {
    /// Materialize — exactly what [`SATPage::map_matrix`] used to return.
    pub fn to_matrix(&self) -> Matrix {
        match *self {
            MapMatrixRef::Stored(m) => m.clone(),
            MapMatrixRef::Identity(n) => mat_identity(n),
            MapMatrixRef::Zero(src, tgt) => mat_zero(src, tgt),
        }
    }

    /// Materialize the TRANSPOSE without building the un-transposed default
    /// first (columns = source basis, i.e. the Python `.T` convention used by
    /// the naturality/Leibniz generators).
    pub fn to_matrix_transposed(&self) -> Matrix {
        match *self {
            MapMatrixRef::Stored(m) => mat_transpose(m),
            MapMatrixRef::Identity(n) => mat_identity(n),
            MapMatrixRef::Zero(src, tgt) => mat_zero(tgt, src),
        }
    }

    /// Apply to a source-coordinate vector (`v * M`), matching
    /// `mat_vec_mul(&self.to_matrix(), v)` without materializing defaults.
    pub fn apply_vec(&self, v: &FpVector) -> FpVector {
        match *self {
            MapMatrixRef::Stored(m) => mat_vec_mul(m, v),
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
    /// Cutoff on total degree `t = s + f`: the single bound used throughout
    /// (matches the Python reference's `SATPage.max_t` / `in_bounds`, which
    /// checks only `s + f <= max_t`; the old `max_n`/`max_s`/`max_f` fields
    /// were dead weight there too — computed and printed but never gating
    /// anything — so they've been dropped here as well).
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

    fn polygon_check(&self, t: Tridegree, _source: bool) -> bool {
        if t.n < -1 || t.s < -1 || t.f < -1 {
            return false;
        }
        let max_t = match self.max_t {
            Some(v) => v,
            None => return true,
        };
        t.s + t.f <= max_t
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

    /// Compute max_t (the s+f cutoff) from dimensions, if not already set.
    pub fn compute_max_values(&mut self) {
        if self.max_t.is_some() {
            return;
        }
        let mut mt = 0i32;
        for (&t, &d) in &self.dimension {
            if d > 0 {
                mt = mt.max(t.s + t.f);
            }
        }
        self.max_t = Some(mt - 2);
    }

    /// Build multiplication pairs (which tridegrees multiply together).
    pub fn build_pairs(&mut self) {
        self.pairs.clear();

        let mut by_n: HashMap<i32, Vec<Tridegree>> = HashMap::new();
        for &t in self.page.keys() {
            by_n.entry(t.n).or_default().push(t);
        }

        let max_t = self.max_t.unwrap_or(i32::MAX);

        let mut keys: Vec<Tridegree> = self.page.keys().copied().collect();
        keys.sort();

        for x in &keys {
            if x.n > max_t {
                continue;
            }
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
