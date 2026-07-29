use fp::matrix::Matrix;
use fp::vector::FpVector;
use hashbrown::{HashMap, HashSet};

use crate::constraints::DiffVar;
use crate::element::Element;
use crate::gf2::*;
use crate::tridegree::Tridegree;

/// Result of solving the GF(2) constraint system.
///
/// Stores a particular solution (offset) and the kernel basis,
/// along with which variables are free (unknown).
#[derive(Clone)]
pub struct SATResult {
    /// Particular solution vector.
    pub offset: FpVector,
    /// Set of variable indices that are free (unknown).
    pub unknown: HashSet<usize>,
    /// Kernel basis matrix (rows are kernel vectors).
    pub kernel: Matrix,
    /// Variable list: global index -> DiffVar.
    pub vars: Vec<DiffVar>,
    /// Reverse map: DiffVar -> global index.
    pub var_index: HashMap<DiffVar, usize>,
}

impl SATResult {
    /// Get the differential matrix at a tridegree.
    ///
    /// Returns the "best guess" matrix using the particular solution.
    /// Entries corresponding to unknown variables use the offset value
    /// (which may be arbitrary for those entries).
    pub fn diff_matrix(&self, t: Tridegree, page_dimensions: &HashMap<Tridegree, usize>) -> Option<Matrix> {
        let _r_unused = 0; // r is not needed here, we use the vars directly
        let src_dim = page_dimensions.get(&t).copied().unwrap_or(0);

        // Find the target dimension by looking at what vars exist
        let n_var = t.n.min(t.s + 2);

        // Find target dim from the vars
        let mut tgt_dim = 0usize;
        for v in &self.vars {
            if v.n == n_var && v.s == t.s && v.f == t.f {
                tgt_dim = tgt_dim.max(v.row as usize + 1);
            }
        }

        if src_dim == 0 || tgt_dim == 0 {
            return Some(mat_zero(tgt_dim, src_dim));
        }

        let first_var = DiffVar::new(n_var, t.s, t.f, 0, 0);
        if !self.var_index.contains_key(&first_var) {
            return None;
        }

        let mut mat = mat_zero(tgt_dim, src_dim);
        for row in 0..tgt_dim {
            for col in 0..src_dim {
                let var = DiffVar::new(n_var, t.s, t.f, row as u16, col as u16);
                if let Some(&idx) = self.var_index.get(&var) {
                    if !self.unknown.contains(&idx) {
                        if vec_get(&self.offset, idx) {
                            mat_set(&mut mat, row, col, true);
                        }
                    }
                }
            }
        }
        Some(mat)
    }

    /// Apply the differential to an element, returning the result and uncertainty.
    pub fn apply_to_element(
        &self,
        elem: &Element,
        page_dimensions: &HashMap<Tridegree, usize>,
        r: i32,
    ) -> Option<DiffApplication> {
        let mat = self.diff_matrix(elem.degree, page_dimensions)?;
        let target_deg = elem.degree.diff_target(r);
        let target_dim = mat.rows();

        let offset_vec = mat_mul_vec(&mat, &elem.vec);
        let offset_elem = Element::new(target_deg, offset_vec);

        // Compute uncertainty: which basis directions are uncertain
        let n_var = elem.degree.n.min(elem.degree.s + 2);
        let _src_dim = mat.columns();
        let mut uncertainty_vecs = Vec::new();

        // For each unknown variable that this element's support touches
        for col in vec_support(&elem.vec) {
            for row in 0..target_dim {
                let var = DiffVar::new(n_var, elem.degree.s, elem.degree.f, row as u16, col as u16);
                if let Some(&idx) = self.var_index.get(&var) {
                    if self.unknown.contains(&idx) {
                        let mut v = vec_zero(target_dim);
                        vec_set(&mut v, row, true);
                        uncertainty_vecs.push(v);
                    }
                }
            }
        }

        // Reduce uncertainty vectors to a basis
        if !uncertainty_vecs.is_empty() {
            let mut mat = mat_from_rows(&uncertainty_vecs, target_dim);
            let (rank, _) = mat_echelon_form(&mut mat);
            uncertainty_vecs = (0..rank)
                .map(|i| mat_get_row(&mat, i))
                .filter(|r| !r.is_zero())
                .collect();
        }

        let uncertainty: Vec<Element> = uncertainty_vecs
            .into_iter()
            .map(|v| Element::new(target_deg, v))
            .collect();

        Some(DiffApplication {
            offset: offset_elem,
            uncertainty,
        })
    }

    /// Check if a tridegree has any unknown entries.
    pub fn is_tridegree_uncertain(&self, t: Tridegree) -> bool {
        let n_var = t.n.min(t.s + 2);
        self.vars.iter().any(|v| {
            v.n == n_var && v.s == t.s && v.f == t.f && {
                let idx = self.var_index[v];
                self.unknown.contains(&idx)
            }
        })
    }

    /// List of tridegrees that have unknown differential entries.
    pub fn unknown_tridegrees(&self) -> Vec<Tridegree> {
        let mut result = HashSet::new();
        for &idx in &self.unknown {
            let v = &self.vars[idx];
            result.insert(Tridegree::new(v.n, v.s, v.f));
        }
        let mut sorted: Vec<Tridegree> = result.into_iter().collect();
        sorted.sort();
        sorted
    }
}

/// Result of applying a differential to an element.
pub struct DiffApplication {
    /// The determined part of d(x).
    pub offset: Element,
    /// Basis for the undetermined part.
    pub uncertainty: Vec<Element>,
}
