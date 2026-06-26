use std::collections::HashMap;

use bivec::BiVec;
use fp::matrix::Matrix;

use crate::algebra::GeneratedAlgebra;
use super::finite_dimensional_module::FiniteDimensionalModule;
use super::Module;

/// Minimal union-find with path halving and union-by-rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }

    fn num_components(&mut self) -> usize {
        let n = self.parent.len();
        (0..n).filter(|&i| self.find(i) == i).count()
    }
}

impl<A: GeneratedAlgebra> FiniteDimensionalModule<A> {
    /// Check whether the module is connected, i.e., every pair of basis elements is linked
    /// through a chain of generator actions.
    pub fn is_connected(&self) -> bool {
        let algebra = self.algebra();
        let min_deg = self.min_degree();
        let max_deg = match self.max_degree() {
            Some(d) => d,
            None => return true, // empty module is vacuously connected
        };

        // Compute cumulative offsets: flat node ID for basis element (d, i) = offset[d] + i
        let mut offsets = BiVec::with_capacity(min_deg, max_deg + 1);
        let mut total = 0usize;
        for d in min_deg..=max_deg {
            offsets.push(total);
            total += self.dimension(d);
        }

        if total <= 1 {
            return true;
        }

        let mut uf = UnionFind::new(total);

        for input_deg in min_deg..=max_deg {
            for output_deg in (input_deg + 1)..=max_deg {
                let op_deg = output_deg - input_deg;
                let input_dim = self.dimension(input_deg);
                let output_dim = self.dimension(output_deg);
                if input_dim == 0 || output_dim == 0 {
                    continue;
                }
                for op_idx in algebra.generators(op_deg) {
                    for input_idx in 0..input_dim {
                        let action = self.action(op_deg, op_idx, input_deg, input_idx);
                        let src = offsets[input_deg] + input_idx;
                        for output_idx in 0..output_dim {
                            if action.entry(output_idx) != 0 {
                                uf.union(src, offsets[output_deg] + output_idx);
                            }
                        }
                    }
                }
            }
        }

        uf.num_components() == 1
    }

    /// Isomorphism-class fingerprint: ranks of each generator's action at each degree.
    ///
    /// Isomorphic modules have identical fingerprints since rank is basis-invariant.
    /// Not a complete invariant, but cheap to compute and groups candidates for the
    /// exact Hom_A check.
    pub fn action_rank_fingerprint(&self) -> Vec<(i32, i32, usize)> {
        let algebra = self.algebra();
        let min_deg = self.min_degree();
        let max_deg = match self.max_degree() {
            Some(d) => d,
            None => return vec![],
        };
        let p = self.prime();
        let mut fingerprint = Vec::new();

        for d in min_deg..=max_deg {
            let n_d = self.dimension(d);
            if n_d == 0 {
                continue;
            }
            for out_deg in (d + 1)..=max_deg {
                let op_deg = out_deg - d;
                let n_out = self.dimension(out_deg);
                if n_out == 0 {
                    continue;
                }
                for op_idx in algebra.generators(op_deg) {
                    // Build action matrix and compute rank via row reduction
                    let mut mat = Matrix::new(p, n_out, n_d);
                    for j in 0..n_d {
                        let action = self.action(op_deg, op_idx, d, j);
                        for i in 0..n_out {
                            mat.row_mut(i).set_entry(j, action.entry(i));
                        }
                    }
                    mat.row_reduce();
                    let rank = (0..n_out).filter(|&i| !mat.row(i).is_zero()).count();
                    if rank > 0 {
                        fingerprint.push((d, op_deg, rank));
                    }
                }
            }
        }
        fingerprint
    }

    /// Check whether the module is indecomposable by computing End_A(M) and searching for
    /// nontrivial idempotents.
    ///
    /// Returns `(result, dim_End_A(M))` where result is:
    /// - `Some(true)` if proven indecomposable
    /// - `Some(false)` if proven decomposable (nontrivial idempotent found)
    /// - `None` if End_A(M) is too large for brute-force idempotent search
    ///
    /// M is indecomposable iff End_A(M) has no nontrivial idempotents.
    pub fn check_indecomposable(&self) -> (Option<bool>, usize) {
        if !self.is_connected() {
            return (Some(false), 0);
        }

        let algebra = self.algebra();
        let p = self.prime();
        let min_deg = self.min_degree();
        let max_deg = self.max_degree().unwrap();

        let mut offsets: Vec<(i32, usize, usize)> = Vec::new();
        let mut total_vars = 0usize;
        for d in min_deg..=max_deg {
            let n = self.dimension(d);
            if n > 0 {
                offsets.push((d, n, total_vars));
                total_vars += n * n;
            }
        }

        if total_vars == 0 {
            return (Some(true), 0);
        }

        let find_offset = |d: i32| -> Option<(usize, usize)> {
            offsets
                .iter()
                .find(|&&(deg, _, _)| deg == d)
                .map(|&(_, n, off)| (n, off))
        };

        let constraints = build_hom_constraints(
            self,
            self,
            &*algebra,
            &offsets,
            &offsets,
            total_vars,
            &find_offset,
            &find_offset,
        );

        if constraints.is_empty() {
            let dim_end = total_vars;
            if dim_end == 1 {
                return (Some(true), 1);
            }
            return check_idempotents_in_kernel_dimension(dim_end, total_vars, &offsets, None);
        }

        let num_constraints = constraints.len();
        let mut c_transpose = vec![vec![0u32; num_constraints]; total_vars];
        for (ci, row) in constraints.iter().enumerate() {
            for (vi, &val) in row.iter().enumerate() {
                c_transpose[vi][ci] = val;
            }
        }

        let (padded_cols, mut aug) = Matrix::augmented_from_vec(p, &c_transpose);
        aug.row_reduce();
        let kernel = aug.compute_kernel(padded_cols);
        let dim_end = kernel.dimension();

        if dim_end <= 1 {
            return (Some(true), dim_end);
        }

        check_idempotents_in_kernel_dimension(dim_end, total_vars, &offsets, Some(&kernel))
    }

    /// M ≅ N iff Hom_A(M, N) contains an invertible element.
    ///
    /// Returns `Some(true)` if isomorphic, `Some(false)` if not,
    /// `None` if the Hom space is too large (dim > 20) to search.
    pub fn check_isomorphic(&self, other: &Self) -> Option<bool> {
        let algebra = self.algebra();
        let p = self.prime();
        let min_deg = self.min_degree();
        let max_deg = match self.max_degree() {
            Some(d) => d,
            None => return Some(self.max_degree() == other.max_degree()),
        };

        // Both must have same graded dims
        for d in min_deg..=max_deg {
            if self.dimension(d) != other.dimension(d) {
                return Some(false);
            }
        }

        // Variables: F_d is n_d × n_d for each degree d. f: M → N.
        let mut offsets: Vec<(i32, usize, usize)> = Vec::new();
        let mut total_vars = 0usize;
        for d in min_deg..=max_deg {
            let nd = self.dimension(d);
            if nd > 0 {
                offsets.push((d, nd, total_vars));
                total_vars += nd * nd;
            }
        }
        if total_vars == 0 {
            return Some(true);
        }

        let find_offset = |d: i32| -> Option<(usize, usize)> {
            offsets
                .iter()
                .find(|&&(deg, _, _)| deg == d)
                .map(|&(_, nd, off)| (nd, off))
        };

        let constraints = build_hom_constraints(
            self,
            other,
            &*algebra,
            &offsets,
            &offsets,
            total_vars,
            &find_offset,
            &find_offset,
        );

        if constraints.is_empty() {
            // Unconstrained: all graded linear maps are A-linear.
            if total_vars > 20 {
                return None;
            }
            return check_invertible_in_space(total_vars, total_vars, &offsets, None);
        }

        let num_constraints = constraints.len();
        let mut c_transpose = vec![vec![0u32; num_constraints]; total_vars];
        for (ci, row) in constraints.iter().enumerate() {
            for (vi, &val) in row.iter().enumerate() {
                c_transpose[vi][ci] = val;
            }
        }
        let (padded_cols, mut aug) = Matrix::augmented_from_vec(p, &c_transpose);
        aug.row_reduce();
        let kernel = aug.compute_kernel(padded_cols);
        let dim = kernel.dimension();
        if dim == 0 {
            return Some(false);
        }
        check_invertible_in_space(dim, total_vars, &offsets, Some(&kernel))
    }

    /// Deduplicate a list of modules up to A-module isomorphism.
    ///
    /// Uses action-rank fingerprints for fast grouping, then exact Hom_A(M,N)
    /// computation within each group.
    pub fn dedup_isomorphism_classes(modules: &[Self]) -> Vec<usize> {
        // Group by fingerprint
        let mut groups: HashMap<Vec<(i32, i32, usize)>, Vec<usize>> = HashMap::new();
        for (i, m) in modules.iter().enumerate() {
            let fp = m.action_rank_fingerprint();
            groups.entry(fp).or_default().push(i);
        }

        let mut representatives: Vec<usize> = Vec::new();
        for (_, group) in &groups {
            // Within each fingerprint group, keep one representative per isomorphism class
            let mut class_reps: Vec<usize> = Vec::new();
            'outer: for &i in group {
                for &rep in &class_reps {
                    match modules[i].check_isomorphic(&modules[rep]) {
                        Some(true) => continue 'outer,
                        Some(false) => {}
                        None => {}
                    }
                }
                class_reps.push(i);
            }
            representatives.extend(class_reps);
        }

        representatives.sort();
        representatives
    }
}

/// Build linear constraints for Hom_A(M, N): the space of A-module maps f: M → N.
///
/// A graded linear map f = {F_d: M_d → N_d} is A-linear iff θ·f = f·θ for each
/// generator θ, i.e. F_{d+k}·A^M_{θ,d} = A^N_{θ,d}·F_d.
fn build_hom_constraints<A: GeneratedAlgebra>(
    source: &FiniteDimensionalModule<A>,
    target: &FiniteDimensionalModule<A>,
    algebra: &A,
    src_offsets: &[(i32, usize, usize)],
    _tgt_offsets: &[(i32, usize, usize)],
    total_vars: usize,
    find_src: &dyn Fn(i32) -> Option<(usize, usize)>,
    find_tgt: &dyn Fn(i32) -> Option<(usize, usize)>,
) -> Vec<Vec<u32>> {
    let max_deg_src = source.max_degree().unwrap();
    let max_deg_tgt = target.max_degree().unwrap();
    let mut constraints: Vec<Vec<u32>> = Vec::new();

    for &(d, n_src_d, _off_src_d) in src_offsets {
        for output_deg in (d + 1)..=max_deg_src.max(max_deg_tgt) {
            let op_deg = output_deg - d;
            let n_src_out = match find_src(output_deg) {
                Some((n, _)) => n,
                None => 0,
            };
            let n_tgt_out = match find_tgt(output_deg) {
                Some((n, _)) => n,
                None => 0,
            };
            if n_tgt_out == 0 && n_src_out == 0 {
                continue;
            }

            for op_idx in algebra.generators(op_deg) {
                // A^M_{θ,d}: n_src_out × n_src_d (source action)
                let mut src_action = vec![vec![0u32; n_src_d]; n_src_out];
                if output_deg <= max_deg_src {
                    for j in 0..n_src_d {
                        let action = source.action(op_deg, op_idx, d, j);
                        for i in 0..n_src_out {
                            src_action[i][j] = action.entry(i);
                        }
                    }
                }

                // A^N_{θ,d}: n_tgt_out × n_tgt_d
                let n_tgt_d = match find_tgt(d) {
                    Some((n, _)) => n,
                    None => continue,
                };
                let mut tgt_action = vec![vec![0u32; n_tgt_d]; n_tgt_out];
                if output_deg <= max_deg_tgt {
                    for j in 0..n_tgt_d {
                        let action = target.action(op_deg, op_idx, d, j);
                        for i in 0..n_tgt_out {
                            tgt_action[i][j] = action.entry(i);
                        }
                    }
                }

                // Constraint: F_{d+k}·A^M = A^N·F_d
                let (_, off_tgt_out) = match find_tgt(output_deg) {
                    Some(v) => v,
                    None => continue,
                };
                let (_, off_tgt_d) = match find_tgt(d) {
                    Some(v) => v,
                    None => continue,
                };

                for i in 0..n_tgt_out {
                    for j in 0..n_src_d {
                        let mut row = vec![0u32; total_vars];

                        // Term 1: Σ_l F_{d+k}[i][l]·A^M[l][j]
                        for l in 0..n_src_out {
                            if src_action[l][j] != 0 {
                                let var = off_tgt_out + i * n_src_out + l;
                                if var < total_vars {
                                    row[var] ^= 1;
                                }
                            }
                        }

                        // Term 2: Σ_l A^N[i][l]·F_d[l][j]
                        for l in 0..n_tgt_d {
                            if tgt_action[i][l] != 0 {
                                let var = off_tgt_d + l * n_src_d + j;
                                if var < total_vars {
                                    row[var] ^= 1;
                                }
                            }
                        }

                        if row.iter().any(|&x| x != 0) {
                            constraints.push(row);
                        }
                    }
                }
            }
        }
    }

    constraints
}

/// Search End_A(M) for nontrivial idempotents via brute force over basis elements.
fn check_idempotents_in_kernel_dimension(
    dim_end: usize,
    total_vars: usize,
    offsets: &[(i32, usize, usize)],
    kernel: Option<&fp::matrix::Subspace>,
) -> (Option<bool>, usize) {
    if dim_end > 20 {
        return (None, dim_end);
    }

    let mut identity = vec![0u32; total_vars];
    for &(_, n, off) in offsets {
        for i in 0..n {
            identity[off + i * n + i] = 1;
        }
    }

    let basis_vecs: Vec<Vec<u32>> = if let Some(k) = kernel {
        k.basis()
            .map(|slice| (0..total_vars).map(|i| slice.entry(i)).collect())
            .collect()
    } else {
        (0..dim_end)
            .map(|b| {
                let mut v = vec![0u32; total_vars];
                v[b] = 1;
                v
            })
            .collect()
    };

    for bits in 1..(1u64 << dim_end) {
        let mut f = vec![0u32; total_vars];
        for (b, bv) in basis_vecs.iter().enumerate() {
            if (bits >> b) & 1 == 1 {
                for (i, &val) in bv.iter().enumerate() {
                    f[i] ^= val;
                }
            }
        }
        if f == identity {
            continue;
        }
        if is_idempotent(&f, offsets) {
            return (Some(false), dim_end);
        }
    }

    (Some(true), dim_end)
}

/// Check f² = f degree-by-degree (matrix multiplication mod 2 per graded block).
fn is_idempotent(f: &[u32], offsets: &[(i32, usize, usize)]) -> bool {
    for &(_, n, off) in offsets {
        for i in 0..n {
            for j in 0..n {
                let mut f_sq = 0u32;
                for l in 0..n {
                    f_sq ^= f[off + i * n + l] & f[off + l * n + j];
                }
                if f_sq != f[off + i * n + j] {
                    return false;
                }
            }
        }
    }
    true
}

/// Search a subspace of graded linear maps for an invertible element.
fn check_invertible_in_space(
    dim: usize,
    total_vars: usize,
    offsets: &[(i32, usize, usize)],
    kernel: Option<&fp::matrix::Subspace>,
) -> Option<bool> {
    if dim > 20 {
        return None;
    }

    let basis_vecs: Vec<Vec<u32>> = if let Some(k) = kernel {
        k.basis()
            .map(|slice| (0..total_vars).map(|i| slice.entry(i)).collect())
            .collect()
    } else {
        (0..dim)
            .map(|b| {
                let mut v = vec![0u32; total_vars];
                v[b] = 1;
                v
            })
            .collect()
    };

    for bits in 1..(1u64 << dim) {
        let mut f = vec![0u32; total_vars];
        for (b, bv) in basis_vecs.iter().enumerate() {
            if (bits >> b) & 1 == 1 {
                for (i, &val) in bv.iter().enumerate() {
                    f[i] ^= val;
                }
            }
        }
        if is_invertible(&f, offsets) {
            return Some(true);
        }
    }
    Some(false)
}

/// Check if a graded linear map is invertible (each degree-block has full rank mod 2).
fn is_invertible(f: &[u32], offsets: &[(i32, usize, usize)]) -> bool {
    for &(_, n, off) in offsets {
        // Gaussian elimination on the n×n block
        let mut mat = vec![0u64; n];
        for i in 0..n {
            for j in 0..n {
                if f[off + i * n + j] != 0 {
                    mat[i] |= 1u64 << j;
                }
            }
        }
        let mut rank = 0;
        for col in 0..n {
            let pivot = (rank..n).find(|&r| mat[r] & (1u64 << col) != 0);
            let pivot = match pivot {
                Some(p) => p,
                None => continue,
            };
            mat.swap(rank, pivot);
            for r in 0..n {
                if r != rank && mat[r] & (1u64 << col) != 0 {
                    mat[r] ^= mat[rank];
                }
            }
            rank += 1;
        }
        if rank < n {
            return false;
        }
    }
    true
}
