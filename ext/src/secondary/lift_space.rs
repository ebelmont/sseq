//! Seed space analysis for Cτ² lifts of bounded module resolutions.
//!
//! A Cτ² lift is a choice of secondary resolution — specifically, a choice of s=2 homotopy
//! ("seed") that satisfies the constraint equations at s=3. Different choices may produce
//! different d₂ differentials.
//!
//! The main type is [`Ct2LiftSpace`], which computes the affine space of valid seeds and
//! provides iteration over [`SecondaryResolution`] objects for each valid lift.

use std::{io, sync::Arc};

use algebra::{
    module::{
        HomModule, Module,
        homomorphism::{FreeModuleHomomorphism, ModuleHomomorphism},
    },
    pair_algebra::PairAlgebra,
    Algebra,
};
use fp::{
    matrix::{Matrix, Subspace},
    vector::FpVector,
};
use sseq::coordinates::{Bidegree, BidegreeGenerator};

use crate::chain_complex::{
    AugmentedChainComplex, BoundedChainComplex, ChainComplex, FreeChainComplex,
};

use super::{SecondaryLift, SecondaryResolution};

// ---------------------------------------------------------------------------
// D2 data types
// ---------------------------------------------------------------------------

/// A single d₂ differential entry.
pub struct D2Entry {
    /// Source class: the generator whose d₂ we're recording.
    pub source: BidegreeGenerator,
    /// Image coefficients in the target bidegree.
    pub coefficients: Vec<u32>,
}

impl D2Entry {
    /// Whether this differential is nonzero.
    pub fn is_nonzero(&self) -> bool {
        self.coefficients.iter().any(|&x| x != 0)
    }
}

/// All d₂ differentials from a secondary resolution.
pub struct D2Data {
    pub entries: Vec<D2Entry>,
}

impl D2Data {
    /// Extract all d₂ from a computed secondary resolution.
    #[tracing::instrument(skip_all)]
    pub fn extract<CC: FreeChainComplex>(
        resolution: &CC,
        lift: &SecondaryResolution<CC>,
    ) -> Self
    where
        CC::Algebra: PairAlgebra,
    {
        let d2_shift = Bidegree::n_s(-1, 2);
        let mut entries = Vec::new();

        for b in resolution.iter_stem() {
            if b.s() < 2 {
                continue;
            }
            if b.t() - 1 > resolution.module(b.s() - 2).max_computed_degree() {
                continue;
            }
            let homotopy = lift.homotopy(b.s());
            let m = homotopy.homotopies.hom_k(b.t() - 1);
            for (i, entry) in m.into_iter().enumerate() {
                entries.push(D2Entry {
                    source: BidegreeGenerator::new(b - d2_shift, i),
                    coefficients: entry,
                });
            }
        }
        D2Data { entries }
    }

    /// Write nonzero d₂ entries to a writer.
    pub fn write_nonzero(&self, writer: &mut dyn io::Write) -> io::Result<()> {
        for entry in &self.entries {
            if entry.is_nonzero() {
                writeln!(
                    writer,
                    "  d_2 x_{} = {:?}",
                    entry.source, entry.coefficients
                )?;
            }
        }
        Ok(())
    }

    /// Number of nonzero differentials.
    pub fn count_nonzero(&self) -> usize {
        self.entries.iter().filter(|e| e.is_nonzero()).count()
    }

    /// Flatten all coefficient vectors into a single flat vector for comparison.
    pub fn flatten(&self) -> Vec<u32> {
        self.entries
            .iter()
            .flat_map(|e| e.coefficients.iter().copied())
            .collect()
    }
}

// Convenience method on SecondaryResolution
impl<CC: FreeChainComplex> SecondaryResolution<CC>
where
    CC::Algebra: PairAlgebra,
{
    /// Extract all d₂ differentials from this secondary resolution.
    pub fn d2(&self) -> D2Data {
        D2Data::extract(&*self.underlying, self)
    }
}

// ---------------------------------------------------------------------------
// Ct2LiftSpace
// ---------------------------------------------------------------------------

/// The space of valid Cτ² lifts of a bounded module resolution.
///
/// Computes the affine subspace of Hom¹(M₂, M) whose elements satisfy the s=3
/// constraint equations, and provides iteration over the resulting
/// [`SecondaryResolution`] objects.
///
/// # Type Parameters
/// - `CC`: The resolution type (a free chain complex).
/// - `M`: The target module type (the module being resolved).
///
/// # Usage
/// ```ignore
/// let lifts = Ct2LiftSpace::new(Arc::clone(&resolution))?;
/// for lift in lifts.iter() {
///     let d2 = lift.d2();
///     // ...
/// }
/// ```
pub struct Ct2LiftSpace<CC: FreeChainComplex, M: Module<Algebra = CC::Algebra>>
where
    CC::Algebra: PairAlgebra,
{
    resolution: Arc<CC>,
    /// The base SecondaryResolution with precomputed composites/intermediates.
    base: SecondaryResolution<CC>,
    /// Dimension of the Hom¹(M₂, M) space.
    dim: usize,
    /// Row-reduced augmented constraint matrix [C | rhs].
    matrix: Matrix,
    /// Pivot columns (constrained variables).
    pivot_cols: Vec<usize>,
    /// Free variable columns.
    free_vars: Vec<usize>,
    /// The particular solution seed (free vars = 0).
    particular_seed: u128,
    /// HomModule used for naming seeds.
    hom: HomModule<M>,
    /// Maximum source degree.
    max_src_deg: i32,
    /// The augmentation map's target module.
    module: Arc<M>,
}

impl<CC, M> Ct2LiftSpace<CC, M>
where
    CC: FreeChainComplex + AugmentedChainComplex<ChainMap = FreeModuleHomomorphism<M>>,
    CC::Algebra: PairAlgebra,
    M: Module<Algebra = CC::Algebra>,
    <CC as AugmentedChainComplex>::TargetComplex: BoundedChainComplex<Module = M>,
{
    /// Compute the space of valid Cτ² lifts for a bounded module resolution.
    ///
    /// This:
    /// 1. Sets up Hom¹(M₂, M) and lifts each basis element
    /// 2. Computes base composites/intermediates (seed-independent)
    /// 3. Builds constraint equations at s=3
    /// 4. Row-reduces to find the affine solution space
    #[tracing::instrument(skip_all)]
    pub fn new(resolution: Arc<CC>) -> anyhow::Result<Self> {
        let target_cc = resolution.target();

        if target_cc.max_s() > 1 {
            anyhow::bail!("Cannot compute secondary space for non-module");
        }

        let module = target_cc.module(0);

        let source = resolution.module(2); // M_2(X)
        let m0 = resolution.module(0); // M_0(X)
        let m1 = resolution.module(1); // M_1(X)

        // For unbounded modules (e.g. classifying spaces), truncate at the
        // resolution's computed range — cell data above that degree is irrelevant.
        let max_nonzero = module
            .max_degree()
            .unwrap_or_else(|| source.max_computed_degree());

        let hom = HomModule::new_with_max(Arc::clone(&source), Arc::clone(&module), max_nonzero);
        resolution.algebra().compute_basis(2 * max_nonzero);
        hom.compute_basis(max_nonzero);

        let g = resolution.chain_map(0);
        let p = source.prime();
        let max_src_deg = source.max_computed_degree();

        let degree = 1;
        let dim = hom.dimension(degree);

        // Precompute lifts for each basis element
        let basis_lifts: Vec<_> = (0..dim)
            .map(|j| {
                let f = FreeModuleHomomorphism::new(Arc::clone(&source), g.target(), degree);
                let gbe = hom.block_structures[degree].index_to_generator_basis_elt(j);
                for gd in f.min_degree()..=max_src_deg {
                    let n = source.number_of_gens_in_degree(gd);
                    let target_dim = module.dimension(gd - degree);
                    let mut rows = Vec::with_capacity(n);
                    for gi in 0..n {
                        let mut row = FpVector::new(p, target_dim);
                        if gd == gbe.generator_degree && gi == gbe.generator_index {
                            row.set_entry(gbe.basis_index, 1);
                        }
                        rows.push(row);
                    }
                    f.add_generators_from_rows(gd, rows);
                }
                f.lift_through(&*g).expect("basis element not in image of g")
            })
            .collect();

        // Compute composites and intermediates once (seed-independent)
        let base = SecondaryResolution::new(Arc::clone(&resolution));
        base.initialize_homotopies();
        base.compute_composites();
        base.compute_intermediates();

        let d3 = resolution.differential(3);
        let d1 = resolution.differential(1);

        // Collect constraint equations over F_p
        let mut constraint_coeffs: Vec<FpVector> = Vec::new();
        let mut constraint_rhs: Vec<u32> = Vec::new();

        let min_t = base.homotopies()[3].homotopies.min_degree();
        let max_t = base.max().t(3);

        for t in min_t..max_t {
            let num_gens_s3 = resolution.module(3).number_of_gens_in_degree(t);
            if num_gens_s3 == 0 {
                continue;
            }

            let target_deg = t - 1;
            let target_dim = m0.dimension(target_deg);
            if target_dim == 0 {
                continue;
            }

            // Build image subspace of d_1 at target_deg
            let m1_dim = m1.dimension(target_deg);
            let mut image = Subspace::new(p, target_dim);
            for basis_idx in 0..m1_dim {
                let mut v = FpVector::new(p, target_dim);
                d1.apply_to_basis_element(v.as_slice_mut(), 1, target_deg, basis_idx);
                image.add_vector(v.as_slice());
            }

            // If image is the entire space, all intermediates are liftable
            if image.dimension() == target_dim {
                continue;
            }

            for idx in 0..num_gens_s3 {
                let bg = BidegreeGenerator::s_t(3, t, idx);
                let mut base_int = base.compute_intermediate(bg);

                let mut contribs: Vec<FpVector> = (0..dim)
                    .map(|j| {
                        let mut contrib = FpVector::new(p, target_dim);
                        basis_lifts[j].apply(
                            contrib.as_slice_mut(),
                            1,
                            t,
                            d3.output(t, idx).as_slice(),
                        );
                        contrib
                    })
                    .collect();

                image.reduce(base_int.as_slice_mut());
                for contrib in &mut contribs {
                    image.reduce(contrib.as_slice_mut());
                }

                for c in 0..target_dim {
                    let rhs_bit = base_int.entry(c);
                    let any_contrib = contribs.iter().any(|v| v.entry(c) != 0);

                    if rhs_bit == 0 && !any_contrib {
                        continue;
                    }

                    if rhs_bit != 0 && !any_contrib {
                        anyhow::bail!(
                            "No valid seeds: unsatisfiable constraint at (s=3, t={t}, idx={idx})"
                        );
                    }

                    let mut row = FpVector::new(p, dim);
                    for j in 0..dim {
                        row.set_entry(j, contribs[j].entry(c));
                    }
                    constraint_coeffs.push(row);
                    constraint_rhs.push(rhs_bit);
                }
            }
        }

        let num_constraints = constraint_coeffs.len();

        // Build augmented matrix [C | rhs] and row reduce
        let mut matrix = Matrix::new(p, std::cmp::max(num_constraints, 1), dim + 1);
        for (i, (row, &rhs)) in constraint_coeffs.iter().zip(&constraint_rhs).enumerate() {
            for j in 0..dim {
                matrix.row_mut(i).set_entry(j, row.entry(j));
            }
            matrix.row_mut(i).set_entry(dim, rhs);
        }
        matrix.initialize_pivots();
        matrix.row_reduce();

        // Check consistency
        if num_constraints > 0 && matrix.pivots()[dim] >= 0 {
            anyhow::bail!("No valid seeds: inconsistent constraint system");
        }

        // Read off solution space
        let mut pivot_cols: Vec<usize> = Vec::new();
        for col in 0..dim {
            if matrix.pivots()[col] >= 0 {
                pivot_cols.push(col);
            }
        }

        let free_vars: Vec<usize> = (0..dim).filter(|j| matrix.pivots()[*j] < 0).collect();

        // Compute particular solution seed (free vars = 0)
        let mut particular_seed: u128 = 0;
        for &col in &pivot_cols {
            let row = matrix.pivots()[col] as usize;
            if matrix.row(row).entry(dim) != 0 {
                particular_seed |= 1u128 << col;
            }
        }

        // basis_lifts is only needed during constraint building; drop it.
        drop(basis_lifts);

        Ok(Ct2LiftSpace {
            resolution,
            base,
            dim,
            matrix,
            pivot_cols,
            free_vars,
            particular_seed,
            hom,
            max_src_deg,
            module,
        })
    }

    /// Dimension of the Hom¹ space (ambient space of seeds).
    pub fn hom_dim(&self) -> usize {
        self.dim
    }

    /// Number of free dimensions = dim - rank.
    pub fn free_dim(&self) -> usize {
        self.free_vars.len()
    }

    /// Rank of the constraint system.
    pub fn rank(&self) -> usize {
        self.pivot_cols.len()
    }

    /// Number of valid seeds = 2^free_dim.
    pub fn num_lifts(&self) -> u128 {
        1u128 << self.free_dim()
    }

    /// The particular solution seed value.
    pub fn particular_seed(&self) -> u128 {
        self.particular_seed
    }

    /// Human-readable name for a seed value.
    pub fn seed_name(&self, seed: u128) -> String {
        if seed == 0 {
            "0".to_string()
        } else {
            let parts: Vec<String> = (0..self.dim)
                .filter(|&idx| (seed >> idx) & 1 != 0)
                .map(|idx| self.hom.basis_element_to_string(1, idx))
                .collect();
            parts.join(" + ")
        }
    }

    /// Build the `FreeModuleHomomorphism` representing a seed's lift through the augmentation.
    pub fn build_seed_lift(
        &self,
        seed: u128,
    ) -> FreeModuleHomomorphism<CC::Module> {
        let source = self.resolution.module(2);
        let g = self.resolution.chain_map(0);
        let p = source.prime();

        let f = FreeModuleHomomorphism::new(Arc::clone(&source), g.target(), 1);
        for gd in f.min_degree()..=self.max_src_deg {
            let n = source.number_of_gens_in_degree(gd);
            let target_dim = self.module.dimension(gd - 1);
            let mut rows = Vec::with_capacity(n);
            for gi in 0..n {
                let mut row = FpVector::new(p, target_dim);
                for idx in 0..self.dim {
                    if (seed >> idx) & 1 != 0 {
                        let gbe =
                            self.hom.block_structures[1].index_to_generator_basis_elt(idx);
                        if gd == gbe.generator_degree && gi == gbe.generator_index {
                            row.add_basis_element(gbe.basis_index, 1);
                        }
                    }
                }
                rows.push(row);
            }
            f.add_generators_from_rows(gd, rows);
        }
        f.lift_through(&*g).expect("seed element not in image of g")
    }

    /// Construct a [`SecondaryResolution`] for a specific seed value.
    ///
    /// Reuses precomputed composites and intermediates from the base.
    #[tracing::instrument(skip(self))]
    pub fn lift(&self, seed: u128) -> SecondaryResolution<CC>
    where
        <CC::Algebra as PairAlgebra>::Element: Clone,
    {
        let h_i = self.build_seed_lift(seed);

        let source = self.resolution.module(2);
        let m0 = self.resolution.module(0);
        let p = source.prime();

        let lift = SecondaryResolution::new(Arc::clone(&self.resolution));
        lift.initialize_homotopies();
        lift.copy_composites_from(&self.base);
        lift.copy_intermediates_from(&self.base);

        // Seed homotopy at s=2
        {
            let hom_field = &lift.homotopies()[2].homotopies;
            let max_seed_deg =
                std::cmp::min(lift.max().t(2) - 1, source.max_computed_degree());
            for d in hom_field.min_degree()..=max_seed_deg {
                let n = source.number_of_gens_in_degree(d);
                let target_dim = m0.dimension(d - hom_field.degree_shift());
                let mut rows = Vec::with_capacity(n);
                for gi in 0..n {
                    let mut row = FpVector::new(p, target_dim);
                    if d >= h_i.min_degree() && d < h_i.next_degree() {
                        let h_out = h_i.output(d, gi);
                        if !h_out.is_zero() {
                            row.add(h_out, 1);
                        }
                    }
                    rows.push(row);
                }
                hom_field.add_generators_from_rows(d, rows);
            }
        }

        // Compute homotopy steps
        let min_t = lift.homotopies()[2].homotopies.min_degree();
        let s_range = lift.homotopies().range();
        let min = Bidegree::s_t(s_range.start + 1, min_t);
        let max = lift.max().restrict(s_range.end);
        sseq::coordinates::iter_s_t(&|b| lift.compute_homotopy_step(b), min, max);

        lift
    }

    /// Iterate over ALL valid seeds. Yields one seed value per valid lift.
    ///
    /// Warning: there are 2^free_dim of these; only practical for small free_dim.
    pub fn seed_iter(&self) -> impl Iterator<Item = u128> + '_ {
        let num = self.num_lifts();
        (0..num).map(|bits| self.seed_from_bits(bits))
    }

    /// Iterate over [`SecondaryResolution`] objects, one per valid seed.
    pub fn iter(&self) -> impl Iterator<Item = SecondaryResolution<CC>> + '_
    where
        <CC::Algebra as PairAlgebra>::Element: Clone,
    {
        self.seed_iter().map(|seed| self.lift(seed))
    }

    /// Convert a bits value (indexing into the free variable space) to a seed.
    pub fn seed_from_bits(&self, bits: u128) -> u128 {
        let mut seed = self.particular_seed;
        for (bit_idx, &f_k) in self.free_vars.iter().enumerate() {
            if (bits >> bit_idx) & 1 != 0 {
                // Build direction vector and back-substitute
                let mut v = vec![0u32; self.dim];
                v[f_k] = 1;
                for &col in self.pivot_cols.iter().rev() {
                    let row_idx = self.matrix.pivots()[col] as usize;
                    let mut val = 0u32;
                    for j in (col + 1)..self.dim {
                        val ^= self.matrix.row(row_idx).entry(j) * v[j];
                    }
                    v[col] = val;
                }
                for j in 0..self.dim {
                    if v[j] != 0 {
                        seed ^= 1u128 << j;
                    }
                }
            }
        }
        seed
    }

    /// Seeds for each free direction (perturbed from the particular solution).
    ///
    /// Useful for d₂ uniqueness analysis without computing all 2^free_dim lifts.
    pub fn direction_seeds(&self) -> Vec<u128> {
        self.free_vars
            .iter()
            .map(|&f_k| {
                let mut v = vec![0u32; self.dim];
                v[f_k] = 1;
                for &col in self.pivot_cols.iter().rev() {
                    let row_idx = self.matrix.pivots()[col] as usize;
                    let mut val = 0u32;
                    for j in (col + 1)..self.dim {
                        val ^= self.matrix.row(row_idx).entry(j) * v[j];
                    }
                    v[col] = val;
                }
                let mut seed = self.particular_seed;
                for j in 0..self.dim {
                    if v[j] != 0 {
                        seed ^= 1u128 << j;
                    }
                }
                seed
            })
            .collect()
    }

    /// Analyze d₂ uniqueness: how many distinct d₂ differentials arise from
    /// different seeds. Returns a [`D2Uniqueness`] result.
    #[tracing::instrument(skip(self))]
    pub fn d2_uniqueness(&self) -> D2Uniqueness
    where
        <CC::Algebra as PairAlgebra>::Element: Clone,
    {
        let d2_base = {
            let lift = self.lift(self.particular_seed);
            D2Data::extract(&*self.resolution, &lift)
        };

        let base_flat = d2_base.flatten();
        let direction_seeds = self.direction_seeds();

        let mut differing_directions: Vec<usize> = Vec::new();
        let mut differing_deltas: Vec<Vec<u32>> = Vec::new();

        let p = self.resolution.module(0).prime();

        for (k, &dir_seed) in direction_seeds.iter().enumerate() {
            let d2_dir = {
                let lift = self.lift(dir_seed);
                D2Data::extract(&*self.resolution, &lift)
            };
            let dir_flat = d2_dir.flatten();

            let delta: Vec<u32> = dir_flat
                .iter()
                .zip(base_flat.iter())
                .map(|(&a, &b)| a ^ b)
                .collect();

            if delta.iter().any(|&x| x != 0) {
                differing_directions.push(k);
                differing_deltas.push(delta);
            }
        }

        // Row-reduce deltas to find linearly independent ones
        let delta_len = base_flat.len();
        let mut independent: Vec<usize> = Vec::new();
        if delta_len > 0 && !differing_deltas.is_empty() {
            let mut basis = Subspace::new(p, delta_len);
            for (i, delta) in differing_deltas.iter().enumerate() {
                let mut v = FpVector::new(p, delta_len);
                for (col, &val) in delta.iter().enumerate() {
                    v.set_entry(col, val);
                }
                let mut v_copy = v.clone();
                basis.reduce(v_copy.as_slice_mut());
                if !v_copy.is_zero() {
                    basis.add_vector(v.as_slice());
                    independent.push(i);
                }
            }
        }

        D2Uniqueness {
            base_d2: d2_base,
            differing_directions,
            independent,
            particular_seed: self.particular_seed,
            direction_seeds,
        }
    }
}

// ---------------------------------------------------------------------------
// D2Uniqueness
// ---------------------------------------------------------------------------

/// Analysis of how d₂ varies across the seed space.
pub struct D2Uniqueness {
    /// d₂ for the particular (base) solution.
    pub base_d2: D2Data,
    /// Indices of free directions that actually change d₂.
    pub differing_directions: Vec<usize>,
    /// Indices into `differing_directions` of linearly independent deltas.
    pub independent: Vec<usize>,
    /// The particular solution seed.
    particular_seed: u128,
    /// Seeds for each free direction.
    direction_seeds: Vec<u128>,
}

impl D2Uniqueness {
    /// Number of distinct d₂ groups = 2^rank.
    pub fn num_groups(&self) -> u128 {
        1u128 << self.independent.len()
    }

    /// Whether all valid lifts produce the same d₂.
    pub fn is_unique(&self) -> bool {
        self.independent.is_empty()
    }

    /// Iterate over (group_index, seed) pairs, one representative per d₂ group.
    pub fn group_seeds(&self) -> impl Iterator<Item = (usize, u128)> + '_ {
        let num_groups = self.num_groups();
        (0..num_groups).map(move |group_bits| {
            let mut group_seed = self.particular_seed;
            for (bit_idx, &indep_idx) in self.independent.iter().enumerate() {
                if (group_bits >> bit_idx) & 1 != 0 {
                    let dir_k = self.differing_directions[indep_idx];
                    group_seed ^= self.particular_seed ^ self.direction_seeds[dir_k];
                }
            }
            (group_bits as usize, group_seed)
        })
    }
}
