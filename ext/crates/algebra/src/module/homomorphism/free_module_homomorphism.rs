use std::sync::Arc;

use crate::{
    algebra::MuAlgebra,
    module::{
        free_module::OperationGeneratorPair,
        homomorphism::{ModuleHomomorphism, ZeroHomomorphism},
        Module, MuFreeModule,
    },
    UnstableAlgebra,
};
use fp::{
    matrix::{MatrixSliceMut, QuasiInverse, Subspace},
    vector::{FpVector, Slice, SliceMut},
};
use once::OnceBiVec;
use std::any::Any;

pub type FreeModuleHomomorphism<M> = MuFreeModuleHomomorphism<false, M>;
pub type UnstableFreeModuleHomomorphism<M> = MuFreeModuleHomomorphism<true, M>;

pub struct MuFreeModuleHomomorphism<const U: bool, M: Module>
where
    M::Algebra: MuAlgebra<U>,
{
    source: Arc<MuFreeModule<U, M::Algebra>>,
    target: Arc<M>,
    pub outputs: OnceBiVec<Vec<FpVector>>, // degree --> input_idx --> output
    pub images: OnceBiVec<Option<Subspace>>,
    pub kernels: OnceBiVec<Option<Subspace>>,
    pub quasi_inverses: OnceBiVec<Option<QuasiInverse>>,
    min_degree: i32,
    /// degree shift, such that ouptut_degree = input_degree - degree_shift
    degree_shift: i32,
}

impl<const U: bool, M: Module> ModuleHomomorphism for MuFreeModuleHomomorphism<U, M>
where
    M::Algebra: MuAlgebra<U>,
{
    type Source = MuFreeModule<U, M::Algebra>;
    type Target = M;

    fn source(&self) -> Arc<Self::Source> {
        Arc::clone(&self.source)
    }

    fn target(&self) -> Arc<Self::Target> {
        Arc::clone(&self.target)
    }

    fn degree_shift(&self) -> i32 {
        self.degree_shift
    }

    fn apply_to_basis_element(
        &self,
        result: SliceMut,
        coeff: u32,
        input_degree: i32,
        input_index: usize,
    ) {
        assert!(input_degree >= self.source.min_degree());
        assert!(input_index < self.source.dimension(input_degree));
        let output_degree = input_degree - self.degree_shift;
        assert_eq!(
            self.target.dimension(output_degree),
            result.as_slice().len()
        );
        let OperationGeneratorPair {
            operation_degree,
            generator_degree,
            operation_index,
            generator_index,
        } = *self.source.index_to_op_gen(input_degree, input_index);

        if generator_degree >= self.min_degree() {
            let output_on_generator = self.output(generator_degree, generator_index);
            self.target.act(
                result,
                coeff,
                operation_degree,
                operation_index,
                generator_degree - self.degree_shift,
                output_on_generator.as_slice(),
            );
        }
    }

    fn quasi_inverse(&self, degree: i32) -> Option<&QuasiInverse> {
        self.quasi_inverses.get(degree).and_then(Option::as_ref)
    }

    fn kernel(&self, degree: i32) -> Option<&Subspace> {
        self.kernels.get(degree).and_then(Option::as_ref)
    }

    fn image(&self, degree: i32) -> Option<&Subspace> {
        self.images.get(degree).and_then(Option::as_ref)
    }

    fn compute_auxiliary_data_through_degree(&self, degree: i32) {
        self.kernels.extend(degree, |i| {
            let (image, kernel, qi) = self.auxiliary_data(i);
            self.images.push_checked(Some(image), i);
            self.quasi_inverses.push_checked(Some(qi), i);
            Some(kernel)
        });
    }
}

impl<const U: bool, M: Module> MuFreeModuleHomomorphism<U, M>
where
    M::Algebra: MuAlgebra<U>,
{
    pub fn new(
        source: Arc<MuFreeModule<U, M::Algebra>>,
        target: Arc<M>,
        degree_shift: i32,
    ) -> Self {
        let min_degree = std::cmp::max(source.min_degree(), target.min_degree() + degree_shift);
        let outputs = OnceBiVec::new(min_degree);
        let kernels = OnceBiVec::new(min_degree);
        let images = OnceBiVec::new(min_degree);
        let quasi_inverses = OnceBiVec::new(min_degree);
        Self {
            source,
            target,
            outputs,
            images,
            kernels,
            quasi_inverses,
            min_degree,
            degree_shift,
        }
    }

    pub fn degree_shift(&self) -> i32 {
        self.degree_shift
    }

    pub fn min_degree(&self) -> i32 {
        self.min_degree
    }

    pub fn next_degree(&self) -> i32 {
        self.outputs.len()
    }

    pub fn output(&self, generator_degree: i32, generator_index: usize) -> &FpVector {
        assert!(
            generator_degree >= self.min_degree(),
            "generator_degree {} less than min degree {}",
            generator_degree,
            self.min_degree()
        );
        assert!(
            generator_index < self.source.number_of_gens_in_degree(generator_degree),
            "generator_index {} greater than number of generators {}",
            generator_index,
            self.source.number_of_gens_in_degree(generator_degree)
        );
        &self.outputs[generator_degree][generator_index]
    }

    pub fn differential_density(&self, degree: i32) -> f32 {
        let outputs = &self.outputs[degree];
        if outputs.is_empty() {
            f32::NAN
        } else {
            outputs.iter().map(FpVector::density).sum::<f32>() / outputs.len() as f32
        }
    }

    pub fn extend_by_zero(&self, degree: i32) {
        let p = self.prime();
        self.outputs.extend(degree, |i| {
            let num_gens = self.source.number_of_gens_in_degree(i);
            let dimension = self.target.dimension(i - self.degree_shift);
            let mut new_outputs: Vec<FpVector> = Vec::with_capacity(num_gens);
            for _ in 0..num_gens {
                new_outputs.push(FpVector::new(p, dimension));
            }
            new_outputs
        });
    }

    pub fn add_generators_from_big_vector(&self, degree: i32, outputs_vectors: Slice) {
        let p = self.prime();
        let new_generators = self.source.number_of_gens_in_degree(degree);
        let target_dimension = self.target.dimension(degree - self.degree_shift);
        let mut new_outputs: Vec<FpVector> = Vec::with_capacity(new_generators);
        for _ in 0..new_generators {
            new_outputs.push(FpVector::new(p, target_dimension));
        }
        if target_dimension == 0 {
            self.outputs.push_checked(new_outputs, degree);
            return;
        }
        for (i, new_output) in new_outputs.iter_mut().enumerate() {
            new_output
                .as_slice_mut()
                .assign(outputs_vectors.slice(target_dimension * i, target_dimension * (i + 1)));
        }
        self.outputs.push_checked(new_outputs, degree);
    }

    /// A MatrixSlice will do but there is no applicaiton of this struct, so it doesn't exist
    /// yet...
    pub fn add_generators_from_matrix_rows(&self, degree: i32, mut matrix: MatrixSliceMut) {
        let p = self.prime();
        let new_generators = self.source.number_of_gens_in_degree(degree);
        let target_dimension = self.target.dimension(degree - self.degree_shift);

        let mut new_outputs: Vec<FpVector> = Vec::with_capacity(new_generators);
        for _ in 0..new_generators {
            new_outputs.push(FpVector::new(p, target_dimension));
        }
        if target_dimension == 0 {
            self.outputs.push_checked(new_outputs, degree);
            return;
        }
        for (i, new_output) in new_outputs.iter_mut().enumerate() {
            new_output.as_slice_mut().assign(matrix.row(i));
        }
        self.outputs.push_checked(new_outputs, degree);
    }

    pub fn add_generators_from_rows(&self, degree: i32, rows: Vec<FpVector>) {
        self.outputs.push_checked(rows, degree);
    }

    /// Add the image of a bidegree out of order. See
    /// [`OnceVec::push_ooo`](once::OnceVec::push_ooo) for details on return value.
    pub fn add_generators_from_rows_ooo(
        &self,
        degree: i32,
        rows: Vec<FpVector>,
    ) -> std::ops::Range<i32> {
        self.outputs.push_ooo(rows, degree)
    }

    /// List of outputs that have been added out of order
    pub fn ooo_outputs(&self) -> Vec<i32> {
        self.outputs.ooo_elements()
    }

    pub fn apply_to_generator(&self, result: &mut FpVector, coeff: u32, degree: i32, idx: usize) {
        let output_on_gen = self.output(degree, idx);
        result.add(output_on_gen, coeff);
    }

    pub fn set_image(&self, degree: i32, image: Option<Subspace>) {
        self.images.push_checked(image, degree);
    }

    pub fn set_kernel(&self, degree: i32, kernel: Option<Subspace>) {
        self.kernels.push_checked(kernel, degree);
    }

    pub fn set_quasi_inverse(&self, degree: i32, quasi_inverse: Option<QuasiInverse>) {
        self.quasi_inverses.push_checked(quasi_inverse, degree);
    }
}
// Ensure that the Module type M implements Any + Send + Sync so we can downcast.
impl<const U: bool, A> MuFreeModuleHomomorphism<U, MuFreeModule<U, A>>
where
    A: MuAlgebra<U>,
{
    // Restrict the method further so that M::Algebra must implement UnstableAlgebra.
    pub fn loops(&self) -> Self
    where
        A: UnstableAlgebra,
    {
        // Create a new instance with copied data
        let mut result = Self {
            degree_shift: self.degree_shift.clone(),
            source: self.source.clone(),
            target: self.target.clone(),
            outputs: self.outputs.clone(),
            images: OnceBiVec::new(self.min_degree),
            kernels: OnceBiVec::new(self.min_degree),
            quasi_inverses: OnceBiVec::new(self.min_degree),
            min_degree: self.min_degree.clone(),
        };

        let source_any: Arc<dyn Any + Send + Sync> = self.source.clone();
        let target_any: Arc<dyn Any + Send + Sync> = self.target.clone();
        // Downcast to the expected concrete type. This requires that the source is actually
        // a MuFreeModule<true, M::Algebra>.
        let source_freemodule = Arc::downcast::<MuFreeModule<true, A>>(source_any)
            .expect("Source is not a MuFreeModule<true, _>");
        // Downcast to the expected concrete type. This requires that the target is actually
        // a MuFreeModule<true, M::Algebra>.
        let target_freemodule = Arc::downcast::<MuFreeModule<true, A>>(target_any)
            .expect("Target is not a MuFreeModule<true, _>");
        // Now that we have the target as a MuFreeModule<true, M::Algebra>, we can safely call
        // its methods.
        let min_degree = result.min_degree();
        let n = (min_degree + 1) / 2;
        for (i, gen) in source_freemodule
            .iter_gens(source_freemodule.max_generator_degree().unwrap())
            .enumerate()
        {
            let deg = gen.0;
            // Get mutable access to outputs for degree `deg`
            if let Some(out_vec) = result.outputs.data.get_mut((deg - min_degree) as usize) {
                if out_vec.is_empty() {
                    continue;
                }
                for idx in 0..out_vec[0].len() {
                    let opgen = target_freemodule.index_to_op_gen(deg, idx);
                    if opgen.looped(source_freemodule.algebra()) {
                        // Modify the FpVector entry at index `idx`
                        out_vec[0].set_entry(idx, 0);
                    }
                }
            }
        }
        result
    }
}
impl<const U: bool, M: Module> ZeroHomomorphism<MuFreeModule<U, M::Algebra>, M>
    for MuFreeModuleHomomorphism<U, M>
where
    M::Algebra: MuAlgebra<U>,
{
    fn zero_homomorphism(
        source: Arc<MuFreeModule<U, M::Algebra>>,
        target: Arc<M>,
        degree_shift: i32,
    ) -> Self {
        Self::new(source, target, degree_shift)
    }
}

impl<const U: bool, A: MuAlgebra<U>> MuFreeModuleHomomorphism<U, MuFreeModule<U, A>> {
    /// Given f: M -> N, compute the dual f*: Hom(N, k) -> Hom(M, k) in source (N) degree t.
    pub fn hom_k(&self, t: i32) -> Vec<Vec<u32>> {
        let source_dim = self.source.number_of_gens_in_degree(t + self.degree_shift);
        let target_dim = self.target.number_of_gens_in_degree(t);
        if target_dim == 0 {
            return vec![];
        }
        let mut result = vec![vec![0; source_dim]; target_dim];

        let offset = self.target.generator_offset(t, t, 0);
        for i in 0..source_dim {
            let output = self.output(t + self.degree_shift, i);
            #[allow(clippy::needless_range_loop)]
            for j in 0..target_dim {
                result[j][i] = output.entry(offset + j);
            }
        }
        result
    }
}
