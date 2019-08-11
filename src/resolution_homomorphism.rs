use std::rc::Weak;
use std::cell::RefCell;

use crate::once::OnceVec;
use crate::fp_vector::{ FpVector, FpVectorT };
use crate::matrix::Matrix;
use crate::module::{Module, OptionModule};
use crate::free_module::FreeModule;
use crate::module_homomorphism::{ZeroHomomorphism, ModuleHomomorphism};
use crate::finite_dimensional_module::FiniteDimensionalModule as FDModule;
use crate::free_module_homomorphism::FreeModuleHomomorphism;
use crate::chain_complex::ChainComplex;
use crate::chain_complex::ChainComplexConcentratedInDegreeZero as CCDZ;
use crate::resolution::Resolution;

pub struct ResolutionHomomorphism<
    S : Module, F1 : ModuleHomomorphism<S, S>, CC1 : ChainComplex<S, F1>,
    T : Module, F2 : ModuleHomomorphism<T, T>, CC2 : ChainComplex<T, F2>
> {
    name : String,
    source : Weak<RefCell<Resolution<S, F1, CC1>>>,
    target : Weak<RefCell<Resolution<T, F2, CC2>>>,
    maps : OnceVec<FreeModuleHomomorphism<FreeModule>>,
    homological_degree_shift : u32,
    internal_degree_shift : i32
}

impl<
    S : Module, F1 : ModuleHomomorphism<S, S>, CC1 : ChainComplex<S, F1>,
    T : Module, F2 : ModuleHomomorphism<T, T>, CC2 : ChainComplex<T, F2>
> ResolutionHomomorphism<S, F1, CC1, T, F2, CC2> {
    pub fn new(
        name : String,
        source : Weak<RefCell<Resolution<S,F1,CC1>>>, target : Weak<RefCell<Resolution<T,F2,CC2>>>,
        homological_degree_shift : u32, internal_degree_shift : i32
    ) -> Self {
        Self {
            name,
            source,
            target,
            maps : OnceVec::new(),
            homological_degree_shift,
            internal_degree_shift
        }
    }

    fn get_map_ensure_length(&self, output_homological_degree : u32) -> &FreeModuleHomomorphism<FreeModule> {
        if output_homological_degree as usize >= self.maps.len() {
            let input_homological_degree = output_homological_degree + self.homological_degree_shift;
            let source_rc = self.source.upgrade().unwrap();
            let target_rc = self.target.upgrade().unwrap();
            self.maps.push(FreeModuleHomomorphism::new(source_rc.borrow().get_module(input_homological_degree), target_rc.borrow().get_module(output_homological_degree), self.internal_degree_shift));
        }
        return &self.maps[output_homological_degree as usize];
    }


    pub fn get_map(&self, output_homological_degree : u32) -> &FreeModuleHomomorphism<FreeModule> {
        &self.maps[output_homological_degree as usize]
    }

    /// Extend the resolution homomorphism such that it is defined on degrees
    /// (`source_homological_degree`, `source_degree`).
    pub fn extend(&self, source_homological_degree : u32, source_degree : i32){
        for i in self.homological_degree_shift ..= source_homological_degree {
            let f_cur = self.get_map_ensure_length(i - self.homological_degree_shift);
            let start_degree = *f_cur.get_lock();
            for j in start_degree + 1 ..= source_degree {
                self.extend_step(i, j, None);
            }
        }
    }

    pub fn extend_step(&self, input_homological_degree : u32, input_internal_degree : i32, extra_images : Option<&mut Matrix>){
        let output_homological_degree = input_homological_degree - self.homological_degree_shift;
        let f_cur = self.get_map_ensure_length(output_homological_degree);
        let computed_degree = *f_cur.get_lock();
        if input_internal_degree <= computed_degree {
            assert!(extra_images.is_none());
            return;
        }
        let num_gens = f_cur.get_source().get_number_of_gens_in_degree(input_internal_degree);
        let mut outputs = self.extend_step_helper(input_homological_degree, input_internal_degree, extra_images);
        let mut lock = f_cur.get_lock();
        if outputs.get_rows() > 0 {
            println!("Extending {} to hom_deg : {}, int_deg : {}", self.name, input_homological_degree, input_internal_degree);
            println!("outputs : {}", outputs);
        }
        f_cur.add_generators_from_matrix_rows(&lock, input_internal_degree, &mut outputs, 0, 0, num_gens);
        *lock += 1;
    }

    fn extend_step_helper(&self, input_homological_degree : u32, input_internal_degree : i32, mut extra_images : Option<&mut Matrix>) -> Matrix {
        let source_rc = self.source.upgrade().unwrap();
        let source = source_rc.borrow();
        let target_rc = self.target.upgrade().unwrap();
        let target = target_rc.borrow();
        let p = source.prime();
        assert!(input_homological_degree >= self.homological_degree_shift);
        let output_homological_degree = input_homological_degree - self.homological_degree_shift;
        let output_internal_degree = input_internal_degree - self.internal_degree_shift;        
        let target_chain_map = target.get_chain_map(output_homological_degree);
        let target_chain_map_qi = target_chain_map.get_quasi_inverse(output_internal_degree);
        let target_cc_dimension = target_chain_map.get_target().get_dimension(output_internal_degree);
        if let Some(extra_images_matrix) = &extra_images {
            assert!(target_cc_dimension <= extra_images_matrix.get_columns());
        }
        let f_cur = self.get_map(output_homological_degree);
        let num_gens = f_cur.get_source().get_number_of_gens_in_degree(input_internal_degree);
        let fx_dimension = f_cur.get_target().get_dimension(output_internal_degree);
        let mut outputs_matrix = Matrix::new(p, num_gens, fx_dimension);
        if num_gens == 0 || fx_dimension == 0 {
            return outputs_matrix;
        }
        if output_homological_degree == 0 {
            if let Some(extra_images_matrix) = extra_images {
                assert!(num_gens <= extra_images_matrix.get_rows(), 
                    format!("num_gens : {} greater than rows : {} hom_deg : {}, int_deg : {}", 
                    num_gens, extra_images_matrix.get_rows(), input_homological_degree, input_internal_degree));
                for k in 0 .. num_gens {
                    let old_slice = extra_images_matrix[k].get_slice();
                    extra_images_matrix[k].set_slice(0, target_cc_dimension);
                    target_chain_map_qi.as_ref().unwrap().apply(&mut outputs_matrix[k], 1, &extra_images_matrix[k]);
                    extra_images_matrix[k].restore_slice(old_slice);
                }
            }
            return outputs_matrix;            
        }
        let d_source = source.get_differential(input_homological_degree);
        let d_target = target.get_differential(output_homological_degree);
        let f_prev = self.get_map(output_homological_degree - 1);
        assert_eq!(d_source.get_source().get_name(), f_cur.get_source().get_name());
        assert_eq!(d_source.get_target().get_name(), f_prev.get_source().get_name());
        assert_eq!(d_target.get_source().get_name(), f_cur.get_target().get_name());
        assert_eq!(d_target.get_target().get_name(), f_prev.get_target().get_name());
        let d_quasi_inverse = d_target.get_quasi_inverse(output_internal_degree).unwrap();
        let dx_dimension = f_prev.get_source().get_dimension(input_internal_degree);
        let fdx_dimension = f_prev.get_target().get_dimension(output_internal_degree);
        let mut dx_vector = FpVector::new(p, dx_dimension);
        let mut fdx_vector = FpVector::new(p, fdx_dimension);
        let mut extra_image_row = 0;
        for k in 0 .. num_gens {
            d_source.apply_to_generator(&mut dx_vector, 1, input_internal_degree, k);
            if dx_vector.is_zero() {
                let extra_image_matrix = extra_images.as_mut().expect("Missing extra image rows");
                let old_slice = extra_image_matrix[extra_image_row].get_slice();
                extra_image_matrix[extra_image_row].set_slice(0, target_cc_dimension);
                target_chain_map_qi.as_ref().unwrap().apply(&mut outputs_matrix[k], 1, &extra_image_matrix[extra_image_row]);
                extra_image_matrix[extra_image_row].restore_slice(old_slice);
                extra_image_row += 1;
            } else {
                f_prev.apply(&mut fdx_vector, 1, input_internal_degree, &dx_vector);
                println!("{:?}", d_quasi_inverse);
                println!("fx_dimension : {}",fx_dimension);
                d_quasi_inverse.apply(&mut outputs_matrix[k], 1, &fdx_vector);
                println!("{}, {}, {}", dx_vector, fdx_vector, outputs_matrix[k]);
                dx_vector.set_to_zero();
                fdx_vector.set_to_zero();
            }
        }
        // let num_extra_image_rows = extra_images.map_or(0, |matrix| matrix.get_rows());
        // assert!(extra_image_row == num_extra_image_rows, "Extra image rows");
        return outputs_matrix;
    }

}

pub type ResolutionHomomorphismToUnit<M, F, CC> = ResolutionHomomorphism<M, F, CC,
    OptionModule<FDModule>,
    ZeroHomomorphism<OptionModule<FDModule>, OptionModule<FDModule>>,
    CCDZ<FDModule>
>;

// FreeModuleHomomorphism *ResolutionHomomorphism_getMap(ResolutionHomomorphism *f, uint homological_degree);
