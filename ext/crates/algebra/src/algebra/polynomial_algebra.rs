use std::collections::HashMap;
use std::fmt;

use once::OnceVec;
use fp::prime::ValidPrime;
use fp::vector::{FpVector, FpVectorT};

use crate::algebra::combinatorics::TruncatedPolynomialPartitions;
use crate::algebra::Algebra;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolynomialAlgebraMonomial {
    pub degree : i32,
    pub poly : FpVector,
    pub ext : FpVector,
    pub valid : bool
}

impl fmt::Display for PolynomialAlgebraMonomial {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "UAM(degree={}, valid={}, poly={}, ext={})", self.degree, self.valid, self.poly, self.ext)?;
        Ok(())
    }
}

impl PolynomialAlgebraMonomial {
    pub fn new(p : ValidPrime) -> Self {
        Self {
            degree : 0xFEDCBA9, // Looks invalid to me!
            poly : FpVector::new(p, 0),
            ext : FpVector::new(ValidPrime::new(2), 0),
            valid : true
        }
    }
}

pub struct PolynomialAlgebraTableEntry {    
    pub index_to_monomial : Vec<PolynomialAlgebraMonomial>, // degree -> index -> AdemBasisElement
    pub monomial_to_index : HashMap<PolynomialAlgebraMonomial, usize>, // degree -> AdemBasisElement -> index
}

impl PolynomialAlgebraTableEntry {
    pub fn new() -> Self {
        Self {
            index_to_monomial : Vec::new(),
            monomial_to_index : HashMap::new()
        }
    }
}

impl std::hash::Hash for PolynomialAlgebraMonomial {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.poly.hash(state);
        self.ext.hash(state);
    }
}

pub trait PolynomialAlgebra : Sized + Send + Sync + 'static {
    fn name(&self) -> String;
    fn prime(&self) -> ValidPrime;

    fn polynomial_partitions(&self) -> &TruncatedPolynomialPartitions;
    fn exterior_partitions(&self) -> &TruncatedPolynomialPartitions;


    fn min_degree(&self) -> i32 { 0 }

    fn polynomial_generators_in_degree(&self, degree : i32) -> usize;
    fn exterior_generators_in_degree(&self, degree : i32) -> usize;
    fn repr_poly_generator(&self, degree : i32, _index : usize) -> (String, u32);
    fn repr_ext_generator(&self, degree : i32, _index : usize) -> String;

    fn basis_table(&self) -> &OnceVec<PolynomialAlgebraTableEntry>;

    fn frobenius_on_generator(&self, degree : i32, index : usize) -> Option<usize>; 
    fn compute_generating_set(&self, degree : i32);
    

    fn compute_basis_step(&self, degree : i32){
        assert!(degree as usize == self.basis_table().len());
        let num_poly_gens = self.polynomial_generators_in_degree(degree);
        let num_ext_gens = self.exterior_generators_in_degree(degree);
        let poly_parts = self.polynomial_partitions();
        let ext_parts = self.exterior_partitions();
        if degree > 0 {
            poly_parts.add_gens_and_calculate_parts(degree, num_poly_gens);
            ext_parts.add_gens_and_calculate_parts(degree, num_ext_gens);
        }
        let mut table = PolynomialAlgebraTableEntry::new();
        for poly_deg in 0 ..= degree {
            let ext_deg = degree - poly_deg;
            for p in poly_parts.parts(poly_deg) {
                for e in ext_parts.parts(ext_deg) {
                    let index = table.index_to_monomial.len();
                    let mut m = PolynomialAlgebraMonomial {
                        degree,
                        poly : p.clone(),
                        ext : e.clone(),
                        valid : true
                    };
                    self.set_monomial_degree(&mut m, degree);
                    println!("==  idx : {}, m : {}", table.index_to_monomial.len(), m);
                    table.monomial_to_index.insert(m.clone(), index);
                    table.index_to_monomial.push(m);
                }
            }
        }
        self.basis_table().push(table);
    }

    
    fn monomial_to_index(&self, monomial : &PolynomialAlgebraMonomial) -> Option<usize> {
        self.basis_table()[monomial.degree as usize].monomial_to_index.get(monomial).map(|x| *x)
    }
    
    fn index_to_monomial(&self, degree : i32, index : usize) -> &PolynomialAlgebraMonomial {
        &self.basis_table()[degree as usize].index_to_monomial[index]
    }

    fn frobenius_monomial(&self, target : &mut FpVector, source : &FpVector) {
        let p = *self.prime() as i32;
        for (i, c) in source.iter_nonzero() {
            let (degree, in_idx) = self.polynomial_partitions().internal_idx_to_gen_deg(i);
            let frob = self.frobenius_on_generator(degree, in_idx);
            if let Some(e) = frob {
                let out_idx = self.polynomial_partitions().gen_deg_idx_to_internal_idx(p*degree, e);
                target.add_basis_element(out_idx, c);
            }
        }
    }

    fn multiply_monomials(&self, target : &mut PolynomialAlgebraMonomial, source : &PolynomialAlgebraMonomial) -> Option<()> {
        self.set_monomial_degree(target, target.degree + source.degree);

        target.ext.set_slice(0, source.ext.dimension());
        target.ext.add_truncate(&source.ext, 1)?;
        target.ext.clear_slice();

        let mut carry_vec = [FpVector::new(self.prime(), target.poly.dimension())];
        let mut source_vec = source.poly.clone();
        source_vec.set_scratch_vector_size(target.poly.dimension());
        let mut carry_q = true;
        while carry_q {
            carry_q = target.poly.add_carry(&source_vec, 1, &mut carry_vec);
            if carry_q {
                source_vec.set_to_zero_pure();
                self.frobenius_monomial(&mut source_vec, &carry_vec[0]);
                carry_vec[0].set_to_zero_pure();
            }
        }
        Some(())
    }

    fn multiply_polynomials(&self, target : &mut FpVector, coeff : u32, left_degree : i32, left : &FpVector, right_degree : i32, right : &FpVector) {
        let p = *self.prime();
        target.extend_dimension(self.dimension(left_degree + right_degree, i32::max_value()));
        for (left_idx, left_entry) in left.iter_nonzero() {
            for (right_idx, right_entry) in right.iter_nonzero() {
                let mut target_mono = self.index_to_monomial(left_degree, left_idx).clone();
                let source_mono = self.index_to_monomial(right_degree, right_idx);
                self.multiply_monomials(&mut target_mono,  &source_mono);
                let idx = self.monomial_to_index(&target_mono).unwrap();
                target.add_basis_element(idx, (left_entry * right_entry * coeff)%p);
            }
        }
    }

    fn multiply_polynomial_by_monomial(&self, target : &mut FpVector, coeff : u32, left_degree : i32, left : &FpVector, right_mono : &PolynomialAlgebraMonomial) {
        let p = *self.prime();
        target.extend_dimension(self.dimension(left_degree + right_mono.degree, i32::max_value()));
        for (left_idx, left_entry) in left.iter_nonzero() {
            let mut target_mono = self.index_to_monomial(left_degree, left_idx).clone();
            println!("left_mono : {}", target_mono);
            println!("right_mono : {}", right_mono);
            self.multiply_monomials(&mut target_mono,  &right_mono);
            println!("target_mono : {}", target_mono);
            let idx = self.monomial_to_index(&target_mono).unwrap();
            target.add_basis_element(idx, (left_entry * 1 * coeff)%p);
        }
    }

    fn set_monomial_degree(&self, mono : &mut PolynomialAlgebraMonomial, degree : i32) {
        mono.degree = degree;
        mono.ext.set_scratch_vector_size(self.exterior_partitions().generators_up_to_degree(mono.degree));
        mono.poly.set_scratch_vector_size(self.polynomial_partitions().generators_up_to_degree(mono.degree));        
    }
}

impl<A : PolynomialAlgebra> Algebra for A {
    fn algebra_type(&self) -> &str {
        &"polynomial"
    }

    fn prime(&self) -> ValidPrime {
        self.prime()
    }
    
    fn compute_basis(&self, degree : i32) {
        self.compute_generating_set(degree);
        for i in self.max_computed_degree() ..= degree {
            self.compute_basis_step(i);
        }
    }

    fn max_computed_degree(&self) -> i32 {
        self.polynomial_partitions().parts.len() as i32 - 1
    }

    fn dimension(&self, degree : i32, _excess : i32) -> usize {
        if degree < 0 { 
            0 
        } else {
            self.basis_table()[degree as usize].index_to_monomial.len()
        }
    }

    fn multiply_basis_elements(&self, result : &mut FpVector, coeff : u32, left_degree : i32, left_idx : usize, right_degree: i32, right_idx : usize, _excess : i32) {
        if coeff == 0 {
            return;
        }
        let mut target = self.index_to_monomial(left_degree, left_idx).clone();
        let source = self.index_to_monomial(right_degree, right_idx);
        if self.multiply_monomials(&mut target, &source).is_some() {
            let idx = self.monomial_to_index(&target).unwrap();
            result.add_basis_element(idx, coeff);
        }
    }
}