use crate::fp_vector::{FpVector, FpVectorT};
use crate::algebra::{Algebra, AlgebraAny, AdemAlgebra, adem_algebra::AdemBasisElement, MilnorAlgebra, milnor_algebra::MilnorBasisElement};
use crate::module::{Module, ZeroModule};
use crate::combinatorics::{binomial, multinomial};

use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

/// This is RP_min^max. THe cohomology is the subquotient of F_2[x, x^{-1}] given by elements of degree
/// between min and max (inclusive)
pub struct RealProjectiveSpace {
    algebra : Arc<AlgebraAny>,
    min : i32,
    max : Option<i32>, // If None,  then RP^oo
}

impl PartialEq for RealProjectiveSpace {
    fn eq(&self, other : &Self) -> bool {
        self.min == other.min &&
            self.max == other.max
    }
}

impl Eq for RealProjectiveSpace {}

impl Module for RealProjectiveSpace {
    fn name(&self) -> &str {
        &"real projective space"
    }

    fn algebra(&self) -> Arc<AlgebraAny> {
        Arc::clone(&self.algebra)
    }

    fn min_degree(&self) -> i32 {
        self.min
    }

    fn dimension(&self, degree : i32) -> usize {
        if degree < self.min {
            return 0;
        }
        if let Some(m) = self.max {
            if degree > m {
                return 0;
            }
        }
        1
    }

    fn basis_element_to_string(&self, degree : i32, _idx : usize) -> String {
        // It is an error to call the function if self.dimension(degree) == 0
        format!("x^{{{}}}", degree)
    }

    fn act_on_basis(&self, result : &mut FpVector, coeff : u32, op_degree : i32, op_index : usize, mod_degree : i32, mod_index : usize){
        assert!(op_index < self.algebra().dimension(op_degree, mod_degree));
        assert!(mod_index < self.dimension(mod_degree));

        let output_degree = mod_degree + op_degree;

        if op_degree == 0 || coeff == 0 || self.dimension(output_degree) == 0 {
            return;
        }

        if match &*self.algebra {
            AlgebraAny::AdemAlgebra(a) => coef_adem(a, op_degree, op_index, mod_degree),
            AlgebraAny::MilnorAlgebra(a) => coef_milnor(a, op_degree, op_index, mod_degree),
            AlgebraAny::Field(_) => true, // For a field, the only operation is the identity.
        } {
            result.add_basis_element(0, 1);
        }
    }
}

// Compute the coefficient of the operation on x^j.
fn coef_adem(algebra : &AdemAlgebra, op_deg : i32, op_idx : usize, mut j : i32) -> bool {
    let elt : &AdemBasisElement = algebra.basis_element_from_index(op_deg, op_idx);
    // Apply Sq^i to x^j and see if it is zero
    for i in elt.ps.iter().rev() {
        let c = if j >= 0 {
            binomial(2, j, *i as i32)
        } else {
            binomial(2, -j + (*i as i32) - 1, *i as i32)
        };
        if c == 0 {
            return false;
        }
        // Somehow j += 1 produces the same answer...
        j += *i as i32;
    }
    true
}

fn coef_milnor(algebra : &MilnorAlgebra, op_deg : i32, op_idx : usize, mut mod_degree : i32) -> bool {
    if mod_degree == 0 {
        return false;
    }

    let elt : &MilnorBasisElement = algebra.basis_element_from_index(op_deg, op_idx);

    let sum : u32 = elt.p_part.iter().sum();
    if mod_degree < 0 {
        mod_degree = sum as i32 - mod_degree - 1;
    } else if mod_degree < sum as i32 {
        return false;
    }

    let mod_degree = mod_degree as u32;

    let mut list = Vec::with_capacity(elt.p_part.len() + 1);
    list.push(mod_degree - sum);
    list.extend(elt.p_part.iter());

    multinomial(2, &list) == 1
}

impl ZeroModule for RealProjectiveSpace {
    fn zero_module(algebra : Arc<AlgebraAny>, min_degree : i32) -> Self {
        Self::new(algebra, min_degree, Some(min_degree - 1))
    }
}

#[derive(Deserialize, Debug)]
struct RPSpec {
    min : i32,
    max : Option<i32>,
}

impl RealProjectiveSpace {
    pub fn new(algebra : Arc<AlgebraAny>, min : i32, max : Option<i32>) -> Self {
        assert_eq!(algebra.prime(), 2);
        if let Some(max) = max {
            assert!(max >= min);
        }
        Self { algebra, min, max }
    }

    pub fn from_json(algebra : Arc<AlgebraAny>, json : &mut Value) -> Result<Self, Box<dyn std::error::Error>> {
        let spec : RPSpec = serde_json::from_value(json.clone())?;
        Ok(Self {
            algebra,
            min : spec.min,
            max : spec.max,
        })
    }
}
