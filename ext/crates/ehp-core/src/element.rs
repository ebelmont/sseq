use fp::vector::FpVector;

use crate::gf2::*;
use crate::tridegree::Tridegree;

/// An element in the spectral sequence at a given tridegree,
/// represented by its coefficient vector over GF(2).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Element {
    pub degree: Tridegree,
    pub vec: FpVector,
}

impl Element {
    pub fn new(degree: Tridegree, vec: FpVector) -> Self {
        Element { degree, vec }
    }

    /// Create a basis element e_i at the given tridegree with given dimension.
    pub fn basis(degree: Tridegree, dim: usize, i: usize) -> Self {
        Element {
            degree,
            vec: vec_basis(dim, i),
        }
    }

    /// Create the zero element at the given tridegree with given dimension.
    pub fn zero(degree: Tridegree, dim: usize) -> Self {
        Element {
            degree,
            vec: vec_zero(dim),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.vec.is_zero()
    }

    /// GF(2) addition of elements in the same tridegree.
    pub fn add(&self, other: &Element) -> Element {
        debug_assert_eq!(self.degree, other.degree);
        Element {
            degree: self.degree,
            vec: vec_xor(&self.vec, &other.vec),
        }
    }

    /// Decompose into basis elements.
    pub fn decompose(&self) -> Vec<Element> {
        let dim = self.vec.len();
        vec_support(&self.vec)
            .map(|i| Element::basis(self.degree, dim, i))
            .collect()
    }

    /// Format as the Python-style string: "n_s_f_i" or "n_s_f_i + n_s_f_j + ..."
    pub fn to_string_repr(&self) -> String {
        if self.is_zero() {
            return "0".to_string();
        }
        let parts: Vec<String> = vec_support(&self.vec)
            .map(|i| {
                if self.vec.len() == 1 {
                    format!("{}_{}", self.degree.n, self.degree.s)
                } else {
                    format!("{}_{}_{}", self.degree.n, self.degree.s, i)
                }
            })
            .collect();
        parts.join(" + ")
    }

    /// String key for Python compatibility: "n_s_f" or "n_s_f_i"
    pub fn to_key(&self) -> String {
        let bits: Vec<usize> = vec_support(&self.vec).collect();
        if bits.len() == 1 && self.vec.len() == 1 {
            format!("{}_{}_{}", self.degree.n, self.degree.s, self.degree.f)
        } else if bits.len() == 1 {
            format!(
                "{}_{}_{}_{}", self.degree.n, self.degree.s, self.degree.f, bits[0]
            )
        } else {
            let parts: Vec<String> = bits
                .iter()
                .map(|&i| {
                    format!(
                        "{}_{}_{}_{}", self.degree.n, self.degree.s, self.degree.f, i
                    )
                })
                .collect();
            parts.join(" + ")
        }
    }
}

impl std::fmt::Display for Element {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        let parts: Vec<String> = vec_support(&self.vec)
            .map(|i| {
                if self.vec.len() == 1 {
                    format!("{}_{}_{}", self.degree.n, self.degree.s, self.degree.f)
                } else {
                    format!("{}_{}_{}_{}", self.degree.n, self.degree.s, self.degree.f, i)
                }
            })
            .collect();
        write!(f, "{}", parts.join(" + "))
    }
}

impl std::ops::Add for &Element {
    type Output = Element;
    fn add(self, rhs: &Element) -> Element {
        self.add(rhs)
    }
}
