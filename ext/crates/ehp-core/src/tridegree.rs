use serde::{Deserialize, Serialize};

/// A tridegree (n, s, f) in the unstable Adams spectral sequence.
///
/// - `n`: sphere of origin
/// - `s`: stem — `U_r^{n,s,f}` detects elements of `π_{n+s}(S^n)`
/// - `f`: Adams filtration
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct Tridegree {
    pub n: i32,
    pub s: i32,
    pub f: i32,
}

impl Tridegree {
    pub fn new(n: i32, s: i32, f: i32) -> Self {
        Tridegree { n, s, f }
    }

    /// The total degree s + f.
    pub fn total(&self) -> i32 {
        self.s + self.f
    }

    /// Target tridegree of a d_r differential from this degree.
    pub fn diff_target(&self, r: i32) -> Tridegree {
        Tridegree::new(self.n, self.s - 1, self.f + r)
    }

    /// Source tridegree of a d_r differential landing in this degree.
    pub fn diff_source(&self, r: i32) -> Tridegree {
        Tridegree::new(self.n, self.s + 1, self.f - r)
    }

    /// Product degree: multiplying elements in (n, s1, f1) and (n+s1, s2, f2)
    /// gives an element in (n, s1+s2, f1+f2).
    pub fn product_degree(&self, other_s: i32, other_f: i32) -> Tridegree {
        Tridegree::new(self.n, self.s + other_s, self.f + other_f)
    }

    /// The "stable" n value: min(n, s+2).
    pub fn stable_n(&self) -> i32 {
        self.n.min(self.s + 2)
    }
}

impl std::fmt::Display for Tridegree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.n, self.s, self.f)
    }
}

impl From<(i32, i32, i32)> for Tridegree {
    fn from((n, s, f): (i32, i32, i32)) -> Self {
        Tridegree::new(n, s, f)
    }
}

impl From<Tridegree> for (i32, i32, i32) {
    fn from(t: Tridegree) -> Self {
        (t.n, t.s, t.f)
    }
}
