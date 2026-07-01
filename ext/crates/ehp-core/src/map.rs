use crate::tridegree::Tridegree;

/// The three standard maps E, H, P in the EHP spectral sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MapKind {
    E,
    H,
    P,
}

impl MapKind {
    /// Target tridegree of the map applied to an element at (n, s, f).
    pub fn target_degree(&self, t: Tridegree) -> Tridegree {
        match self {
            MapKind::E => Tridegree::new(t.n + 1, t.s, t.f),
            MapKind::H => Tridegree::new(2 * t.n - 1, t.s - t.n + 1, t.f - 1),
            MapKind::P => Tridegree::new((t.n - 1) / 2, t.s + (t.n - 1) / 2 - 1, t.f + 2),
        }
    }

    /// Source tridegree given a target tridegree (inverse of target_degree).
    pub fn source_degree(&self, target: Tridegree) -> Tridegree {
        match self {
            MapKind::E => Tridegree::new(target.n - 1, target.s, target.f),
            MapKind::H => {
                let n = (target.n + 1) / 2;
                let s = target.s + n - 1;
                let f = target.f + 1;
                Tridegree::new(n, s, f)
            }
            MapKind::P => {
                let n = 2 * target.n + 1;
                let s = target.s - target.n + 1;
                let f = target.f - 2;
                Tridegree::new(n, s, f)
            }
        }
    }

    /// Check if the map is defined at this source tridegree.
    pub fn domain_check(&self, t: Tridegree) -> bool {
        match self {
            MapKind::E => t.n >= 2,
            MapKind::H => t.n >= 2,
            MapKind::P => t.n >= 5 && t.n % 2 == 1,
        }
    }

    /// All three map kinds.
    pub fn all() -> [MapKind; 3] {
        [MapKind::E, MapKind::H, MapKind::P]
    }

    pub fn name(&self) -> &'static str {
        match self {
            MapKind::E => "E",
            MapKind::H => "H",
            MapKind::P => "P",
        }
    }
}

impl std::fmt::Display for MapKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
