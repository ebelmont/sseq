use crate::tridegree::Tridegree;

/// The standard maps in the EHP spectral sequence.
///
/// `E`, `H`, `P` are the EHP exact-sequence triple. `Lh0` is the "left h0"
/// map ((n,s,f) → (n−1,s,f+1)); it is loaded, induced on page turns, and
/// rendered (stem view), but is deliberately NOT part of the EHP exact
/// sequence and NOT a solver constraint — so it is EXCLUDED from
/// [`MapKind::all`] (the EHP triple used by constraints/fiber/interpage) and
/// only appears via [`MapKind::all_with_lh0`] at the load / induction /
/// serialization / CSV-export sites.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MapKind {
    E,
    H,
    P,
    Lh0,
}

impl MapKind {
    /// Target tridegree of the map applied to an element at (n, s, f).
    pub fn target_degree(&self, t: Tridegree) -> Tridegree {
        match self {
            MapKind::E => Tridegree::new(t.n + 1, t.s, t.f),
            MapKind::H => Tridegree::new(2 * t.n - 1, t.s - t.n + 1, t.f - 1),
            MapKind::P => Tridegree::new((t.n - 1) / 2, t.s + (t.n - 1) / 2 - 1, t.f + 2),
            MapKind::Lh0 => Tridegree::new(t.n - 1, t.s, t.f + 1),
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
            MapKind::Lh0 => Tridegree::new(target.n + 1, target.s, target.f - 1),
        }
    }

    /// Check if the map is defined at this source tridegree.
    pub fn domain_check(&self, t: Tridegree) -> bool {
        match self {
            MapKind::E => t.n >= 2,
            MapKind::H => t.n >= 2,
            MapKind::P => t.n >= 5 && t.n % 2 == 1,
            MapKind::Lh0 => t.n >= 3,
        }
    }

    /// The EHP exact-sequence triple (E, H, P). Does NOT include `Lh0` — see
    /// the type-level note. Used by constraint generation, fiber exactness,
    /// and interpage propagation.
    pub fn all() -> [MapKind; 3] {
        [MapKind::E, MapKind::H, MapKind::P]
    }

    /// The EHP triple plus `Lh0`. Used only where lh0 should participate:
    /// CSV/binary load, induced-map computation, serialization, CSV export.
    pub fn all_with_lh0() -> [MapKind; 4] {
        [MapKind::E, MapKind::H, MapKind::P, MapKind::Lh0]
    }

    pub fn name(&self) -> &'static str {
        match self {
            MapKind::E => "E",
            MapKind::H => "H",
            MapKind::P => "P",
            MapKind::Lh0 => "lh0",
        }
    }
}

impl std::fmt::Display for MapKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
