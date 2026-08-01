//! Degree bookkeeping for the EHP fiber-sequence viewer ("fiber" charts).
//!
//! Convention (fixing the README note): `U_r^{n,s,f}` is the (stem `s`, Adams
//! filtration `f`) part of the E_r page of the unstable Adams spectral
//! sequence for `S^n`; it detects elements of `π_{n+s}(S^n)`. In particular
//! `s` is the STEM, not the total or filtration degree, and the differential
//! is `d_r : U_r^{n,s,f} -> U_r^{n,s-1,f+r}`.
//!
//! For a base sphere N ≥ 2 the chart displays the triple (S^N, S^{N+1},
//! S^{2N+1}) from the fiber sequence S^N -> ΩS^{N+1} -> ΩS^{2N+1}, chained as
//!
//! ```text
//! U_r^{N,σ,f} --E--> U_r^{N+1,σ,f} --H--> U_r^{2N+1,σ-N,f-1} --P--> U_r^{N,σ-1,f+1}
//! ```
//!
//! Master column σ collects S^N and S^{N+1} classes with stem σ together with
//! S^{2N+1} classes with stem σ − N; P is the only map that leaves a master
//! column (σ drops by exactly 1, f rises by 2). Everything here is UI-free so
//! odd-primary EHP can later swap the degree rules in one place.

pub use crate::map::MapKind;
use crate::tridegree::Tridegree;

/// The three sub-column spheres of the display triple, left to right.
pub fn triple_spheres(base_n: i32) -> [i32; 3] {
    [base_n, base_n + 1, 2 * base_n + 1]
}

/// The map drawn from classes on sphere `n` within the triple:
/// E from S^N, H from S^{N+1}, P from S^{2N+1}.
pub fn triple_map(n: i32, base_n: i32) -> Option<MapKind> {
    let [a, b, c] = triple_spheres(base_n);
    if n == a {
        Some(MapKind::E)
    } else if n == b {
        Some(MapKind::H)
    } else if n == c {
        Some(MapKind::P)
    } else {
        None
    }
}

/// The next map in the fiber-sequence chain: E -> H -> P -> E.
pub fn next_map(kind: MapKind) -> MapKind {
    match kind {
        MapKind::E => MapKind::H,
        MapKind::H => MapKind::P,
        MapKind::P => MapKind::E,
        // lh0 is not part of the EHP fiber chain (fiber code iterates
        // MapKind::all(), which excludes it).
        MapKind::Lh0 => unreachable!("lh0 is not part of the EHP fiber sequence"),
    }
}

/// Master column σ of a class in the triple: S^N and S^{N+1} classes sit at
/// their own stem, S^{2N+1} classes at stem + N. `None` if `t.n` is not one
/// of the triple's spheres.
pub fn master_column(t: Tridegree, base_n: i32) -> Option<i32> {
    match sub_column(t, base_n)? {
        2 => Some(t.s + base_n),
        _ => Some(t.s),
    }
}

/// Sub-column index (0 = S^N, 1 = S^{N+1}, 2 = S^{2N+1}) of a class in the
/// triple, or `None` if `t.n` is not one of the triple's spheres.
pub fn sub_column(t: Tridegree, base_n: i32) -> Option<usize> {
    triple_spheres(base_n).iter().position(|&n| n == t.n)
}

/// Source tridegree of the map of the given kind drawn from master column σ
/// at filtration `f` (chart coordinates -> intrinsic tridegree).
pub fn chain_source(kind: MapKind, base_n: i32, sigma: i32, f: i32) -> Tridegree {
    match kind {
        MapKind::E => Tridegree::new(base_n, sigma, f),
        MapKind::H => Tridegree::new(base_n + 1, sigma, f),
        MapKind::P => Tridegree::new(2 * base_n + 1, sigma - base_n, f),
        MapKind::Lh0 => unreachable!("lh0 is not part of the EHP fiber sequence"),
    }
}

/// One exactness comparison of the diagnostics panel: at `middle`, the rank
/// of the incoming map is compared with the kernel dimension of the outgoing
/// map (E2 is exact; E∞ mismatches flag candidate hidden maps).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactnessSquare {
    pub incoming: MapKind,
    pub outgoing: MapKind,
    pub middle: Tridegree,
}

/// The three exactness positions anchored at the chain start `U^{N,σ,f}`:
/// rank E vs dim ker H at `U^{N+1,σ,f}`, rank H vs dim ker P at
/// `U^{2N+1,σ-N,f-1}`, rank P vs dim ker E at `U^{N,σ-1,f+1}`.
pub fn exactness_squares(base_n: i32, sigma: i32, f: i32) -> [ExactnessSquare; 3] {
    let mut deg = chain_source(MapKind::E, base_n, sigma, f);
    let mut kind = MapKind::E;
    std::array::from_fn(|_| {
        deg = kind.target_degree(deg);
        let square = ExactnessSquare {
            incoming: kind,
            outgoing: next_map(kind),
            middle: deg,
        };
        kind = next_map(kind);
        square
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page::{MapMatrixRef, SATPage};

    #[test]
    fn composite_degree_drops_one_master_column() {
        for base_n in 2..=10 {
            for s in 0..=12 {
                for f in 0..=8 {
                    let t0 = Tridegree::new(base_n, s, f);
                    let t1 = MapKind::E.target_degree(t0);
                    let t2 = MapKind::H.target_degree(t1);
                    let t3 = MapKind::P.target_degree(t2);
                    assert_eq!(t1, Tridegree::new(base_n + 1, s, f));
                    assert_eq!(t2, Tridegree::new(2 * base_n + 1, s - base_n, f - 1));
                    assert_eq!(t3, Tridegree::new(base_n, s - 1, f + 1));
                    // σ is constant along E and H, drops by 1 only at P.
                    assert_eq!(master_column(t0, base_n), Some(s));
                    assert_eq!(master_column(t1, base_n), Some(s));
                    assert_eq!(master_column(t2, base_n), Some(s));
                    assert_eq!(master_column(t3, base_n), Some(s - 1));
                    assert_eq!(sub_column(t0, base_n), Some(0));
                    assert_eq!(sub_column(t1, base_n), Some(1));
                    assert_eq!(sub_column(t2, base_n), Some(2));
                    assert_eq!(sub_column(t3, base_n), Some(0));
                }
            }
        }
    }

    #[test]
    fn maps_commute_with_differentials() {
        for kind in MapKind::all() {
            for n in 2..=15 {
                if !kind.domain_check(Tridegree::new(n, 0, 0)) {
                    continue;
                }
                for s in 0..=10 {
                    for f in 0..=6 {
                        let t = Tridegree::new(n, s, f);
                        for r in 2..=8 {
                            assert_eq!(
                                kind.target_degree(t.diff_target(r)),
                                kind.target_degree(t).diff_target(r),
                                "{kind} square fails at {t}, r={r}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn source_degree_inverts_target_degree() {
        for kind in MapKind::all() {
            for n in 2..=21 {
                for s in -3..=10 {
                    for f in 0..=6 {
                        let t = Tridegree::new(n, s, f);
                        if !kind.domain_check(t) {
                            continue;
                        }
                        assert_eq!(kind.source_degree(kind.target_degree(t)), t);
                    }
                }
            }
        }
    }

    #[test]
    fn exactness_squares_match_contract() {
        for base_n in 2..=8 {
            for sigma in 0..=10 {
                for f in 0..=5 {
                    let squares = exactness_squares(base_n, sigma, f);
                    assert_eq!(
                        squares[0],
                        ExactnessSquare {
                            incoming: MapKind::E,
                            outgoing: MapKind::H,
                            middle: Tridegree::new(base_n + 1, sigma, f),
                        }
                    );
                    assert_eq!(
                        squares[1],
                        ExactnessSquare {
                            incoming: MapKind::H,
                            outgoing: MapKind::P,
                            middle: Tridegree::new(2 * base_n + 1, sigma - base_n, f - 1),
                        }
                    );
                    assert_eq!(
                        squares[2],
                        ExactnessSquare {
                            incoming: MapKind::P,
                            outgoing: MapKind::E,
                            middle: Tridegree::new(base_n, sigma - 1, f + 1),
                        }
                    );
                    // Each middle is the target of its incoming map applied to
                    // the previous chain degree.
                    for sq in squares {
                        let src = sq.incoming.source_degree(sq.middle);
                        assert_eq!(sq.incoming.target_degree(src), sq.middle);
                        assert_eq!(triple_map(src.n, base_n), Some(sq.incoming));
                        assert_eq!(triple_map(sq.middle.n, base_n), Some(sq.outgoing));
                    }
                }
            }
        }
    }

    /// Accept 3-part `n_s_f` or 4-part `n_s_f_i` forms of the same class.
    fn name_matches(name: &str, base: &str) -> bool {
        name == base
            || name
                .strip_prefix(base)
                .and_then(|rest| rest.strip_prefix('_'))
                .is_some_and(|idx| !idx.is_empty() && idx.bytes().all(|b| b.is_ascii_digit()))
    }

    fn has_h_row(text: &str, element: &str, image: &str) -> bool {
        text.lines().any(|line| {
            let Some((e, i)) = line.trim().split_once(',') else {
                return false;
            };
            name_matches(e.trim_matches('"'), element)
                && i.trim_matches('"')
                    .split(" + ")
                    .any(|part| name_matches(part.trim(), image))
        })
    }

    #[test]
    fn hopf_invariants_in_e2_data() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/E2/E2_H.csv");
        let Ok(text) = std::fs::read_to_string(path) else {
            eprintln!("skipping hopf_invariants_in_e2_data: {path} not found");
            return;
        };
        // η, ν, σ have Hopf invariant one: H hits the fundamental class.
        assert!(has_h_row(&text, "2_1_1", "3_0_0"), "H(eta) missing");
        assert!(has_h_row(&text, "4_3_1", "7_0_0"), "H(nu) missing");
        assert!(has_h_row(&text, "8_7_1", "15_0_0"), "H(sigma) missing");
    }

    #[test]
    fn whitehead_square_degree() {
        // Degree arithmetic only: vendored E2_P.csv has no rows for the
        // fundamental classes of S^5 or the Hopf spheres (data gap), so this
        // must not be a data test. P(ι_{2n+1}) lands at (n-1, 2) on S^n; for
        // n = 2 that cell is h0h1, i.e. the Whitehead square [ι_2, ι_2] = 2η.
        for n in 2..=30 {
            assert_eq!(
                MapKind::P.target_degree(Tridegree::new(2 * n + 1, 0, 0)),
                Tridegree::new(n, n - 1, 2)
            );
        }
    }

    #[test]
    fn freudenthal_stable_edge() {
        let mut page = SATPage::new(2);
        // n > s+1 with matching src/tgt dims: E defaults to the identity.
        let src = Tridegree::new(8, 3, 2);
        page.dimension.insert(src, 2);
        page.dimension.insert(MapKind::E.target_degree(src), 2);
        match page.map_matrix_ref(MapKind::E, src) {
            MapMatrixRef::Identity(n) => assert_eq!(n, 2),
            _ => panic!("expected identity E default in the stable range"),
        }
        // H vanishes on the nose for s <= n-2: its target stem is negative,
        // so the target tridegree carries no classes.
        for s in 0..=6 {
            let tgt = MapKind::H.target_degree(Tridegree::new(8, s, 2));
            assert!(tgt.s < 0);
            assert_eq!(page.dim_at(tgt), 0);
        }
    }
}
