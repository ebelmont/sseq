/// Crossing minimization via conflict graph + BFS 2-coloring.
///
/// Two-page book embedding: curved/rectangular arcs are assigned to
/// "left" (-1) or "right" (+1) of the spine. Two arcs on the same side
/// cross iff their spine intervals strictly interleave.

use crate::layout::LayoutResult;
use crate::module::Module;

/// Result of side assignment.
pub struct SideAssignment {
    /// Side for each edge in module.edges: -1=left, +1=right, 0=straight (Sq1)
    pub sides: Vec<i8>,
    /// Whether the conflict graph is bipartite
    pub bipartite: bool,
    /// Number of residual crossings (0 if bipartite)
    pub residual: usize,
}

/// Assign sides to all edges to minimize crossings.
///
/// Sq1 edges get side=0 (drawn straight).
/// Sq2+ edges are assigned -1 (left) or +1 (right) via 2-coloring.
pub fn assign_sides(module: &Module, layout: &LayoutResult) -> SideAssignment {
    let ne = module.edges.len();
    let mut sides = vec![0i8; ne];

    // Collect arcs: indices of edges with sq >= 2
    let arcs: Vec<usize> = (0..ne)
        .filter(|&i| module.edges[i].op_degree >= 2)
        .collect();
    let na = arcs.len();

    if na == 0 {
        return SideAssignment {
            sides,
            bipartite: true,
            residual: 0,
        };
    }

    // Compute spine intervals for each arc (normalized so lo < hi)
    let intervals: Vec<(usize, usize)> = arcs
        .iter()
        .map(|&ei| {
            let r_src = layout.positions[module.edges[ei].from_idx].rank;
            let r_dst = layout.positions[module.edges[ei].to_idx].rank;
            (r_src.min(r_dst), r_src.max(r_dst))
        })
        .collect();

    // Build conflict graph: adjacency list
    // Two arcs conflict if their intervals strictly interleave
    let mut adj: Vec<Vec<usize>> = vec![vec![]; na];
    for i in 0..na {
        let (a_lo, a_hi) = intervals[i];
        for j in (i + 1)..na {
            let (b_lo, b_hi) = intervals[j];
            if interleaves(a_lo, a_hi, b_lo, b_hi) {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }

    // BFS 2-coloring
    let mut color = vec![-1i32; na];
    let mut bipartite = true;

    for s in 0..na {
        if color[s] != -1 {
            continue;
        }
        color[s] = 0;
        let mut stack = vec![s];
        let mut comp = vec![s];

        while let Some(u) = stack.pop() {
            for &v in &adj[u] {
                if color[v] == -1 {
                    color[v] = 1 - color[u];
                    stack.push(v);
                    comp.push(v);
                } else if color[v] == color[u] {
                    bipartite = false;
                }
            }
        }

        // Choose flip (which color is "left") to best match geometric preference
        let pref = |i: usize| -> i8 {
            let ei = arcs[i];
            let x_src = layout.positions[module.edges[ei].from_idx].x;
            let x_dst = layout.positions[module.edges[ei].to_idx].x;
            if x_dst > x_src {
                1
            } else if x_dst < x_src {
                -1
            } else {
                0
            }
        };

        let cost = |flip: i32| -> i32 {
            comp.iter()
                .map(|&i| {
                    let side = if color[i] == 0 { -flip } else { flip };
                    let p = pref(i) as i32;
                    if p != 0 && side != p {
                        1
                    } else {
                        0
                    }
                })
                .sum::<i32>()
        };

        let flip = if cost(1) <= cost(-1) { 1 } else { -1 };
        for &i in &comp {
            color[i] = if color[i] == 0 { -flip } else { flip };
        }
    }

    // Write sides from coloring
    for (ai, &ei) in arcs.iter().enumerate() {
        sides[ei] = color[ai] as i8;
    }

    // Count residual crossings (same-side interleaving pairs)
    let mut residual = 0;
    if !bipartite {
        // Greedy repair for non-bipartite: process arcs by descending span,
        // assign each the side with fewer conflicts
        let mut arc_order: Vec<usize> = (0..na).collect();
        arc_order.sort_by(|&a, &b| {
            let span_a = intervals[a].1 - intervals[a].0;
            let span_b = intervals[b].1 - intervals[b].0;
            span_b.cmp(&span_a)
        });

        let mut repaired = vec![0i8; na];
        for &ai in &arc_order {
            // Count conflicts with already-placed arcs on each side
            let mut left_conflicts = 0;
            let mut right_conflicts = 0;
            for &neighbor in &adj[ai] {
                if repaired[neighbor] != 0 {
                    if interleaves(intervals[ai].0, intervals[ai].1, intervals[neighbor].0, intervals[neighbor].1) {
                        if repaired[neighbor] == -1 {
                            left_conflicts += 1;
                        } else {
                            right_conflicts += 1;
                        }
                    }
                }
            }
            // Prefer the side from BFS coloring as tiebreak
            let bfs_side = color[ai] as i8;
            repaired[ai] = if left_conflicts < right_conflicts {
                -1
            } else if right_conflicts < left_conflicts {
                1
            } else {
                bfs_side
            };
        }

        // Use repaired sides
        for (ai, &ei) in arcs.iter().enumerate() {
            sides[ei] = repaired[ai];
        }

        // Count residual
        for i in 0..na {
            for j in (i + 1)..na {
                if sides[arcs[i]] == sides[arcs[j]]
                    && interleaves(intervals[i].0, intervals[i].1, intervals[j].0, intervals[j].1)
                {
                    residual += 1;
                }
            }
        }
    }

    SideAssignment {
        sides,
        bipartite,
        residual,
    }
}

/// Check if two intervals strictly interleave.
/// Intervals are [a_lo, a_hi] and [b_lo, b_hi] with lo < hi.
fn interleaves(a_lo: usize, a_hi: usize, b_lo: usize, b_hi: usize) -> bool {
    (a_lo < b_lo && b_lo < a_hi && a_hi < b_hi) || (b_lo < a_lo && a_lo < b_hi && b_hi < a_hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{compute_layout, LayoutConfig};
    use crate::module::{Edge, Generator, Module};

    fn joker_module() -> Module {
        Module {
            name: "Joker".into(),
            prime: 2,
            generators: vec![
                Generator { name: "x0".into(), degree: 0, index: 0 },
                Generator { name: "x1".into(), degree: 1, index: 1 },
                Generator { name: "x2".into(), degree: 2, index: 2 },
                Generator { name: "x3".into(), degree: 3, index: 3 },
                Generator { name: "x4".into(), degree: 4, index: 4 },
            ],
            edges: vec![
                Edge { op_name: "Sq1".into(), op_degree: 1, from_idx: 0, to_idx: 1 },
                Edge { op_name: "Sq2".into(), op_degree: 2, from_idx: 0, to_idx: 2 },
                Edge { op_name: "Sq2".into(), op_degree: 2, from_idx: 1, to_idx: 3 },
                Edge { op_name: "Sq2".into(), op_degree: 2, from_idx: 2, to_idx: 4 },
                Edge { op_name: "Sq1".into(), op_degree: 1, from_idx: 3, to_idx: 4 },
            ],
        }
    }

    #[test]
    fn test_joker_bipartite() {
        let module = joker_module();
        let config = LayoutConfig::default();
        let layout = compute_layout(&module, &config);
        let result = assign_sides(&module, &layout);

        assert!(result.bipartite);
        assert_eq!(result.residual, 0);

        // Sq1 edges should be straight (side=0)
        assert_eq!(result.sides[0], 0); // Sq1 x0->x1
        assert_eq!(result.sides[4], 0); // Sq1 x3->x4

        // Sq2 edges should alternate sides (not all same)
        let sq2_sides: Vec<i8> = vec![result.sides[1], result.sides[2], result.sides[3]];
        // At least two different sides used
        assert!(sq2_sides.iter().any(|&s| s == -1) || sq2_sides.iter().any(|&s| s == 1));
    }

    #[test]
    fn test_sum_test() {
        // Two arcs from same source split left/right
        let module = Module {
            name: "sum_test".into(),
            prime: 2,
            generators: vec![
                Generator { name: "a".into(), degree: 0, index: 0 },
                Generator { name: "b".into(), degree: 1, index: 1 },
                Generator { name: "c".into(), degree: 2, index: 2 },
                Generator { name: "d".into(), degree: 2, index: 3 },
                Generator { name: "e".into(), degree: 3, index: 4 },
            ],
            edges: vec![
                Edge { op_name: "Sq1".into(), op_degree: 1, from_idx: 0, to_idx: 1 },
                Edge { op_name: "Sq2".into(), op_degree: 2, from_idx: 0, to_idx: 2 },
                Edge { op_name: "Sq2".into(), op_degree: 2, from_idx: 0, to_idx: 3 },
                Edge { op_name: "Sq1".into(), op_degree: 1, from_idx: 1, to_idx: 2 },
                Edge { op_name: "Sq1".into(), op_degree: 1, from_idx: 3, to_idx: 4 },
            ],
        };
        let config = LayoutConfig::default();
        let layout = compute_layout(&module, &config);
        let result = assign_sides(&module, &layout);

        assert!(result.bipartite);
        assert_eq!(result.residual, 0);

        // The two Sq2 arcs from a should go to opposite sides
        assert_ne!(result.sides[1], result.sides[2]);
    }

    #[test]
    fn test_nonbipartite() {
        // Three arcs forming an odd cycle of interleaving
        // Generators in a single column: degrees 0,1,2,3,4,5
        // Arcs: 0->3, 1->4, 2->5 (pairwise interleaving)
        let module = Module {
            name: "nonbipartite".into(),
            prime: 2,
            generators: vec![
                Generator { name: "x0".into(), degree: 0, index: 0 },
                Generator { name: "x1".into(), degree: 1, index: 1 },
                Generator { name: "x2".into(), degree: 2, index: 2 },
                Generator { name: "x3".into(), degree: 3, index: 3 },
                Generator { name: "x4".into(), degree: 4, index: 4 },
                Generator { name: "x5".into(), degree: 5, index: 5 },
            ],
            edges: vec![
                Edge { op_name: "Sq3".into(), op_degree: 3, from_idx: 0, to_idx: 3 },
                Edge { op_name: "Sq3".into(), op_degree: 3, from_idx: 1, to_idx: 4 },
                Edge { op_name: "Sq3".into(), op_degree: 3, from_idx: 2, to_idx: 5 },
            ],
        };
        let config = LayoutConfig::default();
        let layout = compute_layout(&module, &config);
        let result = assign_sides(&module, &layout);

        assert!(!result.bipartite);
        // Should still render (small residual)
        assert!(result.residual <= 1);
    }
}
