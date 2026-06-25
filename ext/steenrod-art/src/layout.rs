/// Layout engine — assigns pixel-coordinate positions to generators.
///
/// Groups generators by degree, centers each group horizontally,
/// then computes a spine ranking for side assignment.

use crate::module::Module;

pub struct LayoutConfig {
    /// Vertical spacing per degree (pixels)
    pub row_h: f32,
    /// Horizontal spacing for same-degree generators (pixels)
    pub col_w: f32,
    /// Border margin (pixels)
    pub margin: f32,
    /// Base canvas width (pixels)
    pub canvas_base_width: f32,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            row_h: 80.0,
            col_w: 70.0,
            margin: 70.0,
            canvas_base_width: 420.0,
        }
    }
}

/// Position and rank of a single generator.
#[derive(Debug, Clone)]
pub struct GenPos {
    pub x: f32,
    pub y: f32,
    /// Position in the spine order (0..n-1)
    pub rank: usize,
}

/// Complete layout result.
pub struct LayoutResult {
    /// Positions indexed by generator index (parallel to Module.generators)
    pub positions: Vec<GenPos>,
    /// Linear spine order: list of generator indices sorted by (degree asc, x asc)
    pub spine: Vec<usize>,
    /// Canvas width in pixels
    pub width: f32,
    /// Canvas height in pixels
    pub height: f32,
}

/// Compute pixel-coordinate layout for all generators.
pub fn compute_layout(module: &Module, config: &LayoutConfig) -> LayoutResult {
    let n = module.generators.len();
    if n == 0 {
        return LayoutResult {
            positions: vec![],
            spine: vec![],
            width: config.canvas_base_width,
            height: config.margin * 2.0,
        };
    }

    // Find degree range
    let min_degree = module.generators.iter().map(|g| g.degree).min().unwrap();
    let max_degree = module.generators.iter().map(|g| g.degree).max().unwrap();

    // Group generators by degree, preserving original order within each degree
    let mut by_degree: Vec<(i32, Vec<usize>)> = Vec::new();
    {
        let mut degree_map: std::collections::BTreeMap<i32, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (i, g) in module.generators.iter().enumerate() {
            degree_map.entry(g.degree).or_default().push(i);
        }
        for (deg, indices) in degree_map {
            by_degree.push((deg, indices));
        }
    }

    // Determine max column count for canvas width
    let max_col_count = by_degree.iter().map(|(_, v)| v.len()).max().unwrap_or(1);
    let width = (config.canvas_base_width)
        .max(max_col_count as f32 * config.col_w + 2.0 * config.margin);
    let center = width / 2.0;
    let height = (max_degree - min_degree) as f32 * config.row_h + 2.0 * config.margin;

    // Assign positions
    let mut positions = vec![
        GenPos {
            x: 0.0,
            y: 0.0,
            rank: 0
        };
        n
    ];

    for &(deg, ref indices) in &by_degree {
        let k = indices.len();
        let y = config.margin + (max_degree - deg) as f32 * config.row_h;
        for (i, &gen_idx) in indices.iter().enumerate() {
            let x = center + (i as f32 - (k as f32 - 1.0) / 2.0) * config.col_w;
            positions[gen_idx].x = x;
            positions[gen_idx].y = y;
        }
    }

    // Spine ranking: sort all generators by (degree asc, x asc)
    let mut spine: Vec<usize> = (0..n).collect();
    spine.sort_by(|&a, &b| {
        let da = module.generators[a].degree;
        let db = module.generators[b].degree;
        da.cmp(&db)
            .then_with(|| positions[a].x.partial_cmp(&positions[b].x).unwrap())
    });

    // Assign ranks
    for (rank, &gen_idx) in spine.iter().enumerate() {
        positions[gen_idx].rank = rank;
    }

    LayoutResult {
        positions,
        spine,
        width,
        height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::{Edge, Generator, Module};

    fn make_c2() -> Module {
        Module {
            name: "C2".into(),
            prime: 2,
            generators: vec![
                Generator {
                    name: "x0".into(),
                    degree: 0,
                    index: 0,
                },
                Generator {
                    name: "x1".into(),
                    degree: 1,
                    index: 1,
                },
            ],
            edges: vec![Edge {
                op_name: "Sq1".into(),
                op_degree: 1,
                from_idx: 0,
                to_idx: 1,
            }],
        }
    }

    #[test]
    fn test_c2_layout() {
        let config = LayoutConfig::default();
        let result = compute_layout(&make_c2(), &config);
        assert_eq!(result.positions.len(), 2);
        // Both at center x (single gen per degree)
        assert_eq!(result.positions[0].x, result.positions[1].x);
        // Higher degree = lower y (drawn higher on canvas = lower y value)
        assert!(result.positions[1].y < result.positions[0].y);
    }

    #[test]
    fn test_same_degree_spread() {
        let m = Module {
            name: "test".into(),
            prime: 2,
            generators: vec![
                Generator {
                    name: "a".into(),
                    degree: 0,
                    index: 0,
                },
                Generator {
                    name: "b".into(),
                    degree: 0,
                    index: 1,
                },
            ],
            edges: vec![],
        };
        let config = LayoutConfig::default();
        let result = compute_layout(&m, &config);
        // Two gens at same degree should be spread apart by COL_W
        let dx = (result.positions[1].x - result.positions[0].x).abs();
        assert!((dx - config.col_w).abs() < 0.01);
    }

    #[test]
    fn test_empty_module() {
        let m = Module {
            name: "empty".into(),
            prime: 2,
            generators: vec![],
            edges: vec![],
        };
        let config = LayoutConfig::default();
        let result = compute_layout(&m, &config);
        assert!(result.positions.is_empty());
        assert!(result.spine.is_empty());
    }

    #[test]
    fn test_spine_ranking() {
        let m = Module {
            name: "test".into(),
            prime: 2,
            generators: vec![
                Generator {
                    name: "x0".into(),
                    degree: 0,
                    index: 0,
                },
                Generator {
                    name: "x1".into(),
                    degree: 1,
                    index: 1,
                },
                Generator {
                    name: "x2".into(),
                    degree: 2,
                    index: 2,
                },
            ],
            edges: vec![],
        };
        let config = LayoutConfig::default();
        let result = compute_layout(&m, &config);
        // Spine should be [0, 1, 2] (degree ascending)
        assert_eq!(result.spine, vec![0, 1, 2]);
        // Ranks should be 0, 1, 2
        assert_eq!(result.positions[0].rank, 0);
        assert_eq!(result.positions[1].rank, 1);
        assert_eq!(result.positions[2].rank, 2);
    }
}
