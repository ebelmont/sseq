/// Built-in example modules.

use steenrod_art::module::{Edge, Generator, Module};

/// The Joker: 5 generators, degrees 0-4, self-dual.
/// Sq1: x0->x1, x3->x4
/// Sq2: x0->x2, x1->x3, x2->x4
pub fn joker() -> Module {
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

/// sum_test: demonstrates sum RHS producing two arcs from same source.
pub fn sum_test() -> Module {
    Module {
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
    }
}

/// higher_sq: demonstrates Sq1/Sq2/Sq3/Sq4 mixed operations.
pub fn higher_sq() -> Module {
    Module {
        name: "higher_sq".into(),
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
            Edge { op_name: "Sq1".into(), op_degree: 1, from_idx: 0, to_idx: 1 },
            Edge { op_name: "Sq2".into(), op_degree: 2, from_idx: 0, to_idx: 2 },
            Edge { op_name: "Sq3".into(), op_degree: 3, from_idx: 0, to_idx: 3 },
            Edge { op_name: "Sq4".into(), op_degree: 4, from_idx: 0, to_idx: 4 },
            Edge { op_name: "Sq1".into(), op_degree: 1, from_idx: 1, to_idx: 2 },
            Edge { op_name: "Sq2".into(), op_degree: 2, from_idx: 1, to_idx: 3 },
            Edge { op_name: "Sq1".into(), op_degree: 1, from_idx: 4, to_idx: 5 },
        ],
    }
}
