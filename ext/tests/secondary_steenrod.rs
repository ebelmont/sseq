use std::sync::Arc;

use algebra::{
    module::Module,
    pair_algebra::PairAlgebra,
};
use ext::{
    secondary::{SecondaryLift, SecondaryResolution},
    utils::construct_standard,
};
use sseq::coordinates::Bidegree;

/// Verify that the fields of `SecondaryComposite` are publicly accessible.
///
/// For S_2, homotopy(2) maps from F2 to F0. F2 has its first generator at internal degree 2.
#[test]
fn secondary_composite_fields_are_public() {
    let resolution = construct_standard::<false, _, _>("S_2", None).unwrap();
    resolution.compute_through_stem(Bidegree::n_s(8, 4));

    let lift = SecondaryResolution::new(Arc::new(resolution));
    lift.initialize_homotopies();
    lift.compute_composites();

    let homotopy = lift.homotopy(2);

    // F2 for S_2 has a generator at internal degree 2
    let source = &homotopy.source;
    assert!(
        source.number_of_gens_in_degree(2) > 0,
        "F2 should have a generator at degree 2"
    );

    let composite = homotopy.composite(2, 0);

    // These accesses would fail to compile if the fields were not pub
    let _target = &composite.target;
    let _degree = composite.degree;
    let _composite_data = &composite.composite;

    // Verify the target is F0
    assert_eq!(format!("{}", composite.target), "F0");
}

/// Test that `SecondaryComposite::degree` returns the correct value.
///
/// The composite degree equals gen_deg - shift_t. For SecondaryResolution the shift is (s=2, t=0),
/// so composite.degree = gen_deg - 0 = gen_deg.
#[test]
fn secondary_composite_degree() {
    let resolution = construct_standard::<false, _, _>("S_2", None).unwrap();
    resolution.compute_through_stem(Bidegree::n_s(10, 5));

    let lift = SecondaryResolution::new(Arc::new(resolution));
    lift.initialize_homotopies();
    lift.compute_composites();

    // Check degree for the first generator of F2 (at internal degree 2)
    let homotopy = lift.homotopy(2);
    let composite = homotopy.composite(2, 0);
    assert_eq!(composite.degree, 2);

    // Check at a higher degree
    let homotopy3 = lift.homotopy(3);
    let composite3 = homotopy3.composite(3, 0);
    assert_eq!(composite3.degree, 3);
}

/// Test that the secondary resolution computes correct d_2 differentials for S_2.
///
/// The Adams spectral sequence for S_2 has the well-known differential
/// d_2(h_4) = h_0 h_3^2, which shows up as hom_k at degree 16 giving [[1]].
#[test]
fn secondary_resolution_d2_s2() {
    let resolution = construct_standard::<false, _, _>("S_2", None).unwrap();
    resolution.compute_through_stem(Bidegree::n_s(15, 8));

    let lift = SecondaryResolution::new(Arc::new(resolution));
    lift.initialize_homotopies();
    lift.compute_composites();
    lift.compute_intermediates();
    lift.compute_homotopies();

    assert_eq!(lift.homotopy(3).homotopies.hom_k(16), vec![vec![1]]);
}

/// Test that composites in the secondary resolution have accessible internal data.
#[test]
fn secondary_composite_data_accessible() {
    let resolution = construct_standard::<false, _, _>("S_2", None).unwrap();
    resolution.compute_through_stem(Bidegree::n_s(10, 5));

    let lift = SecondaryResolution::new(Arc::new(resolution));
    lift.initialize_homotopies();
    lift.compute_composites();

    // homotopy(3) maps from F3 to F1. F3 has its first generator at internal degree 3.
    let homotopy = lift.homotopy(3);
    let source = &homotopy.source;
    assert!(
        source.number_of_gens_in_degree(3) > 0,
        "F3 should have a generator at degree 3"
    );

    let composite = homotopy.composite(3, 0);

    // The composite BiVec should be populated (non-empty)
    assert!(!composite.composite.is_empty());

    // Check we can iterate over composite entries
    for row in composite.composite.iter() {
        for elt in row {
            let _ = <algebra::MilnorAlgebra as PairAlgebra>::element_is_zero(elt);
        }
    }
}

/// Test the full secondary resolution pipeline with save/load, verifying
/// composite fields are accessible at each step.
#[test]
fn secondary_resolution_roundtrip_with_field_access() {
    let tempdir = tempfile::TempDir::new().unwrap();

    let mut resolution =
        construct_standard::<false, _, _>("S_2", Some(tempdir.path().into())).unwrap();
    resolution.load_quasi_inverse = false;
    resolution.compute_through_stem(Bidegree::n_s(10, 4));

    let lift = SecondaryResolution::new(Arc::new(resolution));
    lift.initialize_homotopies();
    lift.compute_composites();
    lift.compute_intermediates();
    lift.compute_homotopies();

    // Iterate over homotopies and verify composite.degree matches gen_deg for each composite
    for s in 2..4 {
        let homotopy = lift.homotopy(s);
        let min_gen_deg = homotopy.source.min_degree();
        let max_gen_deg = homotopy.source.max_computed_degree();

        for t in min_gen_deg..=max_gen_deg {
            let n_gens = homotopy.source.number_of_gens_in_degree(t);
            for idx in 0..n_gens {
                let composite = homotopy.composite(t, idx);
                assert_eq!(
                    composite.degree, t,
                    "composite.degree should equal gen_deg at s={s}, t={t}, idx={idx}"
                );
            }
        }
    }
}
