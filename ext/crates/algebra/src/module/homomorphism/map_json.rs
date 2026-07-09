use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, anyhow};
use bivec::BiVec;
use fp::matrix::Matrix;
use fp::vector::FpVector;
use serde_json::Value;

use crate::algebra::{Algebra, GeneratedAlgebra};
use crate::module::Module;

use super::{FullModuleHomomorphism, ModuleHomomorphism};

/// Parsed specification of a module homomorphism from JSON.
pub struct MapSpec {
    pub p: u32,
    pub name: String,
    pub source_name: String,
    pub target_name: String,
    /// Mathematical convention: source degree t maps to target degree t + degree_shift.
    pub degree_shift: i32,
    pub images: MapImages,
    pub max_degree: Option<i32>,
}

/// The two forms of specifying a homomorphism.
pub enum MapImages {
    /// Human-authored: generator names map to F_p linear combination strings.
    GeneratorImages(BTreeMap<String, String>),
    /// Machine-written: keyed by source degree (as string), values are 2D matrices.
    Matrices(BTreeMap<i32, Vec<Vec<u32>>>),
}

/// Parse a map specification from JSON.
pub fn parse_map_json(json: &Value) -> anyhow::Result<MapSpec> {
    let p = json["p"]
        .as_u64()
        .ok_or_else(|| anyhow!("Missing or invalid 'p' field"))?
        as u32;

    let name = json["name"]
        .as_str()
        .unwrap_or("")
        .to_string();

    let source_name = json["source"]
        .as_str()
        .ok_or_else(|| anyhow!("Missing 'source' field"))?
        .to_string();

    let target_name = json["target"]
        .as_str()
        .ok_or_else(|| anyhow!("Missing 'target' field"))?
        .to_string();

    let degree_shift = json["degree_shift"].as_i64().unwrap_or(0) as i32;

    let max_degree = json["max_degree"].as_i64().map(|x| x as i32);

    let images = if let Some(img_obj) = json.get("images") {
        let obj = img_obj
            .as_object()
            .ok_or_else(|| anyhow!("'images' must be an object"))?;
        let mut map = BTreeMap::new();
        for (k, v) in obj {
            let val = v
                .as_str()
                .ok_or_else(|| anyhow!("Image value for '{k}' must be a string"))?;
            map.insert(k.clone(), val.to_string());
        }
        MapImages::GeneratorImages(map)
    } else if let Some(mat_obj) = json.get("matrices") {
        let obj = mat_obj
            .as_object()
            .ok_or_else(|| anyhow!("'matrices' must be an object"))?;
        let mut map = BTreeMap::new();
        for (k, v) in obj {
            let degree: i32 = k
                .parse()
                .with_context(|| format!("Invalid degree key '{k}' in matrices"))?;
            let rows: Vec<Vec<u32>> = serde_json::from_value(v.clone())
                .with_context(|| format!("Invalid matrix at degree {degree}"))?;
            map.insert(degree, rows);
        }
        MapImages::Matrices(map)
    } else {
        return Err(anyhow!(
            "Map spec must contain either 'images' or 'matrices'"
        ));
    };

    Ok(MapSpec {
        p,
        name,
        source_name,
        target_name,
        degree_shift,
        images,
        max_degree,
    })
}

/// Build a `FullModuleHomomorphism` from a parsed map spec and resolved modules.
///
/// The `degree_shift` in the JSON uses mathematical convention (source degree t maps to
/// target degree t + degree_shift). Internally, `ModuleHomomorphism::degree_shift()` returns
/// `s` where `output = input - s`, so the Rust value is `-degree_shift`.
pub fn build_homomorphism<S: Module, T: Module<Algebra = S::Algebra>>(
    spec: &MapSpec,
    source: Arc<S>,
    target: Arc<T>,
) -> anyhow::Result<FullModuleHomomorphism<S, T>> {
    let p = source.prime();
    // Rust internal shift: output_degree = input_degree - internal_shift
    // Mathematical: output_degree = input_degree + degree_shift
    // So internal_shift = -degree_shift
    let internal_shift = -spec.degree_shift;

    let source_min = source.min_degree();
    let source_max = match (source.max_degree(), spec.max_degree) {
        (Some(m), Some(cap)) => std::cmp::min(m, cap),
        (Some(m), None) => m,
        (None, Some(cap)) => cap,
        (None, None) => {
            return Err(anyhow!(
                "Source module is unbounded; specify 'max_degree' in the map spec"
            ))
        }
    };

    // Target min degree for matrix BiVec
    let target_min = target.min_degree();

    // Compute the range of target degrees we need matrices for.
    // For source degree t, target degree = t + degree_shift = t - internal_shift.
    let out_max = source_max + spec.degree_shift;

    // Ensure bases are computed
    source.compute_basis(source_max);
    target.compute_basis(out_max);

    let matrices_max = out_max;

    let mut matrices = BiVec::with_capacity(target_min, matrices_max + 1);

    // Fill with zero matrices up to the range we need
    for target_deg in target_min..=matrices_max {
        let source_deg = target_deg + internal_shift; // = target_deg - degree_shift
        let source_dim = if source_deg >= source_min && source_deg <= source_max {
            source.dimension(source_deg)
        } else {
            0
        };
        let target_dim = target.dimension(target_deg);
        matrices.push(Matrix::new(p, source_dim, target_dim));
    }

    match &spec.images {
        MapImages::GeneratorImages(gen_images) => {
            // Build a lookup from target basis element names to (degree, index)
            let target_max_deg = match (target.max_degree(), spec.max_degree) {
                (Some(m), Some(cap)) => std::cmp::min(m, cap),
                (Some(m), None) => m,
                (None, Some(cap)) => cap,
                (None, None) => {
                    return Err(anyhow!(
                        "Target module is unbounded; specify 'max_degree' in the map spec"
                    ))
                }
            };

            let mut target_name_to_idx: BTreeMap<String, (i32, usize)> = BTreeMap::new();
            for deg in target.min_degree()..=target_max_deg {
                target.compute_basis(deg);
                for idx in 0..target.dimension(deg) {
                    let name = target.basis_element_to_string(deg, idx);
                    target_name_to_idx.insert(name, (deg, idx));
                }
            }

            // For each source generator, look up its image
            for source_deg in source_min..=source_max {
                let source_dim = source.dimension(source_deg);
                let target_deg = source_deg + spec.degree_shift;

                if target_deg < target_min || target_deg > matrices_max {
                    continue;
                }

                for source_idx in 0..source_dim {
                    let gen_name = source.basis_element_to_string(source_deg, source_idx);

                    let rhs = match gen_images.get(&gen_name) {
                        Some(s) => s.as_str(),
                        None => continue, // omitted generators map to 0
                    };

                    if rhs == "0" {
                        continue;
                    }

                    let matrix = &mut matrices[target_deg];

                    // Parse the RHS as a linear combination.
                    // Basis element names may contain spaces (e.g. "x^{1} y"),
                    // so we first try the full term as a basis name, then try
                    // splitting off a leading coefficient.
                    for item in rhs.split(" + ") {
                        let item = item.trim();
                        let (coef, g) = if target_name_to_idx.contains_key(item) {
                            (1u32, item)
                        } else if let Some((coef_str, rest)) = item.split_once(' ') {
                            if let Ok(c) = str::parse::<u32>(coef_str) {
                                (c, rest.trim())
                            } else {
                                return Err(anyhow!(
                                    "Cannot parse '{item}' in image of '{gen_name}': \
                                     not a known basis element, and '{coef_str}' is not a coefficient"
                                ));
                            }
                        } else {
                            return Err(anyhow!(
                                "Unknown basis element '{item}' in image of '{gen_name}'"
                            ));
                        };
                        let (deg, idx) = target_name_to_idx
                            .get(g)
                            .copied()
                            .ok_or_else(|| {
                                anyhow!("Unknown target basis element '{g}' in image of '{gen_name}'")
                            })?;
                        if deg != target_deg {
                            return Err(anyhow!(
                                "Degree mismatch in image of '{gen_name}': '{g}' has degree {deg} \
                                 but expected degree {target_deg}"
                            ));
                        }
                        matrix.row_mut(source_idx).add_basis_element(idx, coef);
                    }
                }
            }
        }
        MapImages::Matrices(mat_data) => {
            for (&source_deg, rows) in mat_data {
                let target_deg = source_deg + spec.degree_shift;
                if target_deg < target_min || target_deg > matrices_max {
                    continue;
                }

                let source_dim = source.dimension(source_deg);
                let target_dim = target.dimension(target_deg);

                if rows.len() != source_dim {
                    return Err(anyhow!(
                        "Matrix at source degree {source_deg}: expected {source_dim} rows, got {}",
                        rows.len()
                    ));
                }

                let matrix = &mut matrices[target_deg];
                for (i, row) in rows.iter().enumerate() {
                    if row.len() != target_dim {
                        return Err(anyhow!(
                            "Matrix at source degree {source_deg}, row {i}: expected {target_dim} \
                             entries, got {}",
                            row.len()
                        ));
                    }
                    for (j, &val) in row.iter().enumerate() {
                        if val != 0 {
                            matrix.row_mut(i).set_entry(j, val);
                        }
                    }
                }
            }
        }
    }

    Ok(FullModuleHomomorphism::from_matrices(
        source,
        target,
        internal_shift,
        matrices,
    ))
}

/// Verify that a homomorphism is A-linear: f(a·x) = a·f(x) for all algebra
/// generators a and all basis elements x in the source, up to `max_degree`.
///
/// Uses `GeneratedAlgebra::generators()` for the generating set.
pub fn verify_a_linearity<S, T>(
    f: &FullModuleHomomorphism<S, T>,
    max_degree: i32,
) -> anyhow::Result<()>
where
    S: Module,
    T: Module<Algebra = S::Algebra>,
    S::Algebra: GeneratedAlgebra,
{
    let source = f.source();
    let target = f.target();
    let algebra = source.algebra();
    let p = source.prime();
    let degree_shift = f.degree_shift(); // internal convention

    let source_min = source.min_degree();

    // Pre-compute algebra and module bases through the max degree we'll need
    algebra.compute_basis(max_degree);
    target.algebra().compute_basis(max_degree);
    source.compute_basis(max_degree);
    target.compute_basis(max_degree);

    for input_degree in source_min..=max_degree {
        source.compute_basis(input_degree);
        let source_dim = source.dimension(input_degree);
        if source_dim == 0 {
            continue;
        }

        // For each algebra generator degree, check that f commutes with the action
        for op_degree in 1..=max_degree - input_degree {
            let generators = algebra.generators(op_degree);
            if generators.is_empty() {
                continue;
            }

            let output_degree = input_degree + op_degree;
            source.compute_basis(output_degree);
            target.compute_basis(output_degree - degree_shift);
            let target_output_dim = target.dimension(output_degree - degree_shift);
            if target_output_dim == 0 {
                continue;
            }

            // Also need target at (input_degree - degree_shift) for f(x)
            let target_fx_degree = input_degree - degree_shift;
            target.compute_basis(target_fx_degree);

            for &op_idx in &generators {
                for basis_idx in 0..source_dim {
                    // Compute f(a · x)
                    // First: a · x in source
                    let source_output_dim = source.dimension(output_degree);
                    let mut ax = FpVector::new(p, source_output_dim);
                    source.act_on_basis(ax.as_slice_mut(), 1, op_degree, op_idx, input_degree, basis_idx);

                    // Then: f(a · x)
                    let mut f_ax = FpVector::new(p, target_output_dim);
                    for (i, v) in ax.iter_nonzero() {
                        f.apply_to_basis_element(f_ax.as_slice_mut(), v, output_degree, i);
                    }

                    // Compute a · f(x)
                    // First: f(x) in target
                    let target_fx_dim = target.dimension(target_fx_degree);
                    let mut fx = FpVector::new(p, target_fx_dim);
                    f.apply_to_basis_element(fx.as_slice_mut(), 1, input_degree, basis_idx);

                    // Then: a · f(x) in target
                    let mut a_fx = FpVector::new(p, target_output_dim);
                    target.act(
                        a_fx.as_slice_mut(),
                        1,
                        op_degree,
                        op_idx,
                        target_fx_degree,
                        fx.as_slice(),
                    );

                    // Compare
                    if f_ax != a_fx {
                        let op_name = algebra.generator_to_string(op_degree, op_idx);
                        let x_name = source.basis_element_to_string(input_degree, basis_idx);
                        return Err(anyhow!(
                            "A-linearity check failed: f({op_name} · {x_name}) != {op_name} · f({x_name})\n  \
                             f({op_name} · {x_name}) = {}\n  \
                             {op_name} · f({x_name}) = {}",
                            target.element_to_string(output_degree - degree_shift, f_ax.as_slice()),
                            target.element_to_string(output_degree - degree_shift, a_fx.as_slice()),
                        ));
                    }
                }
            }
        }
    }

    Ok(())
}

