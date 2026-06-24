//! Generates a random finite dimensional module over the Steenrod algebra and resolves it.
//!
//! Unlike `FPModule` (finitely presented modules), where Adem relations are automatically
//! satisfied by construction, `FDModule` requires explicitly specifying all Steenrod operations
//! and verifying Adem relations via `check_validity`. This example generates random generator
//! actions, uses `extend_actions` to compute derived operations, and checks that all Adem
//! relations hold. If any relation fails, it retries with a fresh random module.
//!
//! Run with: cargo run --example random_module

use std::io::IsTerminal;
use std::sync::Arc;

use algebra::{
    AdemAlgebra, GeneratedAlgebra, SteenrodAlgebra,
    module::{FDModule, Module, SteenrodModule},
};
use bivec::BiVec;
use ext::chain_complex::{ChainComplex, FreeChainComplex};
use fp::matrix::Matrix;
use fp::prime::ValidPrime;
use rand::Rng;
use sseq::coordinates::Bidegree;

/// Generate a random valid `FDModule` over the Steenrod algebra with the given graded dimensions.
///
/// Repeatedly tries random sparse generator actions until Adem relations are satisfied.
fn generate_random_fd_module(
    rng: &mut impl Rng,
    algebra: &Arc<SteenrodAlgebra>,
    graded_dim: &BiVec<usize>,
    verbose: bool,
) -> FDModule<SteenrodAlgebra> {
    let max_attempts = 1000;
    let sparsity = 0.7; // probability that each action entry is zero

    for attempt in 0..max_attempts {
        let mut module =
            FDModule::new(Arc::clone(algebra), "random_fd".to_string(), graded_dim.clone());

        let min_deg = graded_dim.min_degree();
        let max_deg = graded_dim.len(); // one past the end

        let mut valid = true;

        // Process in the same order as from_json: input_deg from HIGH to LOW,
        // output_deg from LOW to HIGH
        'outer: for input_deg in (min_deg..max_deg).rev() {
            for output_deg in (input_deg + 1)..max_deg {
                let op_deg = output_deg - input_deg;
                let output_dim = module.dimension(output_deg);
                let input_dim = module.dimension(input_deg);

                if output_dim == 0 || input_dim == 0 {
                    continue;
                }

                // Set random actions for generator operations only
                for op_idx in algebra.generators(op_deg) {
                    for input_idx in 0..input_dim {
                        let mut output = vec![0u32; output_dim];
                        for entry in output.iter_mut() {
                            // With probability (1 - sparsity), set a random nonzero value
                            if rng.random_range(0..1000) >= (sparsity * 1000.0) as u32 {
                                *entry = 1; // p=2, so only nonzero value is 1
                            }
                        }
                        module.set_action(op_deg, op_idx, input_deg, input_idx, &output);
                    }
                }

                // Compute derived (non-generator) actions from the generators we just set
                module.extend_actions(input_deg, output_deg);

                // Check Adem relations for this (input_deg, output_deg) pair
                if module.check_validity(input_deg, output_deg).is_err() {
                    valid = false;
                    break 'outer;
                }
            }
        }

        if valid {
            if verbose && attempt > 0 {
                println!("Found valid module after {attempt} retries.");
            }
            return module;
        }
    }

    // Fallback: zero-action module (always valid)
    if verbose {
        println!("Falling back to zero-action module after {max_attempts} attempts.");
    }
    FDModule::new(
        Arc::clone(algebra),
        "random_fd".to_string(),
        graded_dim.clone(),
    )
}

/// Return the ANSI color escape code for a given operation degree.
fn op_color(op_deg: u8) -> &'static str {
    match op_deg {
        0 => "\x1b[1m",   // bold (cells)
        1 => "\x1b[36m",  // cyan (Sq1)
        2 => "\x1b[33m",  // yellow (Sq2)
        4 => "\x1b[35m",  // magenta (Sq4)
        _ => "\x1b[32m",  // green (higher)
    }
}

/// Return the ANSI color for degree labels.
fn label_color() -> &'static str {
    "\x1b[2m" // dim
}

const RESET: &str = "\x1b[0m";

/// Print a cell diagram showing cells at each degree and Steenrod operation connections.
///
/// When stdout is a terminal, uses ANSI colors to distinguish different Sq operations.
fn print_cell_diagram(
    module: &FDModule<SteenrodAlgebra>,
    algebra: &Arc<SteenrodAlgebra>,
) {
    let min_deg = module.min_degree();
    let max_deg = module.max_degree().unwrap();

    let max_dim = (min_deg..=max_deg)
        .map(|d| module.dimension(d))
        .max()
        .unwrap_or(0);

    if max_dim == 0 {
        return;
    }

    let use_color = std::io::stdout().is_terminal();

    // Layout parameters
    let col_sp: usize = 3; // columns between cell centers
    let row_sp: usize = 3; // rows between adjacent degree rows
    let margin: usize = 6; // left margin for "  d | " labels

    let height = ((max_deg - min_deg) as usize) * row_sp + 1;
    let width = margin + (max_dim - 1) * col_sp + 1;

    let deg_row = |d: i32| -> usize { ((max_deg - d) as usize) * row_sp };
    let cell_col = |i: usize| -> usize { margin + i * col_sp };

    // Grid stores (char, op_degree) where op_degree encodes the color category:
    // 0 = cell/label, 1 = Sq1, 2 = Sq2, 4 = Sq4, etc., 255 = blank
    let mut grid = vec![vec![(' ', 255u8); width]; height];

    // Place cells
    for d in min_deg..=max_deg {
        let r = deg_row(d);
        for i in 0..module.dimension(d) {
            let c = cell_col(i);
            if c < width {
                grid[r][c] = ('\u{25cf}', 0); // ● with cell color
            }
        }
    }

    // Collect connections: (src_deg, src_idx, tgt_deg, tgt_idx, op_deg)
    let mut connections: Vec<(i32, usize, i32, usize, i32)> = Vec::new();
    for input_deg in min_deg..=max_deg {
        for output_deg in (input_deg + 1)..=max_deg {
            let op_deg = output_deg - input_deg;
            if module.dimension(output_deg) == 0 || module.dimension(input_deg) == 0 {
                continue;
            }
            for op_idx in algebra.generators(op_deg) {
                for input_idx in 0..module.dimension(input_deg) {
                    let action = module.action(op_deg, op_idx, input_deg, input_idx);
                    for output_idx in 0..module.dimension(output_deg) {
                        if action.entry(output_idx) != 0 {
                            connections
                                .push((input_deg, input_idx, output_deg, output_idx, op_deg));
                        }
                    }
                }
            }
        }
    }

    // Draw connections (longer first so shorter ones overlay on top)
    connections.sort_by(|a, b| b.4.cmp(&a.4));

    for &(src_d, src_i, tgt_d, tgt_i, op_d) in &connections {
        let src_r = deg_row(src_d) as f64;
        let tgt_r = deg_row(tgt_d) as f64;
        let src_c = cell_col(src_i) as f64;
        let tgt_c = cell_col(tgt_i) as f64;

        let dr = src_r - tgt_r;
        let dc = tgt_c - src_c;
        if dr <= 0.0 {
            continue;
        }

        // Draw line between cells, skipping the cell positions themselves
        let steps = dr as usize;
        for step in 1..steps {
            let r = (src_r - step as f64) as usize;
            let c = (src_c + dc * step as f64 / dr).round() as usize;
            if c >= width {
                continue;
            }

            let ch = if dc.abs() < 0.01 {
                '\u{2502}' // │
            } else if dc > 0.0 {
                '/'
            } else {
                '\\'
            };

            // Don't overwrite cells
            if grid[r][c].1 == 255 {
                grid[r][c] = (ch, op_d as u8);
            }
        }
    }

    // Print diagram
    println!("\n  Cell Diagram");
    for r in 0..height {
        // Build prefix (degree label)
        let mut prefix = String::from("    \u{2502} ");
        let mut is_label_row = false;
        for d in min_deg..=max_deg {
            if deg_row(d) == r {
                if use_color {
                    prefix = format!("{}{d:>3}{} \u{2502} ", label_color(), RESET);
                } else {
                    prefix = format!("{d:>3} \u{2502} ");
                }
                is_label_row = true;
                break;
            }
        }
        if !is_label_row && use_color {
            prefix = format!("{}    \u{2502}{} ", label_color(), RESET);
        }

        let mut line = prefix;

        if use_color {
            let mut cur_color: Option<u8> = None;
            for c in margin..width {
                let (ch, od) = grid[r][c];
                if od != 255 {
                    if cur_color != Some(od) {
                        line.push_str(op_color(od));
                        cur_color = Some(od);
                    }
                    line.push(ch);
                } else {
                    if cur_color.is_some() {
                        line.push_str(RESET);
                        cur_color = None;
                    }
                    line.push(ch);
                }
            }
            if cur_color.is_some() {
                line.push_str(RESET);
            }
        } else {
            for c in margin..width {
                line.push(grid[r][c].0);
            }
        }
        println!("{}", line.trim_end());
    }

    // Legend
    let mut ops: Vec<i32> = connections.iter().map(|c| c.4).collect();
    ops.sort();
    ops.dedup();
    if !ops.is_empty() {
        if use_color {
            let legend: Vec<String> = ops
                .iter()
                .map(|d| format!("{}Sq{}{}", op_color(*d as u8), d, RESET))
                .collect();
            println!("        ({})", legend.join(", "));
        } else {
            let legend: Vec<String> = ops.iter().map(|d| format!("Sq{d}")).collect();
            println!("        ({})", legend.join(", "));
        }
    }
}

/// Check whether the module is indecomposable by computing End_A(M) and searching for
/// nontrivial idempotents.
///
/// Returns `(result, dim_End_A(M))` where result is:
/// - `Some(true)` if proven indecomposable
/// - `Some(false)` if proven decomposable (nontrivial idempotent found)
/// - `None` if End_A(M) is too large for brute-force idempotent search
fn check_indecomposable(
    module: &FDModule<SteenrodAlgebra>,
    algebra: &Arc<SteenrodAlgebra>,
) -> (Option<bool>, usize) {
    let p = module.prime();
    let min_deg = module.min_degree();
    let max_deg = module.max_degree().unwrap();

    // Compute offsets and total number of variables.
    // Variables: for each degree d with dimension n_d, we have n_d^2 entries of F_d.
    // Variable for F_d[i][j] is at index offset_d + i * n_d + j.
    let mut offsets: Vec<(i32, usize, usize)> = Vec::new(); // (degree, dim, offset)
    let mut total_vars = 0usize;
    for d in min_deg..=max_deg {
        let n = module.dimension(d);
        if n > 0 {
            offsets.push((d, n, total_vars));
            total_vars += n * n;
        }
    }

    if total_vars == 0 {
        return (Some(true), 0);
    }

    // Helper to find offset entry for a degree
    let find_offset = |d: i32| -> Option<(usize, usize)> {
        offsets.iter().find(|&&(deg, _, _)| deg == d).map(|&(_, n, off)| (n, off))
    };

    // Build constraint rows. Each constraint says:
    //   sum_l F_{d+k}[i][l] * A[l][j] + sum_l A[i][l] * F_d[l][j] = 0  (mod 2)
    // which is linear in the F variables.
    //
    // For each generator theta of degree k, each input degree d, and each (i, j):
    //   i in 0..n_{d+k}, j in 0..n_d
    let mut constraints: Vec<Vec<u32>> = Vec::new();

    for &(d, n_d, off_d) in &offsets {
        for output_deg in (d + 1)..=max_deg {
            let op_deg = output_deg - d;
            let (n_out, off_out) = match find_offset(output_deg) {
                Some(v) => v,
                None => continue,
            };

            for op_idx in algebra.generators(op_deg) {
                // Precompute the action matrix A_{theta,d}: n_out x n_d
                // A[i][j] = module.action(op_deg, op_idx, d, j).entry(i)
                let mut action_mat = vec![vec![0u32; n_d]; n_out];
                for j in 0..n_d {
                    let action = module.action(op_deg, op_idx, d, j);
                    for i in 0..n_out {
                        action_mat[i][j] = action.entry(i);
                    }
                }

                // For each (i, j) with i in 0..n_out, j in 0..n_d:
                for i in 0..n_out {
                    for j in 0..n_d {
                        let mut row = vec![0u32; total_vars];

                        // Term 1: sum_l F_{d+k}[i][l] * A[l][j]
                        for l in 0..n_out {
                            if action_mat[l][j] != 0 {
                                let var = off_out + i * n_out + l;
                                row[var] ^= 1;
                            }
                        }

                        // Term 2: sum_l A[i][l] * F_d[l][j]
                        for l in 0..n_d {
                            if action_mat[i][l] != 0 {
                                let var = off_d + l * n_d + j;
                                row[var] ^= 1;
                            }
                        }

                        // Only add non-trivial constraints
                        if row.iter().any(|&x| x != 0) {
                            constraints.push(row);
                        }
                    }
                }
            }
        }
    }

    // If no constraints, End_A(M) = all linear maps = total_vars dimensional
    if constraints.is_empty() {
        let dim_end = total_vars;
        if dim_end == 1 {
            return (Some(true), 1);
        }
        // Check for idempotents in the unconstrained case
        return check_idempotents_in_kernel_dimension(dim_end, total_vars, &offsets, None);
    }

    // We want the right kernel of C: {x : Cx = 0}.
    // augmented_from_vec + compute_kernel computes the left kernel (row null space),
    // so we transpose C first: left kernel of C^T = right kernel of C.
    let num_constraints = constraints.len();
    let mut c_transpose = vec![vec![0u32; num_constraints]; total_vars];
    for (ci, row) in constraints.iter().enumerate() {
        for (vi, &val) in row.iter().enumerate() {
            c_transpose[vi][ci] = val;
        }
    }

    let (padded_cols, mut aug) = Matrix::augmented_from_vec(p, &c_transpose);
    aug.row_reduce();
    let kernel = aug.compute_kernel(padded_cols);
    let dim_end = kernel.dimension();

    if dim_end == 0 {
        // Only the zero endomorphism — module might be zero or trivial
        return (Some(true), 0);
    }
    if dim_end == 1 {
        return (Some(true), 1);
    }

    check_idempotents_in_kernel_dimension(dim_end, total_vars, &offsets, Some(&kernel))
}

/// Check for nontrivial idempotents given the kernel (or None for unconstrained space).
fn check_idempotents_in_kernel_dimension(
    dim_end: usize,
    total_vars: usize,
    offsets: &[(i32, usize, usize)],
    kernel: Option<&fp::matrix::Subspace>,
) -> (Option<bool>, usize) {
    if dim_end > 20 {
        // Too large to brute force
        return (None, dim_end);
    }

    // Build the identity endomorphism vector
    let mut identity = vec![0u32; total_vars];
    for &(_, n, off) in offsets {
        for i in 0..n {
            identity[off + i * n + i] = 1;
        }
    }

    // Enumerate all 2^dim_end - 1 nonzero elements of End_A(M)
    let num_elements = 1u64 << dim_end;

    // Collect basis vectors
    let basis_vecs: Vec<Vec<u32>> = if let Some(k) = kernel {
        k.basis()
            .map(|slice| (0..total_vars).map(|i| slice.entry(i)).collect())
            .collect()
    } else {
        // Unconstrained: standard basis
        (0..dim_end)
            .map(|b| {
                let mut v = vec![0u32; total_vars];
                v[b] = 1;
                v
            })
            .collect()
    };

    for bits in 1..num_elements {
        // Build the endomorphism f from the basis
        let mut f = vec![0u32; total_vars];
        for (b, bv) in basis_vecs.iter().enumerate() {
            if (bits >> b) & 1 == 1 {
                for (i, &val) in bv.iter().enumerate() {
                    f[i] ^= val;
                }
            }
        }

        // Skip the identity
        if f == identity {
            continue;
        }

        // Check f^2 = f degree by degree
        if is_idempotent(&f, offsets) {
            return (Some(false), dim_end);
        }
    }

    (Some(true), dim_end)
}

/// Check whether an endomorphism (given as a flat variable vector) is idempotent.
fn is_idempotent(f: &[u32], offsets: &[(i32, usize, usize)]) -> bool {
    for &(_, n, off) in offsets {
        // Extract n x n block for this degree
        for i in 0..n {
            for j in 0..n {
                // (F^2)[i][j] = sum_l F[i][l] * F[l][j]
                let mut f_sq = 0u32;
                for l in 0..n {
                    f_sq ^= f[off + i * n + l] & f[off + l * n + j];
                }
                if f_sq != f[off + i * n + j] {
                    return false;
                }
            }
        }
    }
    true
}

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let mut rng = rand::rng();
    let p = ValidPrime::new(2);

    let num_cells: usize = query::with_default("Number of cells (total dimension)", "6", str::parse);
    let max_degree: i32 = query::with_default("Max degree", "4", str::parse);
    let require_indecomposable: bool = query::with_default(
        "Require indecomposable? (y/n)",
        "n",
        |response: &str| {
            if response.starts_with('y') || response.starts_with('n') {
                Ok(response.starts_with('y'))
            } else {
                Err(format!(
                    "unrecognized response '{response}'. Should be '(y)es' or '(n)o'"
                ))
            }
        },
    );

    let algebra = Arc::new(SteenrodAlgebra::AdemAlgebra(AdemAlgebra::new(p, false)));

    println!("\n=== Random Finite Dimensional Steenrod Module (p = 2) ===\n");

    // Distribute num_cells randomly across degrees 0..=max_degree
    let num_degrees = (max_degree + 1) as usize;
    let mut dim_vec = vec![0usize; num_degrees];
    for _ in 0..num_cells {
        let deg = rng.random_range(0..num_degrees);
        dim_vec[deg] += 1;
    }
    let graded_dim = BiVec::from_vec(0, dim_vec);

    println!("Graded dimensions:");
    for (deg, &dim) in graded_dim.iter_enum() {
        if dim > 0 {
            println!("  degree {deg}: {dim}");
        }
    }

    let max_retry = 200;
    let mut module = generate_random_fd_module(&mut rng, &algebra, &graded_dim, true);
    let (mut indecomp_result, mut dim_end) = check_indecomposable(&module, &algebra);

    if require_indecomposable && indecomp_result != Some(true) {
        let mut attempt = 1;
        while attempt < max_retry {
            module = generate_random_fd_module(&mut rng, &algebra, &graded_dim, false);
            let result = check_indecomposable(&module, &algebra);
            indecomp_result = result.0;
            dim_end = result.1;
            if indecomp_result == Some(true) {
                println!("Found indecomposable module after {attempt} retries.");
                break;
            }
            attempt += 1;
        }
        if indecomp_result != Some(true) {
            println!("Could not find indecomposable module after {max_retry} attempts, proceeding with last attempt.");
        }
    }

    // Print nonzero generator actions
    let min_deg = module.min_degree();
    let max_deg = module.max_degree().unwrap();

    println!("\nNonzero generator actions:");
    let mut any_nonzero = false;
    for input_deg in min_deg..=max_deg {
        for output_deg in (input_deg + 1)..=max_deg {
            let op_deg = output_deg - input_deg;
            let output_dim = module.dimension(output_deg);
            if output_dim == 0 {
                continue;
            }
            for op_idx in algebra.generators(op_deg) {
                for input_idx in 0..module.dimension(input_deg) {
                    let action = module.action(op_deg, op_idx, input_deg, input_idx);
                    if !action.is_zero() {
                        any_nonzero = true;
                        println!(
                            "  {} {} = {}",
                            algebra.generator_to_string(op_deg, op_idx),
                            module.basis_element_to_string(input_deg, input_idx),
                            module.element_to_string(output_deg, action.as_slice()),
                        );
                    }
                }
            }
        }
    }
    if !any_nonzero {
        println!("  (all zero)");
    }

    println!("\nAdem relations verified.");

    // Print indecomposability result
    match indecomp_result {
        Some(true) => println!("Module is indecomposable (dim End_A(M) = {dim_end})"),
        Some(false) => println!("Module is decomposable (dim End_A(M) = {dim_end})"),
        None => println!("dim End_A(M) = {dim_end} (too large for idempotent search)"),
    }

    print_cell_diagram(&module, &algebra);

    // Resolve and display Ext chart
    let t_max: i32 = query::with_default("Max t", "30", str::parse);
    let s_max: i32 = query::with_default("Max s", "10", str::parse);

    let module: SteenrodModule = Arc::new(module);
    let module = Arc::new(module);
    let cc: Arc<ext::CCC> = Arc::new(ext::chain_complex::FiniteChainComplex::ccdz(module));
    let res = ext::resolution::Resolution::new(Arc::clone(&cc));

    let max = Bidegree::s_t(s_max, t_max);
    res.compute_through_bidegree(max);

    println!("\nExt chart:");
    println!("{}", res.graded_dimension_string());

    Ok(())
}
