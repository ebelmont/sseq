use std::collections::HashMap as StdHashMap;
use std::time::Instant;

use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use log::error;

use ehp_core::constraints::{self, DiffVar};
use ehp_core::io;
use ehp_core::pageturning;
use ehp_core::solver;

#[derive(Parser)]
#[command(name = "ehp-sat")]
#[command(about = "EHP spectral sequence differential solver (Rust)")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Logging verbosity: -v for info, -vv for debug
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
}

#[derive(Subcommand)]
enum Commands {
    /// Load a page and compute differentials
    Compute {
        /// Path to data directory, file prefix, or .ehp file
        #[arg(short, long)]
        prefix: String,

        /// Page number r (default: 2)
        #[arg(short, long, default_value = "2")]
        r: i32,

        /// Maximum total degree s+f
        #[arg(short = 't', long)]
        max_total: i32,

        /// Output directory for results
        #[arg(short, long, default_value = ".")]
        output: String,

        /// CSV file of known differentials (format: r,n,s,f,row,col,value)
        #[arg(short = 'k', long)]
        known_diffs: Option<String>,
    },

    /// Turn the page: compute E_{r+1} from E_r
    TurnPage {
        /// Path to data directory, file prefix, or .ehp file for E_r
        #[arg(short, long)]
        prefix: String,

        /// Page number r of the source page
        #[arg(short, long, default_value = "2")]
        r: i32,

        /// Maximum total degree s+f
        #[arg(short = 't', long)]
        max_total: i32,

        /// Output path for E_{r+1} (directory for CSV, .ehp file for binary)
        #[arg(short, long)]
        output: String,

        /// CSV file of known differentials (format: r,n,s,f,row,col,value)
        #[arg(short = 'k', long)]
        known_diffs: Option<String>,
    },

    /// Export a page to CSV format
    Export {
        /// Path to data directory, file prefix, or .ehp file
        #[arg(short, long)]
        prefix: String,

        /// Page number r
        #[arg(short, long, default_value = "2")]
        r: i32,

        /// Maximum total degree s+f
        #[arg(short = 't', long)]
        max_total: i32,

        /// Output directory
        #[arg(short, long)]
        output: String,
    },

    /// Show information about a loaded page
    Info {
        /// Path to data directory, file prefix, or .ehp file
        #[arg(short, long)]
        prefix: String,

        /// Page number r
        #[arg(short, long, default_value = "2")]
        r: i32,

        /// Maximum total degree s+f
        #[arg(short = 't', long)]
        max_total: i32,
    },

    /// Convert CSV data to binary .ehp format
    Convert {
        /// Path to data directory or file prefix (CSV input)
        #[arg(short, long)]
        prefix: String,

        /// Page number r
        #[arg(short, long, default_value = "2")]
        r: i32,

        /// Maximum total degree s+f
        #[arg(short = 't', long)]
        max_total: i32,

        /// Output .ehp file path
        #[arg(short, long)]
        output: String,
    },
}

fn main() {
    let cli = Cli::parse();

    let log_level = match cli.verbose {
        0 => "warn",
        1 => "info",
        _ => "debug",
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level)).init();

    let result = match cli.command {
        Commands::Compute {
            prefix,
            r,
            max_total,
            output,
            known_diffs,
        } => cmd_compute(&prefix, r, max_total, &output, known_diffs.as_deref()),
        Commands::TurnPage {
            prefix,
            r,
            max_total,
            output,
            known_diffs,
        } => cmd_turn_page(&prefix, r, max_total, &output, known_diffs.as_deref()),
        Commands::Export {
            prefix,
            r,
            max_total,
            output,
        } => cmd_export(&prefix, r, max_total, &output),
        Commands::Info {
            prefix,
            r,
            max_total,
        } => cmd_info(&prefix, r, max_total),
        Commands::Convert {
            prefix,
            r,
            max_total,
            output,
        } => cmd_convert(&prefix, r, max_total, &output),
    };

    if let Err(e) = result {
        error!("Error: {}", e);
        std::process::exit(1);
    }
}

/// Load known differentials from file, or fall back to hardcoded defaults.
fn load_known_diffs_for_page(r: i32, file: Option<&str>) -> Result<hashbrown::HashMap<DiffVar, bool>, Box<dyn std::error::Error>> {
    if let Some(path) = file {
        let diffs = io::load_known_diffs(path, r)?;
        eprintln!("Loaded {} known differentials from {}", diffs.len(), path);
        Ok(diffs)
    } else {
        let mut diffs = hashbrown::HashMap::new();
        match r {
            2 => { diffs.insert(DiffVar::new(17, 15, 1, 0, 0), true); }
            3 => { diffs.insert(DiffVar::new(17, 15, 2, 0, 0), true); }
            4 => { diffs.insert(DiffVar::new(40, 38, 2, 0, 0), true); }
            _ => {}
        }
        if !diffs.is_empty() {
            eprintln!("Using {} hardcoded known differentials", diffs.len());
        }
        Ok(diffs)
    }
}

fn cmd_compute(prefix: &str, r: i32, max_total: i32, _output: &str, known_diffs_file: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    // Load the page
    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::with_template("{spinner:.green} {msg}").unwrap());
    pb.set_message("Loading page...");

    let page = io::load_page(prefix, r, max_total)?;

    pb.finish_with_message(format!(
        "Loaded E_{} page ({} tridegrees)",
        r,
        page.dimension.values().filter(|&&d| d > 0).count()
    ));

    // Build constraint system
    let cutoff = page.max_s.unwrap_or(0);
    eprintln!("cutoff = {}", cutoff);

    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::with_template("{spinner:.green} {msg}").unwrap());
    pb.set_message("Building variables...");

    let known_diffs = load_known_diffs_for_page(r, known_diffs_file)?;

    pb.set_message("Building constraint system...");
    let system = constraints::build_constraint_system(&page, cutoff, &known_diffs);

    pb.finish_with_message(format!(
        "Constraint system: {} vars, {} constraints",
        system.num_vars,
        system.num_constraints()
    ));

    // Solve
    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::with_template("{spinner:.green} {msg}").unwrap());
    pb.set_message("Solving GF(2) system...");

    match solver::solve(&system) {
        Some(result) => {
            let determined = system.num_vars - result.unknown.len();
            pb.finish_with_message(format!(
                "Solved: {}/{} vars determined ({} unknown)",
                determined,
                system.num_vars,
                result.unknown.len()
            ));

            // Print unknown tridegrees
            let unknown_tris = result.unknown_tridegrees();
            if !unknown_tris.is_empty() {
                eprintln!("Unknown tridegrees ({}):", unknown_tris.len());
                for t in &unknown_tris {
                    eprintln!("  {}", t);
                }
            }

            let elapsed = t0.elapsed();
            eprintln!("Total time: {:.2}s", elapsed.as_secs_f64());
        }
        None => {
            pb.finish_with_message("No solution (system may be inconsistent or empty)");
        }
    }

    Ok(())
}

fn cmd_turn_page(prefix: &str, r: i32, max_total: i32, output: &str, known_diffs_file: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    // Load the page
    eprintln!("Loading E_{} page...", r);
    let page = io::load_page(prefix, r, max_total)?;

    // Build and solve constraint system
    let cutoff = page.max_s.unwrap_or(0);

    let known_diffs = load_known_diffs_for_page(r, known_diffs_file)?;

    eprintln!("Building constraint system...");
    let system = constraints::build_constraint_system(&page, cutoff, &known_diffs);
    eprintln!(
        "  {} vars, {} constraints",
        system.num_vars,
        system.num_constraints()
    );

    eprintln!("Solving...");
    let sat_result = match solver::solve(&system) {
        Some(r) => r,
        None => {
            eprintln!("ERROR: No solution found");
            return Ok(());
        }
    };

    let determined = system.num_vars - sat_result.unknown.len();
    eprintln!(
        "  {}/{} vars determined",
        determined,
        system.num_vars
    );

    // Turn the page
    eprintln!("Computing E_{} from E_{}...", r + 1, r);
    let (next_page, _turned) = pageturning::build_next_page(&page, &sat_result)?;

    // Save — binary if output ends with .ehp, otherwise CSV
    eprintln!("Saving E_{} to {}...", r + 1, output);
    if output.ends_with(".ehp") {
        io::save_to_binary(&next_page, output)?;
    } else {
        io::save_page_json(&next_page, output)?;
    }

    let nonzero_count = next_page.dimension.values().filter(|&&d| d > 0).count();
    eprintln!(
        "E_{} has {} tridegrees with nonzero dimension",
        r + 1,
        nonzero_count
    );

    let elapsed = t0.elapsed();
    eprintln!("Total time: {:.2}s", elapsed.as_secs_f64());

    Ok(())
}

fn cmd_export(prefix: &str, r: i32, max_total: i32, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading E_{} page...", r);
    let page = io::load_page(prefix, r, max_total)?;

    eprintln!("Exporting to {}...", output);
    io::save_page_json(&page, output)?;

    eprintln!("Done.");
    Ok(())
}

fn cmd_info(prefix: &str, r: i32, max_total: i32) -> Result<(), Box<dyn std::error::Error>> {
    let page = io::load_page(prefix, r, max_total)?;

    println!("Page: E_{}", r);
    println!("Max total degree: {}", max_total);
    println!(
        "Max values: n={:?}, s={:?}, f={:?}, t={:?}",
        page.max_n, page.max_s, page.max_f, page.max_t
    );

    let nonzero = page.dimension.values().filter(|&&d| d > 0).count();
    let total_dim: usize = page.dimension.values().sum();
    println!("Tridegrees with elements: {}", nonzero);
    println!("Total dimension: {}", total_dim);
    println!("Products: {}", page.products.len());

    for kind in ehp_core::map::MapKind::all() {
        if let Some(map_table) = page.maps.get(&kind) {
            let count = map_table.matrices.len();
            println!("{} map: {} source tridegrees", kind.name(), count);
        }
    }

    if !page.names.is_empty() {
        println!("Named elements: {}", page.names.len());
    }

    // Show dimension distribution
    let mut by_total: StdHashMap<i32, usize> = StdHashMap::new();
    for (&t, &d) in &page.dimension {
        if d > 0 {
            *by_total.entry(t.total()).or_default() += d;
        }
    }
    let mut totals: Vec<i32> = by_total.keys().copied().collect();
    totals.sort();
    println!("\nDimension by total degree (s+f):");
    for t in totals {
        println!("  t={}: dim={}", t, by_total[&t]);
    }

    Ok(())
}

fn cmd_convert(prefix: &str, r: i32, max_total: i32, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    eprintln!("Loading E_{} page from CSV...", r);
    let page = io::load_from_csv(prefix, r, max_total)?;

    let nonzero = page.dimension.values().filter(|&&d| d > 0).count();
    eprintln!(
        "  {} tridegrees, {} product blocks",
        nonzero,
        page.products.num_blocks()
    );

    eprintln!("Saving binary to {}...", output);
    io::save_to_binary(&page, output)?;

    let elapsed = t0.elapsed();
    eprintln!("Done ({:.2}s)", elapsed.as_secs_f64());

    Ok(())
}
