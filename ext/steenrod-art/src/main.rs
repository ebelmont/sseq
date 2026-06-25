use steenrod_art::{module, layout, crossing, render, sixel};
mod examples;

use std::path::Path;

use anyhow::Result;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // Parse flags
    let mut output_mode = OutputMode::Png;
    let mut rail_strategy = render::RailStrategy::FixedByOrder;
    let mut output_path: Option<String> = None;
    let mut input_files: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--output=sixel" || arg == "--sixel" {
            output_mode = OutputMode::Sixel;
        } else if arg == "--output=png" || arg == "--png" {
            output_mode = OutputMode::Png;
        } else if arg == "--rails=fixed" {
            rail_strategy = render::RailStrategy::FixedByOrder;
        } else if arg == "--rails=enclosure" {
            rail_strategy = render::RailStrategy::EnclosureAware;
        } else if arg.starts_with("--output-file=") || arg.starts_with("-o=") {
            output_path = Some(arg.splitn(2, '=').nth(1).unwrap().to_string());
        } else if arg == "-o" {
            i += 1;
            if i < args.len() {
                output_path = Some(args[i].clone());
            }
        } else if arg == "--help" || arg == "-h" {
            print_usage();
            return Ok(());
        } else if !arg.starts_with('-') {
            input_files.push(arg.clone());
        } else {
            eprintln!("Unknown flag: {arg}");
            print_usage();
            std::process::exit(1);
        }
        i += 1;
    }

    if input_files.is_empty() {
        eprintln!("No input files specified. Using built-in Joker module.");
        let m = examples::joker();
        render_module(&m, output_mode, rail_strategy, output_path.as_deref())?;
        return Ok(());
    }

    for file in &input_files {
        let path = Path::new(file);
        match module::load_module(path) {
            Ok(m) => {
                let out_path = output_path.as_deref().unwrap_or_else(|| {
                    // Default: replace .json extension with .png
                    ""
                });
                let effective_path = if out_path.is_empty() {
                    let stem = path.file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| "output".to_string());
                    Some(format!("{stem}.png"))
                } else {
                    Some(out_path.to_string())
                };
                render_module(&m, output_mode, rail_strategy, effective_path.as_deref())?;
            }
            Err(e) => {
                eprintln!("Error loading {}: {e:#}", path.display());
            }
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum OutputMode {
    Png,
    Sixel,
}

fn render_module(
    module: &module::Module,
    output_mode: OutputMode,
    rail_strategy: render::RailStrategy,
    output_path: Option<&str>,
) -> Result<()> {
    let layout_config = layout::LayoutConfig::default();
    let render_config = render::RenderConfig::default();

    // 1. Compute layout
    let layout_result = layout::compute_layout(module, &layout_config);

    // 2. Assign sides (crossing minimization)
    let assignment = crossing::assign_sides(module, &layout_result);

    eprintln!(
        "{}: {} generators, {} edges, {} arcs, bipartite={}, residual={}",
        module.name,
        module.generators.len(),
        module.edges.len(),
        module.edges.iter().filter(|e| e.op_degree >= 2).count(),
        assignment.bipartite,
        assignment.residual,
    );

    // 3. Render to Pixmap
    let pixmap = render::render_diagram(
        module,
        &layout_result,
        &assignment.sides,
        &render_config,
        rail_strategy,
    );

    // 4. Output
    match output_mode {
        OutputMode::Png => {
            let path = output_path.unwrap_or("output.png");
            pixmap.save_png(path)?;
            eprintln!("Saved PNG to {path}");
        }
        OutputMode::Sixel => {
            let sixel_data = sixel::pixmap_to_sixel(&pixmap);
            use std::io::Write;
            std::io::stdout().write_all(&sixel_data)?;
            std::io::stdout().flush()?;
        }
    }

    Ok(())
}

fn print_usage() {
    eprintln!("Usage: steenrod-art [OPTIONS] [MODULE.json ...]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --output=png        Output PNG file (default)");
    eprintln!("  --output=sixel      Output Sixel inline graphics");
    eprintln!("  --rails=fixed       Fixed rail depth by operation order (default)");
    eprintln!("  --rails=enclosure   Compact enclosure-aware rail depth");
    eprintln!("  -o FILE             Output file path (for PNG)");
    eprintln!("  -h, --help          Show this help");
}
