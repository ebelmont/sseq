use clap::Parser;

use ehp_server::ServerState;

#[derive(Parser)]
#[command(name = "ehp-server")]
#[command(about = "Interactive EHP spectral sequence viewer")]
struct Cli {
    /// Path to data directory, file prefix, or .ehp file
    #[arg(short, long)]
    prefix: String,

    /// Page number r (default: 2)
    #[arg(short, long, default_value = "2")]
    r: i32,

    /// Maximum total degree s+f
    #[arg(short = 't', long)]
    max_total: i32,

    /// WebSocket port (default: 8765)
    #[arg(long, default_value = "8765")]
    port: u16,

    /// HTTP port for serving the viewer page (default: 8080)
    #[arg(long, default_value = "8080")]
    http_port: u16,

    /// Path to SeqSee directory (containing main.py)
    #[arg(long, default_value = "~/seqsee/seqsee")]
    seqsee_dir: String,

    /// CSV file of known differentials
    #[arg(short = 'k', long)]
    known_diffs: Option<String>,

    /// Don't auto-open browser
    #[arg(long)]
    no_open: bool,

    /// Logging verbosity: -v for info, -vv for debug
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let log_level = match cli.verbose {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level)).init();

    // Expand ~ in seqsee_dir
    let seqsee_dir = if cli.seqsee_dir.starts_with("~/") {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{}{}", home, &cli.seqsee_dir[1..])
    } else {
        cli.seqsee_dir.clone()
    };

    let state = ServerState::new(
        &cli.prefix,
        cli.r,
        cli.max_total,
        cli.known_diffs.as_deref(),
        &seqsee_dir,
    )?;

    ehp_server::run_server(state, cli.http_port, cli.port, !cli.no_open).await
}
