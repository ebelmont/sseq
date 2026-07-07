//! Interactive EHP spectral sequence viewer.
//!
//! Loads the E2 page from binary .ehp format, solves constraints,
//! and opens an interactive chart in the browser using SeqSee rendering.
//!
//! Usage:
//!   cargo run -p ehp-server --release --example ehp_viewer
//!
//! By default looks for ~/ehp-sat-rs/data/E2.ehp. Override with EHP_DATA env var.
//! Set EHP_MAX_T to change the max total degree (default 20).
//! Set SEQSEE_DIR to point to the SeqSee directory (default ~/seqsee/seqsee).

use std::time::Instant;

use ehp_server::ServerState;

const DEFAULT_DATA: &str = concat!(env!("HOME"), "/ehp-sat-rs/data/E2.ehp");
const DEFAULT_MAX_T: i32 = 20;
const DEFAULT_SEQSEE: &str = concat!(env!("HOME"), "/seqsee/seqsee");

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let data_path = std::env::var("EHP_DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let max_t: i32 = std::env::var("EHP_MAX_T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_T);
    let r: i32 = std::env::var("EHP_R")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let seqsee_dir =
        std::env::var("SEQSEE_DIR").unwrap_or_else(|_| DEFAULT_SEQSEE.to_string());

    eprintln!("EHP Spectral Sequence Viewer");
    eprintln!("  data:    {}", data_path);
    eprintln!("  page:    E_{}", r);
    eprintln!("  max t:   {}", max_t);
    eprintln!("  seqsee:  {}", seqsee_dir);
    eprintln!();

    let t0 = Instant::now();
    let state = ServerState::new(&data_path, r, max_t, None, &seqsee_dir)?;
    eprintln!("Loaded and solved in {:.2}s", t0.elapsed().as_secs_f64());
    eprintln!();

    ehp_server::run_server(state, 8080, 8765, true).await
}
