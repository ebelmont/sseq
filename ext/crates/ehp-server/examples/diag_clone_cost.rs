//! Micro-benchmark: which field of SATPage makes `clone()` slow?
//! Run: `EHP_MAX_T=50 cargo run -p ehp-server --release --example diag_clone_cost`

use ehp_core::io;
use std::time::Instant;

const DEFAULT_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/E2");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = std::env::var("EHP_DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let max_t: i32 = std::env::var("EHP_MAX_T").ok().and_then(|s| s.parse().ok()).unwrap_or(50);

    eprintln!("Loading E_2 (max_t={max_t})...");
    let page = io::load_page(&data, 2, max_t)?;
    eprintln!(
        "page entries: {}, dims: {}, product blocks: {}, pairs keys: {}, names: {}",
        page.page.len(),
        page.dimension.len(),
        page.products.num_blocks(),
        page.pairs.len(),
        page.names.len(),
    );
    let pairs_total: usize = page.pairs.values().map(|v| v.len()).sum();
    let maps_total: usize = page.maps.values().map(|mt| mt.matrices.len()).sum();
    eprintln!("pairs total entries: {}, map matrices: {}", pairs_total, maps_total);

    let t = Instant::now();
    let c1 = page.page.clone();
    eprintln!("page.page clone:      {:.3}s ({} entries)", t.elapsed().as_secs_f64(), c1.len());

    let t = Instant::now();
    let c2 = page.dimension.clone();
    eprintln!("dimension clone:      {:.3}s ({} entries)", t.elapsed().as_secs_f64(), c2.len());

    let t = Instant::now();
    let c3 = page.products.clone();
    eprintln!("products clone:       {:.3}s ({} blocks)", t.elapsed().as_secs_f64(), c3.num_blocks());

    let t = Instant::now();
    let c4 = page.maps.clone();
    eprintln!("maps clone:           {:.3}s ({} tables)", t.elapsed().as_secs_f64(), c4.len());

    let t = Instant::now();
    let c5 = page.pairs.clone();
    eprintln!("pairs clone:          {:.3}s ({} keys)", t.elapsed().as_secs_f64(), c5.len());

    let t = Instant::now();
    let c6 = page.names.clone();
    eprintln!("names clone:          {:.3}s ({} entries)", t.elapsed().as_secs_f64(), c6.len());

    let t = Instant::now();
    let c7 = page.clone();
    eprintln!("FULL SATPage clone:   {:.3}s", t.elapsed().as_secs_f64());
    drop(c7);

    Ok(())
}
