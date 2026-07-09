//! One-off: dump the E2 constraint system's rows (as DiffVar lists + rhs)
//! so fold-on vs fold-off runs can be diffed. Rows printed sorted.
use ehp_core::{constraints, io};

const DEFAULT_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/E2");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = std::env::var("EHP_DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let max_t: i32 = std::env::var("EHP_MAX_T").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    let page = io::load_page(&data, 2, max_t)?;
    let known = ehp_server::load_known_diffs_for_page(&page, None)?;
    let cutoff = page.max_s.unwrap_or(0);
    let sys = constraints::build_constraint_system(&page, cutoff, &known);
    let mut lines: Vec<String> = Vec::new();
    for (i, row) in sys.rows.iter().enumerate() {
        let terms: Vec<String> = row
            .iter()
            .map(|&j| {
                let v = &sys.vars[j as usize];
                format!("d({},{},{})[{},{}]", v.n, v.s, v.f, v.row, v.col)
            })
            .collect();
        lines.push(format!("{} = {}", terms.join(" + "), sys.rhs[i] as u8));
    }
    lines.sort();
    for l in &lines {
        println!("{}", l);
    }
    Ok(())
}
