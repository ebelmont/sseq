use ehp_core::constraints;
use ehp_core::io;
use ehp_core::map::MapKind;
use ehp_core::page::SATPage;
use ehp_core::tridegree::Tridegree;

const DEFAULT_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/E2");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = std::env::var("EHP_DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let max_t: i32 = 80;
    let page: SATPage = io::load_page(&data, 2, max_t)?;

    let args: Vec<String> = std::env::args().collect();
    let (n, s, f) = if args.len() >= 4 {
        (args[1].parse().unwrap(), args[2].parse().unwrap(), args[3].parse().unwrap())
    } else {
        (10, 49, 17)
    };
    let t = Tridegree::new(n, s, f);
    let kind = MapKind::E;
    println!("t = {:?}", t);
    println!("domain_check: {}", kind.domain_check(t));
    println!("stable skip (n >= s+2): {}", t.n >= t.s + 2);
    let tgt = kind.target_degree(t);
    let diff_src = t.diff_target(2);
    let diff_tgt = tgt.diff_target(2);
    println!("src={:?} diff_src={:?} tgt={:?} diff_tgt={:?}", t, diff_src, tgt, diff_tgt);
    println!("max_t={:?}", page.max_t);
    println!("bound check: src.s+src.f+r-1={} tgt.s+tgt.f+r-1={}", t.s+t.f+2-1, tgt.s+tgt.f+2-1);
    for &td in &[t, diff_src, tgt, diff_tgt] {
        println!(
            "  td={:?} excluded={} poly_source={} poly={} dim={}",
            td, page.is_excluded(td), page.is_in_computed_polygon_source(td), page.is_in_computed_polygon(td), page.dim_at(td)
        );
    }

    let system = constraints::build_constraint_system(&page, page.max_t.unwrap_or(0), &Default::default());
    let out = constraints::make_naturality_constraint_single(&page, t, kind, &system.var_index);
    println!("constraint result: {:?}", out);

    Ok(())
}
