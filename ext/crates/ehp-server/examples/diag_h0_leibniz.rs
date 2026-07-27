//! Diagnose why d2 vars at S46 (45,8) and (46,8) are not forced to zero by
//! h0-Leibniz, and what E3 thinks about the d3 at S46 (44,10).
//!
//! Run: EHP_MAX_T=<t> cargo run -p ehp-server --release --example diag_h0_leibniz

use ehp_core::constraints::{self, make_leibniz_constraint_single, DiffVar};
use ehp_core::gf2::*;
use ehp_core::io;
use ehp_core::map::MapKind;
use ehp_core::pageturning;
use ehp_core::solver;
use ehp_core::tridegree::Tridegree;

const DEFAULT_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/E2");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let data = std::env::var("EHP_DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let max_t: i32 = std::env::var("EHP_MAX_T").ok().and_then(|s| s.parse().ok()).unwrap_or(80);

    eprintln!("Loading E_2 (max_t={max_t})...");
    let e2 = io::load_page(&data, 2, max_t)?;
    let known = ehp_server::load_known_diffs(2, None)?;
    let cutoff = e2.max_t.unwrap_or(0);
    eprintln!("cutoff (max_t) = {cutoff}");

    let sys = constraints::build_constraint_system(&e2, cutoff, &known);
    let res = solver::solve(&sys).expect("E_2 should be SAT");

    let vars_of_interest = [
        DiffVar::new(46, 45, 8, 0, 0),
        DiffVar::new(46, 46, 8, 0, 0),
    ];

    for dv in &vars_of_interest {
        eprintln!("\n=== d_2({},{},{})[{},{}] ===", dv.n, dv.s, dv.f, dv.row, dv.col);
        let Some(&idx) = sys.var_index.get(dv) else {
            eprintln!("  NOT A VARIABLE — make_basis skipped it. dims: src={} tgt={}",
                e2.dim_at(dv.tridegree()),
                e2.dim_at(dv.tridegree().diff_target(2)));
            continue;
        };
        let unknown = res.unknown.contains(&idx);
        eprintln!(
            "  var #{idx}: {} (offset bit {})",
            if unknown { "UNKNOWN" } else { "determined" },
            vec_get(&res.offset, idx) as u8,
        );

        // Constraints in the full system touching this var
        let mut touching = 0;
        let mut sizes = Vec::new();
        for (i, row) in sys.rows.iter().enumerate() {
            if vec_get(row, idx) {
                touching += 1;
                if sizes.len() < 12 {
                    sizes.push((i, vec_popcount(row), sys.rhs[i]));
                }
            }
        }
        eprintln!("  constraints touching it: {touching}; first few (row, support, rhs): {sizes:?}");

        // Candidate h0-style Leibniz pairs: deg2 = var degree, deg1 with
        // th1 + s1 = deg2.n + 1.
        let deg2 = dv.tridegree();
        eprintln!("  -- candidate Leibniz pairs with deg2 = {:?} (th1+s1 = {}) --", deg2, deg2.n + 1);
        let mut candidates: Vec<Tridegree> = e2
            .dimension
            .iter()
            .filter(|(&t, &d)| d > 0 && t.f >= 1 && t.n + t.s == deg2.n + 1 && t.n >= 2)
            .map(|(&t, _)| t)
            .collect();
        candidates.sort();
        eprintln!("  {} candidates", candidates.len());

        for deg1 in candidates {
            let prod_deg = Tridegree::new(deg1.n, deg1.s + deg2.s, deg1.f + deg2.f);
            let prod_dr = prod_deg.diff_target(2);
            let e_deg2 = Tridegree::new(deg2.n + 1, deg2.s, deg2.f);
            let d_deg2 = deg2.diff_target(2);
            let e_target = Tridegree::new(deg2.n + 1, deg2.s - 1, deg2.f + 2);

            let mut notes: Vec<String> = Vec::new();
            if deg1.s + deg1.f > cutoff || prod_deg.s + prod_deg.f > cutoff {
                notes.push(format!("beyond cutoff (s3+f3={})", prod_deg.s + prod_deg.f));
            }
            if !e2.is_in_computed_polygon_source(prod_deg)
                || !e2.is_in_computed_polygon_source(prod_dr)
            {
                notes.push("prod/prod_dr out of polygon".into());
            }
            if e2.dim_at(prod_dr) == 0 {
                notes.push("prod_dr dim 0".into());
            }
            if e2.maps[&MapKind::E].matrix_at(deg2).is_none() {
                notes.push("E MATRIX UNSTORED at deg2 (zero/identity default applies)".into());
            }
            if e2.maps[&MapKind::E].matrix_at(d_deg2).is_none() && e2.dim_at(e_target) > 0 {
                notes.push("E MATRIX UNSTORED at d(deg2)".into());
            }
            // product of deg1 basis with E(deg2 basis col)
            let e_mat = e2.map_matrix(MapKind::E, deg2);
            let mut prod_all_zero = true;
            for i1 in 0..e2.dim_at(deg1) {
                let e_v = mat_vec_mul(&e_mat, &vec_basis(e2.dim_at(deg2), dv.col as usize));
                let p = e2.products.multiply(
                    deg1,
                    &vec_basis(e2.dim_at(deg1), i1),
                    e_deg2,
                    &e_v,
                    e2.dim_at(prod_deg),
                );
                if !p.is_zero() {
                    prod_all_zero = false;
                }
            }
            // Ytilde(deg1, e_target basis): rhs1 products
            let mut ytil_zero = true;
            for i1 in 0..e2.dim_at(deg1) {
                for j in 0..e2.dim_at(e_target) {
                    let p = e2.products.multiply(
                        deg1,
                        &vec_basis(e2.dim_at(deg1), i1),
                        e_target,
                        &vec_basis(e2.dim_at(e_target), j),
                        e2.dim_at(prod_dr),
                    );
                    if !p.is_zero() {
                        ytil_zero = false;
                    }
                }
            }

            let cons = make_leibniz_constraint_single(&e2, deg1, deg2, &sys.var_index, None);
            let with_var = cons.iter().filter(|c| c.contains(&idx)).count();
            eprintln!(
                "  deg1={:?} dims(d1={},prod={},prod_dr={},e_tgt={}) prod_zero={} ytil1_zero={} constraints={} with_var={} {}",
                deg1,
                e2.dim_at(deg1),
                e2.dim_at(prod_deg),
                e2.dim_at(prod_dr),
                e2.dim_at(e_target),
                prod_all_zero,
                ytil_zero,
                cons.len(),
                with_var,
                notes.join("; "),
            );
        }
    }

    // The stable-fold context: d2 sources feeding (n,44,10) for n = 46, 47.
    eprintln!("\n=== d_2 sources feeding (n,44,10) ===");
    for src in [Tridegree::new(46, 45, 8), Tridegree::new(47, 45, 8)] {
        let dv = DiffVar::new(src.n.min(src.s + 2), src.s, src.f, 0, 0);
        match sys.var_index.get(&dv) {
            Some(&i) => eprintln!(
                "  d_2{:?}: var #{} {} (offset {})",
                src,
                i,
                if res.unknown.contains(&i) { "UNKNOWN" } else { "determined" },
                vec_get(&res.offset, i) as u8,
            ),
            None => eprintln!(
                "  d_2{:?}: no variable (src dim {}, tgt dim {})",
                src,
                e2.dim_at(src),
                e2.dim_at(src.diff_target(2)),
            ),
        }
    }

    // E3 status of the d3 at (44,10) on S46 (the rep) and S47 (stable copy)
    eprintln!("\n=== E_3 at (44,10) ===");
    let (e3, _t) = pageturning::build_next_page(&e2, &res).expect("d^2 = 0 on E_2");
    for n in [46, 47] {
        let t3 = Tridegree::new(n, 44, 10);
        eprintln!(
            "  S{}: dim = {}, is_excluded = {} (folds to rep (46,44,10)), target (43,13) dim = {}, target excluded = {}",
            n,
            e3.dim_at(t3),
            e3.is_excluded(t3),
            e3.dim_at(t3.diff_target(3)),
            e3.is_excluded(t3.diff_target(3)),
        );
    }
    let known3 = ehp_server::load_known_diffs(3, None)?;
    let cutoff3 = e3.max_t.unwrap_or(0);
    let sys3 = constraints::build_constraint_system(&e3, cutoff3, &known3);
    let dv3 = DiffVar::new(46, 44, 10, 0, 0);
    match sys3.var_index.get(&dv3) {
        Some(&i3) => {
            let res3 = solver::solve(&sys3);
            match res3 {
                Some(r3) => eprintln!(
                    "  d_3 var exists (#{}): {} (offset {})",
                    i3,
                    if r3.unknown.contains(&i3) { "UNKNOWN" } else { "determined" },
                    vec_get(&r3.offset, i3) as u8,
                ),
                None => eprintln!("  E_3 UNSAT"),
            }
        }
        None => eprintln!(
            "  d_3(46,44,10) is NOT a variable on E_3 (excluded or zero dims) — \
             chart should show it dashed via the exclusion fallback",
        ),
    }

    Ok(())
}
