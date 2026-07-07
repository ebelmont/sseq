//! Deep-dive the E_4 step of interpage propagation for the S^6 (34,10) d_3:
//! replay try_diffs' first iteration by hand and report, for every naturality
//! square touching the un-excluded degrees, whether it produced constraints
//! and (if not) which check killed it.
//!
//! Run: `EHP_MAX_T=50 cargo run -p ehp-server --release --example diag_interpage_deep`

use ehp_core::constraints::{
    self, make_naturality_constraint_single, make_leibniz_constraint_single, DiffVar,
    ExcludedLeibniz,
};
use ehp_core::interpage::{
    build_overlay_page, collect_new_vars, turn_page_local, update_sat_result,
};
use ehp_core::map::MapKind;
use ehp_core::page::SATPage;
use ehp_core::pageturning::{self, TurnedBidegree};
use ehp_core::result::SATResult;
use ehp_core::tridegree::Tridegree;
use ehp_core::{io, solver};
use hashbrown::HashMap;

const DEFAULT_DATA: &str = concat!(env!("HOME"), "/ehp-sat-rs/data/E2.ehp");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let data = std::env::var("EHP_DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let max_t: i32 = std::env::var("EHP_MAX_T").ok().and_then(|s| s.parse().ok()).unwrap_or(50);

    let assumed = DiffVar::new(6, 34, 10, 0, 0);
    let assumed_val = false;

    eprintln!("Loading E_2 (max_t={max_t})...");
    let e2 = io::load_page(&data, 2, max_t)?;
    // solve_and_turn solves the given page and turns to the next:
    // res2 is E2's solve result, e3/t2 the turned page/data, etc.
    let (e3, _res2, _t2) = solve_and_turn(&e2)?;
    let (e4, res3, _t3) = solve_and_turn(&e3)?;

    let sys4 = {
        let known = ehp_server::load_known_diffs(4, None)?;
        constraints::build_constraint_system(&e4, e4.max_s.unwrap_or(0), &known)
    };
    let excl_leib4: ExcludedLeibniz = sys4.excluded_leibniz.clone();
    let res4 = solver::solve(&sys4).expect("E4 SAT");
    eprintln!(
        "E_4: {}/{} determined, {} excluded degrees, {} excluded-leibniz keys",
        sys4.num_vars - res4.unknown.len(),
        sys4.num_vars,
        e4.exclude_set.len(),
        excl_leib4.len()
    );

    // ---- Step 1: update the E3 result with the assumption ----
    let idx = *res3.var_index.get(&assumed).expect("assumed var exists");
    let (learned3, dsat) =
        update_sat_result(&res3, &[(vec![idx], assumed_val)], &[]).expect("E3 consistent");
    eprintln!("\nE_3 learned after assumption ({}):", learned3.len());
    for (v, val) in &learned3 {
        eprintln!("  d_3({},{},{})[{},{}] = {}", v.n, v.s, v.f, v.row, v.col, *val as u8);
    }

    // ---- Step 2: local turn E3 -> E4 ----
    let lt0 = turn_page_local(&e3, &e4, &dsat, None).expect("no d2 error");
    eprintln!("\nun-excluded degrees on E_4 ({}):", lt0.unexclude.len());
    for t in &lt0.unexclude {
        eprintln!(
            "  ({},{},{}): base dim {} -> new dim {}",
            t.n, t.s, t.f,
            e4.dim_at(*t),
            lt0.new_dims.get(t).copied().unwrap_or(0)
        );
    }

    // ---- Step 3: overlay + new vars ----
    let mut lt = lt0;
    let overlay0 = build_overlay_page(&e3, &e4, &lt, &dsat, None, None).expect("no d2 error");
    lt.new_vars = collect_new_vars(&overlay0, &res4, &lt.unexclude);
    eprintln!("\nnew E_4 vars ({}):", lt.new_vars.len());
    for v in &lt.new_vars {
        eprintln!("  d_4({},{},{})[{},{}]", v.n, v.s, v.f, v.row, v.col);
    }

    let old_n = res4.vars.len();
    let mut ext_index = res4.var_index.clone();
    for (j, v) in lt.new_vars.iter().enumerate() {
        ext_index.insert(*v, old_n + j);
    }

    // ---- Step 4: probe every naturality square ----
    let mut overlay = overlay0;
    overlay.max_t = e3.max_t.map(|t| t - (e3.r - 1));
    eprintln!(
        "\noverlay max_t reduced to {:?} (base E4 max_t {:?})",
        overlay.max_t, e4.max_t
    );

    let r1 = overlay.r;
    let name = |v: &DiffVar, i: usize| {
        let known = if i >= old_n {
            "NEWVAR".to_string()
        } else if res4.unknown.contains(&i) {
            "unknown".to_string()
        } else {
            format!("={}", ehp_core::gf2::vec_get(&res4.offset, i) as u8)
        };
        format!("d_4({},{},{})[{},{}]{}", v.n, v.s, v.f, v.row, v.col, known)
    };
    let var_of = |i: usize| -> DiffVar {
        if i < old_n { res4.vars[i] } else { lt.new_vars[i - old_n] }
    };

    for &t in &lt.unexclude {
        for kind in MapKind::all() {
            let deg = t;
            let diff_deg = Tridegree::new(t.n, t.s + 1, t.f - r1);
            let source_deg = match kind {
                MapKind::P => kind.target_degree(t),
                _ => kind.source_degree(t),
            };
            let source_diff_deg =
                Tridegree::new(source_deg.n, source_deg.s + 1, source_deg.f - r1);
            for d in [deg, diff_deg, source_deg, source_diff_deg] {
                let cons = make_naturality_constraint_single(&overlay, d, kind, &ext_index);
                if cons.is_empty() {
                    eprintln!(
                        "SKIP {} square at ({},{},{}): {}",
                        kind.name(), d.n, d.s, d.f,
                        skip_reason(&overlay, d, kind)
                    );
                } else {
                    eprintln!(
                        "OK   {} square at ({},{},{}): {} constraints",
                        kind.name(), d.n, d.s, d.f, cons.len()
                    );
                    for c in &cons {
                        let terms: Vec<String> =
                            c.iter().map(|&i| { let v = var_of(i); name(&v, i) }).collect();
                        eprintln!("     0 = {}", terms.join(" + "));
                    }
                }
            }
        }
    }

    // ---- Step 5: Leibniz replay ----
    eprintln!("\nLeibniz replay:");
    for &t in &lt.unexclude {
        match excl_leib4.get(&t) {
            None => eprintln!("  ({},{},{}): no recorded pairs", t.n, t.s, t.f),
            Some(pairs) => {
                eprintln!("  ({},{},{}): {} recorded pairs", t.n, t.s, t.f, pairs.len());
                for &(d1, d2) in pairs {
                    let cons =
                        make_leibniz_constraint_single(&overlay, d1, d2, &ext_index, None);
                    eprintln!(
                        "    pair ({},{},{}) x ({},{},{}): {} constraints (E matrix at deg2: {})",
                        d1.n, d1.s, d1.f, d2.n, d2.s, d2.f,
                        cons.len(),
                        overlay.maps[&MapKind::E].matrix_at(d2).is_some(),
                    );
                    for c in &cons {
                        let terms: Vec<String> =
                            c.iter().map(|&i| { let v = var_of(i); name(&v, i) }).collect();
                        eprintln!("     0 = {}", terms.join(" + "));
                    }
                }
            }
        }
    }

    Ok(())
}

fn skip_reason(page: &SATPage, t: Tridegree, kind: MapKind) -> String {
    let r = page.r;
    if !kind.domain_check(t) {
        return "domain check".into();
    }
    if kind == MapKind::E && t.n >= t.s + 2 {
        return "E stable".into();
    }
    let (src, diff_src, tgt) = match kind {
        MapKind::P => {
            if t.f - 2 < 0 || t.s - t.n < 0 {
                return "P bounds".into();
            }
            (
                Tridegree::new(2 * t.n + 1, t.s - t.n + 1, t.f - 2),
                Tridegree::new(2 * t.n + 1, t.s - t.n, t.f - 2 + r),
                t,
            )
        }
        _ => (t, t.diff_target(r), kind.target_degree(t)),
    };
    let diff_tgt = tgt.diff_target(r);
    for &td in &[src, diff_src, tgt, diff_tgt] {
        if page.is_excluded(td) {
            return format!("corner ({},{},{}) excluded", td.n, td.s, td.f);
        }
        if !page.is_in_computed_polygon_source(td) || !page.is_in_computed_polygon(td) {
            return format!("corner ({},{},{}) out of polygon", td.n, td.s, td.f);
        }
    }
    let dims = [
        page.dim_at(src),
        page.dim_at(diff_src),
        page.dim_at(tgt),
        page.dim_at(diff_tgt),
    ];
    if dims.iter().any(|&d| d == 0) {
        return format!(
            "zero dim (src {} d_src {} tgt {} d_tgt {})",
            dims[0], dims[1], dims[2], dims[3]
        );
    }
    if page.maps[&kind].matrix_at(src).is_none() {
        return format!(
            "map matrix UNSTORED at src ({},{},{}) — default applies",
            src.n, src.s, src.f
        );
    }
    if page.maps[&kind].matrix_at(diff_src).is_none() {
        return format!(
            "map matrix UNSTORED at diff_src ({},{},{}) — default applies",
            diff_src.n, diff_src.s, diff_src.f
        );
    }
    let ps = page.map_matrix(kind, src);
    let pt = page.map_matrix(kind, diff_src);
    if ps.columns() != dims[2] || ps.rows() != dims[0] {
        return format!(
            "phi_source dims {}x{} vs src {} tgt {}",
            ps.rows(), ps.columns(), dims[0], dims[2]
        );
    }
    if pt.columns() != dims[3] || pt.rows() != dims[1] {
        return format!(
            "phi_target dims {}x{} vs diff_src {} diff_tgt {}",
            pt.rows(), pt.columns(), dims[1], dims[3]
        );
    }
    "constraints degenerate (all-empty entries)".into()
}

fn solve_and_turn(
    page: &SATPage,
) -> Result<(SATPage, SATResult, HashMap<Tridegree, TurnedBidegree>), Box<dyn std::error::Error>> {
    let known = ehp_server::load_known_diffs(page.r, None)?;
    let sys = constraints::build_constraint_system(page, page.max_s.unwrap_or(0), &known);
    let res = solver::solve(&sys).expect("SAT");
    eprintln!(
        "E_{}: {}/{} determined",
        page.r,
        sys.num_vars - res.unknown.len(),
        sys.num_vars
    );
    let (next, turned) = pageturning::build_next_page(page, &res)?;
    Ok((next, res, turned))
}
