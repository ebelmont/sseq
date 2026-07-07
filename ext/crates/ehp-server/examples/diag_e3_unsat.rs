//! Find why E_3 WITH the seed is still UNSAT at high total degree, after the
//! exclude-uncertain-degrees fix. Traces the Leibniz pair(s) that single-term
//! force the seed d_3(17,15,2)=0 and reports, for the product's landing degree,
//! whether it is excluded and the status of the E_2 differentials that bound it.
//!
//! Run: `EHP_MAX_T=90 cargo run -p ehp-server --release --example diag_e3_unsat`

use ehp_core::constraints::{self, DiffVar, make_basis, make_leibniz_constraint_single};
use ehp_core::page::SATPage;
use ehp_core::result::SATResult;
use ehp_core::tridegree::Tridegree;
use ehp_core::{io, pageturning, solver};
use hashbrown::HashMap;

const DEFAULT_DATA: &str = concat!(env!("HOME"), "/ehp-sat-rs/data/E2.ehp");
const SEED: DiffVar = DiffVar { n: 17, s: 15, f: 2, row: 0, col: 0 };

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("error")).init();
    let data = std::env::var("EHP_DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let max_t: i32 = std::env::var("EHP_MAX_T").ok().and_then(|s| s.parse().ok()).unwrap_or(90);

    eprintln!("Loading E_2 (max_t={max_t})...");
    let e2 = io::load_page(&data, 2, max_t)?;
    let e2_known = ehp_server::load_known_diffs(2, None)?;
    let c2 = e2.max_s.unwrap_or(0);
    let sys2 = constraints::build_constraint_system(&e2, c2, &e2_known);
    let res2: SATResult = solver::solve(&sys2).expect("E_2 SAT");
    eprintln!("E_2 solved: {} unknown vars", res2.unknown.len());

    let (e3, _t) = pageturning::build_next_page(&e2, &res2).expect("E_2 d^2 = 0");
    let cutoff = e3.max_s.unwrap_or(0);
    eprintln!("E_3 turned (cutoff={cutoff}, exclude_set={}).\n", e3.exclude_set.len());

    let vars = make_basis(&e3, cutoff);
    let var_index: HashMap<DiffVar, usize> =
        vars.iter().enumerate().map(|(i, v)| (*v, i)).collect();
    let Some(&seed_idx) = var_index.get(&SEED) else {
        eprintln!("seed not a variable (excluded?) — different problem"); return Ok(());
    };
    let y = SEED.tridegree(); // (17,15,2)

    // Only pairs where deg1==Y or deg2==Y can put the seed var into a Leibniz
    // relation. th2 = th1 + s1 - 1.
    //  - deg2 == Y  => th1 + s1 == Y.n + 1
    //  - deg1 == Y  => deg2.n == Y.n + Y.s - 1
    let mut candidates: Vec<(Tridegree, Tridegree)> = Vec::new();
    for (&t, &d) in &e3.dimension {
        if d == 0 { continue; }
        if t.f >= 1 && t.n + t.s == y.n + 1 {
            candidates.push((t, y));                       // deg1=t, deg2=Y
        }
        if t.n == y.n + y.s - 1 {
            candidates.push((y, t));                       // deg1=Y, deg2=t
        }
    }
    eprintln!("Scanning {} candidate Leibniz pairs touching the seed...\n", candidates.len());

    let r = e3.r;
    let mut found = 0;
    for (deg1, deg2) in candidates {
        for cvec in make_leibniz_constraint_single(&e3, deg1, deg2, &var_index, None) {
            if cvec.len() == 1 && cvec[0] == seed_idx {
                found += 1;
                let prod = Tridegree::new(deg1.n, deg1.s + deg2.s, deg1.f + deg2.f);
                let prod_dr = prod.diff_target(r);
                eprintln!("FORCE #{found}: deg1={:?} deg2={:?}", deg1, deg2);
                eprintln!("  prod {:?} (E2 {}, E3 {}, excluded {})",
                    prod, e2.dim_at(prod), e3.dim_at(prod), e3.is_excluded(prod));
                eprintln!("  prod_dr {:?} (E2 {}, E3 {}, excluded {})",
                    prod_dr, e2.dim_at(prod_dr), e3.dim_at(prod_dr), e3.is_excluded(prod_dr));
                d2_status(&e2, &res2, prod, "prod");
                d2_status(&e2, &res2, prod_dr, "prod_dr");
                if found >= 8 { break; }
            }
        }
        if found >= 8 { break; }
    }
    if found == 0 {
        eprintln!("No single-term forcing among seed-touching pairs — forcing is multi-term.");
        eprintln!("(Would need to inspect the combined nullspace.)");
    }
    Ok(())
}

/// Report the E_2 d_2 differentials bounding degree `deg` on E_3: the outgoing
/// d_2 (deg as source) and the incoming d_2 (from (n,s+1,f-2)). Flags UNKNOWN.
fn d2_status(_e2: &SATPage, res2: &SATResult, deg: Tridegree, label: &str) {
    let check = |t: Tridegree, kind: &str| {
        let n_var = t.n.min(t.s + 2);
        let mut unknown = false; let mut det_nonzero = false; let mut present = false;
        for v in &res2.vars {
            if v.n == n_var && v.s == t.s && v.f == t.f {
                present = true;
                if let Some(&i) = res2.var_index.get(v) {
                    if res2.unknown.contains(&i) { unknown = true; }
                    else if ehp_core::gf2::vec_get(&res2.offset, i) { det_nonzero = true; }
                }
            }
        }
        if present {
            eprintln!("    {label} {kind} d2 src {:?}: unknown={unknown} det_nonzero={det_nonzero}{}",
                t, if unknown { "  <-- UNKNOWN (should have been excluded)" } else { "" });
        }
    };
    check(deg, "outgoing");                                  // deg is the source
    check(Tridegree::new(deg.n, deg.s + 1, deg.f - 2), "incoming"); // source of incoming d2
}
