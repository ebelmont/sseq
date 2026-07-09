//! Stage-C0 quantification: where does the RAM go, and how does the
//! constraint system decompose?
//!
//! Builds the page chain exactly like diag_startup (same env: EHP_DATA,
//! EHP_MAX_T, EHP_MAX_R, EHP_OUTSIDE_DIFFS, ...) and reports per page:
//!   - variables, rows, nnz, row-support histogram;
//!   - dense-row bytes (what `ConstraintSystem.rows` costs today) vs sparse
//!     projection (4 B/nnz);
//!   - connected components of the constraint graph (union-find over row
//!     supports): count, and the top components by variable count — this
//!     decides whether component-wise solving kills the solve-matrix peak;
//!   - kernel dimensions after the solve (kernel rows × num_vars bits);
//!   - product/map table sizes (blocks, data words) for RAM attribution.

use std::time::Instant;

use ehp_core::{constraints, io, pageturning, solver};

const DEFAULT_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/E2");

struct UnionFind {
    parent: Vec<usize>,
}
impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
        }
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let data_path = std::env::var("EHP_DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let max_t: i32 = std::env::var("EHP_MAX_T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let max_r: i32 = std::env::var("EHP_MAX_R")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    eprintln!("diag_system_stats: data={} max_t={} max_r={}", data_path, max_t, max_r);
    let t0 = Instant::now();
    let mut current_page = io::load_page(&data_path, 2, max_t)?;
    eprintln!("loaded E_2 in {:.2}s", t0.elapsed().as_secs_f64());

    loop {
        let r = current_page.r;
        let known_diffs = ehp_server::load_known_diffs_for_page(&current_page, None)?;
        let cutoff = current_page.max_s.unwrap_or(0);
        let t1 = Instant::now();
        let system = constraints::build_constraint_system(&current_page, cutoff, &known_diffs);
        let build_s = t1.elapsed().as_secs_f64();

        let nvars = system.num_vars;
        let nrows = system.rows.len();
        let mut nnz = 0usize;
        let mut hist = [0usize; 9]; // support 0..=7, 8 = 8+
        let mut uf = UnionFind::new(nvars);
        for row in &system.rows {
            let supp: Vec<usize> = row.iter().map(|&j| j as usize).collect();
            nnz += supp.len();
            hist[supp.len().min(8)] += 1;
            for w in supp.windows(2) {
                uf.union(w[0], w[1]);
            }
        }
        // Component sizes (variables per root), counting only vars in rows.
        let mut comp_vars: hashbrown::HashMap<usize, usize> = hashbrown::HashMap::new();
        for v in 0..nvars {
            let root = uf.find(v);
            *comp_vars.entry(root).or_insert(0) += 1;
        }
        let mut sizes: Vec<usize> = comp_vars.values().copied().collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        let singletons = sizes.iter().filter(|&&s| s == 1).count();

        let dense_bytes = nrows * nvars.div_ceil(8);
        let sparse_bytes = nnz * 4 + nrows * 16;

        eprintln!("== E_{} ==", r);
        eprintln!(
            "  vars {} | rows {} | nnz {} | build {:.2}s",
            nvars, nrows, nnz, build_s,
        );
        eprintln!("  row-support histogram (8=8+): {:?}", hist);
        eprintln!(
            "  rows RAM: dense {:.1} MB -> sparse {:.1} MB",
            dense_bytes as f64 / 1e6,
            sparse_bytes as f64 / 1e6,
        );
        eprintln!(
            "  components: {} total ({} singleton vars); top sizes: {:?}",
            sizes.len(),
            singletons,
            &sizes[..sizes.len().min(10)],
        );

        let t2 = Instant::now();
        let result = solver::solve_with_d2(&current_page, &system).map(|(res, _)| res);
        let solve_s = t2.elapsed().as_secs_f64();
        match result {
            Some(ref res) => {
                let kr = res.kernel.rows();
                eprintln!(
                    "  solve {:.2}s | determined {}/{} | kernel {} x {} = {:.1} MB",
                    solve_s,
                    nvars - res.unknown.len(),
                    nvars,
                    kr,
                    nvars,
                    (kr * nvars.div_ceil(8)) as f64 / 1e6,
                );
            }
            None => {
                eprintln!("  solve {:.2}s: INCONSISTENT or no vars", solve_s);
                break;
            }
        }

        // Product/map table attribution.
        let mut blocks = 0usize;
        let mut block_rows = 0usize;
        for (_, pm) in current_page.products.iter_blocks() {
            blocks += 1;
            block_rows += pm.rows();
        }
        eprintln!(
            "  products: {} blocks, {} rows (~{:.1} MB incl. overhead) | maps: {} entries",
            blocks,
            block_rows,
            (blocks * 100 + block_rows * 8) as f64 / 1e6,
            current_page
                .maps
                .values()
                .map(|mt| mt.matrices.len())
                .sum::<usize>(),
        );

        if r >= max_r {
            break;
        }
        let result = result.unwrap();
        let (next, _turned) = match pageturning::build_next_page(&current_page, &result) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("turn E_{} failed: {}", r, e);
                break;
            }
        };
        current_page = next;
    }
    eprintln!("total {:.2}s", t0.elapsed().as_secs_f64());
    Ok(())
}
