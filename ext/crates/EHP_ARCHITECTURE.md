# EHP Tooling — Architecture Overview

Quick orientation for the EHP spectral-sequence tooling that lives in this
workspace (`sseq/ext`). This is a Rust port of an older Python implementation.

## What it does

Computes the EHP spectral sequence page by page. Each page `E_r` is a set of
classes indexed by tridegree `(n, s, f)`. Differentials `d_r` are unknowns; the
tool encodes the constraints they must satisfy (naturality, Leibniz, known
seeds) as **linear equations over GF(2)**, solves them, then "turns the page"
(takes homology) to get `E_{r+1}`. Results are exported as CSV/HTML charts.

- Differential geometry: `d_r : (n, s, f) → (n, s-1, f+r)` (see `Tridegree::diff_target`).
- Solving is plain Gaussian elimination over GF(2) (no external SAT solver).

## Crates (workspace `crates/`)

- **`ehp-core`** — the math engine (pages, constraints, solver, page turning, IO, chart export).
- **`ehp-server`** — driver + chart generation. `examples/ehp_chart.rs` is the interactive REPL binary; `src/lib.rs` has `load_known_diffs` and SeqSee/HTML glue.
- **`ehp-cli`** — command-line entry (`src/main.rs`).
- Supporting library crates: `fp` (GF(2) vectors/matrices), `algebra`, `sseq`, `bivec`, etc.

## `ehp-core` modules

| file | role |
|------|------|
| `tridegree.rs` | `Tridegree {n,s,f}`; `diff_target(r)`, `diff_source(r)` |
| `page.rs` | `SATPage`: dimensions, basis, products, maps, polygon bounds (`max_n/s/f/t`, `is_in_computed_polygon_source`, `compute_max_values`) |
| `products.rs` | `ProductTable`: block-keyed `(deg1,deg2)` product matrices. `multiply` returns 0 for an **absent** block (careful: absent ≠ genuine zero) |
| `map.rs` | `MapKind::{E,H,P}` structure maps |
| `constraints.rs` | builds the GF(2) system: `make_naturality_constraints`, `make_leibniz_constraints` / `..._single`, `make_known_constraints`, `build_constraint_system`. `DiffVar {n,s,f,row,col}` is one differential unknown |
| `solver.rs` | `solve()` → `Option<SATResult>` (None = inconsistent/UNSAT) |
| `result.rs` | `SATResult`: `offset` (particular solution), `unknown` (free vars), `kernel`; `diff_matrix(t)` reconstructs a differential (treats **unknown vars as 0**) |
| `pageturning.rs` | `turn_page` / `turn_page_single` (homology `H = ker(d_out)/im(d_in)`), `compute_induced_products`, `compute_induced_maps`, `build_next_page` / `build_page_from_turned` |
| `gf2.rs` | GF(2) linear algebra helpers (`gauss_image` = **column space**, `gauss_right_kernel`, `mat_transpose`, …) |
| `io.rs` | binary `.ehp` + CSV loaders (`load_page`, `load_from_binary`, `load_known_diffs`) |
| `seqsee.rs` | CSV/JSON chart export (`write_ehp_csv`; columns incl. `drinfo`/`drtarget` = determined diffs, `nulldif` = unknown/dashed) |

## Pipeline (in `ehp_chart.rs`)

```
load E2 (io::load_page)
loop:
  build_constraint_system  →  solver::solve
  if UNSAT or no vars → stop            # UNSAT here means no next page
  build_next_page (turn homology, induce products/maps) → E_{r+1}
export CSVs → generate SeqSee charts → REPL
```

Stalls at a page iff that page is UNSAT — trace by checking which constraint
forces the collision (isolate naturality vs Leibniz, then the specific pair).

## Data & the original reference

- Input page: `~/ehp-sat-rs/data/E2.ehp` (binary), generated from CSVs in `~/ehp-sat-rs/data/E2/` (`E2_relations.csv` = products, `E2_rank.csv`, `E2_{E,H,P}.csv`). Product tables store only **nonzero** products (absent row = zero).
- Original Python implementation (ground truth for logic): `~/EHP_SAT/old_sat/` (`SAT_helper.py` `make_constraints_Y` = Leibniz, `uASS.py` `multiply` folds an E-suspension `extra_susp = th1+s1-th2` into each product lookup). Newer Python: `~/EHP_SAT/ehp_sat/`, `~/ehpreprint/python/`.

## Gotchas learned

- **`gauss_image(A)` already returns the column space of `A`** (it transposes internally). To get boundaries `im(d_in)`, call `gauss_image(d_in)` — an extra `mat_transpose` computes the *row* space and silently drops incoming differentials from the homology quotient. (This was the E₃-turning bug.)
- **Uncertain degrees must be excluded when turning.** `diff_matrix` treats **unknown** (undetermined) differentials as zero (result.rs), so `turn_page` mis-computes homology wherever a bounding differential is unknown, and those degrees' products become spuriously nonzero → false Leibniz forcings (e.g. E₃ UNSAT only at large `max_t`, where more E₂ diffs are unknown). Fix: `build_next_page` populates `next.exclude_set` via `make_next_exclude_set` (ports the original `_make_exclude_set` in `~/EHP_SAT/ehp_sat/sat_ss.py`): for each unknown `d_r` at `(n,s,f)`, exclude both source `(n,s,f)` and target `(n,s-1,f+r)`, carrying prior exclusions forward along incoming diffs. Constraint generation (`make_basis`, naturality, Leibniz) skips excluded degrees. Seeds on non-excluded degrees are still enforced.
- Induced products are reduced modulo boundaries in `compute_induced_products_single` — so a correct `boundary_basis` (from a correct `im(d_in)`) is what makes products landing in killed classes vanish on the next page.
- `products.multiply` and `result.diff_matrix` both return 0 for missing/unknown data; when a product/differential "is zero," confirm it's a genuine zero vs. absent block vs. undetermined var.
- Leibniz uses the Ỹtilde product with a built-in suspension; the second factor is stored at `n₂ = n₁ + s₁`. The Rust reproduces the suspension with explicit `map_matrix('E')` compositions.
- See the `EHP Chart REPL` notes in `ext/CLAUDE.md` for REPL commands and chart conventions.
