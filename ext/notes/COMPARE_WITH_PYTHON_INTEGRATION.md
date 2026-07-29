# Integration of `compare_with_python` bug fixes (2026-07-29)

Consolidation of the 14 commits on `eva/compare_with_python` (fork branch,
merge-base `0a00b2443`) into `secondary-steenrod-changes`. Each commit was
classified against the current tree (which had diverged: uf solver rewrite,
interpage propagation, lh0 maps, snapshot system) and integrated, adapted, or
skipped as below. `CACHE_VERSION` was bumped 3 → 4: the fixes change solve
semantics, so every cached page chain from before them is stale.

## Integrated verbatim

| commit | fix | where |
|---|---|---|
| `c40ebc2a3` | Remove bogus `domain_check` gate from P-map naturality constraint generation (Python never wires those lambdas; the gate silently dropped valid P constraints) | `constraints.rs` `make_naturality_constraint_single` |
| `6242ac9a8` | Remove zero-dimension early bailout in naturality generation (a zero-dim target still forces real constraints on source differentials) | `constraints.rs` `make_naturality_constraint_single` |
| `23b86864e` | Crop `turn_page` tridegrees to the old page's own max_t (over-extended dimension tables propagated; caused 4372 E4 rank mismatches at s+f=80) | `pageturning.rs` `turn_page` |
| `d3cee3e70` | Per-page max_t decrement is r−1 (Python's 80→79→77→74 schedule), not a flat −1 | `pageturning.rs` `build_page_from_turned` + `build_next_page` |
| `b024447f0` | Remove spurious shifted-degree filter from `compute_induced_products` | `pageturning.rs` |
| `ca707acfd` | Product bound is the flat `page.max_t − 1` of the OLD page, not `next_page.max_t` (11430 real E4 products were excluded) | `pageturning.rs` `compute_induced_products` |
| `f50471282` | `TurnedBidegree::is_cycle` guard before `quotient()` in induced maps and products (non-cycle vectors previously produced basis-dependent phantom values) | `pageturning.rs` |
| `5c7aed8a2` | `verify_ehp.sh` Rust-vs-Python harness (`ext/scripts/*` + `diag_dump_page_tower.rs`) | new files |

## Integrated with changes

| commit | fix | what changed and why |
|---|---|---|
| `62dd87ab6` | Two Leibniz pair-enumeration bugs: missing `s2 < 1` skip; bogus n-derived `max_data_n` bounds (max_t bounds t=s+f only, never n) | Our loop is rayon-parallel with documented byte-for-byte deterministic output; Eva iterates the `HashMap` directly (unordered). Integrated with **sorted-key iteration** (`th1_keys`) to keep determinism. |
| `2c32c155e` | Remove n-based bound (`x.n > max_s`) from product-triple enumeration | Applied to all three sites (`pageturning.rs`, `page.rs` `build_pairs`, `interpage.rs` overlay pre-filter). In `interpage.rs` we went **further than Eva's tree**: her overlay pre-filter still had the shifted-degree + polygon checks that `b024447f0`/`ca707acfd` removed from the full enumeration, violating its own "identical filters" invariant — ours now matches the fixed enumeration (flat `source_page.max_t − 1` bound). |
| `136bd32d3` | `mat_from_rows` takes `&[FpVector]` instead of owned `Vec` (halves peak memory on the big call sites) | Same signature change; our tree has different/more call sites (uf solver, interpage) — all 14 updated, clones/`to_vec()` removed where present. |
| `e2d263311` | Collapse page bounds to a single max_t cutoff | **Semantic part only**: all *gating* on `max_n`/`max_s`/`max_f` is gone (see `2c32c155e` row). The struct fields and the `.ehp` binary header keep all four values — our io/cache format stores them and removing them would churn the format for no behavioral gain. |

## Already present / superseded (not integrated)

| commit | why skipped |
|---|---|
| `691defbfc` | Sparse GF(2) solver to fix t=100 OOM — subsumed. Our tree already stores constraints sparsely (`Vec<Box<[u32]>>`) and the default `uf` union-find solver keeps sparsity through the solve; Eva's `sparse_gauss_solve` + `sparse-bin-mat` dependency would be a strictly inferior addition. |
| `c715bdd27` | Chart batch-generation speedup — the Rust side (fused CSV scans in `seqsee.rs`) is already present in our tree; the Python side evolved differently here (ehp_batch.py has since grown a backwards-compat fallback). No behavioral gap. |

## Follow-ups

- Run the vendored harness to validate against Python at scale:
  `ext/scripts/verify_ehp.sh` (drives `diag_dump_page_tower` + the compare
  scripts; see script header for usage). SAT at t=20 was verified after
  integration (all pages solve, all vars determined).
- All `.ehp_cache` entries are invalidated by the version bump — the first
  t=100 startup after this will cold-solve and regenerate charts, and results
  (ranks, products, determined diffs) are EXPECTED to differ from the July 21
  session, because these fixes change the constraint system.
