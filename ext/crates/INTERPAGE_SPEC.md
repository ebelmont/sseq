# Specification: `run_interpage.py` — Interpage differential propagation

This document specifies what `run_interpage.py` and the modules it uses
(`ehp_sat/interpage.py`, `ehp_sat/overlay_ss.py`, plus the incremental parts of
`ehp_sat/sat_result.py`) do. It is written for a reader who already fully
understands the vanilla algorithm executed by `run.py`: the SAT encoding of
d_r matrix entries as variables `(n, s, f, i, j)` (i = target basis index,
j = source basis index), the naturality constraints for E/H/P/lh0, the
Leibniz (Ytilde) constraints, the CryptoMiniSat solve producing
`(offset, unknown, AE)`, the exclude-set mechanism, and page turning with
induced products/maps. Vanilla concepts are referenced freely and not
re-explained.

## 1. Purpose

The vanilla pipeline answers: *given the E_r input data plus outside
differentials, which d_r entries are forced?* — one page at a time, each page a
full constraint generation + full SAT solve + full page turn.

`run_interpage.py` answers a different question: *given a **completed** vanilla
run (E2 through E5 already solved and cached on disk), what are the downstream
consequences of asserting a handful of additional differential values?* It
propagates those hypotheses **across pages** — an injected d_2 value can force
d_3, d_4, d_5 values — and it does so **incrementally**: rather than re-running
constraint generation and SAT solving from scratch, it

1. updates the cached solution-space description of each page by pure linear
   algebra over GF(2), plus a single satisfiability check per page, and
2. re-turns pages only at the (usually few) tridegrees that the new information
   affects, using lazy "overlay" pages instead of rebuilding E_{r+1} data.

Everything is done in memory. No cached files are modified; the loaded
`SATSpectralSequence` object is treated as read-only, so many independent
trials can be run against one loaded state (this is what makes the
parallel trial-and-error loop, §6, possible).

The two termination behaviors are the whole point:

* **Consistent:** the run completes and prints every differential (on any page
  up to `max_page`) that became *newly determined* as a consequence of the
  injected values.
* **Contradiction:** `UnsatException` (the augmented SAT system for some page
  is unsatisfiable) or `d2Exception` (a fully-determined composite d∘d is
  nonzero during page turning). A contradiction proves the injected assignment
  is impossible, i.e. the opposite value is a theorem relative to the vanilla
  constraint set + outside diffs.

## 2. Prerequisites: cached artifacts consumed

A complete vanilla run must exist under `save_dir`. `load()` refuses to start
unless `E{r}_data/` exists for every r in 2..max_page (default `max_page=5`).
The artifacts actually consumed per page r are the ones `run.py` writes:

| File in `E{r}_data/` | Used for |
|---|---|
| `diffs`, `offset`, `AE`, `unknown` | reconstructing the page's `SATResult` without solving (`load_computation`) |
| `{E,H,P,lh0}_constraints`, `Y_constraints` | `main_constraints()` returns these from cache instead of regenerating |
| `excluded_Yconstraints` | the Leibniz constraints the vanilla run *skipped* because a participating degree was excluded — replayed here when that degree gets unexcluded |
| `unknown_degrees`, `exclude_list` | rebuilding each page's exclude set on load |

plus the page-level input data `E{r}/E{r}_rank.csv`, `_relations.csv`,
`_{E,H,P,lh0}.csv`, `_names.json` for pages 3..max_page (written by the
vanilla page turns), and the E2 input from `input_dir`. The `outside_diffs/`
directories are re-read every time `main_constraints` is called, exactly as in
vanilla, so outside differentials are always part of the "old" constraints.

## 3. Public API

`run_interpage.py` has no `main`; per the README it is meant to be
`attach()`-ed in an interactive Sage session so the (expensive) load happens
once and many trials run against it.

### 3.1 `load(input_dir, save_dir, total_deg, max_page=5) -> SATSpectralSequence`

Reconstructs the full multi-page state of a finished run, entirely from cache:

* Loads the E2 page from `input_dir` with `max_t = total_deg` and calls
  `compute(recompute=False)`, which short-circuits into `load_computation()`
  (deserializing `SATResult` from `offset`/`AE`/`unknown`/`diffs`).
* For r = 3..max_page: `next_page(recompute=False)` (which loads
  `save_dir/E{r}/` from disk and rebuilds the exclude set from the previous
  page's `unknown_degrees` + `exclude_list`), sets the same shrinking cutoff
  schedule run.py uses — `max_t = total_deg − Σ_{i=3..r}(i−2)` — and again
  loads the cached `SATResult`.

After `load`, `ss.pages[2..max_page]` each hold: the page data (dimensions,
basis, products, map tables), the exclude set, and a `SATResult` `ss.pages[r].d`
describing everything the vanilla run knew about the d_r's.

### 3.2 `try_diffs(ss, filename, return_info=False, max_page=5)`

File-driven front end. `filename` has the `outside_diffs` line format
`r, n, s, f, i, j, val`. All lines must share the same `r` (enforced; the
implementation currently only supports injecting on a single starting page).
Parses into `[((n,s,f,i,j), val), ...]` and delegates to
`try_diffs_from_list(ss, r, ...)`.

### 3.3 `try_diffs_from_list(ss, r_start, diffs, max_page=5, return_info=False)`

The core routine; see §4. `diffs` is a list of `((n, s, f, i, j), value)`
pairs asserting d_{r_start} matrix entries. Prints all newly learned
differentials as `(r, (n, s, f, target_index, source_index), value)`. With
`return_info=True` it additionally returns three dicts keyed by page number:
`all_dsat[r]` (the updated `SATResult` per page), `all_overlay[r]` (the
`OverlaySATPage` per page; for `r_start` itself the base page), and
`all_new_constraints[r]` (the incremental constraints added on that page) —
for interactive inspection.

### 3.4 `trial_error_loop(ss, r, output_filename, max_page=5, min_stem=0, max_stem=70)`

Exhaustive contradiction search over one page's unknowns; see §6.

## 4. Core algorithm (`try_diffs_from_list`)

### 4.1 Stage 0 — injecting on page r_start

Each injected pair `((n,s,f,i,j), val)` is translated through
`ss.pages[r_start].diffs_rev` into a unit constraint `((var_idx,), val)`.
The page's full vanilla constraint set is re-obtained via
`main_constraints(added_constraints={})` — which is cheap here because every
component is read from the `E{r}_data` cache (naturality per map, Y
constraints, then outside/known constraints). Then

```python
learned_diffs, dsat = update_sat_result(page.d, page.dimension,
                                        old_constraints, new_constraints, [])
```

produces an updated `SATResult` and the list of d_{r_start} entries that the
injection newly forces.

**`update_sat_result(dsat, dimensions, old_constraints, new_constraints, new_diffs)`**
(`interpage.py`) is the incremental replacement for `compute_sat`. Recall the
vanilla representation: `dsat.matrix` (the echelonized `AE`) has rows spanning
the *homogeneous* solution space K = ker(A) of the constraint matrix A over
GF(2); a variable is *determined* iff its column in AE is zero; `offset` is one
particular solution of the inhomogeneous system; determined variables take
their offset values. `update_sat_result` computes the same data for the system
extended by k new variables (`new_diffs`) and the rows of `new_constraints`,
without touching A itself:

* **Kernel update (pure linear algebra, no SAT).** Let N = number of existing
  variables, K = `dsat.matrix`. New variables are appended after index N, so
  build `Kaug` = K padded with k zero columns, stacked with the unit vectors
  e_{N+1},…,e_{N+k} (a new, initially unconstrained variable contributes a free
  direction). Rows of Kaug span the homogeneous solutions of the old system
  viewed in the enlarged variable space. Let B be the (homogeneous parts of
  the) new constraints as a sparse matrix. The new homogeneous solution space
  is {v ∈ rowspace(Kaug) : Bv = 0}; writing v = Kaugᵀy, it is spanned by the
  rows of `U * Kaug` where U is a basis of ker(B·Kaugᵀ). Because
  dim(rowspace(Kaug)) — the number of still-free variables plus k — is
  typically far smaller than N, this is cheap.
* **Newly determined variables.** Only columns indexed by
  `dsat.unknown ∪ {N,…,N+k−1}` can change status; a column of the new kernel
  matrix that is identically zero on that index set marks a newly determined
  variable (`find_zero_cols` restricted to those indices).
* **Particular solution.** One call to `solve_sat_inhomogeneous` (a *single*
  CryptoMiniSat solve — no per-variable probing as in vanilla
  `solve_sat_cms`) on `old_constraints + new_constraints` yields the new
  `offset`, or `None` ⇒ **`UnsatException`** ⇒ the injected values are
  contradictory. The value of each newly determined variable is read off the
  offset (1 iff its index is in the offset support), printed as
  `found: d_r(n, s, f, i, j) = v`.
* **Result.** A fresh `SATResult` with merged `diffs_rev` (new variables get
  indices N, N+1, … in list order — this ordering contract is shared with
  `OverlaySATPage.diffs_rev`, §5), the new offset, the shrunken unknown set,
  and the new kernel matrix. The input `dsat` is not mutated.

`record_learned` then evaluates the updated differential matrices
(`dsat(n,s,f)[i,j]`) at each learned entry and accumulates
`(r, (n,s,f,i,j), value)` into the global result list.

### 4.2 Stages r_start+1 .. max_page — localized page turning

For each subsequent page `r`, with `prev_page` the (overlay) version of page
r−1 and `dsat` the updated `SATResult` for d_{r−1}, three steps run:

**Step A — `turn_page_local(prev_page, target_page, dsat)`: find what got
unexcluded and re-turn the page only there.**

Recompute the page-r exclude set from the *updated* unknown set:
`new_exclude_set = _make_exclude_set(r, dsat.unknown_degrees(), prev_page.exclude_set)`.
Since the update only ever removes unknowns, the new exclude set is a subset of
the vanilla one;

```
unexclude_list = target_page.exclude_set − new_exclude_set
```

is exactly the set of tridegrees that were opaque in the vanilla run but whose
incoming and outgoing d_{r−1}'s are now fully determined. (By construction of
`_make_exclude_set`, a degree is outside the exclude set iff both its incoming
and outgoing differentials on the previous page are known — so everything
below is well-defined.)

For each unexcluded `(n,s,f)`:

* `dsat.get_tb(n,s,f, cache)` computes the `TurnedBidegree` (homology basis,
  quotient map, boundary space B) exactly as `_turn_page_single` does in
  vanilla page turning, building the in/out d_{r−1} matrices from the updated
  `SATResult` (zero matrices substituted for out-of-range degrees). If both
  composable differentials are fully known and d∘d ≠ 0, **`d2Exception`** is
  raised — the second contradiction channel.
* The homology dimension is recorded in `new_dims[(n,s,f)]`. If `n = s+2`
  (the stable edge), the same dimension is copied to every past-stable copy
  `(n', s, f)`, `n' > s+2`, present on the previous page — past-stable degrees
  are never in exclude lists (exclusion of them is checked via folding in
  `is_excluded`) but their page data still has to be turned.

Then, on a temporary overlay page carrying these new dimensions, the candidate
**new d_r variables** are enumerated: for each unexcluded `(n,s,f)`,
`make_basis_single` is called for differentials *out of* `(n,s,f)`, *into*
`(n,s,f)` (i.e. sourced at `(n, s+1, f−r)`), and — at the stable edge — out of
`(n+1, s+1, f−r)`. `make_basis_single` is passed the unexclude list so the
usual "skip excluded degrees" filter treats these degrees as live; it still
requires source and target to be in bounds, non-excluded (post-shrink), and of
nonzero dimension. The union is deduplicated. These become the k new SAT
variables of §4.1's kernel-augmentation step.

**Step B — `OverlaySATPage`: a virtual, lazily-corrected E_r page** (§5).

**Step C — `make_new_constraints(...)`: generate only the constraints that
mention an unexcluded degree.**

Two families, mirroring the two vanilla constraint generators:

* *Naturality.* For every unexcluded `(n,s,f)` and every map in
  `{E, H, P, lh0}`, `make_constraints_map_single(overlay, ·, r, map_name)` is
  invoked at four tridegrees: the degree itself, the source of the
  differential into it `(n, s+1, f−r)`, the map-preimage of the degree
  (`source_degree`; for P, `target_degree` — the P convention swap noted in
  the code, since `make_constraints_map_single(·, deg)` means "the square whose
  P-*target* is deg"), and that preimage's differential source. Together these
  four calls cover every naturality square in which the unexcluded degree
  appears as any corner. Squares that still touch an excluded degree are
  skipped by the usual guard inside `make_constraints_map_single` (evaluated
  against the *shrunken* overlay exclude set), i.e. deferred, not lost. The
  overlay's cutoff is set to `source_page.max_t − (r_prev − 1)`, matching the
  vanilla cutoff schedule for page r.
* *Leibniz replay.* The vanilla run recorded, per excluded degree, every
  product pair `(deg1, deg2)` whose Leibniz constraint it skipped
  (`excluded_Yconstraints`, loaded from the target page's cache). For each
  unexcluded degree, exactly those recorded pairs are re-fed to
  `make_leibniz_constraints_single` on the overlay page. Exceptions from
  individual pairs are caught and logged, not fatal. (Known gap, flagged
  `FIXME` in the code: pairs recorded under a *past-stable* key `n > s+2` are
  never replayed, because `unexclude_list` only contains folded degrees
  `n ≤ s+2` while the recorded keys are unfolded.)

All generated relations are deduplicated (order-preserving) and homogenized to
`(clause, 0)` — new information on page r carries no inhomogeneous part; the
inhomogeneity lives entirely in the injected unit clauses on page r_start and
in each page's cached known/outside constraints.

**Step D — incremental solve on page r.** Same call as stage 0, but now with
new variables:

```python
constraints = ss.pages[r].main_constraints(added_constraints={})   # cached vanilla set
learned, dsat = update_sat_result(ss.pages[r].d, curr_page.dimension,
                                  constraints, new_constraints, new_diffs)
```

The first argument is the page's *vanilla* `SATResult` (its kernel matrix
`AE`), extended per §4.1 by the `new_diffs` variables and the Step-C
constraints; the dimensions dict is the overlay's (so the new degrees resolve).
Learned entries are recorded, and `prev_page = curr_page` hands the overlay to
the next iteration, so page r+1 is turned relative to the corrected page r.

Finally the accumulated `all_diffs` list is printed. Nothing on disk changes.

## 5. `OverlaySATPage` and `OverlayMapping` (`overlay_ss.py`)

`OverlayMapping(base, extra)` is a two-layer read-only dict view: lookups hit
`extra` first, then `base`; iteration is over the key union.

`OverlaySATPage(base_page, prev_page, prev_d_result, extra_diffs,
extra_dimensions, recompute_list)` subclasses `SATPage` but holds no data of
its own; it represents "the base E_r page as corrected by the new page-(r−1)
information." Its contract (stated in its docstring): `prev_page` together
with `prev_d_result` needs no further modification — true because new
information never changes anything the vanilla run already computed (§7).

* `__getattr__` delegates every unshimmed attribute to `base` (including
  `max_t`, unless explicitly assigned on the overlay, and `maps`, `d`, etc.).
* `dimension` and `page` are `OverlayMapping`s adding the unexcluded degrees
  (with fresh standard-basis `Element`s of the new dimensions).
* `diffs_rev` appends the `extra_diffs` after the base variables, at indices
  `len(base.diffs_rev) + 0, 1, …` in list order — the same convention
  `update_sat_result` uses, which is what makes constraint generation on the
  overlay produce clauses over the augmented variable space consistently.
* `exclude_set` is the shrunken `_make_exclude_set(r, prev_d_result.unknown_degrees(), prev_page.exclude_set)`.
* `products` is booby-trapped (raises); all product access must go through
  `multiply`.
* `multiply(x, y)` and `apply_map(name, elt)`: if none of the involved degrees
  is in `recompute_list` (checked with stable folding), delegate to the base
  page's stored tables. Otherwise the value is computed **on demand** by
  page-turning arithmetic on the *previous* page: `prev_d_result.recompute_product`
  / `.recompute_map` build the relevant `TurnedBidegree`s via `get_tb`, lift
  basis elements to the previous page, multiply / apply the map there, reduce
  modulo boundaries, and project to the new homology basis — i.e. exactly what
  vanilla `compute_induced_products` / `compute_induced_map` do globally, but
  restricted to one degree and cached (`product_cache`, `map_cache`,
  `tb_cache`). A summand whose value cannot be produced is tolerated silently
  iff the degree is (still) excluded; otherwise it is an error.

## 6. `trial_error_loop`: exhaustive contradiction search

`trial_error_loop(ss, r, output_filename, max_page=5, min_stem=0, max_stem=70)`
turns the contradiction channel into a differential-discovery engine:

1. Collect every still-unknown d_r variable `(n,s,f,i,j)` of page r with stem
   `min_stem ≤ s < max_stem` (the `min_stem` bound exists so the search can be
   run in windowed batches).
2. For each such variable and each value in {0, 1}, run
   `try_diffs_from_list(ss, r, [(diff, value)], max_page)` in a worker process.
3. If the trial raises `UnsatException` or `d2Exception`, append the line
   `"(n, s, f, i, j) value"` to `output_filename` — meaning *this assignment is
   refuted*, so `d_r(n,s,f,i,j)` provably equals the opposite value (these are
   the "differentials learned using [trial and error]" that feed back into the
   dataset as outside diffs). A consistent trial writes nothing. Progress
   (`trying d_r(...) = v (count = k)`) goes to `output_filename + ".log"`.

Parallelization details: the loaded `ss` is published through the module
global `_SS` and shared with workers via `fork` (`ProcessPoolExecutor` with
the fork context); `gc.collect()` + `gc.freeze()` run first so copy-on-write
pages aren't dirtied by the collector in children (the memory-usage fix from
the recent commit history). This sharing is only sound because
`try_diffs_from_list` never mutates `ss` — every trial builds its own
`SATResult`s and overlays.

Note that if *both* values of some variable are refuted, the base data itself
is inconsistent; the loop does not special-case this — both lines simply
appear in the output file.

## 7. Correctness invariants

* **Soundness of learned differentials.** Every constraint used (cached
  vanilla constraints, replayed Leibniz constraints, incrementally generated
  naturality constraints) is a valid consequence of the spectral-sequence
  axioms given the injected values; `update_sat_result` computes exactly the
  determined variables of the union system. So each printed differential is a
  logical consequence of {vanilla input data + outside diffs + injected
  values}.
* **Vanilla results never change, only extend.** New information consists
  solely of *values* for previously-unknown d variables. Any degree that was
  non-excluded in the vanilla run had all incident differentials determined,
  so its turned dimension, induced products, and induced maps were computed
  from fully-known data and remain valid. Hence corrections are confined to
  previously-excluded degrees — precisely `unexclude_list` — which is why the
  overlay can delegate everything else to the base page, and why
  "`prev_page` + `prev_d_result` needs no modification."
* **Monotone exclude sets.** Unknown sets only shrink, so
  `new_exclude_set ⊆ old exclude_set` on every page and `unexclude_list` is
  well-defined; constraints touching still-excluded degrees are skipped now
  exactly as they were in vanilla, so no invalid constraint is ever emitted.
* **Isolation.** No cached file is written and `ss` is not mutated; trials are
  independent and repeatable (the README example: asserting a value and then
  its opposite in two successive trials is *not* a contradiction).

## 8. Known limitations

* All injected differentials must live on a single starting page
  (`try_diffs` enforces it; `try_diffs_from_list` takes one `r_start`).
* Propagation stops at `max_page` (default 5) and requires cached data for
  every page 2..max_page regardless of `r_start`.
* The `FIXME` in `make_new_constraints`: Leibniz constraints recorded under
  past-stable degree keys (`n > s+2`) are not replayed on unexclusion, so a
  (presumably small) class of derivable constraints is currently missed —
  propagation is sound but not maximally complete.
* `main_constraints` is re-read from disk on every trial (and in every
  worker), which is redundant I/O but harmless.
* Learned values are reported by printing (and, in `trial_error_loop`, by
  appended output files); `try_diffs_from_list` returns data only with
  `return_info=True`, and nothing is persisted automatically — feeding learned
  differentials back into a run (e.g. as `outside_diffs`) is a manual step.
