# Session handoff — EHP tooling (July 2026 marathon session)

All EHP docs live in `ext/notes/` (this file included). Read `EHP_ARCHITECTURE.md`
first for base vocabulary, then this. `CLAUDE.md` (in `ext/`) has the per-feature
usage notes; `REFACTORING_ROADMAP.md` has the agreed modularization/generalization
plan; `INTERPAGE_SPEC.md` and `STEM_VIEW_SPEC.md` are the user-provided specs
their features were built to.

## What this session did (chronological, all committed to working tree only)

### Core math engine (`ehp-core`)
1. **E₃-UNSAT root causes fixed** (historical `EHP_E3_UNSAT_HANDOFF.md` removed July 2026 — all its issues fixed; summary here):
   - `solver.rs`: `unknown` was `free_cols` only; correlated pivot columns are
     also undetermined. Now = nonzero columns of the kernel (matches Python
     `find_zero_cols`). This bug produced fake "determined" values everywhere.
   - `pageturning.rs`: turning is **partial-matrix** based (`TurnContext`,
     `get_tb`): determined entries quotient (real boundaries/kills — user
     asserts at partially-unknown degrees take effect), unknown entries = 0
     (possibly-dead classes stay alive). d∘d≠0 with both degrees fully
     determined ⇒ `D2Error` (contradiction channel). NOTE: we deliberately
     moved OFF the Python whole-matrix-zeroing after the user hit its
     downside (added d₃ not quotienting); safety argument in the TurnContext
     doc comment.
   - `make_next_exclude_set`: stable-representative normalization of excluded
     *targets* (handoff case (a)); prior exclusions carried forward.
2. **Leibniz sphere-bound fix** (`constraints.rs`): pair loops were bounded by
   the stem cutoff; h_i pairs live at sphere th₂ = n+s−1 ≫ max_t and were all
   silently dropped (user's missing h₀/h₁ forcings at S46/S47). Bounds are now
   data-driven (`max_data_n`); **the only max_t condition is s+f ≤ cutoff** —
   the user was explicit: never bound computations by n+s or sphere vs max_t.
   UNVERIFIED by a data run yet — `diag_h0_leibniz` example exists for this
   (run with their EHP_MAX_T; prints per-pair skip reasons; suspect list if
   still broken: missing E-map matrices at the class's degree).
3. **User-asserted diffs always representable** (`add_known_diff_vars`):
   forced variable blocks at excluded degrees; `make_known_constraints` warns
   instead of silently dropping.
4. **`interpage.rs`** — full port of `~/EHP_SAT/ehp_sat/interpage.py` per
   `INTERPAGE_SPEC.md`: `update_sat_result` (kernel augmentation; offset via
   in-kernel correction — equivalent to spec's re-solve), `turn_page_local`,
   `build_overlay_page` (eager materialization), `make_new_constraints`
   (4-corner naturality incl. P anchor swap + Leibniz replay), `try_diffs`,
   `trial_error_sweep`. Deviations: no `lh0` map anywhere in the Rust
   pipeline (data has only E/H/P); the spec §8 FIXME (past-stable Leibniz
   replay keys) is FIXED here (recorded under stable reps). Unit tests in
   the module pass.
5. `excluded_leibniz` recorded on `ConstraintSystem`; `seqsee.rs` exclusion
   display fallback (excluded degree + no vars ⇒ dash to ALL potential
   targets — Python `write_spheres` behavior; prevents "excluded" reading as
   "determined zero"; also makes S46/S47 stable-fold displays consistent).

### REPL / driver (`ehp-server/examples/ehp_chart.rs`, now ~3.3k lines — split it, see roadmap)
- `PageState` (+`stem_values`, `excluded_leibniz`), `cascade_resolve` returns
  deduced diffs; stops only when values AND dims AND page content (products/
  maps/exclusions, bitwise) are unchanged (`page_content_equal`); carries
  dim-changed degrees into the next re-turn set. `push_forward` has a
  dimension guard (stale turned data ⇒ skip, not panic).
- Commands: `add/zero/toggle/remove` (validated via `check_diff_addable`),
  `;`-separated **multi-mutation batches** (one cascade), `undo`,
  `try`, `sweep`, `interpage [r]` + `propagate on|off` (deferred mode),
  `mapview <K> <n> [r]` / `mapview all [r]`, `regen`, `save`, `status`, `list`.
  `interpage` runs the cascade with `force=true`: no early stop — every page
  through the last is re-solved/re-turned/rebuilt even if a page shows no
  changes (add-cascades keep the early-stop optimization).
- Charts: cleanup of stale charts+CSVs at startup; batch generation via
  `~/seqsee/seqsee_new/ehp_batch.py` (modes: sphere/stem/sidebyside-manifest,
  one python process per core, prints `OK/FAIL id`); venv python resolved once
  (`poetry env info --executable`).
- Injected chart data (all updated in place after mutations, marker pattern):
  `DIFFDATA` (edges `[src,tgt,det,r]`, incl. prior-page unknowns pushed
  through quotient chains; per-r colors; equal thickness; dashed=unknown),
  `CLASSDIMS` (live class-count overlay: fades dead node indices AND
  structlines via `data-source`/`data-target`; display convention: last
  indices die until manual regen), `MAPDATA` (minimap + `view` availability),
  `PROPDATA` (cmd → deduced diffs; log entries expand; click navigates via
  `#diff=src;tgt` hash → highlight + pan/zoom to midpoint at 1.8×),
  `NAVDATA`/`STEMNAV` (WASD — unified convention on EVERY chart type:
  w/s = dimension-like axis −/+ [sphere n or stem k], a/d = page −/+;
  do not revert any chart type to a different mapping).
- Keys in sphere charts: click+click add (shift-click accumulates sum
  targets), Shift+M minimap, Shift+L log, e/h/p open pre-generated split
  view, Shift+E/H/P = hint only (image mode lives in split view),
  Sphere/Stem template button overridden (was a placeholder alert!) to
  prompt+navigate to stem charts. Split view: WASD siblings (w/s = source
  sphere −/+ [step 2 for P], a/d = page −/+, same as single charts),
  Shift+E/H/P/J toggle image mode. Stem charts: WASD (w/s = stem −/+,
  a/d = page −/+), titled `E_r(π^k)` (template generateTitle stem branch),
  button back to sphere view.
- Stem view per `STEM_VIEW_SPEC.md` (jsonmaker/main.py in the SHARED repo
  `~/seqsee/seqsee_new/` were edited): E column name fix ("E" not "Etarget"),
  cap n ≤ k+2, out-of-range E ⇒ 0.7 arrows, E edges inherit target d_r color,
  O(N) hit map, `diff_d{r}_open/filled` node CSS in all palettes.
- Map views pre-generated at startup (J-map style; `EHP_MAPVIEWS=0` skips).
- Themes: + `access`/`access-dark` (Okabe-Ito colorblind-safe) end to end.

## July 2026 follow-up: interpage "no results" investigation

- **Missing-map-data semantics fixed** (was the main bug family): the original
  treats unstored map matrices as the ZERO map, and E in the stable range
  (n > s+1) as the IDENTITY (`Map.matrix`/`apply_map` in lib.py/sat_ss.py).
  Rust skipped naturality squares / Leibniz pairs entirely when
  `map_matrix` returned `None`, silently dropping every `d·φ = 0`-type forcing.
  Now `SATPage::map_matrix` returns the zero/identity default; the naturality
  and Leibniz generators and `compute_induced_map_single_tb` use it (E-map
  induced entries roughly double — the stable identities now propagate).
- `matrix_mult_left/right` return `Option`: a referenced non-variable
  differential skips the whole square/pair (the original raises) instead of
  silently emitting a corrupted constraint.
- Naturality parity: added the original's `s+f+r-1 <= max_t` margin check;
  base generation iterates n only to s+2 (stable copies share the rep's data —
  with zero-default matrices, generating their squares would be wrong).
- Effect at EHP_MAX_T=50: unknowns E2 119→109, E3 345→338, E4 192→176,
  E5 53→46; `try 3 6 34 10 0 0 0` propagates (2 degrees un-exclude on E4,
  7 on E5, 13 on E6) and deduces new d_4's.
- **lh0 is intentionally absent** from the Rust pipeline (user: "we don't need
  lh0") — but note the Python data/saved runs DO include lh0 maps+constraints,
  so Python determines some values Rust legitimately won't.
- **Known-diffs gap**: Python `main_constraints` loads the accumulated
  `outside_diffs/` knowledge base (~388 constraints at E3 in jun-13-dan);
  Rust `ehp_server::load_known_diffs` seeds only 1 hardcoded diff per page
  unless given a file. Python's recorded d3(6,34,10)=1 forcing
  (jun-13-dan/outside_diffs/d3-by-contradiction.csv) came from an earlier
  sweep; re-running Python today "finds" it only circularly (UNSAT at E3
  against its own outside_diffs entry). So Rust not re-deriving that specific
  forcing at max_t=50 without the knowledge base is expected, per the user.
- Diagnostics: `diag_interpage` (build pages headless, try both values of a
  target diff, DIAG_N/S/F/R env vars) and `diag_interpage_deep` (replays the
  E4 step, prints per-square skip reasons and per-pair Leibniz replay counts).
- **`interpage try [min_stem [max_stem]]`** (REPL): automated fixpoint
  trial-and-error — sweeps every page's unknowns, applies forced values
  (known-diff + undo entry each), cascades between batches, repeats passes
  until nothing new is forced; logs to prop_log/charts and
  `output/interpage_try.log`. Note plain `interpage` (no args) only
  propagates recorded mutations — with none recorded it correctly reports 0
  new diffs (the startup solve already contains everything derivable).
  UNVERIFIED at scale (user runs it).
- **Trial cost fixed (~150×)**: a `try` was ~50s per page step; profiling
  showed 52s of it was `target_page.clone()` in `build_overlay_page` —
  fp `Matrix` carries padded tile-aligned storage, so deep-cloning ~200k
  product blocks + ~50k map matrices is gigabytes of memcpy. `ProductTable`
  and `MapTable` now hold blocks behind `Arc` (clone = refcount bumps;
  mutation copy-on-writes the single block via `Arc::make_mut`/re-insert).
  Also the overlay product recompute enumerates candidate triples directly
  from the recomputed degrees (as x, as y, or as xy) instead of the
  page-wide ~10M-pair scan. Verified at max_t=50: identical learned diffs,
  full 4-page `try` now ~0.2s (was ~2.5min) — sweeps and `interpage try`
  are now minutes, not days. `diag_clone_cost` example benchmarks the
  clone by field if this regresses.
- **Sweep panic fixed** (`mat_mul_vec` dim assert, gf2.rs): trials mixed
  quotient coordinates. A trial's newly determined differentials change the
  next page's homology at affected degrees — a different quotient space
  (different dim, and different basis even at equal dim), so all products/
  maps there must be re-expressed in the new coordinates. The port reused
  startup turned data (`pages[i-1].turned`, stock coordinates) inside
  trials and only refreshed un-excluded degrees; on the 2nd+ propagation
  step (source = overlay) that data is stale — panic when dims differ,
  silently wrong values when they don't. Now `build_overlay_page` NEVER
  consults startup turned data: every needed turned degree is computed
  fresh from the trial's updated result (`lt.tbs` is the cache — same
  TurnContext), matching the original's tb_cache-recomputes-everything
  semantics. It takes `d_result`, returns `Result` (a fresh re-turn can
  hit a genuine d∘d≠0), and `InterpagePage` lost its `turned` field so
  stale data can't be reintroduced. `diag_sweep` example = headless
  `sweep <r>` (DIAG_R/DIAG_MIN_STEM/DIAG_MAX_STEM env vars).

## Pending / unverified
- **No full pipeline run has verified**: the Leibniz bound fix, partial
  turning at scale, stem charts at scale, map-view pregeneration timing, the
  access themes, or max_t=90/120 E₃ SAT. max_t=40 was verified long ago
  (before partial turning). The user runs all long jobs themselves (memory:
  never launch them).
- User-reported, diagnosis pending their run of `diag_h0_leibniz`:
  d₂ (46,45,8)/(46,46,8) unknown (expect: fixed by Leibniz bounds; else check
  E-map matrix presence at those degrees), d₃ story at (44,10) S46/S47
  (expect: display fallback + bounds fix resolves).
- Task #3 in the task list tracks the max_t verification.

## Gotchas for the next agent
- Two SeqSee checkouts exist; the RUNTIME one is `~/seqsee/seqsee_new/`
  (found via `find_seqsee_dir`), NOT `~/SeqSee/`. Its venv python is
  `~/SeqSee/.venv/bin/python` (shared). We edited jsonmaker.py, main.py,
  ehp_batch.py there — it's a shared repo, keep edits surgical.
- Charts are file:// — no fetch/existence probing works cross-browser; the
  REPL must inject any state charts need (that's why `view` availability,
  CLASSDIMS etc. are injected rather than probed).
- `sphere-toggle`/`toggleView()` in the chart template is a placeholder that
  alerts; our injection replaces the button (cloneNode + removeAttribute
  ('onclick')).
- Injected-JS changes require regenerating charts (new binary) AND a
  hard-reload (file:// caching).
- The user is a topologist; trust their math judgments (they were right about
  the missing h₀ forcings, the s+f-only cutoff rule, and partial turning's
  necessity).
- Long-term goal (memory + `REFACTORING_ROADMAP.md`): split ehp_chart.rs,
  then genericize the engine (Degree trait, StructureMap data, product
  normalization) toward other SS diagrams and RO(G) gradings.

## Sweep/startup performance session (July 7 2026)

Goal: `sweep 3` at EHP_MAX_T=100 under 20 min (was days). Achieved ~75×+:
t=50/MAX_R=6 sweep 749.6s → 9.9s; t=70/MAX_R=7 = 2492 trials in ~minutes and
now genuinely forces d_3(30,45,7)[1,1]=0 via d∘d≠0. All changes verified by a
**parity harness**: `diag_sweep_verify` (ehp-server example) prints every
trial's full sorted learned-diff list; old-vs-new logs diffed byte-identical
(668/668 trials at t=50). Use it for ANY future change to the trial path.

What was done (all in ehp-core/src/interpage.rs unless noted):
- Profiled first (samply + `analyze_profile.py` in the session scratchpad):
  ~85% of sweep time was get_tb page-turning churn in build_overlay_page —
  work identical across trials. update_sat_result was 0.01% (kernel-copy
  fears were wrong); solving is never the bottleneck (gauss_solve 0.05%).
- `TbCache`: pre-computed (parallel, then read-only/lock-free) get_tb results
  from the stock first page + base result. Trials consult it only on the
  FIRST page step and only where neither rep(t) nor rep(t_in) is in the
  trial's dirty set (= degrees of assumed+learned vars — exactly where the
  trial result can differ; invariant adversarially verified 4-way). A lazy
  RwLock version was 3× slower under rayon contention — keep it lock-free.
- `SweepCache` = TbCache + `SourceIndex` (stable-copies by (s,f)) +
  per-target-page `PageIndex` (by_n/by_ns partner lists, keys/map-squares/
  product-blocks inverted by stable rep). Valid on every step (stock target
  pages); per-trial dims patch handled as added/removed deltas. GOTCHA: the
  partner lookups must ITERATE (iterator chain), not build filtered Vec
  copies per call — the copy version was a 2.7× whole-sweep regression at
  t=70 (invisible at t=50).
- lt.tbs / overlay tbs hold Arc<TurnedBidegree> (kills deep clones + the
  20% drop cost); product-candidate filters run BEFORE the needed/turned-
  degree collection (this filter reorder was the single biggest win; it also
  narrows incidental D2 detection to the canonical triple set — reviewed,
  accepted); `SATPage::overlay_clone()` skips pairs/names.
- Startup: `build_pairs()` calls removed everywhere (io/pageturning/
  ehp-server/ehp-cli) — `SATPage.pairs` is read by NOTHING (≈13% of startup,
  serial). Field kept but documented as unpopulated.
- **Leibniz hoisting attempt REVERTED** (constraints.rs is back to its
  pre-session-day state): correctness was fine (parity passed) but t=70
  startup went from minutes to 45+ min stuck in build_constraint_system.
  The workload is fail-fast-dominated (most pairs abort via skip() after
  ~zero work); hoisting Ytilde/matrix builds above those exits multiplies
  per-abort cost by ~10⁶⁺ pairs. Memory note ehp-perf-benchmark-at-scale.md.
  ANY perf change here must be benchmarked on a real t≥70 workload, not
  just t=50 parity.
- Diagnosing a "stuck" REPL: `sample <pid> 5` on the live process; 97%+ on
  one core = serial phase; line attribution smears under inlining — confirm
  by mechanism.

Still open: final clean t=70 sweep timing with the shipped lib (last clean
number, 611s, predates the partners-iterator fix; t=50 is 9.9s); user t=100
validation (sweep 3 + chart eyeball); startup profile of
build_page_from_turned (~14%, likely basis-clone/map-assembly serial work +
ProductTable insert hash churn) and parallelizing serial startup phases
(~36% of cores idle during them). The map-view Shift+E/H/P image-mode fix
lives in the sidebyside template at `~/SeqSee/seqsee/` (the seqsee_new path
is a symlink — shared repo, edit surgically).

## Theme refactor (July 2026, runtime-switchable themes session)

- All palettes moved to `~/seqsee/seqsee_new/themes.json` (single source of
  truth; adds `nord`, `nord-light`, `eldritch`). main.py loads it
  (`THEME_PALETTES`/`THEME_ORDER`/`DARK_THEMES` are now derived);
  ehp_chart.rs loads it via `theme_registry()` (OnceLock) — the hand-synced
  `dr_overlay_color` match arms are gone.
- Charts now emit ALL themes as `:root[data-theme=…]` CSS-variable blocks
  (`build_theme_css` in main.py); generated color classes use
  `var(--cc-<alias>)`. Runtime switch = data-theme flip on `<html>` (t/T keys
  or the theme button cycle; localStorage `seqsee-theme` persists across
  charts). The old per-element JS brute-force restyle (`updateThemeColors`)
  was deleted from both templates — the name survives as an alias so on-load
  hooks still work.
- CLICK_SCRIPT: `DIFF_PALETTES` now carries every theme keyed by name
  (`{{THEME_INITIAL}}` fallback); overlay redraw hangs off a MutationObserver
  on data-theme. Panel/log/minimap CSS uses the chart's CSS variables (with
  old-light fallbacks). index.html is theme-aware the same way.
- NOT touched: WebSocket path (template.rs + seqsee.rs JSON color literals),
  navigator/multi-chart/extract_fresh scripts (still light/dark only).
- Charts must be regenerated once (new binary) + hard reload, per the usual
  injected-JS gotcha.

## July 8 2026 session — soundness, performance, and the t=130 path

Read notes/CHANGES_2026-07-07.md §11–§20 for every change with mechanisms,
measurements and revert instructions. Compressed map:

**Soundness (all experimentally confirmed):**
- E4-UNSAT-with-correct-data root-caused and fixed: outside-diffs rows are
  basis-dependent for r≥3; rows at excluded ("partial-quotient") / edge
  degrees are now pruned at load with per-row warnings (EHP_OUTSIDE_PARITY=0
  forces). stable_Dan.csv skipped by default (EHP_OUTSIDE_SKIP).
- Exclusion LEAK fixed in make_leibniz_constraint_single: e_d_deg2 (where
  d(E y) lives) was missing from the participation checks in BOTH pipelines;
  now checked unconditionally. This resolved §8b: asserting d4(3,32,9)=1 is
  now SAT + enforced (+152 determinations). The uncertainty contract (user):
  every consulted degree with prior-page uncertainty must skip the pair.
- EHP_RELAX_TARGET_EXCLUDE default OFF (opt-in =1).
- Orbit-fold lookup dedup FALSIFIED in vivo (§19c) — never re-attempt key-
  presence changes while multiply() treats absent blocks as zero.

**Performance stack (all byte-identical on the 672-trial diag_sweep_verify
harness; identical per-page counts t=50/80):** compact product blocks (E2
load 74s→1.4s), sparse constraint rows (1.28 GB→4.6 MB at t=80), parallel
constraint generation (build 12.8s→0.24s at t=50), content-sharing dedup
(products + maps; ~90% of blocks share), d²=0 one-leg linearization
(EHP_D2_LINEAR; +129/+59/+53 entries at t=80), solver dispatch EHP_SOLVER:
classic (default) | dense (M4RI) | uf (parity union-find; E2@t=80 0.02s vs
70.65s) | verify / verify-uf, warm-start cache EHP_CACHE (t=80 pages warm in
1.08s; REPL chart stamp skips regeneration on unchanged state).

**Features:** `why <r> <n> <s> <f>` (explains un/determined status, traces
exclusions to their lower-page unknowns), shift-tap zero gesture, stem-view
"?" markers + cross-stem add flow + reflected axis + diagonal stacking +
viewport persistence, regen cap removed, seqsee python probe fix.

**Open items, in priority order:**
1. verify-uf at t=100 (user running) — if clean, flip uf to default.
2. §12b certificate math adjudication (user): which datum along the
   d4(33,31,8)→d4(3,32,9) chain is wrong if naturality-through-excluded-
   degrees reasoning were admitted (engine no longer asks; the question
   remains). notes/unsat_cert_t80_2026-07-07.log.
3. d² fixpoint cost at t=100 (~43s on E2) — optimization candidate.
4. Outside-pruning margin calibration (t=50 drops a few previously-enforced
   edge rows; user decision).
5. Deferred designs in CHANGES: block-driven pair enumeration (§ plan file),
   incremental cascade builds, product arena/mmap, Tier-2 SAT backbone
   sweep, basis-aligned translation for pruned outside rows.
6. Peak-RSS attribution at t=100/130 (dedup helps steady-state, not peak).
7. Pre-existing gaps: mapview staleness after mutations; Rust has no
   per-page-turn cutoff decrement for ITS OWN edge determinations (Python
   does) — unaudited.
