# Steenrod Modules

## Parallel Work Mode

When I say "parallel mode" or ask for multiple tasks to be done simultaneously,
help me set up git worktrees so I can run a separate `claude` session in each one.

### Setup

```bash
# From the main repo, create a worktree per task on its own branch
git worktree add ../sseq-<task-name> -b <task-name>

# Then open a claude session in that worktree
cd ../sseq-<task-name>
claude
```

### Avoiding conflicts between parallel sessions

- **Each worktree must work on different files/modules.** Before dispatching,
  split work so tasks touch non-overlapping parts of the codebase.
- When I describe the tasks, suggest a file-disjoint split and flag any
  overlap risks before I start.
- Each session should commit to its own branch before merging.

### Merging back

```bash
# From the main branch
git merge --no-ff <task-name-a>
git merge --no-ff <task-name-b>   # resolve conflicts if any

# Clean up
git worktree remove ../sseq-<task-name-a>
git worktree remove ../sseq-<task-name-b>
git branch -d <task-name-a> <task-name-b>
```

### Protecting uncommitted work

- Never run `git checkout`, `git restore`, or `git stash` without asking first.
- Never overwrite files you didn't create without reading them first.
- Before starting edits, check `git status` for uncommitted changes in files
  you might touch, and warn me if any exist.

### Within a single session

If the tasks don't need file isolation, Claude can dispatch multiple subagents
in parallel inside one session using the Task tool. Prefer this for read-only
research or tasks that touch completely separate files.

## EHP Chart REPL — Progress Notes

Interactive tool for exploring and editing EHP spectral sequence differentials.

**Key files:**
- `crates/ehp-server/examples/ehp_chart.rs` — REPL binary + injected JS
- `crates/ehp-core/src/seqsee.rs` — CSV/JSON export with h_i extensions and nulldif

**What works:**
- REPL commands: `add`, `zero`, `toggle`, `remove`, `undo` (argless = last mutation; `undo <r> <n> <s> <f> <row> <col>` = most recent mutation of that diff), `list`, `status`, `regen [n]`, `save`, `outside [retry|status]`, `quit`
- Multi-page propagation (ports `~/EHP_SAT/ehp_sat/interpage.py`, lives in `crates/ehp-core/src/interpage.rs`):
  - `try <r> <n> <s> <f> <row> <col> <0|1>` — assume a value for one differential and propagate the consequences through all higher pages (incremental kernel update, un-excluding degrees that become certain, re-activating their skipped constraints); reports newly determined diffs or a contradiction (UNSAT / d²≠0), which forces the opposite value.
  - `sweep <r> [min_stem [max_stem]]` — trial-and-error over every unknown d_r × both values in parallel; contradictions mean forced values; results also written to `output/sweep_E<r>.log`.
  - Both-worlds CONSENSUS (2026-07-31, `trial_error_sweep_full` /
    `ConsensusFinding` in interpage.rs, `EHP_CONSENSUS=0` disables): sweeps no
    longer discard the learned sets of CONSISTENT trials — when both values of
    an unknown are consistent, the two worlds' learned values are intersected,
    and any var forced to the SAME value in both worlds is determined
    unconditionally (case analysis). This removes uncertainties that exist only
    because a degree sits in the exclude list of an earlier-page unknown (e.g.
    an "uncertain" d4 that h1-Leibniz forces to 0 whether the obstructing d3 is
    0 or 1). Basis guard: a learned agreement is accepted only if NEITHER world
    changed the dims at the var's source/target degrees (equal dims ⇒ identical
    canonical bases — the stale-blocks argument). `sweep` REPORTS consensus
    values; `interpage try` APPLIES them like forced values (log lines
    "determined (same value in both worlds of …)"; conflicting conclusions
    from different switches, or against a recorded diff, are reported and NOT
    applied — they signal base-system inconsistency). Consensus values can
    land on pages beyond the swept one; the applying cascade's solve_through
    covers the highest such page.
  - Possibility-set consensus tiers 1–2 (2026-08-01, `EHP_POSSIBILITY=0`
    disables; addendum in notes/HIDDEN_VALUE_PROPAGATION_NOTES.md): sweeps
    also project each world's final solution space onto per-tridegree
    variable BLOCKS (explicit cosets, caps 16 bits/rank 12), union the two
    worlds per switch, intersect across switches and the base projection —
    carrying CORRELATIONS per-entry consensus cannot. `interpage try`
    applies entries constant across every surviving matrix (tier 1: same
    guards/undo as consensus values) and logs shrunken sets (tier 2:
    "k of m matrices remain possible"); `sweep` reports both. Tier 3
    (recording the sets' linear hulls as constraints) is deferred to a
    `recorded_constraints` channel shared with the hidden-value system.
  - Zero-map consensus (2026-08-01, `EHP_ZERO_CONSENSUS=0` disables): if in
    BOTH worlds of a switch the d_r block at a degree is provably zero
    (every entry determined 0) or VACUOUS (source classes dead, re-turned
    dim 0), the target is not hit in any possible world — `interpage try`
    records the whole stock-basis block as 0 ("d_r(deg) ≡ 0 recorded …
    target not hit"; `sweep` reports). Sound despite the SOURCE basis
    differing between worlds (zeros never quotient; the only effect is
    un-excluding the TARGET, dim-guarded at the target in both worlds).
    This resolves SELF-obstructed ghosts the per-entry consensus must
    refuse because the var vanishes in one world — e.g. an uncertain d4 at
    (15,32,7) whose only obstruction is the class's own uncertain d3:
    world d3=0 learns d4=0 by Leibniz replay, world d3=1 kills the class
    ⇒ d4 ≡ 0 recorded, the d4 target un-excludes, dashes clear. Now
    PER-ENTRY (one source class can record while a sibling stays unknown;
    per-world guard: entry determined-0 with dims stable at src+tgt, or
    src/tgt dead). JOINT width-2 phase (`EHP_JOINT=0` disables): suppressed
    diffs whose `why`-trace obstruction set is exactly TWO unknown entries
    (any pages) get a 4-world analysis — second switch applied as a STAGED
    assumption (`try_diffs_staged`) when propagation reaches its page;
    harvest runs over the consistent worlds. t=50 `interpage try`:
    114 → 378 values forced (222 ≡0 recordings), fixpoint clean, but wall
    461s → 3190s (joint trials are uncached; optimize only if asked —
    user declared try-speed done). Width ≥3 → tier-3 recorded constraints.
  - Thin trial overlays (2026-08-01): the per-page-step overlay is a LAYERED
    view (`SATPage::thin_overlay` over `OverlayBases` Arc snapshots;
    ProductTable/MapTable base+tombstone fallthrough, mutations patch/COW) —
    build/drop O(patch) instead of the old ~2M-entry clone+drop.
    `EHP_TRIAL_VERIFY=1` rebuilds every step the old eager way and
    cross-checks pointwise (the guard against a missed base block silently
    reading as zero); gated by diag_sweep_verify byte-parity at t=25/50 and
    a clean full-sweep verify run at t=50. `MapTable.matrices` is private
    now — go through matrix_at/iter/remove_matrix.
  - Influence-based pass skipping (2026-08-01): OPT-IN `EHP_TRY_SKIP=1`,
    default OFF and UNVALIDATED (user declared interpage try fast enough —
    the gauntlet was not run). Skips re-trials whose influence cone (kernel
    component + downstream turn/partner closure) misses everything changed
    since their last trial. Before trusting it, run once with
    `EHP_TRY_SKIP_VERIFY=1` (trials everything, reports would-skip outcome
    violations; must be 0).
  - `hidden solve` / `hidden exactness [margin]` (2026-08-01, report-only;
    `ehp-core/src/hidden_solve.rs` = migration steps 1–2 of the hidden-value
    notes): `hidden solve` recasts the v1 Toda closure as a linear system
    over HiddenVar entries — reports UNSAT on mutually inconsistent
    assertions, individually determined entries, solver-only determinations
    (beyond the forward closure), and a v1-oracle containment check (also
    printed by diag_hidden). `hidden exactness` lists terminal-page cells
    where rank(incoming) < dim ker(outgoing) (in-window, un-excluded,
    frontier-margined) — the engine-side pink fiber_hidden candidates.
  - `interpage try [min_stem [max_stem]]` — the automated fixpoint form: sweeps every page's unknowns (lowest r first), APPLIES each forced value (recorded like add/zero, with an undo entry each), cascades immediately so later sweeps see the consequences, and repeats whole passes until nothing new is forced. Findings go to the propagation log, the charts, and `output/interpage_try.log`. Trials are cheap since the overlay page shares product/map blocks via Arc (~0.2s per 4-page try at max_t=50); the final pass is always a full sweep that forces nothing.
  - `propagate off` defers mutations (add/zero/toggle just record, instantly); `interpage [r]` then propagates everything recorded through all pages in one pass, printing every newly determined differential (the on-demand form of the original run_interpage workflow; notes/INTERPAGE_SPEC.md documents the original). `propagate on` restores cascade-per-add.
- Uncertainty-aware page turning: differential matrices are used *partially* — determined
  entries are real kills/boundaries and are quotiented (this is what makes user-asserted
  diffs at partially-unknown degrees take effect on the next page); unknown entries are
  treated as zero (possibly-dead classes stay alive). Degrees with any unknown entry are
  excluded from constraint generation on the next page — that exclusion (plus the solver
  correctly marking correlated pivots unknown) is what keeps large-max_t pages SAT.
- Zero-asserts propagate + refresh displays (2026-07-31 fix): asserting `zero` on an
  UNKNOWN differential used to be a silent no-op downstream — unknown entries already
  turn as 0, so `changed_diff_tridegrees` saw no matrix change and the cascade stopped
  before ever recomputing the next page's exclusions (the excluded degrees stayed
  uncertain), and no chart regenerated (no dim change), leaving the generation-time
  nulldif dashes baked into the SVG. `cascade_resolve` now tracks DETERMINATION-status
  changes (unknown ↔ determined, value flips, vanished vars) per page: they keep the
  cascade alive (so `make_next_exclude_set` re-runs and un-excluded degrees re-activate
  their constraints — this can determine further entries, e.g. sibling components), and
  their degrees (stable-fold-expanded) plus any exclusion-set symmetric difference on
  the rebuilt next page are added to `affected_degrees`, so the charts with baked
  dashed/exclusion-fallback lines regenerate from a fresh CSV. Symmetric: `undo` of the
  zero restores the dashes the same way.
- Incremental cascade page rebuilds (2026-07-31, `EHP_CASCADE_PATCH`, default ON):
  mutation cascades no longer re-induce EVERY map and product on each downstream page
  (the "Computing induced products for ~9M triples" ~40s-per-page step at t=100).
  `pageturning::patch_page_from_turned` patches an Arc-sharing `overlay_clone` of the
  old next-page: dims/basis reset at dirty degrees, induced-map matrices recomputed at
  sources whose source OR target is dirty (visiting DIED degrees too — skipping them
  left stale matrices, and absent-vs-stale E is the identity-default distinction), and
  product blocks recomputed for triples (x,y,xy) touching a dirty degree (same filters
  as the full enumeration: unit rule, turned membership, flat old.max_t−1 bound; stale
  blocks touching dirty degrees removed). Dirty = re-turned degrees ∪ the PREVIOUS
  page's patch-changed degrees (chained via `PatchOutcome.changed`; a full rebuild with
  changes resets the chain → next page rebuilds fully too). Empty dirty set = the
  fast path: skip the rebuild entirely, recompute only the exclusion sets (this is the
  zero-assert case — near-instant again). `interpage` (force) still always fully
  rebuilds; the stable-fold guardrail still runs on patched pages and self-heals with a
  full turn. `EHP_CASCADE_VERIFY=1` runs patch AND full rebuild each step and
  cross-checks structurally (prints per-entry diffs on mismatch, then uses the full
  rebuild); `EHP_CASCADE_PATCH=0` restores full rebuilds. Validated at t=25: verify
  passes on zero/add/undo cascades, and patched-vs-full runs produce byte-identical
  ehp_E{2..5}.csv. NOT yet benchmarked/verified at t≥70 (user runs those — one
  `EHP_CASCADE_VERIFY=1` session at scale recommended before trusting it).
- Undo stack for reverting mutations
- Themes: all palettes live in `~/seqsee/seqsee_new/themes.json` — the single source of
  truth read by both main.py and ehp_chart.rs (`theme_registry`); add a theme by adding
  one JSON entry. Current set: `dark`/`light` (Catppuccin), `teak`/`linen` (Finn Juhl),
  `nord`/`nord-light`, `eldritch` (dark only), and `access-dark`/`access` (colorblind-safe
  Okabe-Ito; d_r colors stay distinguishable under deuteranopia/protanopia/tritanopia).
  `SEQSEE_THEME` picks the startup default; in any open chart, `t`/`T` (or the theme
  button) cycles through all themes at runtime with zero lag — every theme is emitted as a
  `:root[data-theme=…]` CSS-variable block and switching is one data-theme attribute flip
  (choice persists across charts via localStorage; the injected d_r overlay redraws via a
  MutationObserver on data-theme). Older navigator/multi-chart scripts and the WebSocket
  template.rs path still hardcode light/dark only.
- h0/h1/h2/h3 extension edges in charts (via CSV h_itarget columns)
- Unknown/uncertain differentials shown as dashed lines (via nulldif CSV column)
- Click-to-clipboard in HTML charts: Shift+D for diff mode, click source then target, paste `add` command into REPL
- Log panel in HTML (Shift+L) with localStorage persistence
- WASD navigation between spheres/pages (filenames match SeqSee's `S{n}_E{r}.html` pattern).
  UNIFIED convention across ALL chart types (sphere, stem, split-screen map views):
  w/s = dimension-like axis −/+ (sphere n or stem k), a/d = page −/+. Do not "fix" any
  chart type back to a different mapping.
- Maps minimap (Shift+M or "Maps" toolbar button): panel listing E/H/P maps on and into the
  current sphere with generator counts; clicking a row copies a `mapview` command.
  Data injected between `/*MAPDATA*/` markers (`inject_map_info`), refreshed on `regen`.
- `mapview <E|H|P> <source_n> [r]` REPL command: annotates the source sphere's JSON with
  `jmap` targets, runs SeqSee `main.py --sidebyside` (synchronized panes, `data-jmap`
  hover/click highlighting, Shift+J image mode), writes
  `output/charts/map_{kind}_{src}_{tgt}_E{r}.html`, and opens it. Back button returns to
  the source sphere's chart. `mapview all [r]` bulk-generates every view for a page (in
  parallel) so chart-side keyboard shortcuts land on existing files.
- Chart map keys (override SeqSee's plain e/h/p navigation): `e`/`h`/`p` opens the
  split-screen map view for the current sphere (navigates if generated, else copies the
  `mapview` command); `Shift+E/H/P` toggles image-mode highlighting of classes in the
  incoming map's image (per-sphere `imageIds` injected in MAP_INFO). WASD works inside
  split-screen views too (unified convention: w/s = source sphere −/+ [step 2 for P],
  a/d = page −/+ — same as single charts); pan/zoom persists across
  split-screen WASD navigation (sessionStorage `seqsee_sbs_viewport`, restored on load),
  so the view stays fixed while stepping dimensions to watch a map's image change.
- Prior-page uncertainty on current charts: unknown d_k from earlier pages are pushed
  through the turned-page quotient chain and drawn as faint dashed edges in d_k's page
  color (DIFF_EDGES entries are `[src, tgt, determined, r]`).
- Stem view (notes/STEM_VIEW_SPEC.md): one chart per stem k
  (`stem{k}_E{r}.html`, x = n capped at the stable edge k+2, y = filtration). Node states:
  open d_r-colored circle = supports a differential, filled = hit by one — the SAME
  convention applies to classes involved in an UNCERTAIN differential (from nulldif),
  plus a bold "?" glyph offset upper-right (bg-halo, ~4x node radius); E edges inherit
  the target's color; arrows mark suspensions continuing past the cap (they point LEFT,
  toward the stable range — stem charts are mirrored; arrowheads are theme-colored, not
  context-fill, which is Firefox-only). Generated at
  startup alongside sphere charts (jsonmaker/main.py stem mode + `ehp_batch.py stem`),
  indexed on index.html, WASD navigates (unified convention: w/s = stem −/+,
  a/d = page −/+). Titled `E_r(π^k)` (KaTeX `$\mathrm{E}_{r}(\pi^{k})$` via the
  template's generateTitle stem branch).
- lh0 in stem view (added 2026-07-20): stem charts show the lh0 map as a
  **slope-1 diagonal** ((n,s,f)→(n−1,s,f+1)) and, INSTEAD of raw right-h0, a
  **corrected vertical** = `h0 + E∘lh0` (F2 sum; both maps land at (n,s,f+1)).
  Sphere and fiber charts are UNCHANGED (still raw h0). Data path: lh0 is
  `MapKind::Lh0` (map.rs) — excluded from `MapKind::all()` (the EHP triple used
  by the solver/fiber/interpage; lh0 is NOT a solve constraint) and included in
  `MapKind::all_with_lh0()` used at load (E2_lh0.csv), induction
  (`compute_induced_maps`, so it's carried up every page like E/H/P),
  serialization (binary section 11; `CACHE_VERSION` bumped to 3), and CSV export.
  seqsee.rs `write_ehp_csv` emits two columns: `lh0target` (diagonal) and
  `h0lh0target` (= `h0_plus_elh0_target_names`, the XOR; empty where h0=E∘lh0,
  e.g. it VANISHES on the s=0 unit column). jsonmaker.py stem-mode edge_types
  are `["h0lh0","lh0","E"]` (was `["h0","E"]`); in-chart edges are drawn
  node-to-node so the diagonal/vertical geometry is automatic; off-window
  targets become offset arrows (`edge_offset` gains `lh0`/`h0lh0`); the f==0
  identity suppression now targets `h0lh0`. A diagonal only appears when both
  n and n−1 are within the stem's n≤k+2 cap (so lh0 shows from stem k≥3-ish up).
- Fiber-sequence view (crates/fiber_spec.md, incl. deviations list): one read-only chart
  per base sphere N (`fiber{N}_E{r}.html`) showing the EHP triple S^N → ΩS^{N+1} →
  ΩS^{2N+1}. The exact sequence is UNROLLED horizontally — x = position in the fiber
  sequence, one column per map source, so every map steps exactly +1 column and NO line
  crosses more than one column (this is the fix for the old sub-column layout where P
  jumped ~1.5 columns). Column of a class (n,s,f): −3s for S^N, −3s+1 for S^{N+1},
  −3s−3N+2 for S^{2N+1}, shifted so leftmost = 0; columns cycle S^N(E)→S^{N+1}(H)→
  S^{2N+1}(P)→next stem's S^N. E (sky) horizontal, H (maroon) +1col/f−1, P (green)
  +1col/f+2; reading left→right walks the sequence E,H,P,E,H,P (higher stems left, lower
  stems right — forced: keeping P a +1 step requires stems to decrease along the flow).
  Each column tick shows its sphere Sⁿ and its own stem; separators mark each stem's
  E-H-P block. NO "zero-or-hidden" dashed stubs (removed 2026-07-14): on an exact page an
  empty map cell is an exactness-FORCED zero (a class hit by E has H=0 since im E=ker H),
  so a dashed stub there is noise; the genuine anomalies are surfaced by the diagnostics
  panel + E∞ uncertain coloring instead. Only solid off-window arrows remain for nonempty
  targets outside the chart. Stable-range shading stem ≤
  N−2; legend + collapsible exactness-diagnostics panel (dim ker vs rank mismatches =
  candidate hidden maps, window/frontier-guarded). x-coordinate lives in jsonmaker
  `nodes_to_json` fiber branch; per-column descriptors + ticks/decorations in main.py
  `process_json`/`generate_fiber_decorations_svg` + template fiber x-tick branch.
  Degree bookkeeping + tests live in `ehp-core/src/fiber.rs`. Generated at startup via
  `ehp_batch.py fiber` (EHP_FIBERVIEWS=0 skips, folded into the freshness stamp),
  regenerated by `regen` and by mutations (a degree on sphere m dirties bases
  {m, m−1, (m−1)/2 if odd}); WASD w/s = base N −/+, a/d = page −/+ (injected
  /*FIBERNAV*/, capture-phase); click = highlight image chain, shift-click adds
  preimages, Escape clears; sphere-toggle button returns to S{N}_E{r}.html.
  Fiber UX refinements: the legend/diagnostics panel starts hidden (`#fiber-legend
  {display:none}`) and toggles on clicking the KaTeX chart title (#main-title needs
  pointer-events:auto since #title-container is pointer-events:none). Hover tooltip is
  multi-line HTML (`fiber_node_label`): `$name$` / `$S^n$` / `$(n,s,f)$` / per-map
  `E/H/P:` image lines, `<br>`-separated, KaTeX-rendered (fiber-only `#tooltip`
  max-width:22em). Intra-cell jitter is exactness-ordered: `position = has_outgoing −
  has_incoming` over SOLID edges (dashed stubs excluded), so map-targets (hit = ker out)
  sit at the SW/left end of the 45° diagonal and map-sources (support) at the NE/right
  end — shortens every in/out segment. On the E∞ (max) page ONLY, fiber nodes involved
  in an UNCERTAIN Adams differential get the stem-style diff_d{r}_open/filled coloring +
  "?" glyph, so exactness failures traceable to a missed Adams differential are visible.
  CROSS-PAGE: each page CSV's `nulldif` holds only its own d_r (E5 → only d5), so
  `build_fiber_uncertain_multipage` reads the sibling ehp_E{2..max}.csv files and unions
  their uncertain differentials, keyed by TRIDEGREE (sphere_stem_f, robust to per-page
  re-indexing) → a class involved in an uncertain d2/d3/d4/d5 is colored with THAT d_r,
  not only d_max. Gated via `EHP_MAX_PAGE` (set once in ehp_chart.rs from the max page r,
  read in jsonmaker `fiber_is_max_page` which parses this-page r from the ehp_E{r}.csv
  filename). The enlarged "?" glyph is fiber-gated in `generate_nodes_svg` (stem keeps
  the 1.1×-centered form). Hidden-EHP-value candidates (max page): a node neither hit by
  nor supporting an EHP map, NOT uncertain, AND at a frontier-guarded exactness-failure
  cell (a flagged `fiberDiagnostics` cell) is tagged `fiber_hidden` → enlarged + bold pink
  border (#ff6ea6). The flagged-cell requirement is essential — it excludes classes
  stranded only because their EHP maps weren't computed at that high stem (truncation, not
  a real hidden value); without it the frontier fills with false pink dots. Tagging is in
  main.py's fiber jitter block (reuses has_incoming/has_outgoing); no dashed "zero-or-
  hidden" stubs anymore (an empty map cell on an exact page is an exactness-forced zero,
  not a hidden map). Default terminal page is E5 (`EHP_MAX_R` default = 5, ehp_chart.rs).
- EHP map-data gaps + fills (notes/EHP_MAP_DATA_GAPS.md, notes/catalogue_ehp_gaps.py):
  exactness (dim M = rank in + rank out, a theorem on E2) is the arbiter for missing map
  data. The only genuine content gap was the P-images of odd-sphere fundamental classes
  (Whitehead squares [ι_N,ι_N] at (N,N−1,2)); FILLED — 91 rows appended to
  `data/E2/E2_P.csv`, computed as the generator of ker(E) at (N,N−1,2) (forced by
  exactness; validated to take genuine violations 176→0). Also FILLED: h_i·1=h_i products
  on the identity (n,0,0), 775 rows appended to `data/E2/E2_relations.csv`
  (`"n_0_0","n_hi_1","n_hi_1"` for hi∈{0,1,3} where dim>0), so the 1→h0/h1/h2 edges now
  draw on sphere and fiber charts. In STEM view the identity h0 structline is suppressed
  (jsonmaker edges_to_json skips h0 from f=0 sources) to keep stem charts unchanged.
  Both files hash into the warm-start cache so edits auto-invalidate; re-run
  `catalogue_ehp_gaps.py` after data changes. Remaining gaps are truncation frontiers
  (P source f≤46, H sphere≤98, E sphere≤130) and structural absences (P Hopf spheres
  7/15/31/63/127/191 = genuinely zero; isolated H spheres 65/81/89/93/97 = unrecorded
  candidates) — all documented, none are content bugs.
- Hidden EHP map values + Toda propagation (2026-07-31, EXPERIMENTAL,
  `ehp-core/src/hidden.rs` + a `Hidden EHP` section in ehp_chart.rs): assert a hidden
  value of E/H/P on the TERMINAL page (a map value landing δ≥1 filtrations above the
  nominal target — the exactness-forced values the fiber view flags in pink) with
  `hidden P <n> <s> <f> <idx> <tn> <ts> <tf> <tidx>` (idx fields accept sums `0+2`), or
  click it in a fiber chart: Shift+V enters hidden-value mode (click source, shift-click
  accumulates a target sum, plain click finishes; kind inferred from the sphere relation,
  δ≥1 pre-validated) → command lands on the clipboard. The engine then propagates via
  Toda's **P(a∘E²b) = P(a)∘b** over the page's product table (which IS composition:
  factor2 sphere = factor1 n+s) to a transitive fixpoint, enumerating ALL nonzero F2
  vectors b per candidate degree (dim ≤ 10; a sum b0+b1 can pass the gates when neither
  summand does) and deducing ONLY when E²b, a∘E²b, and P(a)∘b are all nonzero (a zero
  algebraic product may hide an uncomputed extension — no conclusion drawn). E²b goes
  through `map_matrix_ref` so stable-range identity E defaults apply. Asserted (not
  deduced) values persist to `data/E2/hidden_EHP.csv` (auto-rewritten; deliberately NOT
  in the cache config-hash, and it must never move into `outside_diffs/`, which IS
  hashed) and reload+re-propagate at startup; rows failing validation (basis drift, r
  mismatch — entries are basis-dependent like outside-diffs) are QUARANTINED with a
  warning, never deleted, and re-admitted by the rebuild when the page state recovers.
  After every diff mutation/interpage/outside-retry cascade, `hidden_after_mutation`
  re-validates + re-deduces. Display: dotted pink (#ff6ea6, matching the fiber_hidden
  candidate color) overlay edges in fiber charts (asserted full-opacity, deduced faint,
  tooltip label) via `/*HIDDENDATA*/` markers; deduced values also land in the prop-log
  panel (entries carry `hidden:true`, click navigates to the fiber chart via `#focus=`).
  `hidden list` / `hidden remove <args>` / `hidden undo` manage the store (deduced values
  are never removed directly — they rebuild from the remaining assertions; the hidden
  undo stack is separate from the diff undo stack by design). Fully modular: nothing
  touches solver/constraints/page-turning; `EHP_HIDDEN=0` hides the whole subsystem;
  snapshots need no changes. `inject_fiber_scripts` now runs unconditionally at startup
  and UPGRADES older-generation fiber scripts in place (replaces the `<script>` block
  holding `/*FIBERNAV*/`) so warm-cache/snapshot charts get the overlay without regen.
  E/H hidden values are stored/displayed but only the P rule propagates (the
  `HiddenRule` trait is the plug point for more Toda relations). Headless check:
  `EHP_MAX_T=… cargo run -p ehp-server --release --example diag_hidden` (defaults to the
  motivating example P(11,13,3)→(5,17,6); DIAG_* env vars override; prints per-gate
  statistics so 0 deductions is distinguishable from a plumbing bug). diag_hidden reuses
  saved pages instead of cold-building when it can: `EHP_SNAPSHOT=<name>` loads
  `snapshots/<name>/pages` verbatim (post-mutation state, no hash check — pass the
  snapshot's EHP_MAX_R if non-default), else the warm-start cache is tried under the
  session's exact env (hash printed on miss); both are cwd-relative, so run it from the
  same directory as the REPL sessions. 9 unit tests in hidden.rs
  (`cargo test -p ehp-core hidden`).
- Live class updates without regen: every chart carries a `/*CLASSDIMS*/` marker (degree →
  current dimension); the injected JS fades node indices beyond the dimension. Updated by
  the same pass as DIFFDATA after add/undo, so killed classes fade on reload — `regen` is
  only needed for exact re-layout/re-labelling (survivors are re-indexed by the quotient).
- Right-click view-jump menu (2026-07-29, `inject_view_menu`): right-clicking a
  class in ANY chart (sphere, stem, fiber) opens a context menu jumping to that
  class's other views — its sphere chart, its stem chart, and the fiber charts
  where its sphere m is the E/H/P source (fiber{m} / fiber{m-1} /
  fiber{(m-1)/2}, each unique; P only for odd m). Only charts that exist are
  offered (`/*VIEWNAV*/` availability injected per file, refreshed on regen).
  Links land with `#focus=<nodeId>`: the target chart highlights the class
  (crimson stroke) and pans to it.
- Unit-class h_i towers on E3+ (2026-07-29): the E2 data seeds h_i-on-identity
  product blocks (read by the chart h_i columns / fiber view); the induced-
  product enumeration skips the unit class (Python parity), which dropped
  those blocks on every turn. `compute_induced_products` now keeps unit
  triples exactly where the old page has a product block (`has_block`), so
  the towers survive to every page. CACHE_VERSION → 5.
- Map-view image mode (Shift+E/H/P/J) highlights differential edges too
  (2026-07-31, `template_sidebyside.html.jinja` edge pre-marking block): the old
  explicit skip of `d{r}`/`n{r}`-classed edges is REMOVED — a differential whose
  source AND target are both in the image gets the structline highlight
  treatment (dashed uncertain edges keep their dash; off-window offset edges
  naturally excluded). TWO STYLES since 2026-08-01: HIGHLIGHT
  (the original — image elements take the highlight color) and FADE (non-image
  content drops to 0.18 opacity, image elements KEEP their original d_r/
  structline colors). Shift+J/E/H/P cycles off → A → B → off where A = the
  last-used style (localStorage `seqsee-image-style`); the active mode still
  survives WASD navigation via sessionStorage. Both styles read the same
  data-in-j-image marks, so they classify identically. Template-only change
  (template_sidebyside.html.jinja) — regen + hard reload required.
- Split-screen map views are pre-generated at startup for every map/sphere/page
  (J-map style; `ehp_batch.py sidebyside` manifest mode; EHP_MAPVIEWS=0 skips), so the
  e/h/p keys and WASD navigation work immediately.
- Sum-target differentials: click source, shift-click several targets, plain-click the
  last one — emits `add ...; add ...` (one per target row, same source column). Any
  ';'-separated add/zero/toggle/remove line is applied as one batch with a single
  cascade re-solve (`process_multi_mutation`).
- Chart-generation timing (2026-07-31, `EHP_TIMING=1`): batch chunks emit
  `TIMESUM` lines (protocol-safe on stdout), aggregated + printed per mode after
  startup/mapview/regen phases; `timed_phase` stopwatches the injection passes;
  per-item `TIME` lines under the flag. Zero effect on chart bytes. Measurements
  + plan live in `notes/CHARTGEN_SPEEDUP_INVESTIGATION.md` — headline: ~95% of
  generation CPU was `jsonmaker.process_csv` re-parsing the same page CSV once
  PER CHART (11–13s/chart at t=100). FIXED 2026-07-31 in the vendored
  `ext/seqsee/jsonmaker.py`: (path,mtime)-keyed CSV cache, `_rows(df)` dict-rows
  cache replacing all `iterrows()`, cached schema, fiber sibling-CSV cache,
  and `jsonschema` validation now opt-in via `SEQSEE_VALIDATE=1`. Verified
  byte-identical on the full t=25 corpus (540 files); ~10–14× less jsonmaker
  CPU at t=25, more at t=100. The external `~/seqsee/seqsee_new` checkout was
  NOT touched (it's older — no fiber mode — and not the runtime copy).
  `regen_affected_charts` also runs its per-page plan entries in PARALLEL now
  (was 4 pages × 3 modes = 12 sequential batch invocations; entries are
  page-independent — own CSV, own chart files). 2026-08-01: the JSON file
  round-trip inside a batch chunk is gone (ehp_batch passes jsonmaker's dict
  straight to main.py `process_json(data=...)`; .json files are still
  WRITTEN — the Rust mapview/annotate path reads them); `sidebyside -` reads
  a JSONL manifest from stdin (on-demand map-view groundwork; Rust-side plan
  in notes/CHARTGEN_SPEEDUP_INVESTIGATION.md); compact_json deliberately NOT
  replaced (byte-format is load-bearing). Byte-parity verified at t=25.
- Chart generation runs sphere-parallel with rayon and calls the SeqSee venv python
  directly (resolved once via `poetry env info --executable`), skipping `poetry run`
  startup per invocation. Stale charts from previous runs are cleared at startup.
  Startup generation is batched: `~/seqsee/seqsee_new/ehp_batch.py` generates a whole
  chunk of spheres (JSON+HTML) in one python process (prints `OK n`/`FAIL n reason`);
  the REPL runs one chunk per CPU in parallel and falls back to per-sphere processes
  if the script is missing.
- Propagation log: each `add`/`zero`/`toggle` records the differentials deduced by the
  cascade (`cascade_resolve` returns them); the cmd→deduced map is injected into every
  chart between `/*PROPDATA*/` markers. Clicking a log entry expands its deduced list;
  clicking a deduced differential navigates to `S{n}_E{r}.html#diff=srcId;tgtId`, which
  switches to that sphere/page chart, highlights source+target, and pans to them
  (`window.panZoom.panBy`). REPL-typed commands appear in the log as "(typed in REPL)".

**Usage:**
```bash
EHP_MAX_T=100 cargo run -p ehp-server --release --example ehp_chart
```

**Solver default (changed 2026-07-20):** an unset `EHP_SOLVER` now means the
fast union-find solver `uf` (was `classic`, the slow per-column elimination).
`EHP_SOLVER=classic` forces the original; `dense`/`verify`/`verify-uf` unchanged.
The solver value is still part of the warm-start cache hash, and `CACHE_VERSION`
was bumped to 2 (old caches invalidated). The startup banner now prints the
resolved solver + config.

**Saving & loading a session (named snapshots):** the warm-start cache
(`output/.ehp_cache/<hash>/`) is keyed on a fragile hash of CSV contents + five
env flags + cwd, so ANY drift (e.g. running with a different `EHP_SOLVER` than
the run that built it) silently forces a full cold recompute + chart regen.
Snapshots are the robust, hash-independent reuse path:
- `snapshot save <name> [force]` (REPL) — bundles the live (post-mutation)
  solved page chain + the whole charts dir + a `manifest.json` into
  `snapshots/<name>/`. The charts are cloned copy-on-write (`cp -c` on APFS):
  instant, ~0 extra bytes until either copy changes — essential given the
  multi-GB charts dir. `force` overwrites an existing snapshot.
- `snapshot list` — names + config (max_t, r-range, theme, chart count,
  mutation count, date).
- `snapshot load <name>` (REPL) — re-execs the binary with `EHP_SNAPSHOT=<name>`.
- `EHP_SNAPSHOT=<name> cargo run ...` (startup) — restores that session
  VERBATIM: the manifest OVERRIDES max_t/start_r/max_r/theme, solved pages load
  from `snapshots/<name>/pages/`, charts clone back into `output/charts`, and the
  config-hash + chart-stamp checks are bypassed entirely — **no solve, no
  page-turn, no chart regeneration.** This is the intended way to "reopen my
  degree-100 charts without recomputing." Snapshot dirs live under `snapshots/`
  (cwd-relative, sibling of `output/`); implementation in `ehp_chart.rs`
  (SessionCfg/SnapshotManifest/save_snapshot/restore_snapshot) + dir-explicit
  `save_pages_to_dir`/`load_pages_from_dir` in `ehp-core/src/cache.rs`.

**Chart archiving in the warm-start cache (2026-07-28):** startup charts are
also cloned (copy-on-write) into `output/.ehp_cache/<hash>/charts/` right after
generation. On a page-cache HIT whose on-disk charts don't match the state
stamp (e.g. another config's run cleared `output/charts`), the archived set is
cloned back in <1s instead of regenerating — chart generation no longer needs
to repeat when alternating between configs. Only stamped (startup-state,
pre-mutation) chart sets are archived or restored; snapshots are unaffected.
Any archived `charts/index.html` (cache or snapshot) is fully self-contained —
open it directly in a browser to view charts without launching the REPL.

**`ehp` launcher & last-session recall (2026-07-28):** every successful startup
records its full env config (all `SESSION_ENV_VARS`: EHP_MAX_T, EHP_OUTSIDE_DIFFS,
solver flags, theme, snapshot name, …) plus the resolved cache hash into
`output/.ehp_last.json`. Binary CLI modes: `-- last` replays that env verbatim
(same config hash → warm cache + archived charts, nothing to remember);
`-- charts` opens the last session's charts (snapshot's, else cache-archived,
else live) in the browser and exits — no solve, no REPL. `ext/ehp` is the
one-word zsh launcher: `ehp` (= resume last), `ehp charts`, `ehp <max_t>`,
`ehp snap <name>`; suggest `alias ehp=~/sseq/ext/ehp`. `EHP_NO_OPEN` is
per-invocation and intentionally NOT recorded/replayed. `snapshot save <name>`
re-points the last-session record at the snapshot, so a later plain `ehp`
reopens the SAVED (post-mutation) state, not the session's startup state.

`EHP_RELAX_TARGET_EXCLUDE=1` opt-in: relaxes the strict exclusion behavior —
degrees excluded only as the *target* of an unknown incoming differential keep
their outgoing d_r variables and Leibniz pairs may constrain them in the
product position (e.g. the h1-Leibniz d3 forcing at S43 (34,4)); see
`make_next_exclude_set` in `crates/ehp-core/src/pageturning.rs`. **Default
OFF** (strict): the relaxation enabled the t=80 outside-diffs E4 UNSAT
(certificate: `notes/unsat_cert_t80_2026-07-07.log`, analysis:
`notes/CHANGES_2026-07-07.md` §11–12c) and its validation gauntlet is
unfinished. When enabled, the carve-out is guarded by an `e_d_deg2`
exclusion check (constraints.rs) — do not remove that guard.

**Known differentials from outside sources:** set `EHP_OUTSIDE_DIFFS=ext/data/outside_diffs`
(vendored in the repo; or any directory) to load externally recorded differentials at startup: every `.csv` in the
directory is scanned for rows `r, n, s, f, row, col, value` (the Python pipeline's
outside_diffs format); rows for each page merge into its known diffs before the solve
(n is normalized to min(n, s+2); a `load <file>`-style per-page file still overrides).
Caveats (2026-07-07, `notes/CHANGES_2026-07-07.md` §11–12):
- `stable_Dan.csv` is SKIPPED by default (user ruled it out as an input);
  `EHP_OUTSIDE_SKIP` holds the comma-separated skip patterns (set to `""` to
  load everything).
- Rows for r≥3 pin `[row,col]` entries that are basis-dependent; rows at
  excluded / over-kept / edge-margin degrees are pruned at load with per-row
  warnings (Python-parity; `EHP_OUTSIDE_PARITY=0` force-enforces everything).
  SAT at one max_t does NOT validate the same rows at a larger max_t.
- `outside retry` (REPL, 2026-07-29): re-attempts the pruned rows against the
  CURRENT page state — after mutations/`interpage try` clear the obstructing
  uncertainties, newly admissible rows are applied like add/zero (one undo
  entry each, cascade per pass, prop-logged as "outside retry (pass N)"),
  iterating until a pass applies nothing (each cascade can un-exclude more
  degrees downstream). Rows whose value the solver already determined
  OPPOSITE are reported as CONFLICTs and not applied. `outside status` is the
  dry-run form.

**Data:** the canonical E2 input lives IN THE REPO at `ext/data/E2` (CSVs) — the
compiled-in default for the REPL and every diag example (via CARGO_MANIFEST_DIR, so
it works from any working directory and for collaborators). `EHP_DATA=<path>`
overrides. The outside-diffs knowledge base is vendored at `ext/data/outside_diffs`.
Loading CSVs directly means the only cutoff is the runtime s+f ≤ EHP_MAX_T; do NOT
point EHP_DATA at an `.ehp` binary unless you know the max-total it was converted
with (the old default `E2.ehp` had one baked in).

**Architecture:** No web server. Charts are local HTML files opened via file://. The REPL runs in the terminal. Clicking nodes in the chart copies an `add` command to the clipboard for pasting into the REPL.
