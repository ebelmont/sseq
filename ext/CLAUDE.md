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
- REPL commands: `add`, `zero`, `toggle`, `remove`, `undo` (argless = last mutation; `undo <r> <n> <s> <f> <row> <col>` = most recent mutation of that diff), `list`, `status`, `regen [n]`, `save`, `quit`
- Multi-page propagation (ports `~/EHP_SAT/ehp_sat/interpage.py`, lives in `crates/ehp-core/src/interpage.rs`):
  - `try <r> <n> <s> <f> <row> <col> <0|1>` — assume a value for one differential and propagate the consequences through all higher pages (incremental kernel update, un-excluding degrees that become certain, re-activating their skipped constraints); reports newly determined diffs or a contradiction (UNSAT / d²≠0), which forces the opposite value.
  - `sweep <r> [min_stem [max_stem]]` — trial-and-error over every unknown d_r × both values in parallel; contradictions mean forced values; results also written to `output/sweep_E<r>.log`.
  - `interpage try [min_stem [max_stem]]` — the automated fixpoint form: sweeps every page's unknowns (lowest r first), APPLIES each forced value (recorded like add/zero, with an undo entry each), cascades immediately so later sweeps see the consequences, and repeats whole passes until nothing new is forced. Findings go to the propagation log, the charts, and `output/interpage_try.log`. Trials are cheap since the overlay page shares product/map blocks via Arc (~0.2s per 4-page try at max_t=50); the final pass is always a full sweep that forces nothing.
  - `propagate off` defers mutations (add/zero/toggle just record, instantly); `interpage [r]` then propagates everything recorded through all pages in one pass, printing every newly determined differential (the on-demand form of the original run_interpage workflow; notes/INTERPAGE_SPEC.md documents the original). `propagate on` restores cascade-per-add.
- Uncertainty-aware page turning: differential matrices are used *partially* — determined
  entries are real kills/boundaries and are quotiented (this is what makes user-asserted
  diffs at partially-unknown degrees take effect on the next page); unknown entries are
  treated as zero (possibly-dead classes stay alive). Degrees with any unknown entry are
  excluded from constraint generation on the next page — that exclusion (plus the solver
  correctly marking correlated pivots unknown) is what keeps large-max_t pages SAT.
- Immediate consistency checking after mutations (warns on UNSAT, suggests `undo`)
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
  open d_r-colored circle = supports a differential, filled = hit by one; E edges inherit
  the target's color; right-arrows mark suspensions continuing past the cap. Generated at
  startup alongside sphere charts (jsonmaker/main.py stem mode + `ehp_batch.py stem`),
  indexed on index.html, WASD navigates (unified convention: w/s = stem −/+,
  a/d = page −/+). Titled `E_r(π^k)` (KaTeX `$\mathrm{E}_{r}(\pi^{k})$` via the
  template's generateTitle stem branch).
- Live class updates without regen: every chart carries a `/*CLASSDIMS*/` marker (degree →
  current dimension); the injected JS fades node indices beyond the dimension. Updated by
  the same pass as DIFFDATA after add/undo, so killed classes fade on reload — `regen` is
  only needed for exact re-layout/re-labelling (survivors are re-indexed by the quotient).
- Split-screen map views are pre-generated at startup for every map/sphere/page
  (J-map style; `ehp_batch.py sidebyside` manifest mode; EHP_MAPVIEWS=0 skips), so the
  e/h/p keys and WASD navigation work immediately.
- Sum-target differentials: click source, shift-click several targets, plain-click the
  last one — emits `add ...; add ...` (one per target row, same source column). Any
  ';'-separated add/zero/toggle/remove line is applied as one batch with a single
  cascade re-solve (`process_multi_mutation`).
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

**Data:** the canonical E2 input lives IN THE REPO at `ext/data/E2` (CSVs) — the
compiled-in default for the REPL and every diag example (via CARGO_MANIFEST_DIR, so
it works from any working directory and for collaborators). `EHP_DATA=<path>`
overrides. The outside-diffs knowledge base is vendored at `ext/data/outside_diffs`.
Loading CSVs directly means the only cutoff is the runtime s+f ≤ EHP_MAX_T; do NOT
point EHP_DATA at an `.ehp` binary unless you know the max-total it was converted
with (the old default `E2.ehp` had one baked in).

**Architecture:** No web server. Charts are local HTML files opened via file://. The REPL runs in the terminal. Clicking nodes in the chart copies an `add` command to the clipboard for pasting into the REPL.
