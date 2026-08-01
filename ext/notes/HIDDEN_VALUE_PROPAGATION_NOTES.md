# Hidden EHP-map values: the original problem + the constraint-system redesign

For future agents. Written 2026-07-31, at the end of the session that built
the first hidden-value engine (ehp-core/src/hidden.rs), the incremental
cascade, and the both-worlds consensus sweep — so the redesign sketch below
is informed by how all of those actually work.

## The original problem

On the terminal page (E5/E∞), exactness of the EHP sequence sometimes forces
a HIDDEN value of an EHP map — a value landing at strictly higher Adams
filtration than the nominal target. Motivating example: the class at
(n,s,f) = (11,13,3) must support a hidden P hitting the unique class at
(5,17,6) (nominal target filtration 5), because exactness admits no other
resolution. The fiber view already FLAGS such situations (pink
`fiber_hidden` candidates from the exactness diagnostics) but nothing
derives them, and once the user asserts one, its consequences should
propagate the way Leibniz propagates differentials, via Toda's composition
relations — v1 implements exactly one: **P(a ∘ E²b) = P(a) ∘ b**, applied
only when every product involved is nonzero on the page (Toda's formulas
hold in homotopy; an algebraically-zero page product may conceal an
uncomputed hidden extension, so no conclusion is drawn from it).

## What exists (v1, keep all of it)

- `ehp-core/src/hidden.rs`: HiddenValue/HiddenStore (F2 vectors, sums OK),
  validation (domain, target-line, δ≥1), `PCompositionRule` forward closure
  (worklist, dedup, all-nonzero-vector b enumeration, nonzero-only gating),
  CSV persistence (`data/E2/hidden_EHP.csv`, asserted rows only, quarantine
  on basis drift), `EHP_HIDDEN` gate. 9 unit tests.
- REPL: `hidden <E|H|P> …` assert/list/remove/undo; rebuild-after-mutation;
  prop-log entries; fiber-chart dotted overlay (`/*HIDDENDATA*/`); Shift+V
  click-to-assert in fiber charts; `diag_hidden` headless example (reuses
  snapshots/warm cache). See CLAUDE.md's hidden-values bullet for details.

## Why v1 is the wrong long-term shape

v1 is a FORWARD REWRITING closure: known values generate more known values.
The differential engine is a CONSTRAINT SYSTEM: unknowns + linear relations
+ a solver, with exclusions for uncertainty, trials for case analysis, and
(now) both-worlds consensus. The forward closure cannot:
- reason BACKWARD (knowing P_h(a∘E²b) constrains P_h(a)∘b — a linear
  relation, not a rewrite);
- DERIVE hidden values from exactness (the original problem! the fiber
  diagnostics' rank bookkeeping is precisely a system of constraints that
  can force existence/values — today it only paints candidates pink);
- do trial-and-error or consensus ("if the hidden value were 0 the homotopy
  sequence could not be exact ⇒ forced nonzero");
- interact coherently with Adams-differential uncertainty.

## The redesign: make it isomorphic to the differential engine

The deep reason this works: **Toda linearity is the Leibniz rule of the
hidden-value world.** Leibniz says d is a derivation over products; Toda
says P is a module map over composition — and the page's ProductTable IS
composition (factor1 at (n,s1,f1), factor2 at sphere n+s1). Both produce
F2-LINEAR constraints whose coefficients are product-multiplication
matrices. So every piece of the differential machinery has a direct analog:

| differential engine                  | hidden-value engine                       |
|--------------------------------------|-------------------------------------------|
| `DiffVar { n,s,f,row,col }`          | `HiddenVar { kind, deg, delta, row, col }` — entries of the hidden part of map K at source degree, filtration jump δ ≥ 1 (δ bounded by the loaded f-window) |
| Leibniz constraints (constraints.rs) | Toda constraints: for pairs (a-deg, b-deg) with product blocks present, `P_h(a·E²b) = P_h(a)·b` linearized exactly like `make_leibniz_constraint_single` (same has_block fail-fast, same product-matrix coefficient construction, same pair enumeration bounded by data) |
| d²=0 / naturality                    | homotopy EXACTNESS of the EHP sequence: at each fiber-sequence cell, rank(incoming incl. hidden parts) = dim ker(outgoing incl. hidden parts). `fiber::exactness_squares` already enumerates the cells; the diagnostics' rank computation becomes constraint generation. THIS is what derives the (11,13,3)→(5,17,6) forcing instead of just flagging it |
| `known_diffs` (user assertions)      | asserted HiddenValues (the existing store/CSV/REPL/Shift+V UI is exactly the right input layer — keep verbatim) |
| solver (solver.rs, uf)               | the SAME solver — the system is XOR-linear over F2; nothing new needed |
| exclude sets (uncertainty)           | degrees whose E∞ basis is uncertain (uncertain Adams differentials, the prior-page exclusion chains) are excluded from hidden constraint generation; determination un-excludes and re-activates (mirror `excluded_leibniz` + the cascade's exclusion recompute) |
| nonzero-only caution                 | constraint EMISSION gating: only emit a Toda constraint when the products it references are nonzero on the page (the same skip a zero block already gets in Leibniz generation — but here it is a SOUNDNESS rule, not an optimization: zero products may hide extensions, so their constraints are simply not knowledge) |
| page tower (E2→E5)                   | the δ-LADDER: hidden values of jump δ are statements in the associated graded after jumps < δ are resolved; solve δ=1 first, higher δ as the "next page" (a hidden value at δ=2 is only meaningful modulo the δ=1 story — same masking structure as page turning) |
| `try`/`sweep`/`interpage try`        | identical machinery over HiddenVars: assume a hidden entry, propagate, contradiction (exactness violated) forces the opposite; both-worlds consensus carries over verbatim |

Migration path (incremental, keeps v1 usable throughout):
1. Define HiddenVar + a constraint generator for the Toda-P rule ONLY,
   feeding the existing ConstraintSystem/solver; assert path unchanged
   (assertions become pinned vars like known_diffs). Validate: the solver's
   determined set must CONTAIN v1's forward closure on the same inputs
   (v1 becomes the regression oracle — same role diag_sweep_verify plays
   for trials).
2. Add exactness constraints from the fiber-diagnostics rank bookkeeping
   (start report-only: print "hidden value forced by exactness at …",
   compare against the pink candidates; then enforce).
3. Port the exclusion/un-exclusion coupling to Adams uncertainty (a hidden
   system re-solve after each cascade, exactly where hidden_after_mutation
   already hooks).
4. Reuse sweep/consensus over HiddenVars (they are generic over "vars +
   linear system" already in spirit; this step is where the
   REFACTORING_ROADMAP genericization pays for itself — the hidden system
   is the second concrete instance and should drive the trait boundaries).
5. More Toda relations (E/H composition rules, EHP-sequence compatibilities)
   = more constraint generators, one function each, like adding a new
   constraint family to constraints.rs.

Cautions learned this session, all of which apply:
- Basis dependence: HiddenVars on E∞ are pinned to the current basis —
  the outside-diffs/consensus dim-guard story repeats here; quarantine +
  re-validate on basis drift (v1's CSV quarantine already models this).
- δ-graded statements must not mix jumps (the associated-graded masking) —
  the δ-ladder ordering is load-bearing, same as never skipping a page.
- Keep `EHP_HIDDEN` (and a future `EHP_HIDDEN_SOLVE`) as kill switches; the
  subsystem must stay deletable without touching the differential engine.
- The UI layer (store, CSV, overlay, Shift+V, prop-log) is
  propagation-core-agnostic — swap the core underneath it, don't rebuild it.

## Addendum (same day): possibility-set consensus (the stronger both-worlds rule)

User-proposed strengthening of the both-worlds consensus, agreed design
(not yet implemented). Statement: the possible values of a differential at
a tridegree must lie in P_w0 ∪ P_w1 for EVERY unknown switch (each world's
possibility set), so ruling a matrix M out in both worlds rules it out
unconditionally — and intersecting over ALL switches combines evidence.

Key math facts settled in discussion:
- Per-ENTRY, the implemented consensus already extracts everything (a
  union of {0}/{1}/{0,1} sets only informs when both are the same
  singleton). The strengthening is about CORRELATIONS — whole matrix
  blocks at a tridegree.
- Single-page slicing yields NOTHING: fixing a switch gives two parallel
  cosets of the base space whose union spans it again. All gain comes from
  world-dependent DOWNSTREAM constraint activation (different
  un-exclusions per world) — so compute at downstream degrees.
- Tractable form: project each world's final affine solution space
  (offset + kernel, from the trial's per-page dsat) onto ONE tridegree's
  variable block (tiny, ≤ ~a dozen bits) → explicit coset; union per
  switch; intersect across switches and with the base possibilities.
  Same dim-stability basis guard as ConsensusFinding.

Consumption tiers:
1. Set collapses to singleton entries → apply via known_diffs (exists).
2. Non-singleton ruled-out sets → report (log + a `why`-style "possible
   values at t: k of 2^n"), prune future trials. No new channels.
3. Linear hull of the set ("row0 ⊕ row1 = 1 in every world") → needs a NEW
   per-page `recorded_constraints: Vec<(Vec<DiffVar>, bool)>` channel fed
   into build_constraint_system, undo-tracked, serialized (CACHE_VERSION
   bump). NOTE: this is the same channel the hidden-value constraint
   system wants — build once, share.

Implementation sketch: try_diffs_full additionally returns per-degree
block projections for affected degrees (block columns of kernel + offset
restriction — cheap); trial_error_sweep_full does union-per-switch then
running intersection per (r, degree); run_interpage_try applies tier 1,
reports tier 2. Tier 3 second. Expected first target: the same excluded-
degree "ghost" differentials the value-consensus already hits, plus cases
where several d3 switches each rule out different candidate d4 matrices.
