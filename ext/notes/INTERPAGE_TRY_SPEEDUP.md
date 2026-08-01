# interpage try speedup — measured profile + designs (2026-07-31)

Full instrumented run at t=100, `interpage try 20 60`: 3 passes, 3164 trials,
262 forced, sweeps 416s + cascades 56s = 472s total. Stage tables from
`trial_stats` (interpage.rs), ceiling lines from run_interpage_try.

## The measured profile

Per-sweep stage tables (CPU-side sums across rayon threads), aggregated:

| page | trials (3 passes) | overlay | teardown | re-solve | first-solve | constraints | turn | total |
|------|-------------------|---------|----------|----------|-------------|-------------|------|-------|
| E2   | 270               | 370s    | 312s     | 71s      | 17s         | 13s         | 2.4s | 802s  |
| E3   | 1346              | 925s    | 219s     | 123s     | 48s         | 128s        | 25s  | 1499s |
| E4   | 738               | 69s     | 48s      | 0.9s     | 6.7s        | 0.4s        | 3.5s | 129s  |
| E5   | 810               | 0       | 0        | 0        | 4.9s        | 0           | 0    | 4.9s  |

Headlines:
- **overlay + teardown = 76–91% of trial CPU.** Teardown (dropping the
  per-step overlay SATPage: ~2M-entry product HashMap clone+drop with Arc
  bumps/decs) is 27–39% on its own — it was the "unaccounted" third.
- **E5 trials cost ~6ms each** (terminal page: no page step, pure incremental
  solve). That is the floor a trial can reach — the solve side is essentially
  free, exactly as in the old SAT-solver days.
- **Skippable ceiling ≈ 100%**: pass 2 = 438/442 re-trialed vars with
  identical outcomes; pass 3 = 438/438. Passes 2–3 cost ~178s of the 416s
  sweep time and forced 4 values (pass 3: zero, by definition).
- Zero-stratum: E4 123/246 (every assumed-0 trial), E3 ~40%, E2 only 16/106
  (dense kernel correlations learn 1s). Forcings overwhelmingly arise as
  "=1 forced (UNSAT when assuming 0)" — the cheap side finds them.

## Design 1 — thin overlay (attacks overlay+teardown for ALL trials)

`build_overlay_page` currently `overlay_clone()`s the whole target page
(~2M-entry HashMap clone; Arc-bump per block) and later drops it, per page
step, per trial. But the ONLY consumers of the overlay are:
- `collect_new_vars` (dims at un-excluded degrees),
- `make_new_constraints` (naturality squares at un-excluded degrees + Leibniz
  pair replay from `excluded_leibniz`) — which consult dims, map matrices,
  and product blocks at an enumerable, SMALL set of degrees.

Replace the full clone with a MINIMAL page: fresh SATPage containing only
the dims (stock + `lt.new_dims` patch) and the blocks the constraint
generators will consult — stock blocks Arc-cloned individually, recomputed
blocks from the existing patch logic. Build and drop become O(needed blocks)
(hundreds) instead of O(page) (millions).

DANGER: `ProductTable::multiply` treats an absent block as ZERO — a missed
block silently corrupts a constraint. Mitigations (both):
1. `EHP_TRIAL_VERIFY=1` mode: build BOTH minimal and full overlays, generate
   constraints from both, byte-compare `new_cons` per page step; report any
   divergence. Run at t=50 and once at t=100.
2. `diag_sweep_verify` byte-parity harness (672 trials t=50) must stay
   byte-identical — it gates ANY change to this path (see SESSION_HANDOFF).
The consultation set must be derived from the SAME enumeration the
generators use (unexclude list + excluded_leibniz pair records + naturality
square degrees + their map/product partner degrees), not re-derived
independently — otherwise the two can drift.

Expected effect: trials approach the E5 floor: per-trial cost dominated by
the incremental solves (~10–100ms) + the genuinely dirty recomputes. Order
~5–10× on sweep time, benefiting pass 1 (which pass-skipping cannot).

Note: `prev_overlay` becomes the trial's source page for the NEXT page step
(lifts/multiplies against it in turn_page_local/compute paths) — the minimal
page must also cover THOSE consultations (turned-degree lifts at the next
step's dirty degrees consult source products/maps). Enumerate or fall back:
v1 can keep the full overlay ONLY when a further page step will occur and
the dirty set is large, else minimal. Measure again after.

## Design 2 — pass skipping (attacks passes 2+; measured ceiling ~100%)

Re-trial var u in pass N+1 only if pass N's changes intersect u's influence
cone. Changed set (already collected per cascade): affected_degrees ∪
PatchOutcome.changed ∪ exclusion symmetric-difference ∪ forced-var degrees,
fold-expanded. Cone(u), conservative over-approximation:
- u's kernel-component degrees on its page (union-find over kernel row
  supports of the CURRENT result), fold-expanded,
- propagated downstream page by page (t → {t, diff_target_r(t)}, fold),
- plus one hop of Leibniz-pair partner degrees (from excluded_leibniz
  records) and naturality partner degrees for everything in the set.
If cone(u) ∩ changed = ∅ for every pass since u's last trial, skip both its
trials (outcome provably unchanged: deterministic pipeline, unchanged
inputs). The terminating pass then trials (almost) nothing — fixpoint
confirmation becomes ~free.
Verification: `EHP_TRY_SKIP_VERIFY=1` runs the skipped trials anyway and
asserts none forces anything (that is precisely what the measured ceiling
lines already demonstrate empirically: 4 outcome changes in 1900 pass-2/3
trials, all 4 within the changed neighborhoods).

## Priority

1. Design 1 (thin overlay) — biggest absolute win, helps pass 1.
2. Design 2 (pass skipping) — deletes most of passes 2+.
3. (Optional, subsumed by 1) zero-stratum shortcut: with the thin overlay,
   zero-stratum trials' remaining cost is already near-floor; a dedicated
   stock-data path may no longer be worth its complexity.

Cascades are a non-issue (56s of 472s, and the cascade patches run in
0.2–4.6s where full rebuilds were ~40s each).

## Instrumentation reference (already in place)

- `interpage::trial_stats` — per-stage atomic timers + assumed-1/learned-1/
  zero-stratum/contradicted classification; reported per sweep by
  run_interpage_try under EHP_TIMING.
- Pass-outcome memo in run_interpage_try — the "IDENTICAL outcomes
  (skippable ceiling)" lines.
- `[timing] E_r cascade patch: Xs (N dirty → M changed)` in cascade_resolve.
