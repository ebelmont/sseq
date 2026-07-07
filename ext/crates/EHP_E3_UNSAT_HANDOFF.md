# EHP: E₃ UNSAT at high total degree — investigation handoff

> **STATUS UPDATE (2026-07-03):** The multi-page propagation port landed
> (`ehp-core/src/interpage.rs` + turning/exclusion fixes below). Two root
> causes were addressed beyond the earlier fixes:
> 1. **Turning semantics** — `turn_page` now zeroes the *whole* differential
>    matrix at any degree with an unknown entry (matching the original
>    `SATResult.turn_page`), instead of quotienting by the partially-determined
>    part; it also raises a d²≠0 contradiction for fully-determined diffs.
> 2. **Stable-representative mismatch** in `make_next_exclude_set` (case (a)
>    below) — excluded *targets* are now stored in representative form.
> Additionally `solver.rs` had a genuine bug: `unknown` was set to the free
> columns only, but pivot columns correlated with free columns are also
> undetermined (the original uses the kernel matrix's nonzero columns). Fixed.
> Verified SAT through E₇ at max_t=40; max_t=90 verification pending.

Read `EHP_ARCHITECTURE.md` (same dir) first for project structure and vocabulary.

## The symptom

Running the page-turning pipeline
(`EHP_MAX_T=<t> cargo run -p ehp-server --release --example ehp_chart`)
computes E₂ → solve → turn → E₃ → solve → … The loop **stops at any page that
comes back UNSAT** (it only turns to the next page if the current one solves).

- `max_t = 40`: **works** — turns cleanly past E₃ (reaches ~E₁₅).
- `max_t = 90`: **E₃ is UNSAT** → pipeline stops at E₃.

At `max_t=90`: `E_3 WITHOUT seeds: SAT (20665/21582 determined)`, but
`E_3 WITH seeds: UNSAT`. So a **derived constraint spuriously forces the seed
differential to zero**, colliding with it.

## The seed and the collision

`load_known_diffs` (ehp-server/src/lib.rs) hardcodes, for r=3, the differential
`d₃(17,15,2)[0,0] = true` (nonzero). **The user states this seed is correct.**
A spurious Leibniz constraint on E₃ forces `d₃(17,15,2)=0`, so adding the seed
makes the system inconsistent.

At `max_t=90` the seed is NOT excluded and IS a live variable, but still
`honoured=false` (UNSAT) — i.e. a genuine derived forcing, not a dropped seed.

## How the spurious forcing works (mechanism, established at max_t=40)

Leibniz on E₃ for a pair `x=deg1, y=deg2` (Ỹtilde product, one E-suspension):
```
d_r(x·E(y)) = d_r(x)·E(y) + x·E(d_r y)
```
When the induced product `x·E(y) = 0` on E₃, the LHS `d_r(0)=0`, leaving
```
d_r(x)·E(y)  +  x·E(d_r y)  =  0
```
If the first term (RHS2) is 0 (its product vanishes) and the second (RHS1) is an
injective form in `d_r(y)`, this collapses to a **single-term relation forcing
`d_r(y)=0`** — here `y=(17,15,2)`.

**The forcing is only spurious when `x·E(y)` is *wrongly* zero on E₃** — i.e. the
product lands in a class that *should* have been killed by a differential but
wasn't quotiented out during the page turn. With a correct turn the product is
nonzero, the LHS carries the `d_r(prod)` variable, and there is no forcing.

So every instance of this bug traces to **page turning computing the wrong
homology** at the product's landing degree.

## Fixes already made (in `ehp-core/src/pageturning.rs`, uncommitted)

1. **`gauss_image` boundary bug (FIXED, this is what made max_t=40 work).**
   `turn_page_single` computed boundaries as `gauss_image(&mat_transpose(d_in))`.
   But `gauss_image(A)` already returns the *column space* of `A` (it transposes
   internally), so the extra transpose gave the *row* space of `d_in` — the wrong
   subspace — and incoming differentials were never quotiented. Fixed to
   `gauss_image(d_in)`. Verified: the turn at `(6,26,8)` now gives `H=0`
   correctly (idx0 non-cycle, idx1 cycle-but-boundary). **This alone fixed
   max_t=40** (the user confirms 40 worked *before* the exclude-set change below).

2. **Exclude uncertain degrees (added for max_t=90 — DID NOT fix 90).**
   `diff_matrix` (result.rs:~62) treats **unknown** (undetermined) differentials
   as zero, so `turn_page` mis-computes homology wherever a bounding d₂ is
   undetermined (max_t=90 has 516 unknown E₂ vars vs few at 40). Ported the
   original's `_make_exclude_set` (`~/EHP_SAT/ehp_sat/sat_ss.py:699`) as
   `make_next_exclude_set`, wired into `build_next_page` to populate
   `next.exclude_set`. For each unknown `d_r` at `(n,s,f)` it excludes source
   `(n,s,f)` and target `(n,s-1,f+r)`; constraint generation
   (`make_basis`/naturality/Leibniz) already skips excluded degrees.
   - Verified at max_t=40: still SAT, seed enforced, `exclude_set=38`. No regression.
   - At max_t=90: `exclude_set=857` but **E₃ still UNSAT**. So a spurious forcing
     remains whose landing degree is NOT being excluded.

## The open problem

At `max_t=90` a spurious `d₃(17,15,2)=0` forcing survives both fixes. It is a
*different* instance from the max_t=40 `(6,26,8)` case. Need to find the exact
Leibniz pair + product landing degree, and determine which of these it is:

- **(a)** The landing degree's bounding differential is **unknown**, but
  `make_next_exclude_set` failed to exclude it. Prime suspect: a
  **stable-degree representative mismatch**. When excluding the *target*
  `(n, s-1, f+2)` of an unknown d₂, if that target is "stable"
  (`n > (s-1)+2 = s+1`), `SATPage::is_excluded` looks it up via representative
  `(s+1, s-1, f+2)`, but we inserted `(n, s-1, f+2)` → membership test misses.
  (Note: the original Python has the same-looking code, so verify before assuming.)
- **(b)** The bounding differential is **determined** but still not quotiented —
  another turning bug beyond `gauss_image`.
- **(c)** The product is **genuinely** zero (correct turn) → then the forcing is
  real and the seed/product data are inconsistent. (User says seed is correct, so
  this would point at the E₂ product data — but (a)/(b) are far more likely.)

## Diagnostic left behind

`ehp-server/examples/diag_e3_unsat.rs` — run:
```
EHP_MAX_T=90 cargo run -p ehp-server --release --example diag_e3_unsat
```
(~226s to load at max_t=90.) It does a **targeted** scan: only Leibniz pairs
where `deg1` or `deg2` = `(17,15,2)` can put the seed var into a relation
(`th2 = th1+s1-1`, so `deg2=Y ⇒ th1+s1=18`; `deg1=Y ⇒ deg2.n=31`). For each
single-term relation forcing the seed it prints the product degree `prod` and its
d_r target `prod_dr`, their E₂/E₃ dims, `is_excluded` status, and the
unknown/determined status of the E₂ d₂'s bounding them. **The `<-- UNKNOWN` or
`excluded=false` line on the culprit degree tells you whether it's case (a).**

### Reproducing / verifying pattern (used throughout)
Load E₂ → `build_constraint_system` → `solver::solve` → `build_next_page` gives
E₃; then build E₃'s system with vs. without `load_known_diffs(3,...)` and compare
SAT/UNSAT. `make_leibniz_constraint_single(page, deg1, deg2, var_index)` returns
the raw GF(2) constraint(s) for one pair; a length-1 constraint `== [seed_idx]`
is a single-term forcing.

## Suggested next steps

1. Run the diagnostic at max_t=90; identify the culprit product landing degree.
2. If its bounding d₂ is UNKNOWN but `is_excluded=false` → fix the stable-degree
   representative mismatch in `make_next_exclude_set` (insert/lookup must agree:
   store the `(s+2,s,f)`-style representative, matching `is_excluded`).
3. If its bounding d₂ is DETERMINED but the class wasn't quotiented → inspect
   `turn_page_single` / `compute_induced_products_single` at that degree
   (as done for `(6,26,8)`: dump `d_out`, `d_in`, cycles, boundaries, `H`).
4. Re-verify max_t=40 stays SAT and the seed is honoured after any change.
5. Cross-check behavior against the original `~/EHP_SAT/ehp_sat/` (interpage.py /
   overlay_ss.py `OverlaySATPage`, sat_ss.py `turn_page`, `degree_is_uncertain_plus`)
   and `~/EHP_SAT/old_sat/SAT_helper.py::make_constraints_Y` (the Leibniz reference).

## Uncommitted code state (this session)

- `ehp-core/src/pageturning.rs`: **gauss_image fix** + **`make_next_exclude_set`**
  and its `build_next_page` wiring. (The two fixes above.)
- `ehp-core/src/io.rs`: an earlier "boundary max_t" experiment was **reverted** —
  file is back to original. (It was a red herring and was zeroing d₂'s.)
- `ehp-server/examples/ehp_chart.rs`: unrelated completed features —
  `EHP_MAX_R` (stop after E₇, default 7), Finn Juhl palettes (`SEQSEE_THEME` =
  `teak`/`linen` alongside `dark`/`light`), per-page differential overlay colors,
  and faded "pending" preview when click-adding a differential.
- `~/SeqSee/seqsee_new/main.py`: Finn Juhl `teak`/`linen` palettes (shared repo).
- `ehp-server/examples/diag_e3_unsat.rs`: the diagnostic (scratch; safe to delete).

## Ground-truth references

- Data: `~/ehp-sat-rs/data/E2.ehp` (binary) + `~/ehp-sat-rs/data/E2/*.csv`
  (products stored only if nonzero; absent row = genuine zero).
- Original Python (authoritative logic): `~/EHP_SAT/` — `ehp_sat/` (newer, has the
  uncertain-degree/overlay machinery) and `old_sat/` (SAT_helper.py, uASS.py).
