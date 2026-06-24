# Implementation guide: automatic realization-obstruction calculator

**Goal.** Given a finitely presented module `M` over the mod-2 Steenrod algebra `A`,
decide (as far as algebra allows) whether `M` can be the `F2` cohomology of a finite
spectrum, using the `sseq` / `ext-rs` library.

This is the Goerss–Hopkins / Blanc–Dwyer–Goerss obstruction theory in its
non-multiplicative (plain spectrum) form. The reference for the indexing is
Lawson, *Calculating obstruction groups for E∞ ring spectra* (arXiv:1709.09629),
Section 3, Example 2.

---

## 0. The math spec (do not deviate from this)

Work entirely inside `Ext` over the Steenrod algebra **with coefficients in `M`
itself**:

```
Ext^{s,t}_A(M, M)
```

In the library's grading, a class lives in `Ext^{s, n+s}` where `n = t - s` is the
**stem**. The obstruction theory reads two stems:

- **Existence obstructions live on stem `n = -2`**: the groups `Ext^{s, s-2}(M,M)`
  for `s = 3, 4, 5, ...`. The first possible one is `Ext^{3,1}(M,M)`.
- **Uniqueness / realization count lives on stem `n = -1`**: `Ext^{s, s-1}(M,M)`,
  `s >= 1`.

Two facts that make this a usable algorithm:

1. **Sufficient condition (cheap, group-only).** If the entire stem `-2` line of
   `Ext(M,M)` vanishes, `M` is realizable and you are done — no obstruction class
   needs to be computed. This disposes of most inputs.
2. **The check is necessary, not just sufficient, only at nonzero spots.** A nonzero
   group on stem `-2` is *where* an obstruction could live; the actual obstruction is
   a specific class (a matric Massey product of the resolution differentials). It can
   still be zero. So nonzero group ⇒ must compute the class.

**Range bound.** A class on stem `-2` at filtration `s` sits in internal degree
`t = s - 2`, and a nonzero map `M -> Omega^t M` needs `t <= diam(M)` where
`diam(M) = (top degree) - (bottom degree)`. So existence obstructions can only appear
for `3 <= s <= diam(M) + 2`. Resolve `M` to filtration `diam(M) + 2` (plus a small
margin) and you have covered every place an existence obstruction can live.
(`h_0`-towers can make the line nonzero in arbitrarily high filtration in principle,
but for a *finite* spectrum the geometric construction has only `dim_F2(M)` cells, so
only finitely many stages can carry genuinely new obstructions. Code the vanishing
test first; only build the higher-stage machinery if a real input forces you past the
first nonzero spot.)

---

## 1. What the library gives you for free vs. what you must build

Native, turnkey (study these as templates):

- `resolve` / `resolve_through_stem` — minimal free resolution `R_• -> M`, saved.
- `num_gens` — `dim Ext^{s,t}(M, F2)` per bidegree. **Note: coefficients in `F2`, not
  `M`.** This is the template for iterating over a resolution, NOT the group we need.
- `lift_hom` — represents a class of `Ext^{s,t}(M,N)` as a map
  `f ∈ Hom_A(R_s, Σ^t N)` (image of each `R_s`-generator in `Σ^t N`) and computes the
  induced map. This is the proof that `Ext(M,N)` classes are first-class objects here.
- `massey` — triple Massey products `<a, b, ->`. Useful later, for the obstruction class.
- `yoneda` — Yoneda representative of an Ext class (cochain-level data).
- `tensor`, `secondary`, `secondary_massey` — not needed for the core algorithm.

**The one thing not turnkey:** the *group* `Ext^{s,t}(M, M)` (coefficients in `M`).
You must build it. It is a finite linear-algebra computation on top of the resolution
the library already produces — see §2.

---

## 2. Computing `Ext^{s,t}(M, M)` (the core new routine)

This is just "apply `Hom_A(-, M)` to the minimal resolution and take cohomology",
exactly as `lift_hom`'s description of `Hom_A(R_s, Σ^t N)` indicates, specialized to
`N = M`.

Setup. Let `R_• -> M` be the minimal resolution. `R_s` is free on a generating set
`{g_{s,j}}` with `deg g_{s,j} = t_{s,j}` (these generators are in bijection with a
basis of `Ext^{s,*}(M, F2)`; you can read them off the resolution object). The
differential `d_{s}: R_s -> R_{s-1}` is stored as a `FreeModuleHomomorphism`; on
generators `d_s(g_{s,i}) = Σ_j a_{ij} g_{s-1,j}` with `a_{ij} ∈ A`.

Cochain complex `C^• = Hom_A(R_•, M)`:

- A map `f: R_s -> M` is determined freely by the images `f(g_{s,j}) ∈ M_{t - t_{s,j}}`
  (freeness ⇒ no constraint on the images). So
  `C^s_t  =  ⊕_j  M_{t - t_{s,j}}`,  finite-dimensional over `F2`.
- Differential `δ^s: C^s -> C^{s+1}` is precomposition with `d_{s+1}`:
  `(δf)(g_{s+1,i}) = Σ_j a_{ij} · f(g_{s,j})`, using the **`A`-action of `M`** to
  evaluate `a_{ij} · (element of M)`. Build `δ^s` as an `F2`-matrix in each internal
  degree `t`.
- `Ext^{s,t}(M,M) = ker(δ^s)_t / im(δ^{s-1})_t`.

Everything needed is already in the data structures: resolution differentials
(matrices over `A`) and the `A`-action on `M` (from the module definition). No new
resolution is required — reuse the saved resolution of `M`.

**Implementation note for Claude Code:** read these source files to fix exact APIs
before writing anything — do not guess signatures:
- `ext/examples/num_gens.rs` (iterating bidegrees of a resolution, reading generators)
- `ext/examples/lift_hom.rs` and `ext/src/resolution_homomorphism.rs`
  (`Hom_A(R_s, Σ^t N)` representation, applying the resolution differential)
- `ext/src/resolution.rs`, `ext/src/chain_complex/` (the `Resolution` /
  `FreeChainComplex` API, `differential(s)`, generator degrees)
- the `algebra` crate (module trait: how to apply an algebra element to a module
  element; how `M`'s action is stored)
- `ext/examples/define_module.rs` and `steenrod_modules/*.json` (module JSON format)

Cross-check your routine against the library: when `N = F2` (`S_2`), your
`dim Ext^{s,t}(M, F2)` must agree with `num_gens` exactly. Make that an automated test.

---

## 3. The obstruction class at a nonzero spot

If stem `-2` is nonzero at some `(s, s-2)`, the group only tells you the obstruction
*could* be nonzero. You must identify the actual class. Do the **lowest nonzero
filtration first**; the obstruction there is canonical (no earlier-stage choices).

Recommended route for the first obstruction (avoids a general Massey implementation):
use the cofiber reformulation from Lawson §3. With `d_1: R_1 -> R_0` the first
resolution differential,
```
0 -> M -> H*(X^(1)) -> Σ ker(d_1) -> 0
```
must split for a realization to exist. The first obstruction is the class of this
extension in `Ext^1(Σ ker(d_1), M)`, transported to `Ext^3(M, Ω M)`. Concretely:

1. Build the module `K = ker(d_1)` (a submodule of the free module `R_1`; finite).
2. Compute `Ext^1(Σ K, M)` with the §2 routine (coefficients in `M`).
3. Identify the specific extension class coming from `X^(1)`. It is zero iff the
   sequence splits iff the first obstruction vanishes.

This turns the first obstruction into one `Ext^1`-between-explicit-modules computation
— the same `Hom_A(-,M)` linear algebra as §2 — rather than a bracket. If you later
need higher stages, they genuinely require tracking the choice of splitting (the
obstruction becomes a higher Massey product with real indeterminacy); at that point
reuse the `massey` example's machinery and the `yoneda` representatives. Do not build
the higher-stage machinery until a real input demands it.

**Honest caveat to encode in comments/output:** realizability is *not* a function of
the bigraded groups alone (two modules with identical `Ext(M,M)` can differ in
realizability — the Massey/`A_∞` structure is irreducible data). So the tool's verdicts
are: `REALIZABLE` (stem `-2` vanishes in range), `OBSTRUCTED` (a nonzero obstruction
*class* found), or `INCONCLUSIVE — group nonzero, class computation needed at (s,t)`.
Never report "realizable" merely because a group is nonzero-but-you-didn't-check.

---

## 4. Program structure

```
input:  module name (e.g. "Ceta", or a JSON path), prime 2
config: max filtration S (default diam(M)+4), max internal degree T

1. Load/parse M. Compute diam(M).
2. Resolve M to (s<=S, t<=T) via the Resolution API (reuse `resolve` logic). Save it.
3. CORE: compute Ext^{s,t}(M,M) for all s<=S, t<=T  (§2).
      - assert dim Ext^{s,t}(M,F2) matches num_gens  (self-test)
4. Read stem -2 line: g(s) = dim Ext^{s, s-2}(M,M), s=3..S.
   Read stem -1 line for the uniqueness report.
5. If g(s)=0 for all s in [3, diam(M)+2]:  report REALIZABLE (+ uniqueness from -1).
   Else: at the smallest s with g(s)>0, run §3 (first obstruction).
      - class = 0  -> advance to next nonzero s (needs choice-tracking; may stop here
                      and report INCONCLUSIVE-BEYOND-FIRST for now)
      - class != 0 -> report OBSTRUCTED at (s, s-2), print the class.
6. Emit a report: verdict, the -2 and -1 lines as a small chart, and the obstruction
   bidegree/class if any.
```

Reuse the library's interactive-prompt / CLI-arg pattern (see `ext` crate docs,
"Prompts and arguments") so the tool can run both interactively and in batch. Build
with `--release --features concurrent`, and `--features nassau` (finite-dimensional
modules at p=2, which is exactly this use case).

---

## 5. Validation suite (must pass before trusting the tool)

| Module | Expected | Why |
|---|---|---|
| `S_2` (`M = F2`) | REALIZABLE | `Ext_A(F2,F2)` is concentrated in stems `>= 0`; stem `-2` empty. Cheapest sanity check. |
| `Ceta` (cells 0,2; `Sq^2`) | REALIZABLE | It is `Σ^{-?} CP^2 / Cη`, a real spectrum. |
| 2-cell `Sq^{2^j}` family, `j = 1,2,3` | REALIZABLE | Hopf invariant one exists (`Sq^2,Sq^4,Sq^8`). |
| 2-cell `Sq^{2^j}` family, `j >= 4` | OBSTRUCTED | Adams: no Hopf invariant one for `Sq^{16}` and up. **This is the gold-standard test** — the tool must find a nonzero obstruction here and nowhere in the `j<=3` cases. |

Define the 2-cell families with `define_module` (two generators in degrees `0` and
`2^j`, action `Sq^{2^j}(x_0) = x_{2^j}`). Automate: the verdict must flip exactly at
`j = 4`. If it does not, the §2 coefficients-in-`M` routine or the §3 class
computation is wrong — debug there, not in the resolution.

---

## 6. Things to get right (failure modes)

- **Coefficients.** The group is `Ext(M,M)`, never `Ext(M,F2)`. `num_gens` is only a
  self-test oracle for the `N=F2` case, not the obstruction group.
- **Variance.** `Ext(-,k)` and `H*(-)` are contravariant (see `lift_hom` docs). Keep
  source/target straight when building `Hom_A(R_s, M)`.
- **Stem vs internal degree.** Library indexes `Ext^{s,n+s}` by stem `n`. Obstructions
  are at `n = -2` (existence) and `n = -1` (uniqueness), i.e. *negative* stems — these
  are exactly the bidegrees `num_gens` would normally never print for the sphere, so
  expect them to be empty for trivial modules.
- **Don't over-claim.** Emit `REALIZABLE` only after the `-2` line is verified zero
  across the full finite range; emit `OBSTRUCTED` only with an actual nonzero class.
  Otherwise `INCONCLUSIVE` with the precise bidegree to inspect.
