# Refactoring Roadmap — from EHP tool to a general SS-diagram engine

Goal: make the `ehp-core`/`ehp-server` code modular enough that (1) future
sessions can improve pieces in isolation, and (2) the paradigm generalizes to
other diagrams of (stable) Adams spectral sequences, and eventually to more
complicated gradings (RO(G)).

## Where the EHP-specific assumptions actually live

An audit of `ehp-core` by module, from most generic to most EHP-bound:

| module | status |
|--------|--------|
| `gf2.rs`, `solver.rs`, `result.rs` | **already generic** — pure GF(2) linear algebra over variable indices; nothing EHP-specific |
| `pageturning.rs`, `interpage.rs` | **generic in spirit** — homology/uncertainty/exclusion/propagation logic only touches degrees through `diff_target`, `diff_source`, and `stable_rep`; those three are the only leaks |
| `page.rs`, `products.rs` | **nearly generic** — dims/basis/products keyed by `Tridegree`; the polygon bounds and the product block convention are EHP-shaped |
| `constraints.rs` | **mixed** — the XOR-system builder and `matrix_mult_left/right` are generic; the Leibniz generator hardcodes the Ỹtilde suspension convention (second factor at `n₂ = n₁+s₁`, composition with `E`); naturality hardcodes the P-map "anchored at target" convention |
| `tridegree.rs`, `map.rs` | **fully EHP** — `Tridegree {n,s,f}`, `n ≤ s+2` stable collapse (Freudenthal), the E/H/P degree formulas (note: H and P are *nonlinear* in `n`) |
| `io.rs`, `seqsee.rs` | **fully EHP** — file formats, `S{n}_{s}_{f}` node naming |

Three cross-cutting facts to preserve in any refactor:

1. **The variable is the atom.** Everything downstream of `DiffVar` (solver,
   result, kernel updates, sweep) never interprets the degree — it only needs
   `Hash + Eq + Ord`. This is why generalizing the degree type is cheap.
2. **`stable_rep` is an *identification*, not geometry.** The `n > s+2 →
   (s+2,s,f)` collapse is the one place the EHP diagram quotients its index
   set. Generalized as an optional `canonical_rep` hook, it covers any
   stabilization-type identification. It currently exists in ~3 copies
   (pageturning, interpage, inline in constraints/page) — deduplicate first.
3. **GF(2) is the one deep scalar assumption.** At odd primes Leibniz has
   signs and the linear algebra is over F_p. Keep p=2 for now, but isolate
   every scalar op behind `gf2.rs` (already true) so a future `fp`-general
   swap is one module.

## Phase 1 — mechanical modularity (no behavior change, do first)

1. **Split `ehp_chart.rs` (~2,700 lines)** into `ehp-server/src/` modules so
   the REPL is a thin loop:
   - `session.rs` — `PageState`, `cascade_resolve`, `page_content_equal`,
     deduced-diff tracking (this is *the* shared state machine; the CLI,
     websocket server, and REPL should all use it instead of re-implementing)
   - `chartgen.rs` — SeqSee pipeline, `ehp_batch.py` chunking, cleanup
   - `inject.rs` — one `inject_marker(html, "DIFFDATA", json)` helper for the
     three marker blocks (DIFF/MAP/PROP) + `CLICK_SCRIPT`
   - `chart.js` — move the injected JS out of the Rust string literal into a
     file pulled in with `include_str!` (editable with JS tooling, diffable)
   - `repl.rs` — command parsing/dispatch, one function per command
2. **Deduplicate** `stable_rep`, `gen_name`/node-ID scheme, and the
   env-var config reads (one `Config` struct built in `main`).
3. **Kill the remaining `SATPage` field-poking** from the driver (e.g.
   `maps.get_mut(...).matrices.remove(...)` in the overlay builder) behind
   small methods, so the page type's invariants live in one file.
4. **Tests as a safety net**: the interpage/constraints unit tests exist; add
   a tiny end-to-end fixture (hand-built 3-page toy SS with known answers)
   exercising solve → turn → cascade → interpage. This fixture later doubles
   as the proof that the generic engine works without EHP data.

## Phase 2 — extract the generic engine (`ss-core`)

Introduce a degree trait and make the engine generic over it:

```rust
pub trait Degree: Copy + Eq + Hash + Ord + Debug {
    /// Target of d_r from this degree.
    fn diff_target(&self, r: i32) -> Self;
    fn diff_source(&self, r: i32) -> Self;
    /// Canonical representative under the diagram's identifications
    /// (EHP: Freudenthal collapse n → min(n, s+2); default: identity).
    fn canonical_rep(&self) -> Self { *self }
}
```

- `DiffVar<D>`, `SATPage<D>`, `ConstraintSystem<D>`, `SATResult<D>`,
  `TurnedBidegree<D>`, `interpage::*<D>` — mechanical substitution;
  the bodies barely change because they already only use the three methods.
- **Structure maps become data, not an enum.** Replace `MapKind` with:

```rust
pub struct StructureMap<D> {
    pub name: &'static str,
    /// May be nonlinear in the diagram coordinate (EHP H: n → 2n-1).
    pub target: Box<dyn Fn(D) -> Option<D> + Sync>,
    pub source: Box<dyn Fn(D) -> Option<D> + Sync>, // inverse where defined
    /// Where the naturality square is anchored when generating constraints
    /// (EHP P uses Target; E and H use Source).
    pub anchor: Anchor,
}
```

- **Normalize products at load time.** The Ỹtilde suspension (compose second
  factor with `E`) should be folded into the stored product table by the EHP
  *loader*, so the engine's Leibniz generator sees a plain graded-bilinear
  pairing `D × D → D` and emits `d(xy) = d(x)y + x·d(y)` generically. This is
  the single highest-leverage change for generality — it moves the last piece
  of EHP mathematics out of the constraint engine. If some theory needs a
  twisted Leibniz rule later, add a `ConstraintProvider` trait the engine
  iterates over (naturality and Leibniz become two built-in providers; a
  theory can add its own).
- **A `Theory` trait ties it together**:

```rust
pub trait Theory {
    type Deg: Degree;
    fn structure_maps(&self) -> &[StructureMap<Self::Deg>];
    fn pairing_target(&self, a: Self::Deg, b: Self::Deg) -> Option<Self::Deg>;
    fn in_bounds(&self, d: Self::Deg, page: &SATPage<Self::Deg>) -> bool; // polygon
    fn load(&self, path: &Path, cap: i32) -> Result<SATPage<Self::Deg>>;
    fn known_diffs(&self, r: i32) -> HashMap<DiffVar<Self::Deg>, bool>;
}
```

Crate layout after phase 2:

```
crates/ss-core      # generic: gf2, solver, result, page<D>, constraints<D>,
                    #          pageturning<D>, interpage<D>, session<D>
crates/ehp-theory   # Tridegree, E/H/P maps, Ỹ normalization, .ehp/CSV io, seeds
crates/ss-charting  # SeqSee export + injection, generic over a NodeNamer +
                    #   a 2D projection of D (see phase 3)
crates/ehp-server   # thin REPL/driver wiring ehp-theory into ss-core/ss-charting
```

The EHP behavior must be bit-for-bit identical after this phase (the phase-1
fixture plus a full max_t=40 run are the acceptance tests).

## Phase 3 — other diagrams, then RO(G)

**Other stable Adams SS diagrams** (the near goal) drop out of phase 2
almost for free: a new theory crate provides its degree type (e.g.
`(object_id, s, f)` for a finite diagram of spectra), its structure maps
(the diagram's maps with their degree shifts), its pairings (module
structures over a ring object), and a loader. The engine — constraints,
solving, uncertainty-aware turning, exclusions, interpage propagation,
trial-and-error sweeps, the deduced-diff log — is unchanged. This is exactly
the machinery you want for, say, cofiber sequences `X → Y → Z` where
naturality of the connecting maps plays the role E/H/P play in EHP.

**RO(G) gradings** need two extensions, both localized:

1. `Degree` impls where the internal grading is a vector in ℤ^k (e.g.
   RO(C₂) = ℤ{1,σ}): `diff_target` shifts by an `r`-dependent vector.
   Nothing in the engine cares that k > 2.
2. **Charting is the real work**: the chart is inherently 2D, so `ss-charting`
   needs a *projection* hook — `fn chart_coords(d: D) -> (i32, i32)` plus a
   *fiber* enumeration (which slices of the k-dim grading land on one chart,
   generalizing "one chart per sphere n"). The sphere-navigation keys (WASD)
   generalize to stepping through fiber slices. Design the `NodeNamer`/
   projection trait in phase 2 with this in mind, but implement RO(G) only
   when a concrete data source exists.

Explicitly out of scope until needed: odd primes (signs + F_p solver — the
`gf2.rs` boundary is the seam), non-abelian gradings, and motivic weights
(these are just k+1-dim gradings, covered by the RO(G) design).

## Suggested order of work

1. Phase 1 (a day of mechanical moves; every later step gets cheaper).
2. Phase 2 degree-genericization *without* touching Leibniz (engine compiles
   for `D = Tridegree` only).
3. Product normalization + generic Leibniz (the delicate step — verify against
   max_t=40/90 runs and the seed).
4. `Theory` trait + crate split.
5. A second toy theory (3-object diagram, synthetic data) as a worked example
   and permanent regression test for genericity.
6. RO(G) charting projection when data exists.
