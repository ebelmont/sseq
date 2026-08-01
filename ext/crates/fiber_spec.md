# Feature: EHP fiber-sequence viewer on the E∞ page (UASS for spheres)

> STATUS (2026-07-13): IMPLEMENTED as the "fiber" view mode. Deviations from
> this spec are listed at the bottom. Degree module: `ehp-core/src/fiber.rs`
> (tests in `fiber::tests`). Renderer: fiber branches in
> `ext/seqsee/{jsonmaker.py,main.py,template.html.jinja,ehp_batch.py}`.
> REPL wiring: `generate_fiber_charts`/`inject_fiber_scripts` in
> `ehp-server/examples/ehp_chart.rs` (EHP_FIBERVIEWS=0 to skip).

## Context
- Repo layout: UASS engine = `ext/crates/ehp-core` (pages, maps, page turning);
  chart renderer = vendored SeqSee at `ext/seqsee` (jsonmaker.py CSV→JSON,
  main.py JSON→HTML, ehp_batch.py batch modes); REPL = 
  `ext/crates/ehp-server/examples/ehp_chart.rs` (chart generation + injected JS).
- Existing algebraic EHP code (REUSED, not reimplemented): `MapKind::{E,H,P}` in
  `ehp-core/src/map.rs` (target_degree/source_degree/domain_check);
  `seqsee::map_target_names` exports each class's map images into the per-page
  CSV columns `E,H,P` (`output/ehp_E{r}.csv`); on r≥3 pages these are the
  induced maps on the page's basis — `pageturning.rs::compute_induced_maps`
  lifts a representative, applies the E2 map, reduces modulo boundaries, and
  drops zero/boundary images, which is exactly this spec's E∞ semantics.
- Class identity/serialization: `S{n}_{s}_{f}` (index suffix `_{i}` appended
  when dim > 1 at the tridegree; 0-based). `Tridegree { n, s, f }` in
  `ehp-core/src/tridegree.rs`. Names re-index across pages (quotient basis).

## Notation
U_r^{n,s,f} = the (stem s, Adams filtration f) part of the E_r page of the unstable
Adams SS for S^n; it detects elements of π_{n+s}(S^n). The differential has degree
  d_r : U_r^{n,s,f} -> U_r^{n,s-1,f+r}.

## Degree contract (2-primary EHP; assert these as unit tests)
For each n, the fiber sequence S^n -> ΩS^{n+1} -> ΩS^{2n+1} gives

  E : U_r^{n,   s, f} -> U_r^{n+1,  s,     f    }
  H : U_r^{n,   s, f} -> U_r^{2n-1, s-n+1, f-1  }        (note: source S^n)
  P : U_r^{2n+1,s, f} -> U_r^{n,    s+n-1, f+2  }

Instantiated for the display triple (S^n, S^{n+1}, S^{2n+1}):

  U_r^{n,s,f} --E--> U_r^{n+1,s,f} --H--> U_r^{2n+1,s-n,f-1} --P--> U_r^{n,s-1,f+1}

Required tests:
- composite E∘then H∘then P has degree (s,f) |-> (s-1, f+1);
- each map commutes with d_r on the stated bidegrees (the three squares);
- η: (n=1) the class at (s,f)=(1,1) on S^2 has H-image (0,0) on S^3;
- ν: (n=3) (3,1) on S^4 |-> (0,0) on S^7; σ: (n=7) (7,1) on S^8 |-> (0,0) on S^15;
- Whitehead square: (0,0) on S^{2n+1} has P-image (n-1,2) on S^n; for n=2 this is
  h0h1 = 2η on S^2;
- Freudenthal: for s <= n-2, E is a bijection on E∞ and H vanishes there.

Put all of this in ONE degree module with no UI dependencies, so odd-primary EHP can be
added later by swapping the degree rules.

## Display  (UPDATED 2026-07-14: unrolled layout — supersedes the sub-column design below)
One view per n. Vertical axis = Adams filtration f (honest, unshifted).

Horizontal axis = **position in the fiber sequence** (the sequence is UNROLLED, one
column per map source), so that every map steps exactly one column right and NO line
crosses more than one column. For a class at (n, s, f) in the base-N chart:

  column(n,s) =  -3s          if n == N     (E-source)
                 -3s + 1       if n == N+1   (H-source)
                 -3s - 3N + 2  if n == 2N+1  (P-source)

(shifted so the leftmost column is 0). Columns cycle S^N (E) → S^{N+1} (H) → S^{2N+1}
(P) → the NEXT stem's S^N, continuing the staircase. Consequences (verify rendering):
- E is horizontal, +1 column, same f.
- H is +1 column, f - 1.
- P is +1 column, f + 2, landing on the next stem's S^N column (NOT a long back-jump).
Higher stems sit left, lower stems right — forced: keeping P a +1 step requires the
represented stem to decrease along the sequence's flow direction. Each column's tick
shows its sphere S^n and its own intrinsic stem; a light separator precedes each S^N
column so a stem's E-H-P triple reads as a block. Distinct color per map + legend.
Multiple classes in one (column, f) cell get the existing jitter; lines anchor to dot
centers.

### ORIGINAL sub-column design (implemented first, then replaced)
The first implementation packed the three spheres into one master column σ (= stem for
S^n/S^{n+1}, stem + n for S^{2n+1}) as three sub-columns [S^n | S^{n+1} | S^{2n+1}] at
x = σ ± 0.28. That made P a ~1.5-column diagonal crossing the intervening columns, which
looked wrong (the degrees were correct, but the reading was cluttered). The unrolled
layout above fixes it. The degree contract (E/H/P target degrees) is unchanged.

## Interaction
- Click a class: highlight it, draw its image(s) under the outgoing map, and (toggle or
  modifier-click) its preimages under the incoming map.
- Hover tooltip: sphere, basis element / name, (n, s, f), and the coordinates of its
  images under E/H/P.
- Overlays: "image of P" (Whitehead-product classes in S^n), "nonzero Hopf invariant"
  (classes in S^{n+1} with H ≠ 0), stable-range shading (s <= n-2).
- Page selector E2 .. E∞ if the existing renderer supports it; default E∞.

## Semantics: E∞ only, no hidden maps
- E, H, P commute with d_r (the squares above), so they induce maps on E_r for all r,
  including E∞. Compute the E∞ map by pushing an E2/E_r representative through the
  existing algebraic map and reducing modulo boundaries on that page.
- If the image is a boundary / zero on E∞, the class has NO drawn image. That means the
  true homotopy image is zero OR hidden. This is a normal outcome, not an error;
  optionally render a dashed stub labeled "zero or hidden".
- DO NOT compute hidden maps/extensions. DO NOT enforce exactness on E∞: the LES is
  exact on E2, but the induced sequence on E∞ need not be. Instead add a diagnostics
  panel that, per tridegree, reports dim ker vs dim im for each map and flags mismatches
  — those cells are exactly the candidate hidden-map locations, for later work.

## Edge cases
- 2n+1 beyond the computed range: show the third sub-column as "out of computed range"
  rather than an empty chart; still draw E.
- Truncate master columns to the range where all three spheres are computed, with a
  visible boundary marker.

## Deliverables
- degree module + tests listed above
- viewer component reusing the existing chart renderer
- README note fixing the convention: U_r^{n,s,f} detects π_{n+s}(S^n); s is the stem.

## Implementation deviations (2026-07-13)
- No `?n=..&page=..` URL state: the whole chart system is file-per-(n,r)
  static HTML (`fiber{N}_E{r}.html`), so the n selector / page selector are
  WASD keys (w/s = base sphere ∓, a/d = page ∓/±, the unified convention) plus
  index.html links; injected NAV lives between `/*FIBERNAV*/` markers.
- No E∞ page exists as a file; the page selector runs over the computed pages
  E2..E_max. Each page's chart shows that page's induced maps (boundary images
  dropped), which is this spec's semantics applied per page.
- Overlays: stable-range shading is implemented (σ ≤ N−2 rect); "image of P" /
  "nonzero Hopf invariant" bulk toggles are NOT — per-class equivalents exist
  via click-highlight (click = image chain, shift-click = + preimages) and
  tooltips listing each class's outgoing image. Bulk toggles are a possible
  follow-up.
- Tick labels: in the unrolled layout every column carries two data-x ticks
  (sphere S^n over its own intrinsic stem); handleCTM repositions any tick
  with a data-x attribute by parseFloat(data-x) so they track zoom/pan.
  (The original sub-column design instead kept coordinate-true integer ticks
  plus a σ−N subtick.)
- Diagnostics window guard: besides the s+f ≤ T window, comparisons are
  skipped beyond each map's source-data frontier (max source f / total among
  nonempty cells) — the vendored E2_P.csv truncates at source f = 46 while
  h0-towers continue, which otherwise flags provably-exact E2 cells.
- Whitehead-square check is degree-arithmetic only: vendored E2_P.csv has no
  rows for the fundamental classes (S^5_0_0 etc.) or Hopf spheres
  (7,15,31,63,127,191) — that data gap is itself surfaced by the diagnostics
  panel as the (5,0,0)/(2,1,2) flags on fiber2_E2.

## UX + semantics refinements (2026-07-14)
- Legend/diagnostics panel hidden by default; click the chart title to toggle it.
- Hover tooltip reformatted to multi-line KaTeX (name / sphere / (n,s,f) / E,H,P images).
- Intra-cell jitter is exactness-ordered: map-targets (hit = ker out) go to the left/SW
  end of the 45° diagonal, map-sources (support) to the right/NE end, shortening every
  incoming/outgoing segment (position = has_outgoing − has_incoming over solid edges).
- On the E∞ (max) page ONLY, nodes involved in an uncertain Adams differential are
  colored with the stem-style d_r color + "?" glyph, so an exactness failure that is
  really a missed Adams differential (rather than a hidden EHP value) is visible. Gated
  by EHP_MAX_PAGE (set in the REPL from the max page r; read in jsonmaker). CROSS-PAGE:
  each page's nulldif holds only its own d_r, so build_fiber_uncertain_multipage unions
  the sibling ehp_E{2..max}.csv nulldifs (keyed by tridegree, robust to re-indexing) → a
  class involved in an uncertain d2/d3/d4/d5 is colored with THAT d_r, not only d_max.
  (Uncertainties are sparse below ~stem 25, so low-stem fiber charts show little color.)
- Removed the "zero-or-hidden" dashed stubs: on an exact page an empty map cell is an
  exactness-forced zero (noise), not a hidden map; the genuine anomalies are shown by the
  diagnostics panel + the uncertain-differential coloring above.
- Hidden-EHP-value candidates (max page): a node neither hit by nor supporting an EHP map
  AND not involved in any uncertain differential AND at a frontier-guarded exactness-
  failure cell (a flagged diagnostics cell) is enlarged with a bold pink border
  (.fiber_hidden). By exactness it's an unexplained failure = a class to inspect for a
  hidden EHP value. The flagged-cell requirement is essential: it excludes classes that
  are "stranded" only because their EHP maps weren't computed at that (high) stem —
  truncation, not a real hidden value.
- Default terminal page is E5 (EHP_MAX_R=5); the E∞ overlays land there.
- Data fills that feed these charts: the Whitehead-square P-rows (ker E generator at
  (N,N−1,2)) and the h_i·1=h_i identity products are now in ext/data/E2 (see
  notes/EHP_MAP_DATA_GAPS.md), so the (N,N−1,2) node is now hit by P and the fundamental
  class emits h0/h1/h2 edges — the E∞ exactness diagnostics drop accordingly.
