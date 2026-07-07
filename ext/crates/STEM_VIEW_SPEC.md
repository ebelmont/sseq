# Stem View — Visual Spec

Stem mode renders one chart per fixed stem `k`, sliced from the master CSV by
`stem == k`. This document specifies what that chart is supposed to look
like. Chrome (grid, axes, pan/zoom, tooltips, theme toggle, viewport
persistence) is identical to sphere-view charts and is not respecified here.

## Layout

- **x-axis = `n`** (one column per sphere S^n), **y-axis = Adams filtration**.
  Axis ticks label `n` values.
- Node ids, labels, `shift`/`position` nudges, and node size are exactly as
  in sphere view.

## Nodes

Every class in the stem-k slice is a dot, in one of three states:

1. **Plain survivor** (neither supports nor is hit by a differential):
   default filled circle in the theme text color — identical to a default
   sphere-view node.

2. **Hit by a d_r differential** (some row in the *full* CSV lists this node
   in its `drtarget`): **filled** circle, fill and stroke both in the d_r
   color. `r` is computed from the filtration jump between the source and
   target node names, exactly as sphere view computes it for `dr` edge
   coloring.

3. **Supports a d_r differential** (the node's own `drtarget` is set):
   **open** circle in the d_r color — no interior fill, but with a stroke
   thick enough that the color reads vividly at normal zoom. Concretely:
   - stroke color = the d_r color;
   - stroke width ≈ 35–45% of the node radius (note default node CSS sets
     `stroke-width: 0`, so this state needs an explicit override);
   - interior: `fill: none` visually, but prefer filling with the theme
     background color so the dot still catches mouse hover for tooltips and
     occludes grid lines passing under it.

   If a node both supports and is hit by a differential, **supporting wins**:
   draw it open in the color of the differential it supports.

The d_r colors are the existing theme-aware `d{r}` palette shared with
sphere-view differential edges: d2 teal, d3 red, d4 green, d5 blue,
d6 yellow, d7 peach, d8 mauve (Catppuccin Latte/Mocha variants per theme).
Source and target of the same differential therefore share a color, and that
color matches the `d{r}` edge one would see in the adjacent sphere-view
charts.

## Edges

Only two edge families appear in stem mode:

1. **E (suspension) edges** — the defining feature of the view. Each `E`
   target inside the slice is a structline from the class in column `n` to
   its suspension in column `n+1` (same stem, so these are near-horizontal,
   rising or flat depending on filtration). **An E edge is colored by the
   class it hits**: it takes the color of its target node's state — the d_r
   color if the target is a differential source or target, the default text
   color if the target is a plain survivor. (This is the existing
   edges-inherit-target-attributes rule; it must apply to the *stem-mode*
   node states above, so a chain of E edges into a dying class visibly
   carries that class's color.)

   A class whose suspensions continue beyond the displayed range (`E` target
   `loc`, or `Einfo` of `loc`/`free`) instead gets a short horizontal arrow
   pointing right (+x, length 0.7 grid units, simple arrowtip), in the
   default edge color.

2. **h0 structlines** — vertical lines within a column (same `n`, filtration
   +1), styled exactly as in sphere view, including the usual
   inherit-target-attributes and `loc`-tower-arrow behavior.

No `dr`, `nulldif`, `h1`, or `h2` edges are ever drawn in stem mode —
differentials are conveyed purely through the node states above, and h1/h2
leave the stem.

## Reading the chart

The intended gestalt: horizontal chains of E edges trace each class through
increasing `n` until it stabilizes; a chain that runs into a colored region
shows exactly where and by which d_r the class dies — an open d_r dot marks
the killer, a filled d_r dot of the same color marks the killed, and the E
edges feeding a colored dot are tinted to match so the eye can follow the
death backwards through the unstable range.
