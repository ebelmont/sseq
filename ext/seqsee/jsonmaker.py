import os
import re
import sys

import pandas as pd
from compact_json import Formatter
from jsonschema import validate

import main as _seqsee_main

# ---------------------------------------------------------------------------
# Per-process caches. A batch process (ehp_batch.py) renders MANY charts from
# the SAME page CSV; re-parsing it per chart dominated generation (~95% of
# CPU at t=25, worse at t=100 — see notes/CHARTGEN_SPEEDUP_INVESTIGATION.md).
# Path-keyed caching is safe here: the Rust caller writes fresh CSVs BEFORE
# spawning batch processes, and each process is short-lived; the mtime key is
# belt-and-braces on top.
# ---------------------------------------------------------------------------

_CSV_CACHE = {}  # path -> (mtime, DataFrame)


def _load_csv_cached(path):
    p = str(path)
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        mtime = None
    hit = _CSV_CACHE.get(p)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    df = pd.read_csv(p)
    _CSV_CACHE[p] = (mtime, df)
    return df


def _rows(df):
    """List-of-dicts view of a frame, cached on the frame itself.

    `df.iterrows()` materializes a Series per row and is ~100x slower than
    iterating plain dicts; dict rows keep the exact `row[key]` / `row.get` /
    KeyError semantics the extraction code relies on.
    """
    rows = df.attrs.get("_rows_cache")
    if rows is None:
        rows = df.to_dict("records")
        df.attrs["_rows_cache"] = rows
    return rows


_SCHEMA_CACHE = None


def load_schema():
    # NB: `main` can't be referenced lazily by module name — this file
    # defines its own `def main()` which shadows the import at call time.
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        _SCHEMA_CACHE = _seqsee_main.load_schema()
    return _SCHEMA_CACHE


# jsonschema validation of every generated chart is a pure development
# assertion (the generator is deterministic) and costs real time per chart —
# opt back in with SEQSEE_VALIDATE=1.
_VALIDATE = bool(os.environ.get("SEQSEE_VALIDATE"))

# Regular expressions for substitutions
substitutions = [
    # Matches any string starting with an underscore, and surrounds it in \overline{}
    (re.compile(r"^_(.*)$"), r"\\overline{\1}"),
    # Matches the dot operator, which is rendered as a centered dot in LaTeX
    (re.compile(r"\."), r"\\cdot"),
    # Matches the string "DD" and replaces it with "{D}". The braces are necessary if we want to
    # handle both "DD" -> "D" and "D" -> "\Delta".
    (re.compile(r"DD"), r"{D}"),
    # Matches the string "D" and replaces it with "\Delta", but only if it is not surrounded by
    # curly braces. The `(?<!...)` and `(?!...)` expressions are negative lookbehind and negative
    # lookahead, respectively. They ensure that whatever we are matching is in the right "context",
    # i.e. in this case not being in braces, but without including that context in the match. This
    # is necessary to avoid replacing the "D" in "{D}".
    (re.compile(r"(?<!{)D(?!})"), r"\\Delta"),
    # Matches the word "t" and replaces it with "\tau". This 't' character needs to be surrounded by
    # either the beginning or end of the line, or a non-word character (something that matches
    # `\W`).
    (re.compile(r"\bt\b"), r"\\tau"),
    # Matches word expressions starting with a non-empty string of latin letters and curly braces
    # (group 1) and ending with a non-empty string of numbers or commas (,) (group 2). We insert an
    # underscore between the two groups and wrap the second one in curly braces. This ensures that
    # subscripts are correctly rendered in LaTeX. NB: Caret (^) is a word separator.
    (re.compile(r"\b([a-zA-Z{}]+)([\d,]+)\b"), r"\1_{\2}"),
    # Matches terms of the form '(letters or curly braces)^(numbers)', and wraps the second group in
    # curly braces. This ensures that superscripts are correctly rendered in LaTeX.
    (re.compile(r"\b([a-zA-Z{}]+)\^(\d+)\b"), r"\1^{\2}"),
]

# Regular expression for detecting "again" suffixes. We strip anything that is whitespace followed
# by any number of non-word characters, then the word "again", and then anything else until the end
# of the line.
detect_again = re.compile(r"\s\W*again.*")

arrow_length = 0.7


def try_get_key(row, key, default=None):
    """
    Get a key from a row.

    Return a default value if the key is not present or the value is `nan`.
    Container values (e.g. a node's `attributes` list) are returned as-is:
    `pd.isna` on a list is element-wise, and truth-testing the resulting
    array raises once the list has 2+ elements (it only ever "worked" for
    the 0/1-element lists stem nodes used to have).
    """

    try:
        val = row[key]
    except KeyError:
        return default
    if isinstance(val, (list, tuple, set, dict)):
        return val
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        return val
    return val


def edge_offset(edge_type, arrow_length=1):
    if edge_type == "h0":
        offset = {"x": 0, "y": 1}
    elif edge_type == "h1":
        offset = {"x": 1, "y": 1}
    elif edge_type == "h2":
        offset = {"x": 3, "y": 1}
    elif edge_type == "dr":
        offset = {"x": -1, "y": 2}
    elif edge_type == "nulldif":
        offset = {"x": -1, "y": 2}
    elif edge_type == "E":
        # E-type edges (stem mode only). Stem charts are mirrored so that n
        # increases right-to-left; a suspension continuing past the stable
        # edge therefore points LEFT (toward the stable range).
        offset = {"x": -1, "y": 0}
    elif edge_type == "lh0":
        # lh0 map (stem mode only): (n,s,f) -> (n-1,s,f+1). n drops by 1 and
        # f rises by 1, so pre-mirror the offset is left-and-up; after the
        # stem x-mirror it renders as a slope-1 diagonal up-and-to-the-right.
        offset = {"x": -1, "y": 1}
    elif edge_type == "h0lh0":
        # Corrected h0 (stem mode): h0 + E∘lh0, (n,s,f) -> (n,s,f+1). Same
        # degree as raw h0, so the same vertical offset.
        offset = {"x": 0, "y": 1}
    elif edge_type == "fiberE":
        # Fiber-view E stub. The fiber sequence is unrolled left-to-right so
        # every map steps exactly one column right; E preserves f.
        offset = {"x": 1, "y": 0}
    elif edge_type == "H":
        # Fiber-view H stub: one column right, dropping f by 1.
        offset = {"x": 1, "y": -1}
    elif edge_type == "P":
        # Fiber-view P stub: one column right (onto the next stem's S^N
        # column), raising f by 2.
        offset = {"x": 1, "y": 2}
    else:
        raise ValueError
    for key in offset:
        offset[key] *= arrow_length
    return offset


def parse_node_coordinates(node_name):
    """Parse node name to extract s, stem, f, index values.

    Handles both legacy names like '4_28_5_1' and prefixed names like
    'S4_28_5_1' or 'SO10_28_5_1'.  The prefix (everything before the first
    digit in the first segment) is stripped before parsing.
    """
    if not node_name or node_name == "loc":
        return None
    parts = node_name.split("_")
    if len(parts) >= 3:
        try:
            # Strip any non-digit prefix from the first part (e.g. "S2" -> "2",
            # "SO10" -> "10").  If the first part is already numeric this is a
            # no-op.
            first = re.sub(r"^[A-Za-z]+", "", parts[0])
            s = int(first)
            stem = int(parts[1])
            f = int(parts[2])
            index = int(parts[3]) if len(parts) > 3 else 0
            return {"s": s, "stem": stem, "f": f, "index": index}
        except ValueError:
            return None
    return None


def build_hit_map(df):
    """Map node name -> d_r height, for every node listed in some drtarget.

    Precomputed once so stem-mode node coloring is O(N) instead of scanning
    the whole frame per node.
    """
    hit = {}
    for row in _rows(df):
        drt = row.get("drtarget")
        if not drt or str(drt) == "nan" or drt == "":
            continue
        src_name, _ = deduplicate_name(row["name"])
        sc = parse_node_coordinates(src_name)
        if not sc:
            continue
        for t in str(drt).split(";"):
            tc = parse_node_coordinates(t)
            if tc:
                hit[t] = tc["f"] - sc["f"]
    return hit


def build_uncertain_target_set(df):
    """Map node name -> d_r height, for every node in some row's `nulldif`.

    `nulldif` holds the possible targets of an UNCERTAIN differential, so
    these nodes are uncertain-targets; the height gives them the same d_r
    coloring as determined targets (plus the "?" glyph). Mirrors
    build_hit_map: precomputed once from the full frame (a stem-k chart's
    uncertain targets come from stem-(k+1) sources), stem mode only.
    """
    targets = {}
    for row in _rows(df):
        nd = row.get("nulldif")
        if not nd or str(nd) == "nan" or nd == "":
            continue
        src_name, _ = deduplicate_name(row["name"])
        sc = parse_node_coordinates(src_name)
        for t in str(nd).split(";"):
            t = t.strip()
            if not t:
                continue
            tc = parse_node_coordinates(t)
            if sc and tc:
                targets.setdefault(t, tc["f"] - sc["f"])
            else:
                targets.setdefault(t, None)
    return targets


def fiber_is_max_page(csv_path):
    """Task 4: True iff this fiber page is the E-infinity (max) page.

    The page r is embedded in the input CSV filename output/ehp_E{r}.csv; the
    highest page being generated is exported by the Rust REPL as the env var
    EHP_MAX_PAGE. Only when both parse and match do we color uncertain
    Adams-differential nodes on the fiber chart. If the env var is absent
    (e.g. a manual jsonmaker run), returns False so non-max coloring stays off.
    """
    import os
    import re as _re

    env = os.environ.get("EHP_MAX_PAGE")
    if not env:
        return False
    m = _re.search(r"ehp_E(\d+)\.csv$", str(csv_path))
    if not m:
        return False
    try:
        return int(m.group(1)) == int(env)
    except ValueError:
        return False


def build_fiber_uncertain_multipage(csv_path, n_base):
    """Aggregate UNCERTAIN Adams differentials across every page E2..E{this}
    for the fiber triple, so the E-infinity chart colors a class involved in an
    uncertain d2/d3/.../d{max}, not only the current page's d{max}.

    Each page's CSV carries only its OWN d_r in `nulldif` (E5 -> only d5), so we
    read the sibling `ehp_E{k}.csv` files (k = 2..this page) from the same
    directory and union them. Returns (unc_src, unc_tgt), dicts keyed by
    tridegree string "sphere_stem_f" -> d_r height: a class with a nonempty
    `nulldif` supports an uncertain d_r (open ring); a class named in some
    `nulldif` is hit by one (filled). Keyed by tridegree, NOT basis index, so it
    survives the per-page quotient re-indexing (a surviving class keeps its
    (n,s,f) even when its index shifts). Empty ({}, {}) if the filename has no
    parseable page number.
    """
    import re as _re

    m = _re.search(r"ehp_E(\d+)\.csv$", str(csv_path))
    if not m:
        return {}, {}
    this_r = int(m.group(1))
    directory = os.path.dirname(str(csv_path))
    triple = set(fiber_spheres(n_base))
    unc_src, unc_tgt = {}, {}
    for k in range(2, this_r + 1):
        page = os.path.join(directory, f"ehp_E{k}.csv")
        # Sphere filtering by `triple` happens on the cached, pre-parsed
        # nulldif rows: every fiber chart of a page re-reads the SAME sibling
        # CSVs, which made fiber the slowest chart mode by far.
        for n, src_key, targets in _fiber_nulldif_rows(page):
            if n not in triple:
                continue
            for tgt_key, height in targets:
                # setdefault: first (lowest-page => lowest-r) wins on ties.
                unc_src.setdefault(src_key, height)
                unc_tgt.setdefault(tgt_key, height)
    return unc_src, unc_tgt


_FIBER_NULLDIF_CACHE = {}  # page path -> (mtime, [(n, src_key, [(tgt_key, height)])])


def _fiber_nulldif_rows(page):
    """Pre-parsed positive-height nulldif rows of one page CSV (cached)."""
    import csv as _csv

    try:
        mtime = os.path.getmtime(page)
    except OSError:
        return []
    hit = _FIBER_NULLDIF_CACHE.get(page)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    rows = []
    with open(page, newline="") as fh:
        for row in _csv.DictReader(fh):
            nd = (row.get("nulldif") or "").strip()
            if not nd or nd == "nan":
                continue
            try:
                n = int(row["n"])
                stem = int(row["stem"])
                f = int(row["Adams filtration"])
            except (KeyError, ValueError, TypeError):
                continue
            src_key = f"{n}_{stem}_{f}"
            targets = []
            for t in nd.split(";"):
                t = t.strip()
                if not t:
                    continue
                tc = parse_node_coordinates(t)
                if not tc:
                    continue
                height = tc["f"] - f  # d_r raises filtration by r
                if height <= 0:
                    continue
                targets.append((f"{tc['s']}_{tc['stem']}_{tc['f']}", height))
            if targets:
                rows.append((n, src_key, targets))
    _FIBER_NULLDIF_CACHE[page] = (mtime, rows)
    return rows


# --------------------------------------------------------------------------
# Fiber view: one chart per base sphere N, showing the display triple
# (S^N, S^{N+1}, S^{2N+1}) of the fiber sequence S^N -> Omega S^{N+1} ->
# Omega S^{2N+1}. The exact sequence is UNROLLED horizontally: the x-axis is
# "position in the fiber sequence", one column per map source, so that every
# map steps exactly one column to the right and no line crosses more than one
# column. Columns cycle S^N (E-source), S^{N+1} (H-source), S^{2N+1}
# (P-source), then P lands on the NEXT stem's S^N column, continuing the
# staircase. Concretely, for a class at (n, s, f) the column is
#   n == N   : -3s          (E: horizontal, +1 col)
#   n == N+1 : -3s + 1       (H: +1 col, f-1)
#   n == 2N+1: -3s - 3N + 2  (P: +1 col, f+2)
# (columns are shifted so the leftmost is 0). Higher stems sit left, lower
# stems right; reading left-to-right walks the sequence E, H, P, E, H, P.
# y = Adams filtration f, unshifted. See crates/fiber_spec.md.
# --------------------------------------------------------------------------

def fiber_spheres(n_base):
    """The display triple of spheres for base sphere N."""
    return (n_base, n_base + 1, 2 * n_base + 1)


def fiber_map_for_sphere(n, n_base):
    """The designated outgoing fiber-sequence map ("E"/"H"/"P") for sphere n
    within the display triple of base sphere n_base, or None."""
    if n == n_base:
        return "E"
    if n == n_base + 1:
        return "H"
    if n == 2 * n_base + 1:
        return "P"
    return None


def fiber_target_tridegree(n, s, f, n_base):
    """Target tridegree (n', s', f') of the designated map out of U^{n,s,f}."""
    if n == n_base:
        return (n_base + 1, s, f)
    if n == n_base + 1:
        return (2 * n_base + 1, s - n_base, f - 1)
    if n == 2 * n_base + 1:
        return (n_base, s + n_base - 1, f + 2)
    return None


def fiber_cell(row, col):
    """The (stripped) contents of a map cell, with nan normalized to ""."""
    val = try_get_key(row, col, "")
    val = str(val).strip()
    return "" if val == "nan" else val


def build_tridegree_dims(df):
    """Map (n, stem, f) -> number of classes (deduplicated rows) in the CSV."""
    dims = {}
    for row in _rows(df):
        _, is_duplicate = deduplicate_name(row["name"])
        if is_duplicate:
            continue
        key = (int(row["n"]), int(row["stem"]), int(row["Adams filtration"]))
        dims[key] = dims.get(key, 0) + 1
    return dims


def _f2_rank(masks):
    """Rank over F2 of a matrix whose rows are given as int bitmasks."""
    pivots = {}
    rank = 0
    for m in masks:
        while m:
            hb = m.bit_length() - 1
            if hb in pivots:
                m ^= pivots[hb]
            else:
                pivots[hb] = m
                rank += 1
                break
    return rank


def fiber_node_label(row, node_name, n_base, dims, existing_label=""):
    """Tooltip label for a fiber-view node (Task 5): a nicely formatted,
    multi-line HTML string rendered by KaTeX in the template.

    Lines (joined by <br>): the class name in $...$; the sphere and tridegree
    wrapped in $...$; then each outgoing map image on its own line labeled
    E/H/P (targets in $...$, keeping the '0 / hidden' / '0' wording). It becomes
    an HTML data-label attribute, so it must contain NO double-quotes.
    """
    n = int(row["n"])
    s = int(row["stem"])
    f = int(row["Adams filtration"])
    # KaTeX-friendly class name (underscores must be escaped inside math mode).
    base = existing_label or "$" + node_name.replace("_", "\\_") + "$"
    lines = [
        base,
        f"$S^{{{n}}}$",
        f"$(n,s,f)=({n},{s},{f})$",
    ]
    map_type = fiber_map_for_sphere(n, n_base)
    if map_type:
        cell = fiber_cell(row, map_type)
        if cell:
            targets = [t.strip() for t in cell.split(";") if t.strip()]
            tex_targets = [
                "$" + t.replace("_", "\\_") + "$" for t in targets
            ]
            lines.append(f"{map_type}: " + " + ".join(tex_targets))
        else:
            tgt = fiber_target_tridegree(n, s, f, n_base)
            if dims.get(tgt, 0) > 0:
                lines.append(f"{map_type}: 0 / hidden")
            else:
                lines.append(f"{map_type}: 0")
    return "<br>".join(lines)


def fiber_edges_to_json(df, nodes, n_base, dims):
    """Edges for fiber view: one map per source sphere (E from S^N, H from
    S^{N+1}, P from S^{2N+1}), colored mapE/mapH/mapP. Nonempty targets outside
    the chart window get an offset arrow.

    No "zero or hidden" dashed stubs: on E2 (and, by exactness, wherever the LES
    is exact) an empty map cell is a zero FORCED by exactness — e.g. a class hit
    by E has H = 0 because im E = ker H — so a dashed stub there is noise, not
    information. The genuinely interesting cases (exactness violations = candidate
    hidden maps / missed differentials) are surfaced instead by the exactness-
    diagnostics panel and the E-infinity uncertain-differential coloring."""
    edges = []
    stub_scale = 0.35 * arrow_length
    offset_types = {"E": "fiberE", "H": "H", "P": "P"}
    for row in _rows(df):
        node_name, is_duplicate = deduplicate_name(row["name"])
        if is_duplicate or node_name not in nodes:
            # Row is outside the current slice (or a duplicate).
            continue
        n = int(row["n"])
        map_type = fiber_map_for_sphere(n, n_base)
        if map_type is None:
            continue
        alias = f"map{map_type}"
        offset_type = offset_types[map_type]
        # Domain check (defensive: always true within the triple for N >= 2)
        if map_type in ("E", "H") and n < 2:
            continue
        if map_type == "P" and (n < 5 or n % 2 == 0):
            continue
        cell = fiber_cell(row, map_type)
        s = int(row["stem"])
        f = int(row["Adams filtration"])
        if cell:
            for target_node in cell.split(";"):
                target_node = target_node.strip()
                if not target_node:
                    continue
                if target_node in nodes:
                    edges.append(
                        {
                            "source": node_name,
                            "target": target_node,
                            "attributes": [alias],
                        }
                    )
                else:
                    # Target outside the chart window (P can leave the s+f
                    # window): short offset arrow toward the target degree.
                    scale = stub_scale if map_type == "P" else arrow_length
                    edges.append(
                        {
                            "source": node_name,
                            "offset": edge_offset(offset_type, scale),
                            "attributes": [alias, {"arrowTip": "simple"}],
                        }
                    )
    return edges


def compute_fiber_diagnostics(df, n_base, total_cap):
    """Per-tridegree F2 ranks of E/H/P and exactness comparisons.

    For each master-column chunk (sigma, f) with sigma + f <= total_cap the
    three exactness positions of

      U^{N,s,f} -E-> U^{N+1,s,f} -H-> U^{2N+1,s-N,f-1} -P-> U^{N,s-1,f+1}

    are compared: rank(map in) vs dim ker(map out). Mismatches are candidate
    hidden-map locations (E_infinity need not be exact); they are only
    flagged when every involved tridegree is inside the window.
    Returns {"cells": [...], "totals": {"E": .., "H": .., "P": .., "flagged": ..}}.
    """
    triple = set(fiber_spheres(n_base))
    dims = {}
    masks = {}
    # Recorded-domain frontier of each map: max source filtration and max
    # source total degree (stem + f) over rows with a NONEMPTY cell. The
    # vendored map data is truncated by generation caps (e.g. E2_P.csv stops
    # at source f = 46 while the h0-towers continue), so an empty cell whose
    # source lies beyond either frontier is "unrecorded", not "zero" —
    # comparisons touching such cells must be skipped or they flag
    # truncation artifacts (a (5,0,f>=47)/(2,1,f>=49) ladder on every page).
    max_src_f = {"E": -1, "H": -1, "P": -1}
    max_src_total = {"E": -1, "H": -1, "P": -1}
    for row in _rows(df):
        _, is_duplicate = deduplicate_name(row["name"])
        if is_duplicate:
            continue
        n = int(row["n"])
        if n not in triple:
            continue
        s = int(row["stem"])
        f = int(row["Adams filtration"])
        tri = (n, s, f)
        dims[tri] = dims.get(tri, 0) + 1
        map_type = fiber_map_for_sphere(n, n_base)
        cell = fiber_cell(row, map_type)
        mask = 0
        if cell:
            for t in cell.split(";"):
                t = t.strip()
                if not t:
                    continue
                tc = parse_node_coordinates(t)
                mask |= 1 << (tc["index"] if tc else 0)
            max_src_f[map_type] = max(max_src_f[map_type], f)
            max_src_total[map_type] = max(max_src_total[map_type], s + f)
        masks.setdefault(tri, []).append(mask)
    ranks = {tri: _f2_rank(ms) for tri, ms in masks.items()}

    def dim(tri):
        return dims.get(tri, 0)

    def rank(tri):
        return ranks.get(tri, 0)

    cells = []
    totals = {"E": 0, "H": 0, "P": 0}
    for (n, _, _), r in ranks.items():
        totals[fiber_map_for_sphere(n, n_base)] += r

    n_mid = n_base + 1
    n_top = 2 * n_base + 1
    for sigma in range(0, total_cap + 1):
        for f in range(0, total_cap - sigma + 1):
            pos_0 = (n_base, sigma, f)
            pos_a = (n_mid, sigma, f)
            pos_b = (n_top, sigma - n_base, f - 1)
            pos_c = (n_base, sigma - 1, f + 1)
            comparisons = [
                # at U^{N+1,sigma,f}: rank E in vs dim ker H
                (pos_a, "E", "H", rank(pos_0), pos_0),
                # at U^{2N+1,sigma-N,f-1}: rank H in vs dim ker P
                (pos_b, "H", "P", rank(pos_a), pos_a),
                # at U^{N,sigma-1,f+1}: rank P in vs dim ker E
                (pos_c, "P", "E", rank(pos_b), pos_b),
            ]

            def beyond_frontier(map_type, src):
                # An empty cell at src is indistinguishable from truncation
                # when src lies beyond the map's recorded-domain frontier.
                return src[2] > max_src_f[map_type] or src[1] + src[2] > max_src_total[map_type]

            for pos, map_in, map_out, rank_in, in_src in comparisons:
                d = dim(pos)
                if d == 0 and rank_in == 0:
                    continue
                # Skip comparisons the truncated map data cannot decide: the
                # map-in edge originates at in_src, the map-out edge at pos
                # itself. If either source is past that map's frontier, its
                # rank/kernel counts are unreliable (truncation, not zero).
                if beyond_frontier(map_in, in_src) or beyond_frontier(map_out, pos):
                    continue
                ker_out = d - rank(pos)
                if rank_in != ker_out:
                    # The cell's own master column: S^N / S^{N+1} classes sit
                    # at their stem, S^{2N+1} classes at stem + N (pos_c lives
                    # one column left of the chain anchor sigma).
                    col = pos[1] + (n_base if pos[0] == n_top else 0)
                    cells.append(
                        {
                            "sphere": pos[0],
                            "stem": pos[1],
                            "f": pos[2],
                            "sigma": col,
                            "dim": d,
                            "map_in": map_in,
                            "map_out": map_out,
                            "rank_in": rank_in,
                            "ker_out": ker_out,
                        }
                    )
    totals["flagged"] = len(cells)
    return {"cells": cells, "totals": totals}


def extract_node_attributes(row, view_mode="sphere", hit_map=None, uncertain_targets=None, fiber_uncertain=None):
    ret = []
    if row.get("tautorsion", []):
        torsion = int(row["tautorsion"])
        if torsion >= 4:
            ret.append("tau4plus")
        elif torsion > 0:
            ret.append(f"tau{torsion}")

    # Add differential coloring for stem mode. Supporting a differential wins
    # over being hit by one (STEM_VIEW_SPEC.md).
    if view_mode == "stem":
        node_name, _ = deduplicate_name(row["name"])
        nd = row.get("nulldif")
        is_unc_src = bool(nd) and str(nd) != "nan" and str(nd).strip() != ""
        is_unc_tgt = bool(uncertain_targets) and node_name in uncertain_targets

        drt = row.get("drtarget")
        if drt and str(drt) != "nan" and drt != "":
            # Node supports a differential — open circle in the d_r color
            sc = parse_node_coordinates(node_name)
            tc = parse_node_coordinates(str(drt).split(";")[0])
            if sc and tc:
                ret.append(f"diff_d{tc['f'] - sc['f']}_open")
        elif hit_map and node_name in hit_map:
            # Node is hit by a differential — filled circle in the d_r color
            ret.append(f"diff_d{hit_map[node_name]}_filled")
        elif is_unc_src:
            # Node may support an uncertain differential — same open-circle
            # d_r convention as a determined source; the "?" glyph conveys
            # the uncertainty.
            sc = parse_node_coordinates(node_name)
            tc = parse_node_coordinates(str(nd).split(";")[0].strip())
            if sc and tc:
                ret.append(f"diff_d{tc['f'] - sc['f']}_open")
        elif is_unc_tgt and uncertain_targets[node_name] is not None:
            # Node may be hit by an uncertain differential — same filled
            # d_r convention as a determined target.
            ret.append(f"diff_d{uncertain_targets[node_name]}_filled")

        # "?" uncertainty markers (stem mode only): a node whose row has a
        # nonempty nulldif may support an uncertain differential; a node
        # listed in some other row's nulldif may be hit by one.
        if is_unc_src:
            ret.append("uncertain_src")
        if is_unc_tgt:
            ret.append("uncertain_tgt")

    # Fiber view, E-infinity (max) page only (Task 4): color nodes involved in
    # an UNCERTAIN Adams differential with their d_r color, so exactness
    # failures attributable to a missed Adams differential are visible. Uses the
    # CROSS-PAGE aggregate (fiber_uncertain), so a class involved in an uncertain
    # d2/d3/d4/d5 is colored with THAT d_r, not only the current page's d_max.
    # Keyed by tridegree (sphere_stem_f). Supporting an uncertain differential
    # (open ring) wins over being hit by one (filled), mirroring stem view.
    if view_mode == "fiber" and fiber_uncertain is not None:
        unc_src, unc_tgt = fiber_uncertain
        try:
            key = f"{int(row['n'])}_{int(row['stem'])}_{int(row['Adams filtration'])}"
        except (KeyError, ValueError, TypeError):
            key = None
        if key is not None and key in unc_src:
            ret.append(f"diff_d{unc_src[key]}_open")
            ret.append("uncertain_src")
        elif key is not None and key in unc_tgt:
            ret.append(f"diff_d{unc_tgt[key]}_filled")
            ret.append("uncertain_tgt")

    return ret


def get_differential_info(row, all_nodes_df, node_name):
    """
    Determine if a node supports or is the target of a differential.
    Returns (diff_type, is_source) or None.
    """
    # Check if this node has a dr target (supports a differential)
    if row.get("drtarget") and str(row["drtarget"]) != "nan" and row["drtarget"] != "":
        # Parse differential height from current node coordinates
        source_coords = parse_node_coordinates(node_name)
        target_name = str(row["drtarget"]).split(";")[0]  # Handle multiple targets
        target_coords = parse_node_coordinates(target_name)
        
        if source_coords and target_coords:
            height = target_coords["f"] - source_coords["f"]
            return (height, True)  # This node supports a d{height} differential
    
    # Check if this node is the target of any dr differential
    for other_row in _rows(all_nodes_df):
        if other_row.get("drtarget") and str(other_row["drtarget"]) != "nan":
            targets = str(other_row["drtarget"]).split(";")
            if node_name in targets:
                # This node is a target - compute differential height
                other_name, _ = deduplicate_name(other_row["name"])
                source_coords = parse_node_coordinates(other_name)
                target_coords = parse_node_coordinates(node_name)
                
                if source_coords and target_coords:
                    height = target_coords["f"] - source_coords["f"]
                    return (height, False)  # This node is target of d{height} differential
    
    return None


def extract_edge_attributes(row, edge_type, nodes):
    ret = []
    # Handle nulldif special case - column is named "nulldif" not "nulldiftarget"
    if edge_type == "nulldif":
        target_node = row["nulldif"]
        target_info = try_get_key(row, f"{edge_type}info")
    elif edge_type == "E":
        # EHP CSVs store suspension targets in the "E" column
        target_node = row["E"] if "E" in row else try_get_key(row, "Etarget", "")
        target_info = try_get_key(row, "Einfo")
    else:
        target_node = row[f"{edge_type}target"]
        # Info fields are sometimes missing, so we default to an empty string
        target_info = try_get_key(row, f"{edge_type}info")
    if edge_type == "dr":
        # Compute height for dr edges and assign specific differential type (d2, d3, d4, etc.)
        source_name, _ = deduplicate_name(row["name"])
        source_coords = parse_node_coordinates(source_name)
        # Handle semicolon-separated targets by using the first target for height calculation
        first_target = target_node.split(";")[0] if ";" in target_node else target_node
        target_coords = parse_node_coordinates(first_target)
        if source_coords and target_coords:
            height = target_coords["f"] - source_coords["f"]
            # Add specific differential type based on height
            if height > 0:
                ret.append(f"d{height}")  # e.g., "d2", "d3", "d4"
    elif edge_type == "E":
        ret.append("E")  # E-type edges for stem mode
        # An E edge is colored by the class it hits: inherit the d_r color of
        # the target node's differential state (STEM_VIEW_SPEC.md).
        first_target = str(target_node).split(";")[0]
        for attr in try_get_key(nodes.get(first_target, {}), "attributes", default=[]):
            if isinstance(attr, str) and attr.startswith("diff_d"):
                ret.append(attr.split("_")[1])  # "diff_d3_open" -> "d3"
    elif edge_type == "nulldif":
        # Compute length for nulldif based on Adams filtration difference
        source_name, _ = deduplicate_name(row["name"])
        source_coords = parse_node_coordinates(source_name)
        # Handle semicolon-separated targets by using the first target for length calculation
        first_target = target_node.split(";")[0] if ";" in target_node else target_node
        target_coords = parse_node_coordinates(first_target)
        if source_coords and target_coords:
            length = abs(target_coords["f"] - source_coords["f"])
            # Add length-based type (n2, n3, n4, etc.)
            if length > 0:
                ret.append(f"n{length}")  # e.g., "n2", "n3", "n4"
    elif target_node in nodes and edge_type not in ["nulldif"]:
        # Edges into a node inherit the attributes of their target, as long as they aren't
        # nulldiff edges (which get their own coloring logic)
        target_node_attributes = try_get_key(
            nodes.get(target_node, {}), "attributes", default=[]
        )
        ret.extend(target_node_attributes)
    if target_node == "loc" or target_info == "loc":
        # This edge is an arrow
        if edge_type == "h1":
            # We treat h1 towers differently because they are also always red
            ret.append("h1tower")
        else:
            ret.append({"arrowTip": "simple"})
    # Only process info fields for non-differential edges (h0, h1, h2, E)
    # For dr and nulldif edges, we compute their types directly above
    if target_info and edge_type not in ["dr", "nulldif"]:
        # The info field has other instructions for the edge. We treat them as aliases and let
        # SeqSee handle them. The only exception is "h", which we need to tag with "edge_type" so
        # the correct alias is applied.
        if isinstance(target_info, float):
            target_info = str(int(target_info))
        extra_attributes = target_info.split(" ")
        if "h" in extra_attributes:
            extra_attributes.remove("h")
            extra_attributes.append(f"h{edge_type}")
        ret.extend(extra_attributes)

    return ret


def label_from_node_name(node_name):
    """Apply substitutions to a node name to generate a label, wrapped in dollar signs for Latex."""
    label = node_name
    for pattern, replacement in substitutions:
        label = pattern.sub(replacement, label)
    if label:
        label = f"${label}$"
    return label


def deduplicate_name(name):
    """Deduplicate names by removing all variations of 'again'. We also return whether the name was
    a duplicate."""
    # I suggest we change the csv format to remove the "again" suffixes, and instead have a
    # semicolon-separated list of names for the targets of the edges. However, the input is
    # from a legacy dataset, so I don't have control over that.

    # First strip outer opening and closing parentheses if both are present. This makes it possible
    # for the regex to only match suffixes, while not affecting the rest of the name.
    if name.startswith("(") and name.endswith(")"):
        name = name[1:-1]
    if detect_again.search(name):
        return (detect_again.sub("", name), True)
    return (name, False)


def nodes_to_json(df, view_mode="sphere", filter_value=None, highlight_mode=None, highlight_targets=None, fiber_uncertain=None):
    nodes = {}
    hit_map = build_hit_map(df) if view_mode == "stem" else None
    uncertain_targets = build_uncertain_target_set(df) if view_mode == "stem" else None
    # Fiber E-infinity coloring (Task 4) is driven by fiber_uncertain, the
    # cross-page uncertain-differential aggregate, passed in only on the max
    # page (None elsewhere, so non-max fiber pages never color).
    fiber_dims = build_tridegree_dims(df) if view_mode == "fiber" else None
    for row in _rows(df):
        # Process node information
        node_name, is_duplicate = deduplicate_name(row["name"])
        if is_duplicate:
            continue

        # Filter by view mode if filter_value is provided
        if filter_value is not None:
            if view_mode == "sphere" and "n" in row and int(row["n"]) != filter_value:
                continue
            elif view_mode == "stem" and int(row["stem"]) != filter_value:
                continue
            elif view_mode == "fiber" and int(row["n"]) not in fiber_spheres(filter_value):
                continue

        # Stem charts stop at the stable edge n = k+2 — stable copies beyond
        # it would just duplicate the last column (STEM_VIEW_SPEC.md).
        if view_mode == "stem" and "n" in row and int(row["n"]) > int(row["stem"]) + 2:
            continue

        # Set coordinates based on view mode
        if view_mode == "sphere":
            x = int(row["stem"])  # s value on x-axis
            y = int(row["Adams filtration"])  # f value on y-axis
        elif view_mode == "stem":
            x = int(row["n"]) if "n" in row else 0  # n value on x-axis
            y = int(row["Adams filtration"])  # f value on y-axis
        elif view_mode == "fiber":
            # Unrolled fiber-sequence column (see module comment above): each
            # of E/H/P steps exactly +1 column right. Shifted to start at 0
            # after the loop. Schema requires integer x — these are integers.
            n = int(row["n"])
            s = int(row["stem"])
            nb = filter_value if filter_value is not None else 0
            if n == nb + 1:
                x = -3 * s + 1
            elif n == 2 * nb + 1:
                x = -3 * s - 3 * nb + 2
            else:  # n == nb (S^N)
                x = -3 * s
            y = int(row["Adams filtration"])  # f value on y-axis, unshifted
        else:
            # Default to sphere mode
            x = int(row["stem"])
            y = int(row["Adams filtration"])

        label = try_get_key(row, "label", "")
        if not label:
            label = ""

        if view_mode == "fiber" and filter_value is not None:
            # Enriched tooltip: LaTeX name, sphere, tridegree, map image
            label = fiber_node_label(row, node_name, filter_value, fiber_dims, label)

        node_data = {
            "x": x,
            "y": y,
            "label": label,
        }
        if try_get_key(row, "weight", None):
            node_data["label"] += (
                f"    ({row['weight']})" if node_data["label"] else f"({row['weight']})"
            )
        if try_get_key(row, "shift", None):
            node_data["position"] = int(row["shift"])
        
        if attributes := extract_node_attributes(row, view_mode, hit_map, uncertain_targets, fiber_uncertain):
            # Only add an attributes key if there are attributes to add
            node_data["attributes"] = attributes

        # Read J-map targets for SO nodes
        j_value = try_get_key(row, "J", "")
        if j_value and str(j_value).strip():
            j_str = str(j_value).strip()
            targets = [t.strip() for t in j_str.split(";") if t.strip()]
            if len(targets) == 1:
                node_data["jmap"] = targets[0]
            elif len(targets) > 1:
                node_data["jmap"] = targets

        nodes[node_name] = node_data

    # Fiber columns are computed as -3s (+offset), i.e. negative for positive
    # stems; shift so the leftmost column is 0.
    if view_mode == "fiber" and nodes:
        min_x = min(nd["x"] for nd in nodes.values())
        if min_x:
            for nd in nodes.values():
                nd["x"] -= min_x
    return nodes


def edges_to_json(df, nodes, view_mode="sphere", filter_value=None):
    # Fiber view has its own edge builder: exactly one map (E/H/P) per
    # source sphere of the display triple, plus zero-or-hidden stubs.
    if view_mode == "fiber" and filter_value is not None:
        return fiber_edges_to_json(df, nodes, filter_value, build_tridegree_dims(df))
    edges = []
    # Define which edge types to process based on view mode
    if view_mode == "sphere":
        edge_types = ["h0", "h1", "h2", "dr", "nulldif"]
    elif view_mode == "stem":
        # Stem view uses the lh0 map (diagonal, n->n-1 f->f+1) and the
        # corrected vertical "h0lh0" = h0 + E∘lh0 (from the Rust CSV's
        # lh0target / h0lh0target columns) INSTEAD of raw h0. E is unchanged.
        # Sphere/fiber views keep raw h0.
        edge_types = ["h0lh0", "lh0", "E"]
    else:
        edge_types = ["h0", "h1", "h2", "dr", "nulldif"]  # default to sphere
        
    for row in _rows(df):
        node_name, _ = deduplicate_name(row["name"])
        if node_name not in nodes:
            # Row is outside the current slice; without this, out-of-slice
            # sources would emit dangling edges (e.g. stem-mode arrows).
            continue
        for edge_type in edge_types:
            # Stem byte-identity: the h_i-on-identity product rows (added for
            # the fiber view) make the unit class (f==0) carry an h0 term,
            # which would otherwise leak a vertical structline into stem-0
            # charts. That term is folded into "h0lh0" now, so suppress the
            # corrected vertical on the unit row (stem mode only). The lh0
            # diagonal is genuine map data and is NOT suppressed.
            if (
                edge_type == "h0lh0"
                and view_mode == "stem"
                and "Adams filtration" in row
                and int(row["Adams filtration"]) == 0
            ):
                continue
            # Handle nulldif special case - column is named "nulldif" not "nulldiftarget"
            if edge_type == "nulldif":
                target_col = "nulldif"
                info_col = f"{edge_type}info"
            elif edge_type == "E":
                # EHP CSVs store suspension targets in the "E" column
                target_col = "E" if "E" in row else "Etarget"
                info_col = "Einfo"
            else:
                target_col = f"{edge_type}target"
                info_col = f"{edge_type}info"
            if target_col not in row:
                # In some CSVs, the target column is missing completely
                continue
            target_value = row[target_col]
            if target_value == "":
                continue  # Skip processing if the value is NaN or empty
            if str(target_value) == "nan":
                continue

            # Removed verbose print for batch processing - uncomment for debugging
            # print(f"{node_name} targeting {str(target_value)}")
            for target_node in str(target_value).split(";"):
                target_info = try_get_key(row, info_col, "")
                edge_data = {"source": node_name}
                if pd.notna(target_node):
                    if target_node in nodes:
                        # This is a structline
                        edge_data["target"] = target_node
                    elif target_node == "loc" or target_info == "loc":
                        # This is an arrow
                        edge_data["offset"] = edge_offset(edge_type, arrow_length)
                    elif edge_type in ("E", "lh0", "h0lh0") and view_mode == "stem":
                        # Map target lies beyond the displayed range (E past the
                        # stable edge; lh0/h0lh0 onto an off-window sphere):
                        # draw a short offset arrow instead of a structline.
                        edge_data["offset"] = edge_offset(edge_type, arrow_length)
                    else:
                        print(
                            f"Invalid target node: ({target_node}) for {edge_type} on ({node_name})"
                        )
                        continue
                elif target_info == "free" or target_info == "loc":
                    # This is also an arrow, but with a different notation
                    edge_data["offset"] = edge_offset(edge_type, arrow_length)
                else:
                    # no edge to be drawn
                    continue

                if attributes := extract_edge_attributes(row, edge_type, nodes=nodes):
                    # Only add an attributes key if there are attributes to add
                    edge_data["attributes"] = attributes

                if (
                    edge_type in ("E", "lh0", "h0lh0")
                    and view_mode == "stem"
                    and "offset" in edge_data
                ):
                    # Off-window map arrows need an explicit arrowtip:
                    # extract_edge_attributes only adds one for "loc" targets,
                    # and this arrow's target is a real node name that is
                    # simply outside the displayed range.
                    attrs = edge_data.setdefault("attributes", [])
                    if not any(isinstance(a, dict) and "arrowTip" in a for a in attrs):
                        attrs.append({"arrowTip": "simple"})

                edges.append(edge_data)
        # Check for `tauextn` if we're printing an E_infinity page
        if target_node := try_get_key(row, "tauextn"):
            if target_node in nodes:
                edges.append(
                    {
                        "source": node_name,
                        "target": target_node,
                        "attributes": ["tauextn"],
                    }
                )
    return edges


def extract_highlight_targets(df, highlight_mode, source_n):
    """Extract the target nodes for highlighting based on the mode."""
    targets = set()
    
    if highlight_mode == "E-forward":
        # E column contains targets in n+1 sphere (forward: where current sphere maps TO)
        for row in _rows(df):
            if int(row["n"]) == source_n:
                e_targets = try_get_key(row, "E", "")
                if e_targets and not pd.isna(e_targets):
                    # Split by comma and clean up
                    for target in str(e_targets).split(','):
                        target = target.strip()
                        if target:
                            targets.add(target)
    
    elif highlight_mode == "E-backward":
        # Find elements that map TO current sphere via E-map (backward: what maps INTO current sphere)
        for row in _rows(df):
            e_targets = try_get_key(row, "E", "")
            if e_targets and not pd.isna(e_targets):
                # Split by comma and check if any target is in our target sphere
                for target in str(e_targets).split(','):
                    target = target.strip()
                    if target:
                        # Parse target to check if it's in our sphere
                        target_parts = target.split('_')
                        if len(target_parts) >= 2 and target_parts[0].isdigit():
                            target_n = int(target_parts[0])
                            if target_n == source_n:  # This element maps into our sphere
                                source_name = row["name"]
                                if pd.notna(source_name):
                                    targets.add(str(source_name).strip())
    
    elif highlight_mode == "H-forward":
        # H column contains targets in 2*n-1 sphere (forward: where current sphere maps TO)
        for row in _rows(df):
            if int(row["n"]) == source_n:
                h_targets = try_get_key(row, "H", "")
                if h_targets and not pd.isna(h_targets):
                    for target in str(h_targets).split(','):
                        target = target.strip()
                        if target:
                            targets.add(target)
    
    elif highlight_mode == "H-backward":
        # Find elements that map TO current sphere via H-map (backward: what maps INTO current sphere)
        for row in _rows(df):
            h_targets = try_get_key(row, "H", "")
            if h_targets and not pd.isna(h_targets):
                for target in str(h_targets).split(','):
                    target = target.strip()
                    if target:
                        # Parse target to check if it's in our sphere
                        target_parts = target.split('_')
                        if len(target_parts) >= 2 and target_parts[0].isdigit():
                            target_n = int(target_parts[0])
                            if target_n == source_n:  # This element maps into our sphere
                                source_name = row["name"]
                                if pd.notna(source_name):
                                    targets.add(str(source_name).strip())
    
    elif highlight_mode == "P-forward":
        # P column contains targets: odd n ≥ 5 → (n-1)/2 sphere
        if source_n % 2 == 1 and source_n >= 5:  # Only for odd n ≥ 5
            for row in _rows(df):
                if int(row["n"]) == source_n:
                    p_targets = try_get_key(row, "P", "")
                    if p_targets and not pd.isna(p_targets):
                        for target in str(p_targets).split(','):
                            target = target.strip()
                            if target:
                                targets.add(target)
    
    elif highlight_mode == "P-backward":
        # Find elements that map TO current sphere via P-map (backward: what maps INTO current sphere)
        # P-map: odd n → (n-1)/2, so to find what maps to even source_n, look for odd n = 2*source_n + 1
        target_odd_n = 2 * source_n + 1
        for row in _rows(df):
            if int(row["n"]) == target_odd_n:  # Look in the odd sphere that maps to us
                p_targets = try_get_key(row, "P", "")
                if p_targets and not pd.isna(p_targets):
                    for target in str(p_targets).split(','):
                        target = target.strip()
                        if target:
                            # Parse target to check if it's in our sphere
                            target_parts = target.split('_')
                            if len(target_parts) >= 2 and target_parts[0].isdigit():
                                target_n = int(target_parts[0])
                                if target_n == source_n:  # This element maps into our sphere
                                    source_name = row["name"]
                                    if pd.notna(source_name):
                                        targets.add(str(source_name).strip())

    elif highlight_mode == "J-forward":
        # J column contains targets in S^n sphere (forward: where SO(n) maps TO)
        for row in _rows(df):
            if int(row["n"]) == source_n:
                j_targets = try_get_key(row, "J", "")
                if j_targets and not pd.isna(j_targets):
                    for target in str(j_targets).split(';'):
                        target = target.strip()
                        if target:
                            targets.add(target)

    return targets



def process_csv(input_file, output_file, view_mode="sphere", filter_value=None, highlight_mode=None, source_csv=None, quiet=False):
    # Define the JSON schema
    schema = load_schema()

    # Load CSV data
    df = _load_csv_cached(input_file)

    # Build a minimal header - let main.py handle theming
    header = {}

    if view_mode == "fiber" and filter_value is not None:
        # Window cap T = max total degree over ALL rows of the CSV; the
        # S^{2N+1} sub-column only exists where (sigma - N) + f <= T.
        total_cap = int((df["stem"] + df["Adams filtration"]).max())
        has_top = bool((df["n"] == 2 * filter_value + 1).any())
        header["metadata"] = {
            "fiberN": int(filter_value),
            "fiberT": total_cap,
            "fiberHasTop": has_top,
            "fiberMaxPage": fiber_is_max_page(input_file),
            "fiberDiagnostics": compute_fiber_diagnostics(df, filter_value, total_cap),
        }

    # Extract highlight targets if in highlight mode
    highlight_targets = None
    if highlight_mode and source_csv:
        # Load source CSV to get the mapping data
        source_df = _load_csv_cached(source_csv)
        # Determine source_n based on the highlight mode and direction
        if highlight_mode == "E-forward" and filter_value:
            source_n = filter_value - 1  # Forward: coming from n-1 to n
        elif highlight_mode == "E-backward" and filter_value:
            source_n = filter_value + 1  # Backward: coming from n+1 to n
        elif highlight_mode == "H-forward" and filter_value:
            # Forward: coming from n to 2n-1, so source_n = (filter_value + 1) // 2
            source_n = (filter_value + 1) // 2
        elif highlight_mode == "H-backward" and filter_value:
            # Backward: coming from (n+1)/2 to n, so source_n = 2*filter_value - 1
            source_n = 2 * filter_value - 1
        elif highlight_mode == "P-forward" and filter_value:
            # Forward: coming from odd n ≥ 5 to (n-1)/2, so source_n = 2*filter_value + 1
            source_n = 2 * filter_value + 1
        elif highlight_mode == "P-backward" and filter_value:
            # Backward: coming from even n to 2n+1, so source_n = (filter_value - 1) / 2
            source_n = (filter_value - 1) // 2
        elif highlight_mode == "J-forward" and filter_value:
            # Forward: J-map from SO(n) to S^n, so source_n = filter_value (same n)
            source_n = filter_value
        else:
            source_n = filter_value

        if source_n:
            highlight_targets = extract_highlight_targets(source_df, highlight_mode, source_n)

    # Process nodes first. Fiber E-infinity coloring (Task 4) is gated on the
    # max page, detected from the input CSV filename + EHP_MAX_PAGE env; on the
    # max page we aggregate uncertain differentials from ALL pages E2..E{max}.
    fiber_uncertain = None
    if view_mode == "fiber" and fiber_is_max_page(input_file):
        fiber_uncertain = build_fiber_uncertain_multipage(input_file, filter_value)
    nodes = nodes_to_json(
        df, view_mode, filter_value, highlight_mode, highlight_targets,
        fiber_uncertain=fiber_uncertain,
    )

    # Process edges after, since they depend on nodes
    edges = edges_to_json(df, nodes, view_mode, filter_value)

    # Combine the data into a single JSON object
    json_data = {
        "$schema": "https://raw.githubusercontent.com/JoeyBF/SeqSee/refs/heads/master/seqsee/input_schema.json",
        "header": header,
        "nodes": nodes,
        "edges": edges,
    }

    # Validation and output
    try:
        if _VALIDATE:
            validate(instance=json_data, schema=schema)
        # NB: compact_json is deliberately NOT replaced by stdlib json — its
        # output is a table-aligned format (padded keys, right-aligned numbers,
        # trailing spaces after line-end commas, length-budget object/array
        # inlining) that json.dumps cannot reproduce byte-identically with any
        # separators/indent combination, and the .json files are consumed
        # downstream (mapview jmap annotation in ehp_chart.rs, sidebyside
        # src/tgt inputs), so their bytes are kept stable. The render path no
        # longer re-reads this file (ehp_batch passes the returned dict to
        # process_json directly).
        formatter = Formatter()
        formatter.indent_spaces = 2
        formatter.dump(json_data, output_file)
        if not quiet:
            print("JSON data successfully generated and validated against the schema.")
        return json_data
    except Exception as e:
        if not quiet:
            print("Validation error:", e)
        raise e


def process_csv_to_dict(input_file, view_mode="sphere", filter_value=None, highlight_mode=None, source_csv=None):
    """
    Process CSV to JSON data structure without writing to file.
    Useful for batch processing and multi-chart generation.
    """
    # Define the JSON schema
    schema = load_schema()

    # Load CSV data
    df = _load_csv_cached(input_file)

    # Build a minimal header - let main.py handle theming
    header = {}

    # Extract highlight targets if in highlight mode
    highlight_targets = None
    if highlight_mode and source_csv:
        # Load source CSV to get the mapping data
        source_df = _load_csv_cached(source_csv)
        # Determine source_n based on the highlight mode and direction
        if highlight_mode == "E-forward" and filter_value:
            source_n = filter_value - 1  # Forward: coming from n-1 to n
        elif highlight_mode == "E-backward" and filter_value:
            source_n = filter_value + 1  # Backward: coming from n+1 to n
        elif highlight_mode == "H-forward" and filter_value:
            # Forward: coming from n to 2n-1, so source_n = (filter_value + 1) // 2
            source_n = (filter_value + 1) // 2
        elif highlight_mode == "H-backward" and filter_value:
            # Backward: coming from (n+1)/2 to n, so source_n = 2*filter_value - 1
            source_n = 2 * filter_value - 1
        elif highlight_mode == "P-forward" and filter_value:
            # Forward: coming from odd n ≥ 5 to (n-1)/2, so source_n = 2*filter_value + 1
            source_n = 2 * filter_value + 1
        elif highlight_mode == "P-backward" and filter_value:
            # Backward: coming from even n to 2n+1, so source_n = (filter_value - 1) / 2
            source_n = (filter_value - 1) // 2
        elif highlight_mode == "J-forward" and filter_value:
            # Forward: J-map from SO(n) to S^n, so source_n = filter_value (same n)
            source_n = filter_value
        else:
            source_n = filter_value
        
        if source_n:
            highlight_targets = extract_highlight_targets(source_df, highlight_mode, source_n)

    # Process nodes first
    nodes = nodes_to_json(df, view_mode, filter_value, highlight_mode, highlight_targets)

    # Process edges after, since they depend on nodes
    edges = edges_to_json(df, nodes, view_mode)

    # Combine the data into a single JSON object
    json_data = {
        "$schema": "https://raw.githubusercontent.com/JoeyBF/SeqSee/refs/heads/master/seqsee/input_schema.json",
        "header": header,
        "nodes": nodes,
        "edges": edges,
    }

    # Validation
    try:
        if _VALIDATE:
            validate(instance=json_data, schema=schema)
        return json_data
    except Exception as e:
        print(f"Validation error: {e}")
        raise e


def batch_process_csv(input_file, output_dir, view_modes=None, filter_ranges=None, quiet=False):
    """
    Batch process a single CSV file into multiple JSON outputs.
    
    Args:
        input_file: Path to CSV file
        output_dir: Directory to write JSON files 
        view_modes: List of view modes ["sphere", "stem"] (default: ["sphere"])
        filter_ranges: Dict with ranges for each view mode (default: sphere=2-72, stem=0-10)
        quiet: If True, suppress progress output
    
    Returns:
        List of generated file paths
    """
    import os
    from pathlib import Path
    
    if view_modes is None:
        view_modes = ["sphere"]
    
    if filter_ranges is None:
        filter_ranges = {
            "sphere": range(2, 73),  # S2 through S72
            "stem": range(0, 11)     # stem 0 through 10
        }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    input_path = Path(input_file)
    # Extract E number from filename (e.g., E2.csv -> 2)
    er_num = int(input_path.stem[1:]) if input_path.stem.startswith('E') and input_path.stem[1:].isdigit() else 2
    
    generated_files = []
    
    for view_mode in view_modes:
        if view_mode not in filter_ranges:
            continue
            
        for filter_value in filter_ranges[view_mode]:
            if view_mode == "sphere":
                output_file = output_dir / f"S{filter_value}_E{er_num}.json"
            else:
                output_file = output_dir / f"S{filter_value}_E{er_num}_{view_mode}.json"
            
            try:
                process_csv(
                    input_file=str(input_file),
                    output_file=str(output_file),
                    view_mode=view_mode,
                    filter_value=filter_value,
                    quiet=quiet
                )
                generated_files.append(str(output_file))
                if not quiet:
                    print(f"  ✅ Generated {output_file.name}")
            except Exception as e:
                if not quiet:
                    print(f"  ❌ Failed to generate {output_file.name}: {e}")
    
    return generated_files


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 7:
        print("Usage: jsonmaker <input.csv> <output.json> [view_mode] [filter_value] [highlight_mode] [source_csv]")
        print("  view_mode: 'sphere', 'stem' or 'fiber' (default: sphere)")
        print("  filter_value: integer to filter by (n for sphere, s for stem, base N for fiber)")
        print("  highlight_mode: 'E', 'H', or 'P' for highlighting (optional)")
        print("  source_csv: CSV file containing the source data for highlighting (optional)")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    view_mode = sys.argv[3] if len(sys.argv) > 3 else "sphere"
    filter_value = int(sys.argv[4]) if len(sys.argv) > 4 else None
    highlight_mode = sys.argv[5] if len(sys.argv) > 5 else None
    source_csv = sys.argv[6] if len(sys.argv) > 6 else None

    process_csv(input_file, output_file, view_mode, filter_value, highlight_mode, source_csv)


if __name__ == "__main__":
    main()
