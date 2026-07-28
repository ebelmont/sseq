"""Shared recursive unknown-affected filtering for Rust vs Python EHP comparisons.

Background (notes/SESSION_HANDOFF_2026-07-26_E4_rank_100pct.md): a Rust/Python
mismatch at page R is not necessarily a bug. It can be the documented,
intentional TurnContext divergence (crates/ehp-core/src/pageturning.rs:197-215)
-- Rust quotients by a *determined* differential entry even at a degree that
also has some *unknown* entry; Python's original code zeroes the whole matrix
at any such degree instead. That divergence is not confined to the page where
the unknown entry lives: an unknown d_2 at E2 can make E3_rank.csv differ,
which then makes E4_rank.csv (and every product/map built on top of it)
differ too, and so on up the tower.

Filtering a page-R comparison against only page R's own unknown list
under-excludes and produces false "bugs" (this happened twice: once when E4's
rank.csv was checked with only E4's own unknowns considered, and the residual
was wrongly hand-waved as a "max_t edge effect" rather than traced). This
module builds the exclusion set recursively across the whole tower instead.

**The "own unknowns" filter must union BOTH sides' unknown sets, not just
Rust's.** This was written after seeing a real gap (rust-only-determined
degrees) between Rust's and Python's E2/E3 unknown sets at DEFAULT Rust
config, e.g. "48 rank mismatches" at E3 that looked "unexplained" without
this union. **That gap turned out to be a Rust config bug, not a legitimate
asymmetry -- see CLAUDE.md's `EHP_D2_LINEAR=0` note.** With the correct
config (`EHP_D2_LINEAR=0` + `EHP_OUTSIDE_DIFFS`/`EHP_OUTSIDE_SKIP=""`),
Rust's and Python's own unknown sets come out byte/set-IDENTICAL at both E2
and E3 (confirmed 2026-07-26) -- there is no more rust-vs-python asymmetry
in "own unknowns" to union away. The union logic here is kept anyway (it's
a correct no-op when both sides already agree, and is cheap insurance
against this regressing, or against a genuine future asymmetry at a later
page), but don't expect it to be doing real work at E2/E3 with the correct
config -- if it IS excluding something nonzero there, that's now a signal
worth investigating rather than assuming it's the old, understood gap.

The genuinely-remaining, still-real divergence at E3-and-up is a different,
unrelated, already-accepted one: the documented TurnContext partial-vs-
whole-matrix-zeroing policy (see the paragraph above) firing on E2's own
unknown differentials -- BOTH sides have the identical 2085-entry E2
unknown set (confirmed), but Rust's page-turn quotients by determined
matrix entries at an unknown-affected degree while Python zeroes the whole
matrix there, producing real, expected rank/product/map differences
downstream. That's what this filter is still catching at E3 (1450 rank
mismatches, all explained; 0 unexplained, confirmed 2026-07-26 with the
corrected config) -- not a Rust-knows-more asymmetry.

**Use `UnknownAffectedTower` as the DEFAULT filter for every comparison from
E3 upward** (rank, products, E/H/P maps). Do not reintroduce a
single-level-only or single-side-only filter for a new page without a
documented reason.

**Use `is_excluded_any`, not `is_excluded`, for filtering comparisons.**
`is_excluded(t, r)` only checks the single MOST RECENT level -- its
docstring explains a real, confirmed gap (a degree's dimension count can
match between rust and python even when the class underneath is still
built on an EARLIER, unresolved page's uncertainty; a rank match doesn't
imply the basis/lift is trustworthy). `is_excluded_any(t)` checks every
level added to the tower so far and is the correct default. `is_excluded`
is kept only for add_page's own internal sanity check.

Usage sketch for comparing page R (built via d_R from page R-1):

    tower = UnknownAffectedTower()
    tower.add_page(2, "output/rust_E2_data", "output/rust_E2_data",
                    "data/E2/E2_rank.csv", "data/E2/E2_rank.csv", max_t=80)  # same file both sides -> mism empty
    tower.add_page(3, "output/rust_E3_data", "<python>/E3_data",
                    "output/rust_E3_page/E3_rank.csv", "<python>/E3/E3_rank.csv", max_t=79)
    ...
    tower.is_excluded_any((n, s, f))
"""
import csv


def stable_rep(t):
    n, s, f = t
    return (s + 2, s, f) if n > s + 2 else (n, s, f)


def load_rank(path):
    d = {}
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            d[(int(row["n"]), int(row["s"]), int(row["f"]))] = int(row["dimension"])
    return d


def _one_side_unknown_reps(diffs_path, unknown_path):
    """stable_reps whose own outgoing d_r is unknown, per ONE side's SAT
    solve. Internal helper -- see own_unknown_reps for the both-sides union
    that should actually be used for filtering."""
    diffs = []
    with open(diffs_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n, s, f, row, col = (int(x) for x in line.strip("()").split(","))
            diffs.append((n, s, f))
    unknown_idx = set()
    with open(unknown_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                unknown_idx.add(int(line))
    return {stable_rep(diffs[i]) for i in unknown_idx}


def own_unknown_reps(rust_data_dir, py_data_dir, r, rank_path=None, max_t=None):
    """Stable_reps whose own outgoing d_r is unknown on EITHER side (from
    page r's SAT solve -- reads {rust,py}_data_dir/diffs and /unknown), plus
    the max_t-truncation heuristic: a live class whose only outgoing d_r
    target is off the edge of the tracked data (no diff-var block at all, but
    the target itself has genuine positive dimension) is also "uncertain,
    don't expect an exact match" -- not the same as a target that's absent
    because its own dimension is provably zero.

    Unions both sides' unknown sets (not just Rust's) as defense in depth --
    with the correct Rust config (EHP_D2_LINEAR=0 + EHP_OUTSIDE_DIFFS, see
    CLAUDE.md) the two sides' own unknown sets come out identical at E2/E3
    anyway, so this union is normally a no-op; see module docstring for the
    full history (an earlier, WRONG Rust config produced a real rust-vs-
    python asymmetry here that this union was written to paper over).
    """
    reps = _one_side_unknown_reps(f"{rust_data_dir}/diffs", f"{rust_data_dir}/unknown")
    reps |= _one_side_unknown_reps(f"{py_data_dir}/diffs", f"{py_data_dir}/unknown")

    if rank_path and max_t:
        diff_stable_reps = set()
        with open(f"{rust_data_dir}/diffs") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                n, s, f, row, col = (int(x) for x in line.strip("()").split(","))
                diff_stable_reps.add(stable_rep((n, s, f)))
        dims = {k: v for k, v in load_rank(rank_path).items() if v > 0}
        for (n, s, f), dim in dims.items():
            if s + f > max_t:
                continue
            rep = stable_rep((n, s, f))
            if rep in diff_stable_reps:
                continue
            target = (n, s - 1, f + r)
            if dims.get(target, 0) > 0 and (target[1] + target[2]) > max_t:
                reps.add(rep)
    return reps


def _affected(t, reps, r):
    n, s, f = t
    return stable_rep((n, s, f)) in reps or stable_rep((n, s + 1, f - r)) in reps


class UnknownAffectedTower:
    """Accumulates the recursive unknown-affected picture page by page.

    IMPORTANT invariant: a tridegree t living on the page created by d_r only
    has a well-defined "source degree" (n, s+1, f-r) one level down, via that
    SAME r -- there is no valid direct formula linking t to a page two or
    more levels below. Reaching further back is handled by induction instead:
    each page's own rank-mismatch set (`_mism`) is required (and checked, via
    the `unexplained` return of add_page) to already be fully explained by
    the page below it, so checking t against only the IMMEDIATELY PRECEDING
    page's own-unknowns and mismatch set is sufficient -- do not loop over
    the whole chain with a single r, that silently produces wrong answers
    (checked and reverted: see notes/SESSION_HANDOFF_2026-07-26_E4_rank_100pct.md).

    Call add_page() once per page in increasing r order (r=2 for E2's own d_2
    unknowns / the page that creates E3, r=3 for E3's own d_3 unknowns / the
    page that creates E4, etc.), then use is_excluded() to test a tridegree
    on the page created by the MOST RECENTLY added r.
    """

    def __init__(self):
        self._own = {}    # r -> reps (page r's own unknown-affected tridegrees)
        self._mism = {}   # r -> reps (page r's rust-vs-python rank.csv mismatch)
        self._chain = []  # r values added so far, in order

    def add_page(self, r, rust_data_dir, py_data_dir, rust_rank_path, py_rank_path, max_t):
        """r: the d_r whose SAT solve produced this page's own unknowns.
        rust_data_dir/py_data_dir: dirs holding that page's own diffs/unknown
        files on EACH side (e.g. r=3 for E3, whose d_3 determines E4) -- both
        are read and unioned, see own_unknown_reps.
        rust_rank_path/py_rank_path: this page's OWN rank.csv (e.g. E3_rank.csv).
        For the base page (E2), pass the SAME path for both data dirs and
        both rank paths -- there is no rust-vs-python split at the input-data
        level, so both the own-unknown union and the mismatch set come out
        trivial/empty automatically.

        Returns this page's rank mismatches NOT explained by the page below it
        in the chain. Should be empty in steady state -- a nonempty result
        means a real, uninvestigated bug is hiding in the tower and would
        otherwise get silently masked as "unknown-affected" at every later
        page built on top of it. Treat a nonempty return as a hard stop:
        investigate before trusting anything added after it.
        """
        own = own_unknown_reps(rust_data_dir, py_data_dir, r, rank_path=rust_rank_path, max_t=max_t)
        rust = load_rank(rust_rank_path)
        py = load_rank(py_rank_path)
        mism = {k for k in set(rust) | set(py) if rust.get(k, 0) != py.get(k, 0)}

        # Sanity check against the page immediately below (its own r, not
        # this page's r -- see is_excluded's docstring). No earlier page
        # exists for the base page, so nothing to check there.
        prev_r = self._chain[-1] if self._chain else None
        unexplained = [k for k in mism if not (prev_r is not None and self.is_excluded(k, prev_r))]

        self._own[r] = own
        self._mism[r] = mism
        self._chain.append(r)
        return unexplained

    def _check_level(self, t, r):
        """Core single-level check, no restriction on r -- used both by
        is_excluded (restricted to the most recent level, for the
        add_page sanity check) and is_excluded_any (all levels)."""
        return _affected(t, self._own[r], r) or _affected(t, self._mism[r], r)

    def is_excluded(self, t, r):
        """t: tridegree on the page created by d_r (r must be the r of the
        MOST RECENTLY added page -- i.e. the transition immediately below t).
        Excluded if t (or its d_r source) is affected by that page's own
        unknowns, or that page's own rank comparison already diverges there
        (which transitively carries every earlier level's divergence, by the
        add_page sanity-check invariant).

        DO NOT use this for filtering comparisons at r > 2 -- use
        is_excluded_any instead. This single-level check has a real gap:
        the add_page sanity-check invariant ("upstream uncertainty always
        shows up as a rank mismatch at the next level, so checking only the
        immediately preceding level is sufficient by induction") is true for
        RANK comparisons but FALSE in general -- a degree's dimension COUNT
        can agree between rust and python even when the specific basis
        vectors / lift construction underneath it is still built on an
        earlier, unresolved page's uncertainty (confirmed 2026-07-27: E3's
        dimension at (4,69,8) matches exactly on both sides, 3=3, yet that
        degree's class is built on top of an E2 degree whose own d_2 is
        unknown-affected -- is_excluded(·, r=3) misses this entirely since
        neither _own[3] nor _mism[3] catches it, only is_excluded(·, r=2)
        does). This method is kept only for add_page's own internal sanity
        check, which genuinely does mean "immediately preceding level."
        """
        if not self._chain or self._chain[-1] != r:
            raise ValueError(f"is_excluded(r={r}) must match the most recently added page "
                              f"(chain={self._chain}) -- see class docstring")
        return self._check_level(t, r)

    def is_excluded_any(self, t):
        """t: a tridegree (same (n,s,f) meaning at every level, since
        page-turning preserves tridegree identity). True if t is
        unknown-affected at ANY level added so far, not just the most
        recent one -- the correct filter for comparisons above E3, see
        is_excluded's docstring for why checking only the most recent level
        is insufficient. This is the check comparison scripts should use."""
        return any(self._check_level(t, r) for r in self._chain)

    def suppresses_variable(self, t, page_r):
        """t: tridegree living on the page whose own d_{page_r} is being
        generated (e.g. page_r=3 for E3's own constraint/diffvar system).

        Both Rust's and Python's make_basis_single (constraints.rs /
        sat_backend.py) skip creating a d_{page_r} variable block at t if
        EITHER t itself OR t's own d_{page_r} target `(n, s-1, f+page_r)` is
        in that page's exclude_set. A degree can end up excluded on only one
        side because the exclude_set is built purely from that side's own
        upstream unknowns -- confirmed 2026-07-26 at E3: all 215 degrees
        excluded on Python only (not Rust) were fully explained by
        is_excluded_any on either the degree or its upstream source. So a
        variable-block mismatch at t is "explained" if EITHER t or t's
        page_r-target is unknown-affected at any level."""
        n, s, f = t
        target = (n, s - 1, f + page_r)
        return self.is_excluded_any(t) or self.is_excluded_any(target)
