"""Compare Rust vs Python constraint systems (E/H/P/Y_constraints) for any
page R, excluding constraints touching tridegrees affected by the recursive
unknown-affected divergence (see scripts/unknown_filter.py).

Generic over the page tower -- pass one --page entry per page from E2 up to
(but not including) the target page R; no per-page hardcoding. Replaces the
old page-specific compare_e3_constraints.py / compare_e4_constraints.py,
which each had two bugs:

  1. They compared constraints as frozensets of RAW SOLVER-INTERNAL VARIABLE
     INDICES (positions into each side's own `diffs` file) directly across
     rust and python. Those indices are not comparable across the two
     systems -- rust's and python's `diffs` files can differ in length
     (e.g. rust had 224 extra E3 variables in one comparison) and once they
     diverge at some index, every constraint referencing a later variable is
     silently misaligned. This made ~86% of constraints look like
     "mismatches" that were actually just index-space noise. Fixed here by
     canonicalizing every index to its (n, s, f, row, col) tuple (via each
     side's own diffs file) before building the comparison frozensets --
     that tuple is the only representation shared between the two sides.

  2. They filtered by only the SINGLE IMMEDIATELY PRECEDING page's own
     unknowns (e.g. E4 constraints filtered only against E3's own unknown
     list), not the full recursive tower back to E2. unknown_filter.py's own
     docstring warns this under-excludes and produces false "bugs" starting
     at E4. Fixed here by using UnknownAffectedTower, the same recursive
     filter compare_rank.py/compare_tables.py already use.

     (An apparent single-SIDE gap -- Rust knowing more than Python -- was
     also chased here on 2026-07-26 and initially attributed to legitimate
     extra Rust deduction; that diagnosis was WRONG, a symptom of a confounded
     Rust config (missing EHP_D2_LINEAR=0), not a real asymmetry -- see
     CLAUDE.md. With the correct config, E2/E3's own constraint families
     compare byte-identical with 0 raw mismatches and no filtering needed
     at all; UnknownAffectedTower's `suppresses_variable` check below is
     kept as defense in depth for later pages, not because it's currently
     load-bearing at E2/E3.)

Usage (E2->E3->E4, comparing E4's own constraint system):
    python3 scripts/compare_constraints.py --max-t 80 \\
        --page 2 output/rust_E2_data ~/uass/git-SAT/80_no_lh0_3/E2_data data/E2/E2_rank.csv data/E2/E2_rank.csv \\
        --page 3 output/rust_E3_data ~/uass/git-SAT/80_no_lh0_3/E3_data output/rust_E3_page/E3_rank.csv ~/uass/git-SAT/80_no_lh0_3/E3/E3_rank.csv \\
        output/rust_E4_data ~/uass/git-SAT/80_no_lh0_3/E4_data

For the base page (E2), pass the SAME path for RUST_DATA_DIR/PY_DATA_DIR and
for RUST_RANK/PY_RANK -- there's no rust/python split at the input-data level,
so its own mismatch set comes out empty automatically.

To compare E3's own constraints instead, use a single --page 2 entry and
point the trailing positional args at E3's data dirs.
"""
import argparse
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from unknown_filter import UnknownAffectedTower


def load_diffs(path):
    diffs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n, s, f, row, col = (int(x) for x in line.strip("()").split(","))
            diffs.append((n, s, f, row, col))
    return diffs


def load_constraint_file(path, diffs):
    # Each line is a constraint: a list of raw solver-variable indices into
    # `diffs`. Those indices are internal to one side's own variable
    # numbering and are NOT comparable across rust/python (their diffs files
    # can differ in length/order), so canonicalize each index to its
    # (n, s, f, row, col) degree-tuple before building the frozenset -- that
    # tuple is the only representation shared between the two sides.
    cons = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            idxs = [int(x) for x in line.strip("[]").split(",") if x.strip() != ""]
            cons.append(frozenset(diffs[i] for i in idxs))
    return cons


def touches_unknown(fs, tower, page_r):
    return any(tower.suppresses_variable((n, s, f), page_r) for (n, s, f, _row, _col) in fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--page", nargs=5, action="append", default=[],
        metavar=("R", "RUST_DATA_DIR", "PY_DATA_DIR", "RUST_RANK", "PY_RANK"),
        help="one entry per page below the target, in increasing r order. "
             "RUST_DATA_DIR/PY_DATA_DIR hold that page's own diffs/unknown files (its d_R "
             "solve) on each side -- both are read and unioned when filtering, since a degree "
             "unknown on EITHER side is uncertain, see scripts/unknown_filter.py. "
             "RUST_RANK/PY_RANK are that page's own rank.csv (same path for both on the base page).",
    )
    ap.add_argument("target_rust_dir", help="e.g. output/rust_E4_data")
    ap.add_argument("target_py_dir", help="e.g. ~/uass/git-SAT/80_no_lh0_3/E4_data")
    ap.add_argument("--max-t", type=int, default=80)
    args = ap.parse_args()

    # No --page entries means the target IS the base page (E2): there's no
    # earlier page to build an unknown-affected tower from, so `tower` stays
    # empty and touches_unknown() is trivially False for everything -- an
    # unfiltered, exact comparison, which is correct for E2 (both sides load
    # the same input CSVs, so there's no legitimate source of divergence).
    ok = True
    tower = UnknownAffectedTower()
    for r, rust_data_dir, py_data_dir, rust_rank, py_rank in args.page:
        r = int(r)
        unexplained = tower.add_page(
            r, rust_data_dir, py_data_dir, rust_rank, py_rank, max_t=args.max_t,
        )
        if unexplained:
            ok = False
            print(f"WARNING: {len(unexplained)} mismatches at page r={r} are NOT explained "
                  f"by the page below it -- investigate before trusting these results.",
                  file=sys.stderr)

    page_r = int(args.page[-1][0]) + 1 if args.page else 2
    rust_dir, py_dir = args.target_rust_dir, args.target_py_dir

    rust_diffs = load_diffs(f"{rust_dir}/diffs")
    py_diffs = load_diffs(f"{py_dir}/diffs")
    diffs_equal = set(rust_diffs) == set(py_diffs)
    print(f"diffs: rust {len(rust_diffs)}, python {len(py_diffs)}, sets equal: {diffs_equal}")
    if not diffs_equal:
        ok = False

    for name in ["E_constraints", "H_constraints", "P_constraints", "Y_constraints"]:
        rust = load_constraint_file(f"{rust_dir}/{name}", rust_diffs)
        py = load_constraint_file(f"{py_dir}/{name}", py_diffs)
        rust_set = set(rust)
        py_set = set(py)
        rust_only = rust_set - py_set
        py_only = py_set - rust_set
        print(f"\n{name}: rust {len(rust)} lines ({len(rust_set)} unique), python {len(py)} lines ({len(py_set)} unique)")
        print(f"  rust-only: {len(rust_only)}, python-only: {len(py_only)}")
        clean_rust_only = [c for c in rust_only if not touches_unknown(c, tower, page_r)]
        clean_py_only = [c for c in py_only if not touches_unknown(c, tower, page_r)]
        print(f"  clean (non-unknown-affected) rust-only: {len(clean_rust_only)}, python-only: {len(clean_py_only)}")
        if clean_rust_only or clean_py_only:
            ok = False
        for c in clean_rust_only[:10]:
            print("    rust-only:", sorted(c))
        for c in clean_py_only[:10]:
            print("    python-only:", sorted(c))

    # unknown/offset: report set equality for `unknown`, and for `offset`
    # confirm any raw difference is confined to unknown (free) variables --
    # offset is just ONE particular solution among many when free variables
    # exist, so a difference there is only a real problem if it touches a
    # variable that's determined on both sides.
    def load_sparse(path):
        with open(path) as fh:
            return [int(x.strip().strip("[],")) for x in fh if x.strip().strip("[],") != ""]

    r_unknown = {rust_diffs[i] for i in load_sparse(f"{rust_dir}/unknown")}
    p_unknown = {py_diffs[i] for i in load_sparse(f"{py_dir}/unknown")}
    print(f"\nunknown: rust {len(r_unknown)}, python {len(p_unknown)}, sets equal: {r_unknown == p_unknown}")

    r_offset = {rust_diffs[i] for i in load_sparse(f"{rust_dir}/offset")}
    p_offset = {py_diffs[i] for i in load_sparse(f"{py_dir}/offset")}
    offset_diff = (r_offset - p_offset) | (p_offset - r_offset)
    offset_diff_non_unknown = offset_diff - r_unknown - p_unknown
    print(f"offset: symmetric diff {len(offset_diff)}, of which non-unknown-affected: {len(offset_diff_non_unknown)}")
    if offset_diff_non_unknown:
        ok = False
        for x in sorted(offset_diff_non_unknown)[:10]:
            print("    non-unknown offset diff:", x)

    print(f"\nE_{page_r} constraints: {'CLEAN' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
