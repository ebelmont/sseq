"""Compare Rust vs Python rank.csv at any page R, correctly excluding
tridegrees affected by the recursive unknown-affected divergence (see
scripts/unknown_filter.py for why this has to be recursive, not just a check
against the immediately-prior page's own unknowns).

Generic over the page tower -- pass one --page entry per page from E2 up to
(but not including) the target page R; no per-page hardcoding.

Usage (E2->E3->E4, matching this project's earlier E4 investigation):
    python3 scripts/compare_rank.py --max-t 80 \\
        --page 2 output/rust_E2_data ~/uass/git-SAT/80_no_lh0_3/E2_data data/E2/E2_rank.csv data/E2/E2_rank.csv \\
        --page 3 output/rust_E3_data ~/uass/git-SAT/80_no_lh0_3/E3_data output/rust_E3_page/E3_rank.csv ~/uass/git-SAT/80_no_lh0_3/E3/E3_rank.csv \\
        output/rust_E4_page/E4_rank.csv ~/uass/git-SAT/80_no_lh0_3/E4/E4_rank.csv

For the base page (E2), pass the SAME path for RUST_DATA_DIR/PY_DATA_DIR and
for RUST_RANK/PY_RANK -- there's no rust/python split at the input-data level
(both sides load the same E2 data), so its own mismatch set comes out empty
automatically.

To check E5 instead, just add a --page 4 ... entry and point the two
trailing positional args at E5_rank.csv.
"""
import argparse
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from unknown_filter import UnknownAffectedTower, load_rank


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
    ap.add_argument("target_rust_rank")
    ap.add_argument("target_py_rank")
    ap.add_argument("--max-t", type=int, default=80)
    args = ap.parse_args()

    if not args.page:
        sys.exit("need at least one --page entry (the page immediately below the target)")

    tower = UnknownAffectedTower()
    ok = True
    for r, rust_data_dir, py_data_dir, rust_rank, py_rank in args.page:
        r = int(r)
        tower_unexplained = tower.add_page(
            r, rust_data_dir, py_data_dir, rust_rank, py_rank, max_t=args.max_t,
        )
        if tower_unexplained:
            ok = False
            print(f"WARNING: {len(tower_unexplained)} mismatches at page r={r} are NOT explained "
                  f"by the page below it -- investigate before trusting anything built on top.",
                  file=sys.stderr)
            for k in sorted(tower_unexplained)[:10]:
                print("  ", k, file=sys.stderr)

    rust_t = load_rank(args.target_rust_rank)
    py_t = load_rank(args.target_py_rank)
    keys = set(rust_t) | set(py_t)
    all_mism = [(k, rust_t.get(k, 0), py_t.get(k, 0)) for k in keys if rust_t.get(k, 0) != py_t.get(k, 0)]
    unexplained = [(k, rv, pv) for (k, rv, pv) in all_mism if not tower.is_excluded_any(k)]

    print(f"Total mismatches: {len(all_mism)}")
    print(f"Explained by recursive unknown-affected divergence: {len(all_mism) - len(unexplained)}")
    print(f"UNEXPLAINED (genuine candidates for a real bug): {len(unexplained)}")
    for k, rv, pv in sorted(unexplained)[:50]:
        print(f"  {k}: rust={rv} python={pv}")
    if unexplained:
        ok = False

    print(f"\nrank: {'CLEAN' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
