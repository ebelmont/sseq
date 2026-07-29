import argparse
import csv
import re
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from unknown_filter import UnknownAffectedTower

SFI = re.compile(r"^(\d+)_(\d+)_(\d+)(?:_(\d+))?$")

def parse_factor(s):
    m = SFI.match(s)
    n, sdeg, f, i = m.groups()
    return (int(n), int(sdeg), int(f), int(i) if i is not None else 0)

def parse_result(s):
    if s.strip() == "0":
        return frozenset()
    return frozenset(parse_factor(t.strip()) for t in s.split(" + "))

def load_products_csv(path):
    table = {}
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            f1 = parse_factor(row["factor1"])
            f2 = parse_factor(row["factor2"])
            res = parse_result(row["result"])
            table[(f1, f2)] = res
    return table

def load_map_csv(path):
    table = {}
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            el = parse_factor(row["element"])
            img = parse_result(row["image"])
            table[el] = img
    return table

def touches_unknown(degs, tower):
    return any(tower.is_excluded_any((n, s, f)) for (n, s, f, i) in degs)

def compare_products(rust_path, py_path, tower):
    rust = load_products_csv(rust_path)
    py = load_products_csv(py_path)
    all_keys = set(rust) | set(py)
    mismatches = []
    clean_mismatches = []
    for k in all_keys:
        # Absent and explicit-empty-set both mean "maps/multiplies to zero" --
        # Python's CSV writer omits zero-image/zero-product rows entirely,
        # Rust's writes them explicitly (confirmed 2026-07-27,
        # notes/SESSION_3STAGE_E2E5_t100.md: every "real" mismatch at E3/E4/E5
        # fell in exactly this category, spot-checked against matching
        # dimensions on both sides -- a CSV convention gap, not a math bug).
        # Normalize by treating a missing key as frozenset() so this no
        # longer shows up as a mismatch at all.
        rv = rust.get(k, frozenset())
        pv = py.get(k, frozenset())
        if rv == pv:
            continue
        f1, f2 = k
        degs = [f1, f2]
        if rv:
            degs += list(rv)
        if pv:
            degs += list(pv)
        mismatches.append((k, rv, pv))
        if not touches_unknown(degs, tower):
            clean_mismatches.append((k, rv, pv))
    return len(all_keys), len(mismatches), clean_mismatches

def compare_map(rust_path, py_path, tower):
    rust = load_map_csv(rust_path)
    py = load_map_csv(py_path)
    all_keys = set(rust) | set(py)
    mismatches = []
    clean_mismatches = []
    for k in all_keys:
        # See compare_products' comment: absent == explicit-zero, normalize
        # both sides through the same default so this isn't a mismatch.
        rv = rust.get(k, frozenset())
        pv = py.get(k, frozenset())
        if rv == pv:
            continue
        degs = [k]
        if rv:
            degs += list(rv)
        if pv:
            degs += list(pv)
        mismatches.append((k, rv, pv))
        if not touches_unknown(degs, tower):
            clean_mismatches.append((k, rv, pv))
    return len(all_keys), len(mismatches), clean_mismatches

def categorize_map(rust_path, py_path, tower):
    rust = load_map_csv(rust_path)
    py = load_map_csv(py_path)
    all_keys = set(rust) | set(py)
    cats = defaultdict(int)
    examples = defaultdict(list)
    for k in all_keys:
        rv = rust.get(k, frozenset())
        pv = py.get(k, frozenset())
        if rv == pv:
            continue
        degs = [k]
        if rv:
            degs += list(rv)
        if pv:
            degs += list(pv)
        if touches_unknown(degs, tower):
            continue
        if not rv and pv:
            cat = "rust_zero_python_nonzero"
        elif not pv and rv:
            cat = "python_zero_rust_nonzero"
        else:
            cat = "both_nonzero_different_value"
        cats[cat] += 1
        if len(examples[cat]) < 5:
            examples[cat].append((k, rv, pv))
    return cats, examples

def categorize_products(rust_path, py_path, tower):
    rust = load_products_csv(rust_path)
    py = load_products_csv(py_path)
    all_keys = set(rust) | set(py)
    cats = defaultdict(int)
    examples = defaultdict(list)
    for k in all_keys:
        # Absent and explicit-empty-set both mean "maps/multiplies to zero" --
        # Python's CSV writer omits zero-image/zero-product rows entirely,
        # Rust's writes them explicitly (confirmed 2026-07-27,
        # notes/SESSION_3STAGE_E2E5_t100.md: every "real" mismatch at E3/E4/E5
        # fell in exactly this category, spot-checked against matching
        # dimensions on both sides -- a CSV convention gap, not a math bug).
        # Normalize by treating a missing key as frozenset() so this no
        # longer shows up as a mismatch at all.
        rv = rust.get(k, frozenset())
        pv = py.get(k, frozenset())
        if rv == pv:
            continue
        f1, f2 = k
        degs = [f1, f2]
        if rv:
            degs += list(rv)
        if pv:
            degs += list(pv)
        if touches_unknown(degs, tower):
            continue
        if not rv and pv:
            cat = "rust_zero_python_nonzero"
        elif not pv and rv:
            cat = "python_zero_rust_nonzero"
        else:
            cat = "both_nonzero_different_value"
        cats[cat] += 1
        if len(examples[cat]) < 5:
            examples[cat].append((k, rv, pv))
    return cats, examples

def _build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Compare Rust vs Python products/E/H/P maps at any page, "
                     "excluding tridegrees affected by the recursive unknown-affected "
                     "divergence (see scripts/unknown_filter.py). Generic over the page "
                     "tower -- no per-page hardcoding.",
    )
    ap.add_argument(
        "--page", nargs=5, action="append", default=[],
        metavar=("R", "RUST_DATA_DIR", "PY_DATA_DIR", "RUST_RANK", "PY_RANK"),
        help="one entry per page below the target, in increasing r order. "
             "RUST_DATA_DIR/PY_DATA_DIR hold that page's own diffs/unknown files (its d_R "
             "solve) on each side -- both are read and unioned when filtering, since a degree "
             "unknown on EITHER side is uncertain, see scripts/unknown_filter.py. "
             "RUST_RANK/PY_RANK are that page's own rank.csv (same path for both on the base page).",
    )
    ap.add_argument("target_prefix", help="e.g. E4 -- looks for {target_prefix}_relations.csv / _E.csv / _H.csv / _P.csv")
    ap.add_argument("rust_dir", help="e.g. output/rust_E4_page")
    ap.add_argument("py_dir", help="e.g. ~/uass/git-SAT/80_no_lh0_3/E4")
    ap.add_argument("--max-t", type=int, default=80)
    return ap


if __name__ == "__main__":
    # Usage (E2->E3->E4):
    #   python3 scripts/compare_tables.py \
    #     --page 2 output/rust_E2_data ~/uass/git-SAT/80_no_lh0_3/E2_data data/E2/E2_rank.csv data/E2/E2_rank.csv \
    #     --page 3 output/rust_E3_data ~/uass/git-SAT/80_no_lh0_3/E3_data output/rust_E3_page/E3_rank.csv ~/uass/git-SAT/80_no_lh0_3/E3/E3_rank.csv \
    #     E4 output/rust_E4_page ~/uass/git-SAT/80_no_lh0_3/E4
    args = _build_arg_parser().parse_args()
    if not args.page:
        sys.exit("need at least one --page entry (the page immediately below the target)")

    tower = UnknownAffectedTower()
    ok = True
    for r, rust_data_dir, py_data_dir, rust_rank, py_rank in args.page:
        r = int(r)
        unexplained = tower.add_page(
            r, rust_data_dir, py_data_dir, rust_rank, py_rank, max_t=args.max_t,
        )
        if unexplained:
            ok = False
            print(f"WARNING: {len(unexplained)} mismatches at page r={r} are NOT explained "
                  f"by the page below it -- investigate before trusting these results.", file=sys.stderr)

    last_r = int(args.page[-1][0])
    rust_dir, py_dir, prefix = args.rust_dir, args.py_dir, args.target_prefix

    n_keys, n_mis, clean = compare_products(f"{rust_dir}/{prefix}_relations.csv", f"{py_dir}/{prefix}_relations.csv", tower)
    print(f"\nProducts: {n_keys} total keys, {n_mis} mismatches, {len(clean)} NOT unknown-affected (real)")
    if clean:
        ok = False
    cats, examples = categorize_products(f"{rust_dir}/{prefix}_relations.csv", f"{py_dir}/{prefix}_relations.csv", tower)
    for cat, cnt in cats.items():
        print(f"    {cat}: {cnt}")
        for ex in examples[cat][:5]:
            print("      ", ex)

    for kind in ["E", "H", "P"]:
        n_keys, n_mis, clean = compare_map(f"{rust_dir}/{prefix}_{kind}.csv", f"{py_dir}/{prefix}_{kind}.csv", tower)
        print(f"\n{kind} map: {n_keys} total keys, {n_mis} mismatches, {len(clean)} NOT unknown-affected (real)")
        if clean:
            ok = False
        cats, examples = categorize_map(f"{rust_dir}/{prefix}_{kind}.csv", f"{py_dir}/{prefix}_{kind}.csv", tower)
        for cat, cnt in cats.items():
            print(f"    {cat}: {cnt}")
            for ex in examples[cat][:5]:
                print("      ", ex)

    print(f"\n{prefix} tables: {'CLEAN' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
