#!/usr/bin/env bash
# One-command Rust-vs-Python EHP correctness check. Builds the Rust page
# tower once (diag_dump_page_tower), then runs every comparison stage
# (constraints, rank, products/E/H/P maps) for every page from E2 up to
# EHP_R_MAX, and prints a single final CLEAN/FAIL verdict.
#
# Usage:
#   scripts/verify_ehp.sh MAX_T R_MAX PY_REF_DIR [OUTSIDE_DIFFS] [RUST_DATA]
#
# MAX_T        EHP_MAX_T to run at (e.g. 60, 80, 100).
# R_MAX        highest page to walk/compare (e.g. 5 for E2..E5).
# PY_REF_DIR   Python reference dir containing E2_data/, E3_data/, E3/,
#              E4_data/, E4/, ... (e.g. ~/uass/git-SAT/100_no_lh0_0).
# OUTSIDE_DIFFS  optional, default ~/uass/git-SAT/outside_diffs.
# RUST_DATA      optional, default data/E2 (the Rust-side input CSVs).
#
# Exit code 0 iff every stage at every page came back CLEAN. On CLEAN,
# nothing further needs figuring out -- the run is a genuine pass.

set -u
cd "$(dirname "$0")/.."

MAX_T="${1:?usage: verify_ehp.sh MAX_T R_MAX PY_REF_DIR [OUTSIDE_DIFFS] [RUST_DATA]}"
R_MAX="${2:?usage: verify_ehp.sh MAX_T R_MAX PY_REF_DIR [OUTSIDE_DIFFS] [RUST_DATA]}"
PY_REF_DIR="${3:?usage: verify_ehp.sh MAX_T R_MAX PY_REF_DIR [OUTSIDE_DIFFS] [RUST_DATA]}"
OUTSIDE_DIFFS="${4:-$HOME/uass/git-SAT/outside_diffs}"
RUST_DATA="${5:-data/E2}"

PY_REF_DIR="${PY_REF_DIR%/}"

OVERALL_OK=1

echo "=== Building diag_dump_page_tower (release) ==="
cargo build -p ehp-server --release --example diag_dump_page_tower || exit 1

echo "=== Clearing stale output/rust_E{2..$R_MAX}_{data,page} ==="
for r in $(seq 2 "$R_MAX"); do
  command rm -rf "output/rust_E${r}_data" "output/rust_E${r}_page"
done

echo "=== Walking E_2 through E_${R_MAX} once (EHP_MAX_T=$MAX_T) ==="
EHP_DATA="$RUST_DATA" EHP_MAX_T="$MAX_T" EHP_R_MAX="$R_MAX" \
  EHP_OUTSIDE_DIFFS="$OUTSIDE_DIFFS" EHP_OUTSIDE_SKIP="" EHP_D2_LINEAR=0 \
  ./target/release/examples/diag_dump_page_tower
if [ $? -ne 0 ]; then
  echo "diag_dump_page_tower FAILED (UNSAT or contradiction) -- aborting."
  exit 1
fi

# Build the --page chain incrementally as we go: PAGE_ARGS accumulates one
# "--page r rust_data py_data rust_rank py_rank" group per completed page.
PAGE_ARGS=()

for r in $(seq 2 "$R_MAX"); do
  echo
  echo "=== Stage 1 (constraints): E_${r} ==="
  if [ "$r" -eq 2 ]; then
    python3 scripts/compare_constraints.py --max-t "$MAX_T" \
      "output/rust_E2_data" "$PY_REF_DIR/E2_data"
  else
    python3 scripts/compare_constraints.py --max-t "$MAX_T" \
      "${PAGE_ARGS[@]}" \
      "output/rust_E${r}_data" "$PY_REF_DIR/E${r}_data"
  fi
  [ $? -ne 0 ] && OVERALL_OK=0

  if [ "$r" -ge 3 ]; then
    echo
    echo "=== Stage 3 (rank): E_${r} ==="
    python3 scripts/compare_rank.py --max-t "$MAX_T" \
      "${PAGE_ARGS[@]}" \
      "output/rust_E${r}_page/E${r}_rank.csv" "$PY_REF_DIR/E${r}/E${r}_rank.csv"
    [ $? -ne 0 ] && OVERALL_OK=0

    echo
    echo "=== Stage 3 (products/E/H/P maps): E_${r} ==="
    python3 scripts/compare_tables.py "E${r}" "output/rust_E${r}_page" "$PY_REF_DIR/E${r}" --max-t "$MAX_T" \
      "${PAGE_ARGS[@]}"
    [ $? -ne 0 ] && OVERALL_OK=0
  fi

  # Extend the chain with this page's own data, for the NEXT page's checks.
  if [ "$r" -eq 2 ]; then
    PAGE_ARGS+=(--page 2 "output/rust_E2_data" "$PY_REF_DIR/E2_data" "$RUST_DATA/E2_rank.csv" "$RUST_DATA/E2_rank.csv")
  else
    PAGE_ARGS+=(--page "$r" "output/rust_E${r}_data" "$PY_REF_DIR/E${r}_data" "output/rust_E${r}_page/E${r}_rank.csv" "$PY_REF_DIR/E${r}/E${r}_rank.csv")
  fi
done

echo
if [ "$OVERALL_OK" -eq 1 ]; then
  echo "=== OVERALL: CLEAN (E2..E${R_MAX} at max_t=${MAX_T} vs $PY_REF_DIR) ==="
  exit 0
else
  echo "=== OVERALL: FAIL -- see stage output above for the specific mismatch(es) ==="
  exit 1
fi
