#!/usr/bin/env python
"""Batch-generate EHP charts in a single process.

Called by the ehp_chart REPL to amortize interpreter startup and imports over
many charts (previously each chart spawned fresh python processes).

Usage:
  ehp_batch.py sphere <input.csv> <output_dir> <theme> <r> <n1> [n2 ...]
  ehp_batch.py stem <input.csv> <output_dir> <theme> <r> <k1> [k2 ...]
  ehp_batch.py sidebyside <manifest.jsonl>

sphere mode writes S{n}_E{r}.json/.html; stem mode writes stem{k}_E{r}.json/
.html (one chart per stem, x-axis = n, per STEM_VIEW_SPEC.md). sidebyside
mode reads one JSON object per line: {"src": ..., "tgt": ..., "out": ...,
"theme": ..., "back": ...} and generates each split-screen map view.

Prints one line per item: "OK <id>" or "FAIL <id> <reason>" (parsed by the
Rust caller). For sphere/stem the id is n/k; for sidebyside it is the line
number (0-based).
"""
import json
import sys

import pandas as pd

import jsonmaker
import main as seqsee_main


def _report(ok, ident, err=None):
    if ok:
        print(f"OK {ident}", flush=True)
    else:
        print(f"FAIL {ident} {err}", flush=True)


def run_slices(mode, csv_path, out_dir, theme, r, values):
    prefix = "S" if mode == "sphere" else "stem"
    # Load once per batch process instead of once per sphere/stem: with
    # hundreds of values sharing one CSV, re-parsing it from disk on every
    # iteration (as process_csv does by default) was the dominant cost of
    # "batched" chart generation, dwarfing the interpreter-startup savings
    # the batching was meant to capture.
    df = pd.read_csv(csv_path)
    for v in values:
        json_path = f"{out_dir}/{prefix}{v}_E{r}.json"
        html_path = f"{out_dir}/{prefix}{v}_E{r}.html"
        try:
            jsonmaker.process_csv(csv_path, json_path, mode, v, quiet=True, df=df)
            seqsee_main.process_json(json_path, html_path, theme, mode, v)
            _report(True, v)
        except KeyboardInterrupt:
            raise
        except BaseException as e:  # includes SystemExit from validation errors
            _report(False, v, e)


def run_sidebyside(manifest_path):
    with open(manifest_path) as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        try:
            job = json.loads(line)
            seqsee_main.generate_sidebyside_html(
                job["src"], job["tgt"], job["out"],
                job.get("theme", "light"), job.get("back", ""),
            )
            _report(True, i)
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            _report(False, i, e)


def run():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    mode = sys.argv[1]
    if mode in ("sphere", "stem"):
        if len(sys.argv) < 7:
            print(__doc__)
            sys.exit(1)
        run_slices(mode, sys.argv[2], sys.argv[3], sys.argv[4],
                   int(sys.argv[5]), [int(x) for x in sys.argv[6:]])
    elif mode == "sidebyside":
        run_sidebyside(sys.argv[2])
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    run()
