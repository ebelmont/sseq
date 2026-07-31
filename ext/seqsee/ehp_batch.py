#!/usr/bin/env python
"""Batch-generate EHP charts in a single process.

Called by the ehp_chart REPL to amortize interpreter startup and imports over
many charts (previously each chart spawned fresh python processes).

Usage:
  ehp_batch.py sphere <input.csv> <output_dir> <theme> <r> <n1> [n2 ...]
  ehp_batch.py stem <input.csv> <output_dir> <theme> <r> <k1> [k2 ...]
  ehp_batch.py fiber <input.csv> <output_dir> <theme> <r> <N1> [N2 ...]
  ehp_batch.py sidebyside <manifest.jsonl>

sphere mode writes S{n}_E{r}.json/.html; stem mode writes stem{k}_E{r}.json/
.html (one chart per stem, x-axis = n, per STEM_VIEW_SPEC.md); fiber mode
writes fiber{N}_E{r}.json/.html (one chart per base sphere N >= 2, showing
the fiber-sequence triple S^N, S^{N+1}, S^{2N+1} with E/H/P edges, per
crates/fiber_spec.md). sidebyside mode reads one JSON object per line:
{"src": ..., "tgt": ..., "out": ..., "theme": ..., "back": ...} and
generates each split-screen map view.

Prints one line per item: "OK <id>" or "FAIL <id> <reason>" (parsed by the
Rust caller). For sphere/stem the id is n/k; for sidebyside it is the line
number (0-based).

Timing: every chunk also prints one "TIMESUM <mode> items=<k> json=<s>
render=<s> bytes=<n>" line (cumulative jsonmaker time, HTML-render time, and
output bytes; the Rust caller aggregates these — unknown line prefixes are
ignored by older parsers, so this is protocol-safe). With EHP_TIMING set in
the environment, a per-item "TIME <mode> <id> ..." line is printed too.
"""
import json
import os
import sys
import time

import jsonmaker
import main as seqsee_main

PER_ITEM_TIMING = bool(os.environ.get("EHP_TIMING"))


def _report(ok, ident, err=None):
    if ok:
        print(f"OK {ident}", flush=True)
    else:
        print(f"FAIL {ident} {err}", flush=True)


def _size_of(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _timesum(mode, items, json_s, render_s, nbytes):
    print(
        f"TIMESUM {mode} items={items} json={json_s:.3f} render={render_s:.3f} "
        f"bytes={nbytes}",
        flush=True,
    )


def run_slices(mode, csv_path, out_dir, theme, r, values):
    prefix = {"sphere": "S", "stem": "stem", "fiber": "fiber"}[mode]
    items = 0
    json_s = render_s = 0.0
    nbytes = 0
    for v in values:
        json_path = f"{out_dir}/{prefix}{v}_E{r}.json"
        html_path = f"{out_dir}/{prefix}{v}_E{r}.html"
        try:
            t0 = time.perf_counter()
            try:
                jsonmaker.process_csv(csv_path, json_path, mode, v, quiet=True)
            except TypeError:
                # older process_csv without the quiet kwarg
                jsonmaker.process_csv(csv_path, json_path, mode, v)
            t1 = time.perf_counter()
            seqsee_main.process_json(json_path, html_path, theme, mode, v)
            t2 = time.perf_counter()
            size = _size_of(html_path)
            items += 1
            json_s += t1 - t0
            render_s += t2 - t1
            nbytes += size
            if PER_ITEM_TIMING:
                print(
                    f"TIME {mode} {v} json={t1 - t0:.3f} render={t2 - t1:.3f} "
                    f"bytes={size}",
                    flush=True,
                )
            _report(True, v)
        except KeyboardInterrupt:
            raise
        except BaseException as e:  # includes SystemExit from validation errors
            _report(False, v, e)
    _timesum(mode, items, json_s, render_s, nbytes)


def run_sidebyside(manifest_path):
    with open(manifest_path) as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    items = 0
    render_s = 0.0
    nbytes = 0
    for i, line in enumerate(lines):
        try:
            job = json.loads(line)
            t0 = time.perf_counter()
            seqsee_main.generate_sidebyside_html(
                job["src"], job["tgt"], job["out"],
                job.get("theme", "light"), job.get("back", ""),
            )
            t1 = time.perf_counter()
            size = _size_of(job["out"])
            items += 1
            render_s += t1 - t0
            nbytes += size
            if PER_ITEM_TIMING:
                print(
                    f"TIME sidebyside {i} render={t1 - t0:.3f} bytes={size}",
                    flush=True,
                )
            _report(True, i)
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            _report(False, i, e)
    _timesum("sidebyside", items, 0.0, render_s, nbytes)


def run():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    mode = sys.argv[1]
    if mode in ("sphere", "stem", "fiber"):
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
