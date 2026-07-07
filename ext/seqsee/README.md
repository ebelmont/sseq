# Vendored SeqSee chart-generation components

Vendored on 2026-07-07 from the user's `~/seqsee/seqsee_new` checkout (a fork
of the upstream [JoeyBF/SeqSee](https://github.com/JoeyBF/SeqSee) repo;
`template_sidebyside.html.jinja` was a symlink into the sibling
`~/SeqSee/seqsee` upstream checkout and is copied here as a regular file).
Only the minimal closure needed by the EHP chart pipeline is included — the
rest of the SeqSee repo (tikz tooling, navigators, test charts, venv) is
deliberately excluded.

## Contents

- `ehp_batch.py` — batch entry point invoked by `ehp_chart.rs`
  (modes: `sphere`, `stem`, `sidebyside`)
- `jsonmaker.py` — CSV → chart JSON
- `main.py` — chart JSON → HTML (also `generate_sidebyside_html`)
- `template.html.jinja`, `template_sidebyside.html.jinja` — HTML templates
  (loaded relative to `main.py`; all JS/CSS/font assets come from CDNs,
  nothing else is loaded from disk)
- `input_schema.json` — chart JSON schema, loaded at import time by `main.py`
- `themes.json` — theme palettes; the single source of truth read by BOTH
  `main.py` (per-theme CSS variable blocks) and `ehp_chart.rs`
  (`theme_registry`)
- `requirements.txt` — third-party deps (`pip install -r requirements.txt`)

## How it is found

`ehp_chart.rs` resolves the SeqSee directory in this order:
`EHP_SEQSEE` env var → this vendored directory
(`<manifest>/../../seqsee`, i.e. `ext/seqsee`) → external checkouts
(`SEQSEE_DIR`, `../seqsee/seqsee_new`, `~/seqsee/seqsee_new`, …).
Python: `EHP_PYTHON` env var if set; for this vendored copy (detected by the
presence of `requirements.txt`) plain `python3` is used — install the deps
into whatever `python3`/`EHP_PYTHON` resolves to; external checkouts keep
the poetry venv probe.

## Updating the vendored copy

Re-copy the files from `~/seqsee/seqsee_new` (dereference the
`template_sidebyside.html.jinja` symlink):

```bash
cd ~/seqsee/seqsee_new
cp main.py jsonmaker.py ehp_batch.py template.html.jinja themes.json \
   input_schema.json ~/sseq/ext/seqsee/
cp -L template_sidebyside.html.jinja ~/sseq/ext/seqsee/
```

If the poetry dependencies in the seqsee repo's `pyproject.toml` change,
mirror them in `requirements.txt`.
