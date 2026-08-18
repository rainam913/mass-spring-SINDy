# Vendored copy — provenance

This directory is an unmodified checkout of the canonical SINDy implementation.

| field | value |
|---|---|
| upstream | `https://github.com/dynamicslab/pysindy` |
| tag | `v2.1.0` |
| commit | `1edf31260fc000692776b1f4259877e8b85e56e5` |
| commit date | 2026-01-05 18:08:58 −0800 |
| subject | `feat: remove Pipeline (#668)` |
| license | MIT — see `LICENSE` |

## What was removed, and why

Only for size. Nothing under `pysindy/` or `test/` was touched.

- `.git/` — history, ~90 MB
- `examples/` — notebooks and their cached outputs, ~115 MB
- `docs/` — built documentation, ~3.6 MB
- `asv_bench/` — benchmark harness

Everything retained is byte-identical to the tag: `pysindy/` (the package
source), `test/` (the upstream suite), `LICENSE`, `README.rst`,
`pyproject.toml`, `citation.cff`, `MANIFEST.in`.

## How to use it

`requirements.txt` installs the released `pysindy==2.1.0`, which is built from
this same tag. Import the installed package; read this tree when you need to
know what a keyword argument actually does rather than what the docstring says
it does.

`SPEC.md` §0.4 asks you to determine, by reading `pysindy/optimizers/`, whether
the `unbias` path refits the surviving columns or rescales the coefficients it
already has. Read the source, then confirm your reading numerically.

To run the upstream suite:

```bash
python3 -m pytest pysindy_upstream/test -q
```

This is optional and slow. It tests pysindy, not this study.

## What it is not

This is the tool. It is not the authors' code and it contains nothing about the
cart, the spring, or the experiment. Nothing in this directory is evidence about
what the answer should be.
