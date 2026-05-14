# Versioned git hooks

This directory holds project-versioned git hooks. They are **not** enabled by default — git only runs hooks from `.git/hooks/` unless `core.hooksPath` is pointed elsewhere.

## One-time activation (per clone)

```bash
git config core.hooksPath .githooks
```

After that, every `git push` will run `pre-push`, which executes the corp-extractor regression suite (`statement-extractor-lib/tests/test_regression_fixtures.py` with `-m slow`). The push aborts if any fixture fails.

## What runs on push

`pre-push` cd's into `statement-extractor-lib/` and runs:

```bash
uv run pytest tests/test_regression_fixtures.py -m slow -q
```

The first invocation on a fresh machine may take several minutes because the pipeline downloads:

- `corp-entity-db` artefacts (entity database + USearch indexes)
- `embeddinggemma` embedding weights
- The Gemma-3-12B disambiguator GGUF
- T5-Gemma2 splitter weights
- GLiNER2 weights

Subsequent runs are dominated by model load (~30 s).

## Bypassing

Intentional bypass (e.g. docs-only pushes you're confident about):

```bash
git push --no-verify
```

Don't make `--no-verify` a habit — the suite exists to keep entity-resolution regressions out of `main`.
