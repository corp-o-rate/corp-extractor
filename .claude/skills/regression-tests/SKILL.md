---
name: regression-tests
description: Manage YAML-fixture regression tests for the corp-extractor pipeline. Use when adding a new regression case, debugging a failing fixture, running the slow suite, or interpreting pre-push hook output.
allowed-tools: Bash(uv:*), Bash(pytest:*), Bash(git:*), Read, Edit, Write
---

# corp-extractor regression-test suite

Fixture-driven, end-to-end regression tests that run the **real** `ExtractionPipeline` against a known article and assert structured rules on the output. Each known bug becomes a permanent YAML fixture. The suite gates `git push` via a local pre-push hook.

## Where things live

- Fixtures: `statement-extractor-lib/tests/fixtures/regression/*.yaml`
- Loader / assertion helper: `statement-extractor-lib/tests/_regression.py`
- Pytest runner: `statement-extractor-lib/tests/test_regression_fixtures.py`
- Pre-push hook: `.githooks/pre-push` (activate per clone with `git config core.hooksPath .githooks`)

The runner is parametrised — any new `*.yaml` is auto-discovered. The pipeline fixture is **module-scoped**, so adding more fixtures costs only the extra `pipeline.process(text)` time, not another full model load.

## Fixture schema

```yaml
name: <slug>                       # matches the filename stem; used as the pytest test id
description: |                     # optional — explain the bug and intent
  Why this case exists.
text: |                            # the input text passed to ExtractionPipeline.process()
  The article body, multi-line, verbatim.
expect:
  min_statements: 5                # optional; default 0
  entities:
    - text: "Surface Form"         # case-insensitive; matched against CanonicalEntity.qualified_entity.original_text
      must_resolve: true           # ≥1 matching CanonicalEntity has non-null canonical_match.canonical_id
      canonical_id_matches: "^wikidata:Q"            # regex; ≥1 must match
      canonical_name_matches: "(?i)apple inc"        # regex; ≥1 must match
      canonical_id_must_not_match: "^UK-CH:"         # regex; EVERY match must NOT match
      canonical_name_must_not_match: "(?i)evil corp" # regex; EVERY match must NOT match
      reason: "free-text note shown on failure"
```

### Rule semantics (cheat sheet)

| Field | Quantifier | Meaning |
|---|---|---|
| `must_resolve` | ∃ | at least one CanonicalEntity for this surface text resolved to **some** canonical_id |
| `canonical_id_matches` | ∃ | at least one matching CanonicalEntity has a canonical_id satisfying the regex |
| `canonical_name_matches` | ∃ | same, for canonical_name |
| `canonical_id_must_not_match` | ∀ | **every** matching CanonicalEntity has canonical_id NOT satisfying the regex (null is OK) |
| `canonical_name_must_not_match` | ∀ | same, for canonical_name |
| `min_statements` | – | `len(ctx.labeled_statements) >= N` |

If a fixture references a surface text that the pipeline didn't extract at all, that is a failure — the entity disappearing is itself a regression worth catching.

Regex tips: anchor with `^` for canonical-id prefixes (`^wikidata:Q`, `^UK-CH:`, `^SEC-CIK:`). Use `(?i)` for case-insensitive name matches. The failure message includes the observed `canonical_id` / `canonical_name`, so writing a bad regex is debuggable.

## Adding a new regression case

1. Save the input text to a new YAML file under `tests/fixtures/regression/`. Filename should be a short slug (e.g. `apple_iphone_pricing.yaml`); `name:` inside must match the stem.
2. Decide rules:
   - For each **mismatch bug** (entity X wrongly resolved to canonical Y), write `canonical_id_must_not_match` or `canonical_name_must_not_match`.
   - For each **expected resolution** that should remain stable, write `must_resolve: true` and (when known) `canonical_id_matches: "^wikidata:Q"` or the relevant prefix.
   - Add `min_statements:` if the bug also caused over- or under-extraction.
3. Validate the YAML loads correctly:
   ```bash
   cd statement-extractor-lib
   uv run python -c "from tests._regression import RegressionFixture, discover_fixtures; [print(p.name, '->', RegressionFixture.from_yaml(p).name, len(RegressionFixture.from_yaml(p).expect.entities), 'rules') for p in discover_fixtures()]"
   ```
4. Run just the new fixture:
   ```bash
   uv run pytest tests/test_regression_fixtures.py -v -m slow -k "<slug>"
   ```

## Running the suite

```bash
cd statement-extractor-lib
uv run pytest tests/test_regression_fixtures.py -v -m slow            # all fixtures
uv run pytest tests/test_regression_fixtures.py -m slow -k electric   # one
uv run pytest -m "not slow"                                            # everything except slow
```

First run on a new machine downloads several GB (entity DB + embeddinggemma + Gemma-3-12B GGUF + T5-Gemma + GLiNER2). Subsequent runs ≈ 30 s model load + 1–2 s/entity for qualifier LLM calls.

## Debugging a failing fixture

The assertion message lists every failed rule with the observed canonical_id / canonical_name. The common failure modes:

- **`must_resolve: true` failed with all `canonical_id=None`** — the qualifier rejected. Two pre-LLM places to look first:
  1. Was the entity even returned by the embedding DB? Check the qualifier's debug log for "Found N candidates".
  2. Did `min_similarity` filter cut the right candidate before the boost was applied? Famous-people queries (name + extracted role + extracted org) often score 0.4–0.48 raw against a DB record that uses a different role/org; the +0.25 exact-name boost brings them to ~0.7. Person qualifier filters on the boosted score; if you're modifying the org qualifier (`embedding_company.py`), do the same.
- **`canonical_id_must_not_match` violated** — the LLM disambiguator picked a wrong-kind-of-thing. Look at `COMPANY_MATCH_PROMPT` / `PERSON_MATCH_PROMPT`. Don't relax the rule unless the canonical match is actually correct.
- **Surface text never appears in `ctx.canonical_entities`** — extraction (Stage 1–2) lost the entity. Likely a splitter or GLiNER2 issue, not a qualifier one.

Useful one-shot probe script template for the person qualifier (drop in `/tmp/`):

```python
from statement_extractor.plugins.qualifiers.person import PersonQualifierPlugin, PERSON_MATCH_PROMPT
plugin = PersonQualifierPlugin(use_llm=True, use_database=True, auto_download_db=True)
emb = plugin._get_embedder().embed("Mary Barra | CEO | General Motors")
for rec, sim in plugin._get_database().search(emb, top_k=10):
    print(f"{sim:.3f}  {rec.name!r}  wikidata={rec.source_id!r}  role={rec.known_for_role!r}  org={rec.known_for_org_name!r}")
```

Run with `uv run python /tmp/probe.py` from `statement-extractor-lib/`. This skips the full pipeline and is the fastest way to isolate "DB has the wrong record" from "filter cut a good record" from "LLM rejected a good candidate".

## Pre-push hook

The hook (`.githooks/pre-push`) runs this suite on every `git push`. Activate once per clone:

```bash
git config core.hooksPath .githooks
```

To bypass intentionally (docs-only push, hot-fix, etc.):

```bash
git push --no-verify
```

**Common gotcha**: the suite takes ~9 minutes; if `~/.ssh/config` has no `ServerAliveInterval` for `github.com`, the SSH connection idles out and the post-hook push fails with SIGPIPE / exit 141 **even when the suite passed**. Fix in `~/.ssh/config`:

```
Host github.com
  ServerAliveInterval 30
  ServerAliveCountMax 20
```

## When NOT to use this suite

- **Quick unit tests for a single plugin** — use `tests/test_*.py` with mocks. The regression suite is for end-to-end behavioural anchors, not isolated function tests.
- **Library-internal refactors with no observable behaviour change** — running the slow suite is wasteful; rely on `pytest -m "not slow"`.
- **Reproducing a bug before you have a fix** — start with the `/tmp/probe.py` pattern above. Promote to a YAML fixture only once you understand which assertion captures the bug class.
