---
description: Update all documentation files with recent changes
allowed-tools: Read, Edit, Write, Glob, Grep
---

Please update all the docs (.mdx and .md) with all the changes.

## Files to update

1. **Root documentation:**
   - `CLAUDE.md` - Claude Code guidance
   - `README.md` - Main project README
   - `ENTITY_DATABASE.md` - Pointer/overview of the entity database (the
     code itself lives in the separate `corp-entity-db` package as of
     v0.10.0; this file should reference that and keep only project-level
     usage notes)

2. **Python library documentation:**
   - `statement-extractor-lib/README.md` - Library README
   - `statement-extractor-lib/CLAUDE.md` - Claude Code Guidance
   - The library version in `statement-extractor-lib/pyproject.toml` and
     `statement-extractor-lib/src/statement_extractor/__init__.py` must
     match the latest version on PyPI
     (https://pypi.org/project/corp-extractor/) when documenting features
     by version.

3. **Cerebrium Documentation (production deployment):**
   - `cerebrium/README.md`
   - `cerebrium/cerebrium.toml` — keep `corp-extractor` pin in
     `[cerebrium.dependencies.pip]` aligned with the latest PyPI release.

4. **Runpod Documentation (legacy, superseded by Cerebrium):**
   - `runpod/README.md` — retained for historical reference; flag the
     superseded-by-Cerebrium notice if missing.

5. **Local Server Documentation:**
   - `local-server/README.md`

6. **Website documentation (MDX):**
   - `src/app/docs/sections/api-reference.mdx`
   - `src/app/docs/sections/cli.mdx`
   - `src/app/docs/sections/configuration.mdx`
   - `src/app/docs/sections/core-concepts.mdx`
   - `src/app/docs/sections/deployment.mdx` — Cerebrium is the primary
     production path; RunPod section should mark itself legacy.
   - `src/app/docs/sections/entity-database.mdx`
   - `src/app/docs/sections/entity-types.mdx`
   - `src/app/docs/sections/examples.mdx`
   - `src/app/docs/sections/getting-started.mdx`
   - `src/components/documentation.tsx`
   - `src/components/llm-prompts.tsx`
   - `src/components/pipeline-diagram.tsx`

7. **Notebooks:**

   Please update the documentation (and code if required) in the notebooks
   in the `notebooks` directory (e.g. `notebooks/statement_extractor_demo.ipynb`).

8. **Environment / wiring:**
   - `.env.example` — must list the current set of env vars used by
     `src/app/api/extract/route.ts` (`CEREBRIUM_EXTRACT_URL`,
     `CEREBRIUM_EXTRACT_URL_URL`, `CEREBRIUM_TOKEN`, `LOCAL_MODEL_URL`,
     plus Supabase/HF/etc.).

## Specific periodic updates

### Entity database

The entity database lives in the separate `corp-entity-db` package
(extracted in `corp-extractor` v0.10.0). For approximate sizes, schema
version, and feature flags, the source of truth is
https://corp-entity-db.vercel.app/.

Update these files when corp-entity-db's data shape, sizes, or schema
version changes:

   - `src/app/docs/sections/core-concepts.mdx`
   - `src/app/docs/sections/entity-database.mdx`
   - `ENTITY_DATABASE.md`
   - `statement-extractor-lib/README.md` (the "Entity DB extracted" feature
     bullet and any version-by-version notes)
   - `CLAUDE.md` (the version-history block)

### Deployment paths

Production deploys to Cerebrium (one app per Cerebrium project, sharing
`/persistent-storage` with the corp-entity-db app). The frontend API
route (`src/app/api/extract/route.ts`) is sync-with-retry; cold-start is
absorbed by a localStorage-gated browser warmup ping in
`src/app/page.tsx`. Deployment is automated by the workflow at
`.github/workflows/cerebrium-deploy.yml` on pushes that touch
`cerebrium/**`. Keep the Cerebrium and RunPod docs aligned with this
posture (Cerebrium = current, RunPod = legacy).

## Process

$ARGUMENTS$

1. First, review recent code changes to understand what needs documenting.
2. Search for any inconsistencies between code and documentation.
3. Update all relevant documentation files.
4. Ensure version numbers (corp-extractor, corp-entity-db),
   feature descriptions, and examples are accurate. Cross-check against:
     - `statement-extractor-lib/pyproject.toml`
     - `cerebrium/cerebrium.toml`
     - `runpod/Dockerfile`
     - the latest tag in `git tag --list`
5. Verify pipeline stages, plugins, and API signatures match the code in
   `statement-extractor-lib/src/statement_extractor/`.
6. Confirm `.env.example` matches the env vars actually read in
   `src/app/api/`.
