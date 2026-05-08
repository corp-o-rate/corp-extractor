# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Global Preferences

### Core Principles

* I use zsh shell.
* **Fail Fast** - raise exceptions and let them bubble up. Avoid try/catch blocks unless at the top level.
* Don't add fallbacks or backwards compatibility unless instructed explicitly.
* Don't change tests to fit the code. If tests fail, **fix the code** not the test.
* We don't do silent failures - all failures MUST appear in logs or cause the application to fail.
* Everything should be strongly typed, use Pydantic models not dicts.
* Use mermaid for markdown docs when diagrams are needed.
* This is startup code - prefer lean, simple, and to the point over enterprise abstractions.
* I like logging statements, please log progress where possible.
* DO NOT REPEAT existing code (DRY) - prefer tweaking existing implementations over adding new code.

### Instruction Following

* **Be explicit and specific**: Clear, thorough implementation expected.
* **Action-oriented by default**: Proceed with implementation rather than only suggesting.
* **Concise but informative**: Brief progress summaries, avoid unnecessary verbosity.
* **Flowing prose over excessive formatting**: Use clear paragraphs. Reserve markdown primarily for `inline code`, code blocks, and simple headings.

## Project Overview

Statement Extractor is a web demo for the T5-Gemma 2 statement extraction model. It transforms unstructured text into structured subject-predicate-object triples with entity type recognition.

## Commands

### Frontend (Next.js)
```bash
pnpm install     # Install dependencies
pnpm dev         # Start dev server at localhost:3000
pnpm build       # Production build
pnpm lint        # Run ESLint
```

### Local Model Server
```bash
cd local-server
uv sync                        # Install Python dependencies
uv run python server.py        # Start FastAPI server at localhost:8000
```

### Cerebrium Deployment
```bash
cd cerebrium
# Deploy into the SAME Cerebrium project as corp-entity-db so the
# /persistent-storage volume (entities-v6.db, USearch indexes, embeddinggemma)
# is shared. Set HF_TOKEN secret for gated models before first deploy.
cerebrium projects current                    # confirm correct project
cerebrium secrets set HF_TOKEN <token>
cerebrium deploy
cerebrium logs statement-extractor --follow
```
See `cerebrium/README.md` for full deployment notes.

### Upload Model to HuggingFace
```bash
cd scripts
uv sync
uv run python upload_model.py
```

### Python Library (corp-extractor)
```bash
cd statement-extractor-lib
uv sync                        # Install dependencies
uv run pytest                  # Run tests
uv build                       # Build package
uv publish                     # Publish to PyPI (requires credentials)

# CLI commands (after install)
corp-extractor split "text"    # Simple extraction
corp-extractor pipeline "text" # Full 5-stage pipeline
corp-extractor plugins list    # List available plugins

# Persistent server (keeps models warm, avoids ~30s startup per invocation)
corp-extractor serve                                     # Start on localhost:8111
corp-extractor --server pipeline "text"                  # Delegate to server
corp-extractor --server-url http://gpu:8111 split "text" # Custom server URL
```

## Architecture

### Deployment Modes
The frontend can connect to the model via two backends (configured by environment variables):
1. **Cerebrium Serverless** (`CEREBRIUM_EXTRACT_URL`, `CEREBRIUM_EXTRACT_URL_URL`, `CEREBRIUM_TOKEN`) - Production, pay-per-use GPU. Synchronous request/response with one-shot retry on Vercel timeout (`maxDuration=300`). Replica is pre-warmed by a localStorage-gated browser ping on page load (TTL 1h). Deployed into the same Cerebrium project as `corp-entity-db` so `/persistent-storage` (entity DB + USearch indexes + embeddinggemma model) is shared.
2. **Local Server** (`LOCAL_MODEL_URL`) - Self-hosted FastAPI server.

### Directory Structure
- `src/` - Next.js frontend (React 19, Tailwind CSS, D3.js for graph visualization)
- `src/app/api/extract/` - API route that proxies to model backends (sync, with retry on cold-boot timeout)
- `cerebrium/` - Cerebrium app (`main.py` + `cerebrium.toml`) — replaces the old runpod/ deployment
- `local-server/` - FastAPI server for local model inference (uv-managed)
- `runpod/` - Legacy RunPod handler. Superseded by `cerebrium/`; retained for reference.
- `scripts/` - HuggingFace upload utilities (uv-managed)
- `statement-extractor-lib/` - Python library for statement extraction (PyPI package)

### Model I/O Format
- **Input**: Text wrapped in `<page>` tags
- **Output**: XML with `<statements>` containing `<stmt>` elements with `<subject>`, `<object>`, `<predicate>`, `<text>`
- Entity types: ORG, PERSON, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, DATE, MONEY, PERCENT, QUANTITY

### Key Technical Notes
- Uses [Diverse Beam Search](https://arxiv.org/abs/1610.02424) (Vijayakumar et al., 2016) for high-quality extraction
- T5Gemma2 requires `transformers` dev version from GitHub (not PyPI)
- Cerebrium app shares `/persistent-storage` with the `corp-entity-db` app in the same project; `cerebrium/main.py` redirects `HF_HOME` and `statement_extractor.database.hub.DEFAULT_CACHE_DIR` to the volume so entity DB + embeddinggemma weights are reused as-is
- Legacy RunPod build required `--platform linux/amd64` when building Docker on Mac (kept in `runpod/` for reference)
- Model uses bfloat16 on GPU, float32 on CPU
- Generation stops at `</statements>` tag to prevent runaway output
- **v0.10.0**: Entity database (organizations, people, roles, locations) extracted into the [corp-entity-db](https://pypi.org/project/corp-entity-db/) PyPI package. `corp-extractor` now depends on `corp-entity-db>=0.1.0` for qualifier plugins; the bundled importers (`gleif`, `sec_edgar`, `companies_house`, `wikidata*`), `db` CLI subcommands, schema_v2 migration, and `canonicalization.py` were removed. Re-export shims in `database/store.py` and `database/resolver.py` keep `from statement_extractor.database.store import OrganizationDatabase` working for downstream code. `EntityQualifiers.resolved_org` now uses `corp_entity_db.models.ResolvedOrganization` directly (was a duplicate definition that failed Pydantic V2's class-identity check). `PersonRecord.known_for_org` was renamed to `known_for_org_name` upstream — qualifier plugin updated.
- **v0.9.8**: Python API server delegation via `server_url=` parameter on `extract_statements()`, `ExtractionPipeline`, and `DocumentPipeline`. New `client.py` module handles HTTP delegation and Pydantic model reconstruction. All backends (server, RunPod, local-server) now return standardized `model_dump()` format — no more XML strings or custom wrappers. CLI reconstructs full Pydantic models from server JSON, eliminating duplicate formatting code.
- **v0.9.7**: Persistent local server (`corp-extractor serve`) with FastAPI. Keeps T5-Gemma, GLiNER2, embedding models, and USearch indexes warm in memory. Endpoints: `GET /` (health), `POST /pipeline`, `POST /split`, `POST /document`. CLI `--server` / `--server-url` flags and `CORP_EXTRACTOR_SERVER` env var to delegate processing to server. Default port 8111.
- **v0.9.6**: Database v3 schema — `db_info` metadata table with `schema_version=3`. Lite databases drop ALL embedding tables (float32 + scalar int8) — uses USearch index files for search. `db upload` now includes USearch `.bin` files. `db download` fetches USearch indexes alongside database. Filenames: `entities-v3.db` / `entities-v3-lite.db`. Global `--db-version` CLI flag for backwards compatibility (`corp-extractor --db-version=2 db download`). Symlink `entities-v2.db` → v3 file on download for backwards compat.
- **v0.9.5**: USearch HNSW indexes for sub-millisecond search on 50M+ vectors. 3-thread parallel Wikidata dump import (reader/embedder/writer). Multi-record person import (one person per position+org). Auto-canonicalization after dump import with recency tiebreaker. New CLI: `db post-import`, `db build-index`, `db rebuild-vec`. `--hybrid` flag for text+embeddings search. `fast-import` extras (orjson, indexed_bzip2). Zstandard (.zst) dump support. New `hf_classifier` labeler plugin.
- **v0.9.4**: Database v2 schema with normalized FK references (replaces TEXT enums with INTEGER FKs). New tables: `roles`, `locations`, `location_types`. Scalar (int8) embeddings for 75% storage reduction (~92% recall). New CLI: `db migrate-v2`, `db backfill-scalar`, `db search-roles`, `db search-locations`. Locations import via `--locations` flag.
- **v0.9.3**: Added SEC Form 4 officers import (`import-sec-officers`) and Companies House officers import (`import-ch-officers`). People now sourced from wikidata, sec_edgar, and companies_house. People canonicalization with priority wikidata > sec_edgar > companies_house.
- **v0.9.1**: Added Wikidata dump importer (`import-wikidata-dump`) for large imports without SPARQL timeouts. Uses aria2c for fast parallel downloads. Extracts people via occupation (P106) and position dates (P580/P582).
- **v0.9.0**: Added person database with Wikidata import and person qualification with canonical IDs. People import auto-inserts discovered organizations with `known_for_org_id` FK. New flags: `--skip-existing`, `--enrich-dates`
- **v0.8.0**: Merged qualification and canonicalization into single stage; added EntityType classification
- **v0.5.0**: Introduces plugin-based pipeline architecture
- **v0.4.0**: Uses GLiNER2 (205M params) for entity recognition and relation extraction instead of spaCy
- GLiNER2 is CPU-optimized and handles NER, relation extraction, and structured data extraction

### Pipeline Architecture (v0.8.0)
The library provides a 5-stage extraction pipeline:

| Stage | Name | Description | Key Tech |
|-------|------|-------------|----------|
| 1 | Splitting | Text → raw triples | T5-Gemma2 |
| 2 | Extraction | Raw triples → typed statements | GLiNER2 |
| 3 | Entity Qualification | Add identifiers + canonical names | Embedding DB |
| 4 | Labeling | Add sentiment, relation type | Classification |
| 5 | Taxonomy | Classify against large taxonomies | MNLI, Embeddings |

**Built-in plugins:**
- **Splitters**: `t5_gemma_splitter`
- **Extractors**: `gliner2_extractor`
- **Qualifiers**: `person_qualifier` (Wikidata person database lookup), `embedding_company_qualifier` (organization database lookup)
- **Labelers**: `sentiment_labeler`, `confidence_labeler`, `relation_type_labeler`, `hf_classifier` (custom HuggingFace models, not auto-registered)
- **Taxonomy**: `embedding_taxonomy_classifier` (default), `mnli_taxonomy_classifier`
- **PDF**: `pypdf_parser` (default) - PDF text extraction with PyMuPDF + Tesseract OCR fallback; `glm_ocr_parser` - GLM-OCR 0.9B VLM for high-quality OCR (scans, tables, formulas)
- **Scrapers**: `http_scraper` - URL/web page scraping

### Entity Database

The entity database supports both **organizations** and **people** (v0.9.0).

**Organization EntityType Classification:**

| EntityType | Description | Examples |
|------------|-------------|----------|
| `business` | Commercial companies | Apple Inc., Amazon |
| `fund` | Investment funds, ETFs | Vanguard S&P 500 ETF |
| `branch` | Branch offices | Deutsche Bank London |
| `nonprofit` | Non-profit organizations | Red Cross |
| `ngo` | Non-governmental orgs | Greenpeace |
| `foundation` | Charitable foundations | Gates Foundation |
| `government` | Government agencies | SEC, FDA |
| `international_org` | International organizations | UN, WHO, IMF |
| `educational` | Schools, universities | MIT, Stanford |
| `research` | Research institutes | CERN, NIH |
| `healthcare` | Hospitals, health orgs | Mayo Clinic |
| `media` | Studios, record labels | Warner Bros |
| `sports` | Sports clubs/teams | Manchester United |
| `political_party` | Political parties | Democratic Party |
| `trade_union` | Labor unions | AFL-CIO |

**Person Database (v0.9.2):**

The person database stores notable people from Wikidata with role/org context for disambiguation:

| PersonType | Description | Examples |
|------------|-------------|----------|
| `executive` | C-suite, board members | Tim Cook, Satya Nadella |
| `politician` | Elected officials (presidents, MPs, mayors) | Joe Biden, Angela Merkel |
| `government` | Civil servants, diplomats, appointed officials | Ambassadors, agency heads |
| `military` | Military officers, armed forces personnel | Generals, admirals |
| `legal` | Judges, lawyers, legal professionals | Supreme Court justices |
| `professional` | Known for profession (doctors, engineers) | Famous surgeons, architects |
| `athlete` | Sports figures | LeBron James, Lionel Messi |
| `artist` | Traditional creatives (musicians, actors, painters) | Tom Hanks, Taylor Swift |
| `media` | Internet/social media personalities | YouTubers, influencers, podcasters |
| `academic` | Professors, researchers | Neil deGrasse Tyson |
| `scientist` | Scientists, inventors | Research scientists |
| `journalist` | Reporters, news presenters | Anderson Cooper |
| `entrepreneur` | Founders, business owners | Mark Zuckerberg |
| `activist` | Advocates, campaigners | Greta Thunberg |

Person records include `birth_date` and `death_date` fields, with an `is_historic` property that returns True for deceased individuals.

### GLiNER2 Integration (v0.4.0)
The library uses GLiNER2 for:
1. **Entity extraction**: Refines subject/object boundaries from T5-Gemma output
2. **Relation extraction**: When `predicates` list is provided, uses GLiNER2's relation extraction
3. **Entity scoring**: Scores how "entity-like" subjects/objects are (replaces spaCy POS tagging)

Two extraction modes:
- **With predicates list**: Uses `extract_relations()` for predefined relation types
- **Without predicates**: Uses entity extraction to refine boundaries + predicate split for verb extraction

### Python Library API

**Simple extraction:**
```python
from statement_extractor import extract_statements

result = extract_statements("Apple announced a new iPhone.")
for stmt in result:
    print(f"{stmt.subject.text} -> {stmt.predicate} -> {stmt.object.text}")
```

**Server delegation (v0.9.8):**
```python
from statement_extractor import extract_statements
from statement_extractor.pipeline import ExtractionPipeline
from statement_extractor.document import DocumentPipeline

# Delegate to a running server instead of loading models locally
result = extract_statements("Apple announced iPhone.", server_url="http://localhost:8111")

# Pipeline with server delegation
pipeline = ExtractionPipeline(server_url="http://localhost:8111")
ctx = pipeline.process("Amazon CEO Andy Jassy announced...")

# Document pipeline with server delegation
doc_pipeline = DocumentPipeline(server_url="http://localhost:8111")
ctx = doc_pipeline.process(document)
```

**Full pipeline (v0.5.0):**
```python
from statement_extractor.pipeline import ExtractionPipeline, PipelineConfig

# Run full pipeline
pipeline = ExtractionPipeline()
ctx = pipeline.process("Amazon CEO Andy Jassy announced...")

# Access results
for stmt in ctx.labeled_statements:
    print(f"{stmt.subject_fqn} -> {stmt.statement.predicate} -> {stmt.object_fqn}")

# With configuration
config = PipelineConfig(
    enabled_stages={1, 2, 3},  # Skip labeling and taxonomy
    disabled_plugins={"person_qualifier"},
)
pipeline = ExtractionPipeline(config)
```

**CLI usage:**
```bash
corp-extractor split "text"              # Simple extraction
corp-extractor pipeline "text"           # Full pipeline
corp-extractor pipeline "text" --stages 1-3
corp-extractor plugins list              # List plugins

# Persistent server (v0.9.7) — keeps models warm for fast repeated use
corp-extractor serve                     # Start on localhost:8111
corp-extractor serve --port 9000         # Custom port
corp-extractor --server pipeline "text"  # Delegate to server
corp-extractor --server-url http://gpu:8111 split "text"  # Custom server URL
# Or set CORP_EXTRACTOR_SERVER=http://localhost:8111 in environment

# Document processing (v0.7.0)
corp-extractor document process article.txt
corp-extractor document process report.pdf                                       # Local PDF (auto-detected)
corp-extractor document process report.pdf --pdf-parser glm_ocr_parser           # Use GLM-OCR VLM parser
corp-extractor document process https://example.com/article
corp-extractor document process https://example.com/report.pdf --use-ocr

# Entity database — moved out to the corp-entity-db project in v0.10.0.
# All db CLI commands now live there. See https://corp-entity-db.vercel.app/
# for the canonical CLI reference, search, download, and build instructions.
```
