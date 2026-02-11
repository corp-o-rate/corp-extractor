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

### RunPod Deployment
```bash
cd runpod
# Build for Linux/amd64 (required on Mac)
docker build --platform linux/amd64 -t statement-extractor-runpod .
```

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
```

## Architecture

### Three Deployment Modes
The frontend can connect to the model via three backends (configured by environment variables):
1. **RunPod Serverless** (`RUNPOD_ENDPOINT_ID`, `RUNPOD_API_KEY`) - Production, pay-per-use GPU
2. **Local Server** (`LOCAL_MODEL_URL`) - Self-hosted FastAPI server
3. **HuggingFace Inference** - Fallback, rate-limited

### Directory Structure
- `src/` - Next.js frontend (React 19, Tailwind CSS, D3.js for graph visualization)
- `src/app/api/extract/` - API route that proxies to model backends
- `local-server/` - FastAPI server for local model inference (uv-managed)
- `runpod/` - Docker + handler for RunPod serverless deployment
- `scripts/` - HuggingFace upload utilities (uv-managed)
- `statement-extractor-lib/` - Python library for statement extraction (PyPI package)

### Model I/O Format
- **Input**: Text wrapped in `<page>` tags
- **Output**: XML with `<statements>` containing `<stmt>` elements with `<subject>`, `<object>`, `<predicate>`, `<text>`
- Entity types: ORG, PERSON, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, DATE, MONEY, PERCENT, QUANTITY

### Key Technical Notes
- Uses [Diverse Beam Search](https://arxiv.org/abs/1610.02424) (Vijayakumar et al., 2016) for high-quality extraction
- T5Gemma2 requires `transformers` dev version from GitHub (not PyPI)
- RunPod requires `--platform linux/amd64` when building Docker on Mac
- Model uses bfloat16 on GPU, float32 on CPU
- Generation stops at `</statements>` tag to prevent runaway output
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

# Document processing (v0.7.0)
corp-extractor document process article.txt
corp-extractor document process report.pdf                                       # Local PDF (auto-detected)
corp-extractor document process report.pdf --pdf-parser glm_ocr_parser           # Use GLM-OCR VLM parser
corp-extractor document process https://example.com/article
corp-extractor document process https://example.com/report.pdf --use-ocr

# Entity database
corp-extractor db import-sec --download  # Bulk SEC data (73K filers)
corp-extractor db import-people --all    # Notable people from Wikidata (v0.9.0)
corp-extractor db import-people --type executive --skip-existing  # Skip existing records
corp-extractor db import-people --type executive --enrich-dates   # Fetch role dates (slower)
corp-extractor db import-sec-officers --start-year 2023 --limit 10000  # SEC Form 4 officers (v0.9.3)
corp-extractor db import-ch-officers --file officers.zip --limit 10000  # CH officers (v0.9.3)
corp-extractor db import-wikidata-dump --download --limit 50000   # From Wikidata dump (v0.9.1)
corp-extractor db import-wikidata-dump --dump /path/to/dump.bz2 --people --no-orgs  # Local dump
corp-extractor db import-wikidata-dump --dump dump.bz2 --resume  # Resume from file position
corp-extractor db import-wikidata-dump --dump dump.bz2 --skip-updates  # Skip existing Q codes
corp-extractor db import-wikidata-dump --download --require-enwiki  # Only orgs with English Wikipedia
corp-extractor db import-wikidata-dump --dump dump.bz2 --locations --no-people --no-orgs  # Locations only (v0.9.4)
corp-extractor db canonicalize           # Link equivalent records across sources
corp-extractor db post-import            # Run after any import: embeddings + USearch index + VACUUM
corp-extractor db post-import --no-orgs  # People only
corp-extractor db build-index            # Build USearch HNSW index for fast ANN search
corp-extractor db rebuild-vec            # Rebuild vec0 tables with distance_metric=cosine
corp-extractor db status                 # Show database statistics
corp-extractor db status --for-llm       # Output schema and enum tables for LLM docs
corp-extractor db search "Microsoft"     # Search organizations (USearch HNSW)
corp-extractor db search "Microsoft" --hybrid  # Hybrid text + embeddings search
corp-extractor db search-people "Tim Cook"  # Search people (v0.9.0)
corp-extractor db search-roles "CEO"     # Search roles (v0.9.4)
corp-extractor db search-locations "California"  # Search locations (v0.9.4)
corp-extractor db upload                 # Upload with lite variant + USearch indexes
corp-extractor db download               # Download lite version + USearch indexes (default)
corp-extractor db download --full        # Download full version + USearch indexes
corp-extractor --db-version=2 db download  # Download v2 database files
corp-extractor db migrate-v2 entities.db entities-v2.db  # Migrate to v2 schema (v0.9.4)
corp-extractor db backfill-scalar        # Generate int8 embeddings (v0.9.4)
```
