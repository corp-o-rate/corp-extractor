# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Corp-Extractor (`corp-extractor` on PyPI) is a Python library that extracts structured subject-predicate-object statements from unstructured text using T5-Gemma2 and GLiNER2 models.

## Commands

```bash
uv sync                              # Install dependencies
uv run pytest                        # Run all tests
uv run pytest tests/test_pipeline.py # Run single test file
uv run pytest -m "not slow"          # Skip slow tests (model loading)
uv run pytest -k "test_name"         # Run specific test by name

uv run ruff check .                  # Lint
uv run mypy src/                     # Type check

uv build                             # Build package
uv publish                           # Publish to PyPI

# CLI testing
uv run corp-extractor split "text"           # Simple extraction (Stage 1)
uv run corp-extractor pipeline "text"        # Full 5-stage pipeline
uv run corp-extractor pipeline "text" -v     # Verbose with debug logs
uv run corp-extractor plugins list           # List registered plugins

# Persistent server (v0.9.7) — keeps models warm for fast repeated use
uv run corp-extractor serve                  # Start on localhost:8111
uv run corp-extractor serve --port 9000      # Custom port
uv run corp-extractor serve --no-warmup      # Skip eager model loading
uv run corp-extractor --server pipeline "text"  # Delegate to running server
uv run corp-extractor --server-url http://gpu:8111 split "text"  # Custom server URL

# Entity database commands
uv run corp-extractor db status              # Show database stats
uv run corp-extractor db status --for-llm    # Output schema/enum tables for LLM docs
uv run corp-extractor db search "Microsoft"  # Search for organization (USearch HNSW)
uv run corp-extractor db search "Microsoft" --hybrid  # Hybrid text + embeddings search
uv run corp-extractor db search-people "Tim Cook"  # Search for person (v0.9.0)
uv run corp-extractor db import-gleif --download --limit 10000  # Import GLEIF (2.6M records)
uv run corp-extractor db import-sec --download                  # Import SEC bulk (73K filers)
uv run corp-extractor db import-sec-officers --limit 10000      # Import SEC Form 4 officers/directors (v0.9.3)
uv run corp-extractor db import-sec-officers --start-year 2023 --resume  # Resume from progress
uv run corp-extractor db import-ch-officers --file officers.zip --limit 10000  # Import CH officers (v0.9.3)
uv run corp-extractor db import-ch-officers --file officers.zip --resume  # Resume from progress
uv run corp-extractor db import-companies-house --download --limit 10000  # Import UK companies
uv run corp-extractor db import-wikidata --limit 5000  # Import Wikidata orgs
uv run corp-extractor db import-people --type executive --limit 5000  # Import notable people (v0.9.0)
uv run corp-extractor db import-people --all --limit 10000            # All person types (v0.9.0)
uv run corp-extractor db import-people --type executive --skip-existing  # Skip existing records
uv run corp-extractor db import-people --type executive --enrich-dates   # Fetch role start/end dates (slower)
uv run corp-extractor db import-wikidata-dump --download --limit 50000   # Import from Wikidata dump (v0.9.1)
uv run corp-extractor db import-wikidata-dump --dump /path/to/dump.json.bz2 --people --no-orgs  # From local dump
uv run corp-extractor db import-wikidata-dump --dump dump.json.bz2 --resume  # Resume from file position
uv run corp-extractor db import-wikidata-dump --dump dump.json.bz2 --skip-updates  # Skip existing Q codes
uv run corp-extractor db import-wikidata-dump --download --require-enwiki  # Only orgs with English Wikipedia
uv run corp-extractor db import-wikidata-dump --dump dump.bz2 --locations --no-people --no-orgs  # Locations only (v0.9.4)
uv run corp-extractor db canonicalize        # Link equivalent records across sources
uv run corp-extractor db build-index         # Build USearch HNSW index for fast ANN search
uv run corp-extractor db rebuild-vec         # Rebuild vec0 tables with distance_metric=cosine
uv run corp-extractor db search-roles "CEO"  # Search roles (v0.9.4)
uv run corp-extractor db search-locations "California"  # Search locations (v0.9.4)
uv run corp-extractor db upload              # Upload with lite variant + USearch indexes
uv run corp-extractor db download            # Download lite version + USearch indexes (default)
uv run corp-extractor db download --full     # Download full version + USearch indexes
uv run corp-extractor --db-version=2 db download  # Download v2 database files
uv run corp-extractor db create-lite entities-v3.db  # Create lite version (drops embeddings)
uv run corp-extractor db compress entities.db     # Compress with gzip
uv run corp-extractor db migrate-v2 entities.db entities-v2.db  # Migrate to v2 schema (v0.9.4)
uv run corp-extractor db backfill-scalar     # Generate int8 embeddings (v0.9.4)
uv run corp-extractor db post-import         # Run after any import: embeddings + USearch + VACUUM
uv run corp-extractor db post-import --no-orgs  # People only

# Document processing commands
uv run corp-extractor document process article.txt
uv run corp-extractor document process report.pdf                                 # Local PDF support
uv run corp-extractor document process report.pdf --pdf-parser glm_ocr_parser     # GLM-OCR VLM parser
uv run corp-extractor document process https://example.com/article
uv run corp-extractor document process https://example.com/report.pdf --use-ocr
uv run corp-extractor document chunk article.txt --max-tokens 500
```

## Architecture

### 5-Stage Pipeline

The extraction pipeline processes text through sequential stages, each with its own plugin type:

| Stage | Plugin Type | Purpose | Interface |
|-------|-------------|---------|-----------|
| 1 | `BaseSplitterPlugin` | Text → `SplitSentence[]` | `split(text, ctx)` |
| 2 | `BaseExtractorPlugin` | `SplitSentence[]` → `PipelineStatement[]` | `extract(sentences, ctx)` |
| 3 | `BaseQualifierPlugin` | Entity → `CanonicalEntity` | `qualify(entity, ctx)` |
| 4 | `BaseLabelerPlugin` | Statement → `StatementLabel` | `label(stmt, subj, obj, ctx)` |
| 5 | `BaseTaxonomyPlugin` | Statement → `TaxonomyResult[]` | `classify(stmt, subj, obj, ctx)` |

### Plugin Registration

Plugins auto-register via decorators when their modules are imported:

```python
from statement_extractor.pipeline.registry import PluginRegistry

@PluginRegistry.labeler
class MyLabeler(BaseLabelerPlugin):
    @property
    def name(self) -> str:
        return "my_labeler"

    @property
    def label_type(self) -> str:
        return "my_label"
    # ...
```

Plugins are sorted by `priority` property (lower = runs first). Default is 100.

### Key Source Files

- `pipeline/orchestrator.py` - Main pipeline coordinator, runs stages in order
- `pipeline/registry.py` - Plugin registration with `@PluginRegistry.<stage>` decorators
- `pipeline/context.py` - `PipelineContext` that flows through all stages
- `plugins/base.py` - Abstract base classes for all plugin types
- `plugins/extractors/gliner2.py` - GLiNER2 entity/relation extraction (Stage 2)
- `plugins/taxonomy/embedding.py` - Embedding-based taxonomy classification (Stage 5)
- `cli.py` - CLI entry point (thin wrapper, delegates to `commands/`)
- `commands/` - CLI command package:
  - `__init__.py` - Main click group + command registration
  - `_common.py` - Shared utilities (`_configure_logging`, `_resolve_db_path`, `_server_request`, etc.)
  - `split.py`, `pipeline.py`, `plugins.py`, `serve.py`, `document.py` - Top-level commands
  - `db/__init__.py` - `db` group + subcommand registration
  - `db/imports.py` - Import commands (GLEIF, SEC, Companies House, Wikidata, people, locations)
  - `db/wikidata_dump.py` - Wikidata dump import (3-thread reader/embedder/writer pipeline)
  - `db/search.py` - Search commands (orgs, people, roles, locations, perf test)
  - `db/management.py` - Status, canonicalize, download, upload, migrations, index building
  - `db/repair.py` - Repair/fix-resume commands for people records
- `server.py` - FastAPI persistent server (`corp-extractor serve`). Endpoints: `GET /` (health), `POST /pipeline`, `POST /split`, `POST /document`. All endpoints return Pydantic `model_dump()` JSON. Keeps models warm in memory. CLI delegates via `--server` / `--server-url` / `CORP_EXTRACTOR_SERVER` env var.
- `client.py` - HTTP client for server delegation (v0.9.8). Functions: `server_split()`, `server_pipeline()`, `server_document()`. Sends requests to a running server and reconstructs full Pydantic models via `model_validate()`. Used by `extract_statements(server_url=...)`, `ExtractionPipeline(server_url=...)`, `DocumentPipeline(server_url=...)`.

### Entity Database Module

The `database/` module provides organization, person, role, and location embedding storage and search:

- `database/models.py` - `CompanyRecord`, `CompanyMatch`, `PersonRecord`, `PersonMatch`, `PersonType`, `RoleRecord`, `LocationRecord`, `SimplifiedLocationType`, `DatabaseStats`, `EntityType` Pydantic models
- `database/store.py` - `OrganizationDatabase`, `PersonDatabase`, `RolesDatabase`, `LocationsDatabase` SQLite+USearch storage (shared connection pool, HNSW indexes for ANN search)
- `database/embeddings.py` - `CompanyEmbedder` using google/embeddinggemma-300m (supports both float32 and int8 quantization, batch_size=192)
- `database/hub.py` - HuggingFace Hub upload/download with lite/compressed variants
- `database/resolver.py` - `OrganizationResolver` shared utility for org lookups (used by person.py and embedding_company.py)
- `database/schema_v2.py` - DDL for v2 normalized schema with INTEGER FK references (v0.9.4)
- `database/seed_data.py` - Enum lookup tables and pycountry integration (v0.9.4)
- `database/migrate_v2.py` - Migration script from v1 to v2 schema (v0.9.4)
- `database/importers/` - Data source importers (all support `from_date`/`to_date`):
  - `gleif.py` - GLEIF LEI data (XML/JSON, 2.6M records) - `from_date` from InitialRegistrationDate
  - `sec_edgar.py` - SEC bulk submissions.zip (73K filers) - `from_date` from oldest filingDate
  - `sec_form4.py` - SEC Form 4 insider filings for officers/directors (v0.9.3) - extracts from quarterly indexes
  - `companies_house.py` - UK Companies House bulk data (5.5M records) - `from_date`/`to_date` from incorporation/dissolution
  - `companies_house_officers.py` - UK Companies House officers bulk data (Prod195, v0.9.3) - requires request to CH, 27.5M people
  - `wikidata.py` - Wikidata SPARQL queries (35+ entity types) - `from_date`/`to_date` from P571/P576
  - `wikidata_people.py` - Wikidata SPARQL queries for notable people - `from_date`/`to_date` from position qualifiers
  - `wikidata_dump.py` - Wikidata JSON dump importer (~100GB) for people (36M), orgs (1.5M), and locations without SPARQL timeouts (v0.9.1, locations in v0.9.4). 3-thread parallel import (reader/embedder/writer). Multi-record person import (one per position+org). Supports orjson, indexed_bzip2, zstandard.

**EntityType Classification:**
Each organization record is classified with an `entity_type` field:
- Business: `business`, `fund`, `branch`
- Non-profit: `nonprofit`, `ngo`, `foundation`, `trade_union`
- Government: `government`, `international_org`, `political_party`
- Other: `educational`, `research`, `healthcare`, `media`, `sports`, `religious`, `unknown`

**PersonType Classification (v0.9.2):**
Each person record is classified with a `person_type` field:
- `executive` - CEOs, board members, C-suite
- `politician` - Elected officials (presidents, MPs, mayors)
- `government` - Civil servants, diplomats, appointed officials
- `military` - Military officers, armed forces personnel
- `legal` - Judges, lawyers, legal professionals
- `professional` - Known for profession (doctors, engineers, architects)
- `academic` - Professors, researchers
- `artist` - Traditional creatives (musicians, actors, painters, writers)
- `media` - Internet/social media personalities (YouTubers, influencers)
- `athlete` - Sports figures
- `entrepreneur` - Founders, business owners
- `journalist` - Reporters, news presenters, columnists
- `activist` - Advocates, campaigners
- `scientist` - Scientists, inventors
- `unknown` - Type not determined

**People Database Features (v0.9.2):**
- Organizations discovered during people import are auto-inserted into the organizations table
- `known_for_org_id` foreign key links people to organizations table
- Same person can have multiple records with different role/org combinations (unique on source_id + role + org)
- `--skip-existing` flag to skip existing records instead of updating
- `--enrich-dates` flag to fetch individual role start/end dates (slower, queries per person)
- `birth_date` and `death_date` fields track life dates; `is_historic` property for deceased people
- `--require-enwiki` flag to only import orgs with English Wikipedia articles

**Wikidata Dump Import Features (v0.9.5):**
- 3-thread parallel pipeline: reader thread → embedder thread → writer thread for overlapping I/O, CPU encoding, and disk writes
- Multi-record person import: creates one record per position+org combination (max 10 per person, deduped by role_qid+org_qid)
- Reverse org→person mappings: extracts P580/P582 date qualifiers from executive properties (P169 CEO, P488 chairperson, etc.)
- Auto-canonicalization after import with recency tiebreaker (most recent from_date preferred)
- Optional `orjson` for ~3x faster JSON parsing (`pip install corp-extractor[fast-import]`)
- Optional `indexed_bzip2` for parallel bzip2 decompression (6x faster with N cores)
- Zstandard (.zst/.zstd) compressed dumps supported
- Single-pass import: people, orgs, and locations are extracted in one scan of the ~100GB dump file
- `--locations` flag to import geopolitical entities (countries, states, cities) with hierarchy
- `--resume` flag to resume from last position in file (tracks entity index in progress file)
- `--skip-updates` flag to skip Q codes already in database (no updates to existing records)
- Progress file stored at `~/.cache/corp-extractor/wikidata-dump-progress.json`
- Progress saved after each batch for reliable resume on interruption
- All import commands output reminder to run `db post-import` after completion

**Organization Canonicalization (v0.9.2):**
The `db canonicalize` command links equivalent records across sources:
- Records matched by LEI, ticker, CIK (globally unique, no region check)
- Records matched by normalized name + region (using pycountry for region normalization)
- Source priority: gleif > sec_edgar > companies_house > wikipedia
- Enables prominence-based search re-ranking that boosts companies with records from multiple sources

**People Canonicalization (v0.9.3):**
The `db canonicalize` command also links equivalent people records:
- Records matched by normalized name + same organization (using org canonical group)
- Records matched by normalized name + overlapping date ranges
- Source priority: wikidata > sec_edgar > companies_house

**Database Schema v3 (v0.9.6):**
- `db_info` metadata table with `schema_version=3` for version detection
- Lite databases drop ALL embedding tables (float32 + scalar int8) — uses USearch index files for search
- `db upload` includes USearch `.bin` files alongside database variants
- `db download` fetches USearch indexes alongside database
- Global `--db-version` CLI flag for backwards compatibility (e.g. `corp-extractor --db-version=2 db download`)
- Symlink `entities-v2.db` → v3 file created on download for backwards compat

**Database Schema v2 (v0.9.4):**
The database uses a normalized schema with INTEGER FK references instead of TEXT enum columns:
- Enum lookup tables: `source_types`, `people_types`, `organization_types`, `location_types`, `simplified_location_types`
- New tables: `roles` (job titles with QID), `locations` (countries/states/cities with hierarchy)
- QIDs stored as integers (Q prefix stripped) in `qid` column
- Human-readable views: `organizations_view`, `people_view`, `roles_view`, `locations_view`
- Both float32 and int8 scalar embeddings supported (75% storage reduction with ~92% recall)
- USearch HNSW indexes for sub-millisecond approximate nearest neighbor search on 50M+ vectors
- USearch index files: `people_usearch.bin`, `organizations_usearch.bin` (same dir as DB)
- USearch `expansion_search` must be set to 200 after `Index.restore()` (not persisted by default)
- Search modes: embeddings-only (default, uses USearch) or hybrid text+embeddings (`--hybrid` flag)
- vec0 tables now use `distance_metric=cosine` for indexed KNN support
- SQLite pragmas: 256MB mmap, 500MB page cache, WAL journal mode for write-heavy operations
- Default database path: `~/.cache/corp-extractor/entities-v3.db`

**Database variants:**
- `entities-v3.db` - Full v3 database with all embedding tables
- `entities-v3-lite.db` - Lite version without record data or embedding tables (default download)
- `organizations_usearch.bin` - USearch HNSW index for organizations
- `people_usearch.bin` - USearch HNSW index for people
- `entities-v2.db` - Legacy v2 symlink (backwards compatibility)

### Document Processing Module

The `document/` module provides document-level extraction:

- `document/chunker.py` - Token-based text chunking with overlap
- `document/context.py` - `DocumentContext` for tracking extraction state
- `document/deduplicator.py` - Cross-chunk statement deduplication
- `document/html_extractor.py` - HTML content extraction (Readability-style)
- `document/loader.py` - URL and file loading with content type detection
- `document/pipeline.py` - `DocumentPipeline` orchestrator
- `document/summarizer.py` - Document summarization

**PDF and Scraper Plugins:**
- `plugins/pdf/pypdf.py` - Default PDF parser using PyMuPDF with Tesseract OCR fallback (priority 100)
- `plugins/pdf/glm_ocr.py` - GLM-OCR 0.9B VLM parser for high-quality OCR of scans, tables, formulas (priority 200). Renders pages to images, runs in-process via HuggingFace transformers. Select with `--pdf-parser glm_ocr_parser`.
- `plugins/scrapers/http.py` - HTTP/URL scraping with httpx

### Data Models Flow

```
Text → SplitSentence → PipelineStatement → QualifiedEntity → CanonicalEntity → LabeledStatement
       (Stage 1)       (with ExtractedEntity)                                   (with TaxonomyResult[])
                       (Stage 2)             (Stage 3)        (Stage 3)          (Stage 4-5)
```

Models are defined in `models/` subdirectory: `entity.py`, `statement.py`, `qualifiers.py`, `canonical.py`, `labels.py`.

Note: `RawTriple` is a deprecated alias for `SplitSentence` (backwards compatibility).

### GLiNER2 Relation Extraction

The `gliner2_extractor` uses 324 default predicates from `data/default_predicates.json` organized into 21 categories:

- **All matching relations returned** - Every relation above confidence threshold is kept (not just the best one)
- **Category-based iteration** - Processes each category separately to stay under GLiNER2's ~25 label limit
- **Confidence filtering** - Relations below `min_confidence` (default 0.75) are filtered out
- **Entity type inference** - Subject/object types determined via separate GLiNER2 entity extraction on source text
- **Classification schemas** - Labeler plugins can provide schemas that run in Stage 2 (stored in `ctx.classification_results`)

### Taxonomy Classification

Stage 5 uses embedding-based classification against `data/statement_taxonomy.json` (ESG topics). Results are stored both in `ctx.taxonomy_results` and on each `LabeledStatement.taxonomy_results`.

## Testing

Tests use pytest markers:
- `@pytest.mark.slow` - Tests that load models (skip with `-m "not slow"`)
- `@pytest.mark.pipeline` - Pipeline architecture tests

Fixtures in `conftest.py` provide `sample_source_text` and `sample_statements`.
