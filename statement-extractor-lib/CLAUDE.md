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

# Entity database — moved out to the separate corp-entity-db project in
# v0.10.0. See https://corp-entity-db.vercel.app/ for the CLI reference,
# search/download instructions, and build pipeline.

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

As of v0.10.0 the entity database lives in the separate
[corp-entity-db project](https://corp-entity-db.vercel.app/). Within
this lib, `database/` is a thin re-export shim:

- `database/store.py` - re-exports `OrganizationDatabase`, `PersonDatabase`,
  `RolesDatabase`, `LocationsDatabase`, `CompanyMatch`, `PersonMatch`,
  `DatabaseStats`, factories from `corp_entity_db`
- `database/resolver.py` - re-exports `OrganizationResolver`,
  `get_organization_resolver`
- `database/embeddings.py` - thin alias around `corp_entity_db.embeddings`
- `database/hub.py` - thin alias around `corp_entity_db.hub`
- `database/models.py` - thin alias around `corp_entity_db.models`
  (`CompanyRecord`, `PersonRecord`, `EntityType`, `PersonType`, etc.)

Importers, schema migration, the `db` CLI subgroup, and canonicalization
all moved to corp-entity-db.

EntityType / PersonType classifications, schema details, canonicalization
rules, dump-import internals, and database variants are all owned by the
[corp-entity-db project](https://corp-entity-db.vercel.app/). Refer there
when changes to those areas need documentation.

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
