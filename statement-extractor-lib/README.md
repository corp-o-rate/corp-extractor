# Corp Extractor

Analyze complex text to extract relationship information about people and organizations. Runs entirely on your hardware (RTX 4090+, Apple M1 16GB+) with no external API dependencies. Uses fine-tuned T5-Gemma 2 for statement splitting and coreference resolution, plus GLiNER2 for entity extraction. Includes a database of 9.7M+ organizations and 63M+ people with USearch HNSW indexes for fast entity qualification.

[![PyPI version](https://img.shields.io/pypi/v/corp-extractor.svg)](https://pypi.org/project/corp-extractor/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/corp-extractor.svg)](https://pypi.org/project/corp-extractor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Entity DB extracted to `corp-entity-db`** *(v0.10.0)*: Organization/person/role/location storage, importers, canonicalization, and search now live in a separate [corp-entity-db](https://pypi.org/project/corp-entity-db/) package. `corp-extractor` depends on it for qualification but no longer ships its own importer/CLI commands. Use the `corp-entity-db` CLI directly for DB management.
- **Database v3** *(v0.9.6)*: Lite databases drop all embedding tables — USearch indexes distributed with `db download`/`db upload`. Global `--db-version` flag for version control.
- **USearch HNSW Indexes** *(v0.9.5)*: Sub-millisecond approximate nearest neighbor search on 50M+ vectors with pre-built HNSW indexes
- **Parallel Import** *(v0.9.5)*: 3-thread pipeline (reader/embedder/writer) for fast Wikidata dump import with optional orjson and indexed_bzip2
- **Database v2 Schema** *(v0.9.4)*: Normalized schema with INTEGER FK references, new roles/locations tables, int8 scalar embeddings (75% smaller)
- **Person Database** *(v0.9.2)*: Qualify notable people (executives, politicians, athletes, etc.) against Wikidata with canonical IDs
- **Organization Canonicalization** *(v0.9.2)*: Link equivalent records across sources (LEI, ticker, CIK, name matching)
- **5-Stage Pipeline** *(v0.8.0)*: Modular plugin-based architecture for full entity resolution
- **Document Processing** *(v0.7.0)*: Process documents, URLs, and PDFs with chunking and deduplication
- **Entity Embedding Database** *(v0.6.0)*: Fast entity qualification using vector similarity (9.7M organizations, 63M people)
- **Structured Extraction**: Converts unstructured text into subject-predicate-object triples
- **Entity Type Recognition**: Identifies 12 entity types (ORG, PERSON, GPE, LOC, PRODUCT, EVENT, etc.)
- **Entity Qualification** *(v0.8.0)*: Adds identifiers (LEI, ticker, company numbers), canonical names, and FQN via embedding database
- **Statement Labeling** *(v0.5.0)*: Sentiment analysis, relation type classification, confidence scoring
- **GLiNER2 Integration** *(v0.4.0)*: Uses GLiNER2 (205M params) for relation extraction with 324 predicates across 21 categories
- **All Relations Returned**: Every matching relation above confidence threshold (0.75) is returned, not just the best one
- **Predicate Taxonomies**: Normalize predicates to canonical forms via embeddings
- **Command Line Interface**: Full-featured CLI with `split`, `pipeline`, `document`, and `db` commands
- **Multiple Output Formats**: Get results as Pydantic models, JSON, XML, or dictionaries

## Installation

```bash
pip install corp-extractor
```

The GLiNER2 model (205M params) is downloaded automatically on first use.

**Note**: This package requires `transformers>=5.0.0` for T5-Gemma2 model support.

**For GPU support**, install PyTorch with CUDA first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install corp-extractor
```

**For Apple Silicon (M1/M2/M3)**, MPS acceleration is automatically detected:
```bash
pip install corp-extractor  # MPS used automatically
```

**For fast Wikidata dump import** (optional, ~3x JSON parsing + 6x decompression):
```bash
pip install "corp-extractor[fast-import]"  # Adds orjson + indexed_bzip2
```

## Quick Start

```python
from statement_extractor import extract_statements

result = extract_statements("""
    Apple Inc. announced the iPhone 15 at their September event.
    Tim Cook presented the new features to customers worldwide.
""")

for stmt in result:
    print(f"{stmt.subject.text} ({stmt.subject.type})")
    print(f"  --[{stmt.predicate}]--> {stmt.object.text}")
    print(f"  Confidence: {stmt.confidence_score:.2f}")  # NEW in v0.2.0
```

## Command Line Interface

The library includes a CLI for quick extraction from the terminal.

### Install Globally (Recommended)

For best results, install globally first:

```bash
# Using uv (recommended)
uv tool install "corp-extractor[embeddings]"

# Using pipx
pipx install "corp-extractor[embeddings]"

# Using pip
pip install "corp-extractor[embeddings]"

# Then use anywhere
corp-extractor "Your text here"
```

### Quick Run with uvx

Run directly without installing using [uv](https://docs.astral.sh/uv/):

```bash
uvx corp-extractor "Apple announced a new iPhone."
```

**Note**: First run downloads the model (~1.5GB) which may take a few minutes.

### Usage Examples

The CLI provides commands for extraction (`split`, `pipeline`), document processing, database management, and a persistent server.

```bash
# Simple extraction (Stage 1 only, fast)
corp-extractor split "Apple Inc. announced the iPhone 15."
corp-extractor split -f article.txt --json

# Full 5-stage pipeline (entity resolution, labeling, taxonomy)
corp-extractor pipeline "Amazon CEO Andy Jassy announced plans to hire workers."
corp-extractor pipeline -f article.txt --stages 1-3
corp-extractor pipeline "..." --disable-plugins sec_edgar

# Persistent server — keeps models warm for fast repeated use (v0.9.7)
corp-extractor serve                                        # Start on localhost:8111
corp-extractor --server pipeline "Amazon CEO Andy Jassy..."  # Delegate to server
corp-extractor --server-url http://gpu:8111 split "text"     # Custom server URL

# Plugin management
corp-extractor plugins list
corp-extractor plugins list --stage 3
corp-extractor plugins info gleif_qualifier
```

### Split Command (Simple Extraction)

```bash
corp-extractor split "Tim Cook is CEO of Apple." --json
corp-extractor split -f article.txt --beams 8 --verbose
cat article.txt | corp-extractor split -
```

### Pipeline Command (Full Entity Resolution)

```bash
# Run all 5 stages
corp-extractor pipeline "Apple CEO Tim Cook announced..."

# Run specific stages
corp-extractor pipeline "..." --stages 1-3         # Stages 1, 2, 3
corp-extractor pipeline "..." --stages 1,2,5       # Stages 1, 2, 5
corp-extractor pipeline "..." --skip-stages 4,5    # Skip stages 4 and 5

# Plugin selection
corp-extractor pipeline "..." --plugins gleif,companies_house
corp-extractor pipeline "..." --disable-plugins sec_edgar
```

### CLI Reference

```
Usage: corp-extractor [OPTIONS] COMMAND [ARGS]...

Global Options:
  --server                     Use local server at http://localhost:8111
  --server-url URL             Use server at custom URL
  --db-version N               Database schema version (default: latest)

Commands:
  split      Simple extraction (T5-Gemma only)
  pipeline   Full 5-stage pipeline with entity resolution
  document   Document processing (files, URLs, PDFs)
  db         Entity database management
  serve      Start persistent local server
  plugins    List or inspect available plugins

Split Options:
  -f, --file PATH              Read input from file
  -o, --output [table|json|xml] Output format (default: table)
  --json / --xml               Output format shortcuts
  -b, --beams INTEGER          Number of beams (default: 4)
  --no-gliner                  Disable GLiNER2 extraction
  --predicates TEXT            Comma-separated predicates for relation extraction
  --device [auto|cuda|mps|cpu] Device to use (default: auto)
  -v, --verbose                Show confidence scores and metadata

Pipeline Options:
  --stages TEXT                Stages to run (e.g., '1-3' or '1,2,5')
  --skip-stages TEXT           Stages to skip (e.g., '4,5')
  --plugins TEXT               Enable only these plugins (comma-separated)
  --disable-plugins TEXT       Disable these plugins (comma-separated)
  -o, --output [table|json|yaml|triples]  Output format

Serve Options:
  --host TEXT                  Bind address (default: 0.0.0.0)
  --port INTEGER               Port number (default: 8111)
  --no-warmup                  Skip eager model loading
  -v, --verbose                Enable debug logging

Environment Variables:
  CORP_EXTRACTOR_SERVER        Server URL fallback (e.g., http://localhost:8111)
```

## Persistent Server (v0.9.7)

The `corp-extractor serve` command starts a FastAPI server that keeps all models (T5-Gemma, GLiNER2, embedding models, USearch indexes) warm in memory. This eliminates the ~30s startup cost for repeated CLI invocations.

```bash
# Start the server
corp-extractor serve                    # Default: 0.0.0.0:8111
corp-extractor serve --port 9000        # Custom port
corp-extractor serve --no-warmup        # Skip eager model loading

# Use from CLI (in another terminal)
corp-extractor --server pipeline "Apple CEO Tim Cook announced..."
corp-extractor --server split -f article.txt --json
corp-extractor --server document process article.txt

# Or set the environment variable
export CORP_EXTRACTOR_SERVER=http://localhost:8111
corp-extractor pipeline "text"          # Automatically delegates to server
```

### Python API Server Delegation (v0.9.8)

All Python API functions accept a `server_url` parameter to delegate processing to a running server instead of loading models locally. This is useful for machines without a GPU or when you want to share a single model instance.

```python
from statement_extractor import extract_statements
from statement_extractor.pipeline import ExtractionPipeline
from statement_extractor.document import DocumentPipeline

# Simple extraction via server
result = extract_statements("Apple announced iPhone.", server_url="http://localhost:8111")
for stmt in result:
    print(f"{stmt.subject.text} -> {stmt.predicate} -> {stmt.object.text}")

# Full pipeline via server
pipeline = ExtractionPipeline(server_url="http://localhost:8111")
ctx = pipeline.process("Amazon CEO Andy Jassy announced...")
for stmt in ctx.labeled_statements:
    print(f"{stmt.subject_fqn} -> {stmt.statement.predicate} -> {stmt.object_fqn}")

# Document pipeline via server
doc_pipeline = DocumentPipeline(server_url="http://localhost:8111")
ctx = doc_pipeline.process(document)
```

When `server_url` is provided, no local models are loaded — the HTTP client sends the request to the server and reconstructs the full Pydantic response objects (`ExtractionResult`, `PipelineContext`, `DocumentContext`) from the JSON response.

> **Note:** The `server_url` parameter is Python API only. For CLI delegation, use `--server` / `--server-url` flags or the `CORP_EXTRACTOR_SERVER` environment variable.

### Server Endpoints

The server exposes three POST endpoints that mirror the CLI commands:

| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check — device info, loaded models, registered plugins |
| `POST /pipeline` | Full extraction pipeline (returns `PipelineContext` model_dump) |
| `POST /split` | Stage 1 extraction only (returns `ExtractionResult` model_dump) |
| `POST /document` | Document pipeline for text input (returns `DocumentContext` model_dump) |

All endpoints return standardized Pydantic `model_dump()` JSON. You can also call the server directly with any HTTP client:

```bash
curl http://localhost:8111/                          # Health check
curl -X POST http://localhost:8111/split \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple announced a new iPhone."}'
```

## Quality Scoring & Beam Merging

By default, the library:
- **Scores each triple** using semantic similarity (50%) + GLiNER2 entity recognition (50%)
- **Merges top beams** instead of selecting one, improving coverage
- **Uses embeddings** to detect semantically similar predicates ("bought" ≈ "acquired")

```python
from statement_extractor import ExtractionOptions, ScoringConfig

# Precision mode - filter low-confidence triples
scoring = ScoringConfig(min_confidence=0.7)
options = ExtractionOptions(scoring_config=scoring)
result = extract_statements(text, options)

# Access confidence scores
for stmt in result:
    print(f"{stmt} (confidence: {stmt.confidence_score:.2f})")
```

## New in v0.2.0: Predicate Taxonomies

Normalize predicates to canonical forms using embedding similarity:

```python
from statement_extractor import PredicateTaxonomy, ExtractionOptions

taxonomy = PredicateTaxonomy(predicates=[
    "acquired", "founded", "works_for", "announced",
    "invested_in", "partnered_with"
])

options = ExtractionOptions(predicate_taxonomy=taxonomy)
result = extract_statements(text, options)

# "bought" -> "acquired" via embedding similarity
for stmt in result:
    if stmt.canonical_predicate:
        print(f"{stmt.predicate} -> {stmt.canonical_predicate}")
```

## New in v0.2.2: Contextualized Matching

Predicate canonicalization and deduplication now use **contextualized matching**:
- Compares full "Subject Predicate Object" strings against source text
- Better accuracy because predicates are evaluated in context
- When duplicates are found, keeps the statement with the best match to source text

This means "Apple bought Beats" vs "Apple acquired Beats" are compared holistically, not just "bought" vs "acquired".

## New in v0.2.3: Entity Type Merging & Reversal Detection

### Entity Type Merging

When deduplicating statements, entity types are now automatically merged. If one statement has `UNKNOWN` type and a duplicate has a specific type (like `ORG` or `PERSON`), the specific type is preserved:

```python
# Before deduplication:
# Statement 1: AtlasBio Labs (UNKNOWN) --sued by--> CuraPharm (ORG)
# Statement 2: AtlasBio Labs (ORG) --sued by--> CuraPharm (ORG)

# After deduplication:
# Single statement: AtlasBio Labs (ORG) --sued by--> CuraPharm (ORG)
```

### Subject-Object Reversal Detection

The library now detects when subject and object may have been extracted in the wrong order by comparing embeddings against source text:

```python
from statement_extractor import PredicateComparer

comparer = PredicateComparer()

# Automatically detect and fix reversals
fixed_statements = comparer.detect_and_fix_reversals(statements)

for stmt in fixed_statements:
    if stmt.was_reversed:
        print(f"Fixed reversal: {stmt}")
```

**How it works:**
1. For each statement with source text, compares:
   - "Subject Predicate Object" embedding vs source text
   - "Object Predicate Subject" embedding vs source text
2. If the reversed form has higher similarity, swaps subject and object
3. Sets `was_reversed=True` to indicate the correction

During deduplication, reversed duplicates (e.g., "A -> P -> B" and "B -> P -> A") are now detected and merged, with the correct orientation determined by source text similarity.

## Pipeline Architecture

The library uses a **5-stage plugin-based pipeline** for comprehensive entity resolution, statement enrichment, and taxonomy classification.

### Pipeline Stages

| Stage | Name | Input | Output | Key Tech |
|-------|------|-------|--------|----------|
| 1 | Splitting | Text | `SplitSentence[]` | T5-Gemma2 |
| 2 | Extraction | `SplitSentence[]` | `PipelineStatement[]` | GLiNER2 |
| 3 | Qualification | Entities | `CanonicalEntity[]` | Embedding DB |
| 4 | Labeling | Statements | `LabeledStatement[]` | Sentiment, etc. |
| 5 | Taxonomy | Statements | `TaxonomyResult[]` | Embeddings |

### Pipeline Python API

```python
from statement_extractor.pipeline import ExtractionPipeline, PipelineConfig

# Run full pipeline
pipeline = ExtractionPipeline()
ctx = pipeline.process("Amazon CEO Andy Jassy announced plans to hire workers.")

# Access results at each stage
print(f"Split sentences: {len(ctx.split_sentences)}")
print(f"Statements: {len(ctx.statements)}")
print(f"Labeled: {len(ctx.labeled_statements)}")

# Output with fully qualified names
for stmt in ctx.labeled_statements:
    print(f"{stmt.subject_fqn} --[{stmt.statement.predicate}]--> {stmt.object_fqn}")
    # e.g., "Andy Jassy (CEO, Amazon) --[announced]--> plans to hire workers"
```

### Pipeline Configuration

```python
from statement_extractor.pipeline import PipelineConfig, ExtractionPipeline

# Run only specific stages
config = PipelineConfig(
    enabled_stages={1, 2, 3},  # Skip labeling and taxonomy
    disabled_plugins={"person_qualifier"},  # Disable specific plugins
)
pipeline = ExtractionPipeline(config)
ctx = pipeline.process(text)

# Alternative: create config from stage string
config = PipelineConfig.from_stage_string("1-3")  # Stages 1, 2, 3
```

### Built-in Plugins

**Splitters (Stage 1):**
- `t5_gemma_splitter` - T5-Gemma2 statement extraction

**Extractors (Stage 2):**
- `gliner2_extractor` - GLiNER2 entity recognition and relation extraction

**Qualifiers (Stage 3):**
- `person_qualifier` - PERSON → role, org, canonical ID via Wikidata person database *(enhanced in v0.9.0)*
- `embedding_company_qualifier` - ORG → canonical name, identifiers (LEI, CIK, company number), and FQN via embedding database

**Labelers (Stage 4):**
- `sentiment_labeler` - Statement sentiment analysis
- `confidence_labeler` - Confidence scoring
- `relation_type_labeler` - Relation type classification

**Taxonomy Classifiers (Stage 5):**
- `mnli_taxonomy_classifier` - MNLI zero-shot classification against ESG taxonomy
- `embedding_taxonomy_classifier` - Embedding similarity-based taxonomy classification

Taxonomy classifiers return **multiple labels** per statement above the confidence threshold.

**PDF Parsers:**
- `pypdf_parser` (priority 100, default) - PyMuPDF text extraction with Tesseract OCR fallback
- `glm_ocr_parser` (priority 200) - GLM-OCR 0.9B VLM for high-quality OCR of scans, tables, and formulas

**Scrapers:**
- `http_scraper` - URL/web page scraping with httpx

## Entity Database

The library includes an **entity embedding database** for fast entity qualification using vector similarity search. It stores records from authoritative sources (GLEIF, SEC, Companies House, Wikidata) with 768-dimensional embeddings for semantic matching.

**Quick start:**
```bash
corp-extractor db download              # Download pre-built database + USearch indexes
corp-extractor db search "Microsoft"    # Search organizations
corp-extractor db search-people "Tim Cook"  # Search people
corp-extractor db search-roles "CEO"    # Search roles (v0.9.4)
corp-extractor db search-locations "California"  # Search locations (v0.9.4)
```

For comprehensive documentation including schema, CLI reference, Python API, and build instructions, see **[ENTITY_DATABASE.md](./ENTITY_DATABASE.md)**.

## New in v0.9.5: USearch HNSW Indexes & Parallel Import

v0.9.5 introduces **USearch HNSW indexes** for sub-millisecond approximate nearest neighbor search and a **3-thread parallel import pipeline** for fast Wikidata dump imports.

- **USearch HNSW indexes**: Pre-built indexes for both people and organizations enable sub-millisecond search on 50M+ vectors
- **3-thread parallel import**: Reader, embedder, and writer threads work concurrently during Wikidata dump import
- **Multi-record person import**: One person now yields multiple records (one per position+org combination, max 10)
- **Auto-canonicalization**: Runs automatically after dump import with recency tiebreaker (most recent from_date preferred)
- **`--hybrid` search flag**: Combine text filtering with USearch embeddings search for higher precision
- **Fast import extras**: Optional `orjson` (~3x JSON parsing) and `indexed_bzip2` (6x decompression)
- **Zstandard support**: Import from `.zst`/`.zstd` compressed dumps
- **`db post-import`**: New command to run after any import (generates embeddings, builds USearch indexes, runs VACUUM)
- **`db build-index`**: Build or rebuild USearch HNSW indexes
- **`hf_classifier` labeler**: New general-purpose HuggingFace sequence classification labeler plugin

```bash
# After any import, run post-import to build indexes
corp-extractor db post-import

# Or build just the USearch index
corp-extractor db build-index

# Search with hybrid mode
corp-extractor db search "Microsoft" --hybrid

# Install fast-import extras
pip install "corp-extractor[fast-import]"
```

## New in v0.9.4: Database v2 Schema

v0.9.4 introduces a **normalized v2 schema** with significant improvements:

- **INTEGER FK references** replace TEXT enum columns for better query performance
- **New enum lookup tables**: `source_types`, `people_types`, `organization_types`, `location_types`
- **New tables**: `roles` (job titles with Wikidata QID), `locations` (countries/states/cities with hierarchy)
- **Scalar (int8) embeddings**: 75% storage reduction with ~92% recall at top-100
- **QID as integers**: Wikidata QIDs stored as integers (Q prefix stripped)
- **Human-readable views**: `organizations_view`, `people_view`, `roles_view`, `locations_view`

**Migration:**
```bash
# Migrate existing v1 database to v2
corp-extractor db migrate-v2 entities.db entities-v2.db

# Generate int8 scalar embeddings
corp-extractor db backfill-scalar
```

**Default database path**: `~/.cache/corp-extractor/entities-v3.db`

## New in v0.6.0: Entity Embedding Database

v0.6.0 introduces an **entity embedding database** for fast entity qualification using vector similarity search.

### Data Sources

**Organizations:**

| Source | Records | Identifier | EntityType Mapping |
|--------|---------|------------|-------------------|
| Companies House | 5.5M | UK Company Number | Maps company_type to business/nonprofit |
| GLEIF | 2.6M | LEI (Legal Entity Identifier) | GENERAL→business, FUND→fund, BRANCH→branch, INTERNATIONAL_ORGANIZATION→international_org |
| Wikidata | 1.5M | Wikidata QID | 35+ query types mapped to EntityType |
| SEC Edgar | 73K | CIK (Central Index Key) | business (or fund via SIC codes) |

**People** *(v0.9.0)*:

| Source | Records | Identifier | PersonType Classification |
|--------|---------|------------|--------------------------|
| Companies House | 27.5M | Person number | UK company officers |
| Wikidata | 36M | Wikidata QID | executive, politician, athlete, artist, academic, scientist, journalist, entrepreneur, activist |
| Wikidata (Dump) | All humans with enwiki | Wikidata QID | Classified from positions (P39) and occupations (P106) |

**Date Fields**: All importers now include `from_date` and `to_date` where available:
- **GLEIF**: LEI registration date
- **SEC Edgar**: First SEC filing date
- **Companies House**: Incorporation and dissolution dates
- **Wikidata Orgs**: Inception (P571) and dissolution (P576) dates
- **Wikidata People**: Position start (P580) and end (P582) dates

**Note**: The same person can have multiple records with different role/org combinations (unique on `source_id + role + org`). Organizations discovered during people import are automatically inserted into the organizations table with `known_for_org_id` foreign key linking people to their organizations.

### EntityType Classification

Each organization record is classified with an `entity_type` field:

| Category | Types |
|----------|-------|
| Business | `business`, `fund`, `branch` |
| Non-profit | `nonprofit`, `ngo`, `foundation`, `trade_union` |
| Government | `government`, `international_org`, `political_party` |
| Other | `educational`, `research`, `healthcare`, `media`, `sports`, `religious`, `unknown` |

### Building the Database

```bash
# Import organizations from authoritative sources
corp-extractor db import-gleif --download
corp-extractor db import-sec --download      # Bulk submissions.zip (~100K+ filers)
corp-extractor db import-companies-house --download
corp-extractor db import-wikidata --limit 50000

# Import notable people (v0.9.0)
corp-extractor db import-people --type executive --limit 5000
corp-extractor db import-people --all --limit 10000  # All person types
corp-extractor db import-people --type executive --skip-existing  # Skip existing records
corp-extractor db import-people --type executive --enrich-dates   # Fetch role start/end dates

# Import from Wikidata dump (v0.9.1) - avoids SPARQL timeouts
corp-extractor db import-wikidata-dump --download --limit 50000   # Downloads ~100GB dump
corp-extractor db import-wikidata-dump --dump /path/to/dump.bz2 --people --no-orgs  # Local dump
corp-extractor db import-wikidata-dump --dump dump.bz2 --locations --no-people --no-orgs  # Locations only (v0.9.4)

# Migrate to v2 schema (v0.9.4)
corp-extractor db migrate-v2 entities.db entities-v2.db
corp-extractor db backfill-scalar        # Generate int8 embeddings (75% smaller)

# Post-import: generate embeddings, build USearch indexes, VACUUM (v0.9.5)
corp-extractor db post-import
corp-extractor db post-import --no-orgs  # People only

# Build USearch HNSW index for fast ANN search (v0.9.5)
corp-extractor db build-index

# Check status
corp-extractor db status

# Search for an organization
corp-extractor db search "Microsoft"

# Search for a person (v0.9.0)
corp-extractor db search-people "Tim Cook"
```

### Using in Pipeline

The database is automatically used by the `embedding_company_qualifier` plugin for Stage 3 (Qualification):

```python
from statement_extractor.pipeline import ExtractionPipeline

pipeline = ExtractionPipeline()
ctx = pipeline.process("Microsoft acquired Activision Blizzard.")

for stmt in ctx.labeled_statements:
    print(f"{stmt.subject_fqn}")  # e.g., "Microsoft (sec_edgar:0000789019)"
```

### Publishing to HuggingFace

```bash
# Upload database with all variants (full, lite, USearch indexes)
export HF_TOKEN="hf_..."
corp-extractor db upload                     # Uses default cache location
corp-extractor db upload entities-v3.db      # Or specify path
corp-extractor db upload --no-lite           # Skip lite version

# Download pre-built database + USearch indexes (lite version by default)
corp-extractor db download                   # Lite version + USearch indexes
corp-extractor db download --full            # Full version + USearch indexes
corp-extractor --db-version=2 db download    # Download v2 files

# Local database management
corp-extractor db create-lite entities-v3.db # Create lite version (drops embeddings)
```

See [ENTITY_DATABASE.md](./ENTITY_DATABASE.md) for complete build and publish instructions.

## New in v0.7.0: Document Processing

v0.7.0 introduces **document-level processing** for handling files, URLs, and PDFs with automatic chunking, deduplication, and citation tracking.

### Document CLI

```bash
# Process local files
corp-extractor document process article.txt
corp-extractor document process report.txt --title "Annual Report" --year 2024

# Process local PDFs (auto-detected by .pdf extension)
corp-extractor document process report.pdf
corp-extractor document process report.pdf --pdf-parser glm_ocr_parser

# Process URLs (web pages and PDFs)
corp-extractor document process https://example.com/article
corp-extractor document process https://example.com/report.pdf --use-ocr

# Configure chunking
corp-extractor document process article.txt --max-tokens 500 --overlap 50

# Preview chunking without extraction
corp-extractor document chunk article.txt --max-tokens 500
```

### Document Python API

```python
from statement_extractor.document import DocumentPipeline, DocumentPipelineConfig, Document
from statement_extractor.models.document import ChunkingConfig

# Configure document processing
config = DocumentPipelineConfig(
    chunking=ChunkingConfig(target_tokens=1000, overlap_tokens=100),
    generate_summary=True,
    deduplicate_across_chunks=True,
)

pipeline = DocumentPipeline(config)

# Process text
document = Document.from_text("Your long document text...", title="My Document")
ctx = pipeline.process(document)

# Process URL (async)
ctx = await pipeline.process_url("https://example.com/article")

# Access results
print(f"Chunks: {ctx.chunk_count}")
print(f"Statements: {ctx.statement_count}")
print(f"Duplicates removed: {ctx.duplicates_removed}")

for stmt in ctx.labeled_statements:
    print(f"{stmt.subject_fqn} --[{stmt.statement.predicate}]--> {stmt.object_fqn}")
    if stmt.citation:
        print(f"  Citation: {stmt.citation}")
```

### PDF Processing

PDFs are automatically parsed using PyMuPDF (default `pypdf_parser`). For scanned documents, image-heavy PDFs, tables, and formulas, use the GLM-OCR VLM parser:

```bash
# Process a local PDF (uses default pypdf_parser)
corp-extractor document process report.pdf

# Use GLM-OCR 0.9B VLM for high-quality OCR
corp-extractor document process scanned.pdf --pdf-parser glm_ocr_parser

# Force Tesseract OCR with the default parser
corp-extractor document process scanned.pdf --use-ocr

# Process PDF from URL
corp-extractor document process https://example.com/report.pdf --pdf-parser glm_ocr_parser
```

**Available PDF parsers** (listed via `corp-extractor plugins list`):

| Parser | Priority | Description |
|--------|----------|-------------|
| `pypdf_parser` | 100 (default) | PyMuPDF text extraction with Tesseract OCR fallback |
| `glm_ocr_parser` | 200 | GLM-OCR 0.9B VLM — renders pages to images, outputs markdown. Best for scans, tables, formulas. |

## New in v0.4.0: GLiNER2 Integration

v0.4.0 replaces spaCy with **GLiNER2** (205M params) for entity recognition and relation extraction. GLiNER2 is a unified model that handles NER, text classification, structured data extraction, and relation extraction with CPU-optimized inference.

### Why GLiNER2?

The T5-Gemma model excels at:
- **Atomic sentence splitting** - breaking complex text into simple statements
- **Coreference resolution** - resolving pronouns to named entities

GLiNER2 handles:
- **Relation extraction** - using 324 default predicates across 21 categories
- **Entity type inference** - classifying subjects/objects as ORG, PERSON, GPE, etc.
- **Confidence scoring** - real confidence values for each extracted relation

### Default Predicates

GLiNER2 uses **324 predicates** organized into 21 categories (ownership, employment, funding, etc.). These are loaded from `default_predicates.json` and include descriptions and confidence thresholds.

**Key features:**
- **All matches returned** - Every matching relation is returned, not just the best one
- **Category-based extraction** - Iterates through categories to stay under GLiNER2's ~25 label limit
- **Custom predicate files** - Provide your own JSON file with custom predicates

### Extraction Modes

**Mode 1: Default Predicates** (recommended)
```python
from statement_extractor import extract_statements

# Uses 324 built-in predicates automatically
result = extract_statements("John works for Apple Inc. in Cupertino.")
# Returns ALL matching relations
```

**Mode 2: Custom Predicate List**
```python
from statement_extractor import extract_statements, ExtractionOptions

options = ExtractionOptions(predicates=["works_for", "founded", "acquired", "headquartered_in"])
result = extract_statements("John works for Apple Inc. in Cupertino.", options)
```

Or via CLI:
```bash
corp-extractor "John works for Apple Inc." --predicates "works_for,founded,acquired"
```

**Mode 3: Custom Predicate File**
```python
from statement_extractor.pipeline import ExtractionPipeline, PipelineConfig

config = PipelineConfig(
    extractor_options={"predicates_file": "/path/to/custom_predicates.json"}
)
pipeline = ExtractionPipeline(config)
ctx = pipeline.process("John works for Apple Inc.")
```

Or via CLI:
```bash
corp-extractor pipeline "John works for Apple Inc." --predicates-file custom_predicates.json
```

### How Extraction Works

The pipeline uses a two-stage approach:

1. **Stage 1 (T5-Gemma2)**: Splits text into atomic sentences with coreference resolution
2. **Stage 2 (GLiNER2)**: Extracts subject-predicate-object relations from each sentence

**Key behavior:**
- **All matching relations returned** - Every relation above the confidence threshold (default 0.75) is returned, not just the best one
- **Category-based extraction** - Iterates through 21 predicate categories to stay under GLiNER2's ~25 label limit
- **Confidence from GLiNER2** - Each relation includes the confidence score from GLiNER2's extraction

```python
for stmt in result:
    print(f"{stmt.subject.text} --[{stmt.predicate}]--> {stmt.object.text}")
    print(f"  Confidence: {stmt.confidence_score:.2f}")
    print(f"  Category: {stmt.predicate_category}")  # e.g., "employment_leadership"
```

### Entity Type Inference

For each subject and object, GLiNER2 performs entity extraction on the source sentence to determine the entity type:

| GLiNER2 Type | Mapped EntityType |
|--------------|-------------------|
| person | PERSON |
| organization, company | ORG |
| location, city, country | GPE/LOC |
| product | PRODUCT |
| event | EVENT |
| date | DATE |
| money | MONEY |
| quantity | QUANTITY |
| (unrecognized) | UNKNOWN |

## Disable Embeddings

```python
options = ExtractionOptions(
    embedding_dedup=False,  # Use exact text matching
    merge_beams=False,      # Select single best beam
)
result = extract_statements(text, options)
```

## Output Formats

```python
from statement_extractor import (
    extract_statements,
    extract_statements_as_json,
    extract_statements_as_xml,
    extract_statements_as_dict,
)

# Pydantic models (default)
result = extract_statements(text)

# JSON string
json_output = extract_statements_as_json(text)

# Raw XML (model's native format)
xml_output = extract_statements_as_xml(text)

# Python dictionary
dict_output = extract_statements_as_dict(text)
```

## Batch Processing

```python
from statement_extractor import StatementExtractor

extractor = StatementExtractor(device="cuda")  # or "mps" (Apple Silicon) or "cpu"

texts = ["Text 1...", "Text 2...", "Text 3..."]
for text in texts:
    result = extractor.extract(text)
    print(f"Found {len(result)} statements")
```

## Entity Types

| Type | Description | Example |
|------|-------------|---------|
| `ORG` | Organizations | Apple Inc., United Nations |
| `PERSON` | People | Tim Cook, Elon Musk |
| `GPE` | Geopolitical entities | USA, California, Paris |
| `LOC` | Non-GPE locations | Mount Everest, Pacific Ocean |
| `PRODUCT` | Products | iPhone, Model S |
| `EVENT` | Events | World Cup, CES 2024 |
| `WORK_OF_ART` | Creative works | Mona Lisa, Game of Thrones |
| `LAW` | Legal documents | GDPR, Clean Air Act |
| `DATE` | Dates | 2024, January 15 |
| `MONEY` | Monetary values | $50 million, €100 |
| `PERCENT` | Percentages | 25%, 0.5% |
| `QUANTITY` | Quantities | 500 employees, 1.5 tons |
| `UNKNOWN` | Unrecognized | (fallback) |

## How It Works

The library uses a **5-stage pipeline** architecture:

1. **Stage 1 - Splitting (T5-Gemma2)**: Splits text into atomic sentences using Diverse Beam Search ([Vijayakumar et al., 2016](https://arxiv.org/abs/1610.02424)) with coreference resolution
2. **Stage 2 - Extraction (GLiNER2)**: Extracts subject-predicate-object relations using 324 predicates across 21 categories. Returns ALL relations above confidence threshold (0.75)
3. **Stage 3 - Qualification**: Looks up entities against 9.7M+ organizations and 63M+ people in the embedding database, adding canonical IDs and FQNs
4. **Stage 4 - Labeling**: Adds sentiment, confidence, and relation type labels (classifications pre-computed in Stage 2)
5. **Stage 5 - Taxonomy**: Classifies statements against taxonomies (e.g., ESG topics) using embedding similarity

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 5.0+
- Pydantic 2.0+
- sentence-transformers 2.2+
- GLiNER2 (model downloaded automatically on first use)
- ~2GB VRAM (GPU) or ~4GB RAM (CPU)

## Links

- [Model on HuggingFace](https://huggingface.co/Corp-o-Rate-Community/statement-extractor)
- [Web Demo](https://statement-extractor.corp-o-rate.com)
- [Diverse Beam Search Paper](https://arxiv.org/abs/1610.02424)
- [Corp-o-Rate](https://corp-o-rate.com)

## License

MIT License - see LICENSE file for details.
