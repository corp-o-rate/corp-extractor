# Statement Extractor

A Python library and web demo for extracting relationship information about people and organizations from complex text. Runs entirely on your hardware (RTX 4090+, Apple M1 16GB+) with no external API dependencies.

Uses fine-tuned [T5-Gemma 2](https://blog.google/technology/developers/t5gemma-2/) for statement splitting and coreference resolution (trained on 70,000+ pages), plus [GLiNER2](https://github.com/urchade/GLiNER) for entity extraction. Includes a database of 9.7M+ organizations and 63M+ people with USearch HNSW indexes for fast entity qualification (~100GB disk for all models and data).

## Features

- **Statement Extraction**: Transform unstructured text into structured subject-predicate-object triples
- **5-Stage Pipeline** *(v0.8.0)*: Plugin-based architecture with entity qualification, labeling, and taxonomy classification
- **Entity DB extracted to `corp-entity-db`** *(v0.10.0)*: Organizations, people, roles, and locations now live in the separate [corp-entity-db](https://pypi.org/project/corp-entity-db/) PyPI package. Use the `corp-entity-db` CLI for DB management; `corp-extractor` consumes it for qualification only.
- **Database v3 Schema** *(v0.9.6)*: Lite databases drop all embedding tables — USearch indexes for search. Global `--db-version` flag for backwards compatibility.
- **USearch HNSW Indexes** *(v0.9.5)*: Sub-millisecond search on 50M+ vectors with pre-built HNSW indexes
- **Entity Database** *(v0.9.6)*: 9.7M+ organizations and 63M+ people with USearch HNSW indexes for fast entity qualification
- **EntityType Classification** *(v0.8.0)*: Classify organizations as business, nonprofit, government, educational, etc.
- **Entity Recognition**: Automatic identification of entity types (ORG, PERSON, GPE, EVENT, etc.)
- **Relationship Graph**: Interactive D3.js visualization of entity relationships
- **Coreference Resolution**: Pronouns are resolved to their referenced entities
- **Local Execution**: No external services required—runs entirely on your hardware

## Quick Start

### Online Demo

Visit [extractor.corp-o-rate.com](https://extractor.corp-o-rate.com) to try the demo.

### Run Locally

```bash
# Clone the repository
git clone https://github.com/corp-o-rate/statement-extractor
cd statement-extractor

# Install dependencies
pnpm install

# Start the dev server
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Model Information

- **Architecture**: T5-Gemma 2 (540M parameters)
- **Training Data**: 77,515 examples from corporate and news documents
- **Final Eval Loss**: 0.209
- **Input Format**: Text wrapped in `<page>` tags
- **Output Format**: XML with extracted statements

### HuggingFace Model

The model is available on HuggingFace: [Corp-o-Rate-Community/statement-extractor](https://huggingface.co/Corp-o-Rate-Community/statement-extractor)

## Usage

### Python Library (Recommended)

Install the Python library for easy CLI and API access:

```bash
pip install corp-extractor
```

**CLI Usage:**

```bash
# Simple extraction (fast)
corp-extractor split "Apple Inc. announced a new iPhone."

# Full 5-stage pipeline with entity resolution
corp-extractor pipeline "Apple CEO Tim Cook announced..."
corp-extractor pipeline -f article.txt --stages 1-3

# Process local PDFs and documents
corp-extractor document process report.pdf
corp-extractor document process report.pdf --pdf-parser glm_ocr_parser

# Persistent server mode (keeps models warm for fast repeated use)
corp-extractor serve                                          # Start server on port 8111
corp-extractor --server pipeline "Apple CEO Tim Cook..."      # Delegate to server

# List available plugins
corp-extractor plugins list
```

**Python API:**

```python
from statement_extractor import extract_statements

# Simple extraction
result = extract_statements("Apple Inc. announced a new iPhone.")
for stmt in result:
    print(f"{stmt.subject.text} -> {stmt.predicate} -> {stmt.object.text}")

# Full pipeline (v0.5.0)
from statement_extractor.pipeline import ExtractionPipeline

pipeline = ExtractionPipeline()
ctx = pipeline.process("Apple CEO Tim Cook announced...")
for stmt in ctx.labeled_statements:
    print(f"{stmt.subject_fqn} -> {stmt.statement.predicate} -> {stmt.object_fqn}")

# Delegate to a running server (v0.9.8) — no local GPU needed
result = extract_statements("text", server_url="http://localhost:8111")
pipeline = ExtractionPipeline(server_url="http://localhost:8111")
```

See [statement-extractor-lib/README.md](statement-extractor-lib/README.md) for full pipeline documentation.

### Entity Database (`corp-entity-db` package, v0.10.0+)

As of v0.10.0 the entity database is a separate project — see the
[corp-entity-db project](https://corp-entity-db.vercel.app/) for search,
download, build, and CLI documentation. `corp-extractor` depends on it
for entity qualification; you don't need to touch it directly to use
the extraction pipeline.

See [ENTITY_DATABASE.md](ENTITY_DATABASE.md) for the project-level overview
of how `corp-extractor` consumes the database for qualification.

### Direct Model Access

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model = AutoModelForSeq2SeqLM.from_pretrained(
    "Corp-o-Rate-Community/statement-extractor",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "Corp-o-Rate-Community/statement-extractor",
    trust_remote_code=True,
)

text = "Apple Inc. announced a commitment to carbon neutrality by 2030."
inputs = tokenizer(f"<page>{text}</page>", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=2048, num_beams=4)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

### Output Format

```xml
<statements>
  <stmt>
    <subject type="ORG">Apple Inc.</subject>
    <object type="EVENT">carbon neutrality by 2030</object>
    <predicate>committed to</predicate>
    <text>Apple Inc. committed to achieving carbon neutrality by 2030.</text>
  </stmt>
</statements>
```

### Entity Types

| Type | Description |
|------|-------------|
| ORG | Organizations (companies, agencies) |
| PERSON | People (names, titles) |
| GPE | Geopolitical entities (countries, cities) |
| LOC | Locations (mountains, rivers) |
| PRODUCT | Products (devices, services) |
| EVENT | Events (announcements, meetings) |
| WORK_OF_ART | Creative works (reports, books) |
| LAW | Legal documents |
| DATE | Dates and time periods |
| MONEY | Monetary values |
| PERCENT | Percentages |
| QUANTITY | Quantities and measurements |

## Deployment Options

### Cerebrium Serverless (Recommended for Production)

Production deploys to [Cerebrium](https://www.cerebrium.ai/) into the same
project as the corp-entity-db app, sharing `/persistent-storage` so model
weights and the entity DB are reused across both apps.

```bash
cd cerebrium
cerebrium projects current             # confirm correct project
cerebrium secrets set HF_TOKEN <token> # gated model downloads
cerebrium deploy
```

Endpoints (auth: `Authorization: Bearer <CEREBRIUM_TOKEN>`):

```
POST https://api.aws.us-east-1.cerebrium.ai/v4/<project-id>/statement-extractor/extract
POST https://api.aws.us-east-1.cerebrium.ai/v4/<project-id>/statement-extractor/extract_url
```

Calls run synchronously and return the full payload (no polling). The
frontend API route (`src/app/api/extract/route.ts`) wraps each call with a
one-shot retry on Vercel timeout, and `src/app/page.tsx` fires a
localStorage-gated browser warm-up ping on page load so cold-boot is
absorbed before the user submits a real query. Auto-deploy is wired up
via `.github/workflows/cerebrium-deploy.yml` on pushes that touch
`cerebrium/**`.

See [cerebrium/README.md](cerebrium/README.md) for full deployment notes,
GPU choices, and troubleshooting.

### RunPod Serverless (Legacy)

Superseded by Cerebrium. The container build and handler in
[runpod/](runpod/) are retained for reference; they are not the active
production path. See [runpod/README.md](runpod/README.md).

### Local Server

For unlimited usage without API rate limits, run the model locally using [uv](https://github.com/astral-sh/uv):

```bash
cd local-server
cp .env.example .env  # Edit to set MODEL_PATH
uv sync
uv run python server.py
```

See [local-server/README.md](local-server/README.md) for details.

## Upload Model to HuggingFace

```bash
cd scripts
cp .env.example .env  # Set HF_TOKEN
uv sync
uv run python upload_model.py
```

## Environment Variables

See [`.env.example`](.env.example) for the canonical list. Key variables:

| Variable | Description |
|----------|-------------|
| `CEREBRIUM_EXTRACT_URL` | Cerebrium `/extract` endpoint URL (production) |
| `CEREBRIUM_EXTRACT_URL_URL` | Cerebrium `/extract_url` endpoint URL (production) |
| `CEREBRIUM_TOKEN` | Cerebrium service-account token or per-app inference key |
| `LOCAL_MODEL_URL` | Local server URL for the web demo (e.g., `http://localhost:8000`) |
| `CORP_EXTRACTOR_SERVER` | Corp-extractor persistent server URL (e.g., `http://localhost:8111`) |
| `HF_TOKEN` | HuggingFace token for gated model downloads |

## Tech Stack

- [Next.js](https://nextjs.org/) - React framework
- [Tailwind CSS](https://tailwindcss.com/) - Styling
- [D3.js](https://d3js.org/) - Graph visualization
- [uv](https://github.com/astral-sh/uv) - Python package manager
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) - Model inference
- [Vercel](https://vercel.com/) - Deployment

## About corp-o-rate

Statement Extractor is part of [corp-o-rate.com](https://corp-o-rate.com) - an AI-powered platform for ESG analysis and corporate accountability. Our models extract structured statements from corporate reports, identifying claims, commitments, and impacts.

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or pull request.