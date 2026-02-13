# Statement Extractor Local Server

A FastAPI server for running the T5-Gemma 2 statement extraction model locally.

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- 8GB+ RAM (16GB recommended)
- CUDA GPU recommended for fast inference

## Installation

1. Install uv (if not already installed):

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

2. Install dependencies:

```bash
cd local-server
uv sync
```

3. Download the model from HuggingFace:

```bash
# Using huggingface-cli
uv run huggingface-cli download Corp-o-Rate-Community/statement-extractor --local-dir ../model

# Or using git-lfs
git lfs install
git clone https://huggingface.co/Corp-o-Rate-Community/statement-extractor ../model
```

## Configuration

Copy the example environment file and customize:

```bash
cp .env.example .env
```

Edit `.env`:
```bash
MODEL_PATH=/path/to/model
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=INFO
```

## Usage

### Using .env file (recommended)

```bash
uv run python server.py
```

### Using CLI arguments

```bash
uv run python server.py --model-path ../model --port 8000
```

### With custom .env file

```bash
uv run python server.py --env-file /path/to/.env
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/
```

Response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "model_path": "/path/to/model"
}
```

### Extract Statements

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple Inc. announced a commitment to carbon neutrality by 2030."}'
```

Response (Pydantic `ExtractionResult.model_dump()` format):
```json
{
  "statements": [
    {
      "subject": {"text": "Apple Inc.", "type": "ORG"},
      "predicate": "committed to",
      "object": {"text": "carbon neutrality by 2030", "type": "EVENT"},
      "source_text": "Apple Inc. announced a commitment to carbon neutrality by 2030.",
      "confidence_score": 0.85
    }
  ],
  "source_text": "<page>Apple Inc. announced a commitment to carbon neutrality by 2030.</page>",
  "cached": false
}
```

## Hardware Requirements

| Configuration | RAM | Inference Time |
|--------------|-----|----------------|
| CPU only | 8GB+ | ~30s |
| CUDA GPU | 16GB+ | ~2s |
| Apple MPS | 16GB+ | ~5s |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `~/corp_var/models/page_splitter` | Path to the model directory |
| `PORT` | `8000` | Port to run the server on |
| `HOST` | `0.0.0.0` | Host to bind to |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Integration with Frontend

Set the `LOCAL_MODEL_URL` environment variable in the frontend:

```bash
echo "LOCAL_MODEL_URL=http://localhost:8000" >> ../.env.local
```

Then restart the frontend dev server:

```bash
cd ..
pnpm dev
```
