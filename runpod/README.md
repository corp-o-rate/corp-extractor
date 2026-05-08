# RunPod Serverless Deployment (legacy)

> **Superseded by `cerebrium/`**. Production now runs on Cerebrium so the
> deployment shares `/persistent-storage` with the corp-entity-db app
> (entity DB + USearch indexes + embeddinggemma weights). This directory is
> kept for reference and is no longer the active production path. See
> `cerebrium/README.md`.

Deploy the Statement Extractor model to RunPod Serverless for production use.

## Prerequisites

1. [RunPod account](https://runpod.io?ref=sjoylkgj)
2. [Docker](https://docker.com) installed locally
3. [Docker Hub](https://hub.docker.com) account (or another container registry)

> **Note**: The model uses the T5Gemma2 architecture which requires the development version of `transformers` from GitHub. The Dockerfile handles this automatically.

## Deployment Steps

### 1. Build the Docker Image

```bash
cd runpod

# Build the image (use --platform flag on Mac)
docker build --platform linux/amd64 -t statement-extractor-runpod .

# Tag for your registry
docker tag statement-extractor-runpod:latest YOUR_DOCKERHUB_USERNAME/statement-extractor-runpod:latest

# Push to registry
docker push YOUR_DOCKERHUB_USERNAME/statement-extractor-runpod:latest
```

> **Note for Mac users**: The `--platform linux/amd64` flag is required because RunPod runs on x86_64 Linux servers.

### 2. Create RunPod Serverless Endpoint

1. Go to [RunPod Console](https://www.runpod.io/console/serverless?ref=sjoylkgj)
2. Click **"New Endpoint"**
3. Configure:
   - **Name**: `statement-extractor`
   - **Container Image**: `YOUR_DOCKERHUB_USERNAME/statement-extractor-runpod:latest`
   - **GPU**: Select a GPU (RTX 3090 or better recommended)
   - **Active Workers**: 0 (scales to 0 when idle)
   - **Max Workers**: 1-3 (depending on expected load)
   - **Idle Timeout**: 5 seconds
   - **Execution Timeout**: 60 seconds

4. Click **"Create Endpoint"**

### 3. Get Your Credentials

After creating the endpoint:
- **Endpoint ID**: Shown in the endpoint details (e.g., `abc123xyz`)
- **API Key**: Go to Settings → API Keys → Create new key

### 4. Configure Vercel

Add these environment variables to your Vercel project:

```
RUNPOD_ENDPOINT_ID=your_endpoint_id
RUNPOD_API_KEY=your_api_key
```

Go to: Vercel Dashboard → Your Project → Settings → Environment Variables

### 5. Test the Endpoint

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "<page>Apple Inc. announced a commitment to carbon neutrality by 2030.</page>"}}'
```

Expected response (Pydantic `model_dump()` format):
```json
{
  "id": "xxx",
  "status": "COMPLETED",
  "output": {
    "statements": [
      {
        "subject": {"text": "Apple Inc.", "type": "ORG"},
        "predicate": "committed to",
        "object": {"text": "carbon neutrality by 2030", "type": "EVENT"},
        "source_text": "Apple Inc. announced a commitment to carbon neutrality by 2030.",
        "confidence_score": 0.85
      }
    ],
    "source_text": "<page>Apple Inc. announced a commitment to carbon neutrality by 2030.</page>"
  }
}
```

## Pricing

RunPod Serverless pricing (approximate):
- **RTX 3090**: ~$0.00031/sec (~$1.12/hr when active)
- **RTX 4090**: ~$0.00044/sec (~$1.58/hr when active)
- **A100 40GB**: ~$0.00081/sec (~$2.92/hr when active)

You only pay when the model is processing requests. With 0 active workers, idle cost is $0.

## Estimated Costs

| Usage | RTX 3090 Cost |
|-------|---------------|
| 100 requests/day (~2s each) | ~$0.19/month |
| 1,000 requests/day | ~$1.86/month |
| 10,000 requests/day | ~$18.60/month |

## Troubleshooting

### Cold Starts

First request after idle may take 30-60 seconds (loading model into GPU memory). Subsequent requests are ~2 seconds.

To reduce cold starts:
- Set **Active Workers** to 1 (costs ~$27/month for RTX 3090)
- Or accept cold starts for low-traffic sites

### Out of Memory

If you see OOM errors:
- Use a GPU with more VRAM (A100 40GB recommended for reliability)
- Or reduce `max_new_tokens` in the handler

### Logs

View logs in RunPod Console → Endpoints → Your Endpoint → Logs

## Local Testing

Test the handler locally before deploying:

```bash
cd runpod

# Install dependencies
pip install runpod transformers torch

# Run locally (requires GPU)
python handler.py
```

Then test with:
```bash
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "<page>Test text here.</page>"}}'
```
