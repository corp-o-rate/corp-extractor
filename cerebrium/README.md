# Cerebrium deployment

Replaces `runpod/`. Deploys into the **same Cerebrium project** as `corp-entity-db`
so both apps share `/persistent-storage` (one volume per project per region).
The volume already holds the entity DB + USearch indexes + embeddinggemma-300m
weights; this app adds T5-Gemma2, Gemma-3-12B GGUF, and GLiNER2 to the same cache.

## One-time setup

```bash
# Confirm your active project matches corp-entity-db's project id:
cerebrium projects current

# (If wrong) switch:
cerebrium projects set <corp-entity-db-project-id>

# Confirm the volume already has the entity-db artifacts:
cerebrium ls /persistent-storage/

# Set the HF_TOKEN secret (gated google/embeddinggemma-300m + gemma-3-12b
# GGUF + the T5-Gemma2 model in Corp-o-Rate-Community/statement-extractor):
cerebrium secrets set HF_TOKEN <your-token>
```

The corp-entity-db volume was resized to ~200 GB. T5-Gemma2 (~20 GB) +
Gemma-3-12B-q4 (~7 GB) + GLiNER2 (~1 GB) push the working set to ~230 GB.
Bump the volume before first deploy if it has less than ~40 GB free — see
the cerebrium skill for the resize REST API.

## Deploy

```bash
cd cerebrium
cerebrium deploy
cerebrium logs statement-extractor --follow
```

Endpoints after deploy (hostname uses `api.<provider>.<region>.cerebrium.ai`
— check the deploy log; for the current project that's
`api.aws.us-east-1.cerebrium.ai`):

- `POST https://api.aws.us-east-1.cerebrium.ai/v4/<project-id>/statement-extractor/extract`
- `POST https://api.aws.us-east-1.cerebrium.ai/v4/<project-id>/statement-extractor/extract_url`

For the current corp-o-rate project the project id is `p-7f30f35c`. The
GitHub Action at `.github/workflows/cerebrium-deploy.yml` auto-deploys
on pushes to `main` that touch `cerebrium/**`.

Auth: `Authorization: Bearer <CEREBRIUM_TOKEN>`.

## First request (cold)

The first request after a fresh deploy triggers `_initialize()` in `main.py`,
which loads T5-Gemma2 and the GLiNER2/embedding/qualifier stacks. If those
weights aren't already on the volume, expect ~5–15 minutes the first time
while they download into `/persistent-storage/hf/`. Subsequent boots are warm.

```bash
curl -X POST -H "Authorization: Bearer $CEREBRIUM_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"Apple announced a new iPhone."}' \
  https://api.aws.us-east-1.cerebrium.ai/v4/<project-id>/statement-extractor/extract
```

Cerebrium returns a `{run_id, result, run_time_ms}` envelope; the handler's
payload is in `.result`.

## Hardware

- Currently `ADA_L40` (48 GB) — largest GPU available on the Cerebrium
  hobby plan. T5-Gemma2 in bf16 is ~20 GB, GLiNER2 + embedders ~3 GB,
  with comfortable headroom. The Gemma-3-12B GGUF qualifier runs CPU-only
  via llama-cpp-python and doesn't share VRAM.
- `AMPERE_A100_40GB` / `AMPERE_A100_80GB` are available on paid plans if
  more headroom is needed.
- `ADA_L4` (24 GB) is the cheapest hobby option that still fits T5-Gemma2.

## Troubleshooting

- `replica_concurrency = 1`: T5-Gemma2 generation isn't safe for concurrent
  GPU access on a single device. Don't raise this without auditing the
  inference path.
- If the first request times out from the frontend (Vercel `maxDuration=300`),
  the API route automatically retries once — the second attempt usually
  hits a now-warm replica. The browser also fires a localStorage-gated
  warm-up ping on page load (TTL 1h) to spin the replica up before the user
  submits a real query.
- To inspect what the volume contains:
  `cerebrium ls /persistent-storage/hf/hub/`
