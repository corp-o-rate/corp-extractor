"""Cerebrium handler for statement-extractor.

Cerebrium auto-exposes every public top-level function as a POST endpoint, so
`extract(...)` and `extract_url(...)` below become:

  POST https://api.cortex.cerebrium.ai/v4/<project>/statement-extractor/extract
  POST https://api.cortex.cerebrium.ai/v4/<project>/statement-extractor/extract_url

Request body keys map directly to function kwargs. Both endpoints return
Pydantic `model_dump()` shaped JSON (matching the runpod handler's old return
shape) so the frontend rendering logic is unchanged.

Sharing storage with corp-entity-db: this app deploys into the same project
and mounts the same /persistent-storage volume at runtime. We point HF_HOME
and `statement_extractor.database.hub.DEFAULT_CACHE_DIR` at the volume so the
embeddinggemma model + entity DB + USearch indexes that corp-entity-db
already downloaded are reused as-is, and the new T5-Gemma + GGUF + GLiNER2
weights also persist there for warm boots.
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Volume probe + cache redirection. Mirrors corp-entity-db/cerebrium/main.py.
# Must run BEFORE any HF / statement_extractor imports so the cache env vars
# take effect on first model download.
# ---------------------------------------------------------------------------
print("=" * 60)
print("[init] probing /persistent-storage for project volume")
_VOLUME: Optional[Path] = None
for _p in ("/persistent-storage", "/workspace"):
    _P = Path(_p)
    if _P.is_dir():
        _free_gb = shutil.disk_usage(_p).free / 1024 ** 3
        _is_mount = os.path.ismount(_p)
        print(f"[init]   {_p}: exists is_mount={_is_mount} free={_free_gb:.1f} GB")
        if _free_gb >= 40 and _VOLUME is None:
            _VOLUME = _P
            print(f"[init] selected volume: {_p}")
    else:
        print(f"[init]   {_p}: missing")

if _VOLUME is not None:
    _hf_dir = _VOLUME / "hf"
    _hf_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(_hf_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(_hf_dir)
    os.environ["HF_HUB_CACHE"] = str(_hf_dir / "hub")
    os.environ["XDG_CACHE_HOME"] = str(_VOLUME)
    print(f"[init] HF_HOME = {_hf_dir}")

    # IMPORTANT: write to corp_entity_db.hub directly. The
    # statement_extractor.database.hub shim re-exports DEFAULT_CACHE_DIR
    # as a bound name, so setting it on the shim has no effect on the
    # downloader functions that read corp_entity_db.hub.DEFAULT_CACHE_DIR.
    import corp_entity_db.hub as _ce_hub
    _ce_hub.DEFAULT_CACHE_DIR = _VOLUME
    print(f"[init] corp_entity_db.hub.DEFAULT_CACHE_DIR = {_VOLUME}")
    # Keep the shim in sync too so any caller importing through the
    # statement_extractor namespace sees the same value.
    import statement_extractor.database.hub as _hub
    _hub.DEFAULT_CACHE_DIR = _VOLUME
else:
    print("[init] WARNING: no volume with >=40 GB free — first request will fail "
          "or run out of disk. Resize the project volume before sending requests.")

_hf_present = bool(os.environ.get("HF_TOKEN"))
print(f"[init] HF_TOKEN env var present: {_hf_present}")
if not _hf_present:
    print("[init] WARNING: HF_TOKEN not in environment — gated model downloads will fail")
print("=" * 60)

# ---------------------------------------------------------------------------
# Heavy imports AFTER cache redirection.
# ---------------------------------------------------------------------------
from statement_extractor.document import (
    DocumentPipeline,
    DocumentPipelineConfig,
    URLLoaderConfig,
)
from statement_extractor.models.document import ChunkingConfig
from statement_extractor.pipeline import ExtractionPipeline, PipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "Corp-o-Rate-Community/statement-extractor")
MAX_CACHE_SIZE_BYTES = int(os.environ.get("MAX_CACHE_SIZE_BYTES", 10 * 1024 * 1024 * 1024))
DEFAULT_NUM_BEAMS = int(os.environ.get("NUM_BEAMS", 4))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 2048))

# Canonical predicates for corporate/ESG domain (copied verbatim from the
# RunPod handler — keeps prediction surface identical).
CANONICAL_PREDICATES = [
    "owned by", "parent organization", "ultimate parent", "has subsidiary",
    "acquired", "merged with", "spun off from", "controlling shareholder",
    "minority shareholder", "beneficial owner", "person of significant control",
    "nominee for", "succeeded by", "preceded by",
    "investor", "funded by", "creditor", "debtor", "vc backed by", "pe backed by",
    "ipo underwriter", "bond holder",
    "chief executive officer", "chief financial officer", "chief operating officer",
    "founder", "board member", "chairperson", "company secretary", "employee",
    "former employee", "former director", "advisor", "consultant",
    "division of", "department of",
    "supplier", "customer", "manufacturer", "distributor", "contractor",
    "outsources to", "subcontractor", "raw material source",
    "headquarters", "located in", "operates in", "facility in", "registered in",
    "tax residence", "offshore entity in", "branch in", "citizenship", "formed in", "residence",
    "sued", "sued by", "fined by", "regulated by", "licensed by", "sanctioned by",
    "investigated by", "settled with", "consent decree with", "debarred by",
    "lobbies", "donated to", "endorsed by", "member of", "sponsored by",
    "lobbied by", "pac contribution", "revolving door",
    "polluted", "affected community", "displaced", "deforested", "benefited",
    "restored", "employed in", "invested in community", "violated rights",
    "emitted ghg", "water usage", "waste disposal",
    "brand of", "product of", "trademark of", "licensed from", "white label for",
    "recalls", "developer", "publisher",
    "partner", "joint venture with", "franchisee of", "distributor for",
    "licensed to", "exclusive dealer", "operator",
    "spouse", "relative", "associate", "co-founder", "classmate", "club member",
    "industry", "competitor", "similar to", "same sector", "peer of", "instance of",
    "mentioned with", "accused of", "praised for", "criticized for",
    "announced", "rumored", "participant",
]

# ---------------------------------------------------------------------------
# Lazy global state. Init under a lock so concurrent first-requests don't
# each kick off the same multi-GB model download.
# ---------------------------------------------------------------------------
_pipeline: Optional[ExtractionPipeline] = None
_doc_pipeline: Optional[DocumentPipeline] = None
_inference_lock = threading.Lock()  # serialise GPU access (replica_concurrency=1, but be defensive)
_init_lock = threading.Lock()
_initialized = False

# In-memory LRU for hot prompts. Bounded by MAX_CACHE_SIZE_BYTES.
_result_cache: "OrderedDict[str, str]" = OrderedDict()
_cache_size_bytes = 0


def _cache_key(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def _evict_if_needed(new_entry_size: int) -> None:
    global _cache_size_bytes
    while _result_cache and (_cache_size_bytes + new_entry_size) > MAX_CACHE_SIZE_BYTES:
        oldest_key, oldest_value = _result_cache.popitem(last=False)
        _cache_size_bytes -= len(oldest_key.encode()) + len(oldest_value.encode())


def _cache_put(key: str, payload: str) -> None:
    global _cache_size_bytes
    size = len(key.encode()) + len(payload.encode())
    _evict_if_needed(size)
    _result_cache[key] = payload
    _cache_size_bytes += size


def _initialize() -> None:
    """Lazy: runs on the first call to extract() or extract_url().

    Done lazily (rather than at import time) so the replica boots even if
    something is wrong with the volume — operator can fix without a crash
    loop. Lock prevents concurrent first-requests from doubling up the
    multi-GB downloads.
    """
    global _pipeline, _doc_pipeline, _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return

        # Build the pipelines — the underlying T5-Gemma + GLiNER2 + entity DB
        # are loaded lazily by their respective plugins on first request, so
        # cold-start cost is shifted to that first call (vs. eager loading
        # here, which would double-load if we kept a separate StatementExtractor).
        t0 = time.time()
        logger.info("Initialising text pipeline (Stages 1-5)...")
        pipeline_config = PipelineConfig(
            disabled_plugins={"mnli_taxonomy_classifier"},  # use embedding classifier (faster)
        )
        _pipeline = ExtractionPipeline(pipeline_config)
        logger.info(f"Text pipeline ready in {time.time() - t0:.1f}s")

        t1 = time.time()
        logger.info("Initialising document pipeline...")
        doc_config = DocumentPipelineConfig(
            chunking=ChunkingConfig(target_tokens=1000, overlap_tokens=100),
            generate_summary=True,
            deduplicate_across_chunks=True,
            pipeline_config=pipeline_config,
        )
        _doc_pipeline = DocumentPipeline(doc_config)
        logger.info(f"Document pipeline ready in {time.time() - t1:.1f}s")

        _initialized = True


def _extract_sync(text: str, use_canonical_predicates: bool, similarity_threshold: float) -> dict:
    """Run the full 5-stage pipeline so the response includes qualified entities,
    canonical IDs, labels (sentiment/relation_type), and taxonomy classifications."""
    assert _pipeline is not None
    metadata: dict = {
        "splitter_options": {
            "num_beams": DEFAULT_NUM_BEAMS,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        },
        "extractor_options": {
            "similarity_threshold": similarity_threshold,
        },
    }
    if use_canonical_predicates:
        metadata["extractor_options"]["canonical_predicates"] = CANONICAL_PREDICATES

    ctx = _pipeline.process(text, metadata=metadata)
    # PipelineContext.model_dump() emits labeled_statements with
    # subject_canonical / object_canonical / labels / taxonomy_results — the
    # shape the frontend parser expects (parseModelDumpLabeledStatements).
    # Exclude ctx-level scratchpad dicts: `taxonomy_results` has tuple keys
    # (unserialisable to JSON) and `classification_results` is internal-only.
    return ctx.model_dump(
        mode="json",
        exclude={"taxonomy_results", "classification_results"},
    )


def _process_url_sync(url: str, use_ocr: bool, max_tokens: int, overlap_tokens: int) -> dict:
    # Override chunking config per-request so frontend params take effect.
    assert _doc_pipeline is not None
    _doc_pipeline.config.chunking = ChunkingConfig(
        target_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    loader_config = URLLoaderConfig(use_ocr=use_ocr)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        ctx = loop.run_until_complete(_doc_pipeline.process_url(url, loader_config))
    finally:
        loop.close()

    # Exclude chunk_contexts (PipelineContexts contain non-serializable fields).
    return ctx.model_dump(mode="json", exclude={"chunk_contexts"})


# ---------------------------------------------------------------------------
# Public endpoints — Cerebrium auto-exposes these as POST routes.
# ---------------------------------------------------------------------------


def extract(
    text: str = "",
    useCanonicalPredicates: bool = False,
    similarityThreshold: float = 0.5,
) -> dict:
    """Text extraction endpoint.

    POST body kwargs:
      text: str                          — required; will be wrapped in <page>
                                            tags by the frontend before sending
      useCanonicalPredicates: bool=False — match against CANONICAL_PREDICATES
      similarityThreshold: float=0.5     — predicate match threshold (0-1)
    """
    _initialize()

    text = (text or "").strip()
    if not text:
        return {"error": "No text provided. Send {\"text\": \"...\"}"}

    use_canonical = bool(useCanonicalPredicates)
    threshold = max(0.0, min(1.0, float(similarityThreshold)))

    cache_key = _cache_key("extract", use_canonical, round(threshold, 3), text)
    if cache_key in _result_cache:
        _result_cache.move_to_end(cache_key)
        cached = json.loads(_result_cache[cache_key])
        cached["cached"] = True
        return cached

    logger.info(f"extract: len={len(text)} canonical={use_canonical} thresh={threshold}")
    t0 = time.time()
    with _inference_lock:
        result = _extract_sync(text, use_canonical, threshold)
    elapsed = time.time() - t0
    logger.info(f"extract complete in {elapsed:.2f}s")

    payload = json.dumps(result)
    _cache_put(cache_key, payload)
    return result


def extract_url(
    url: str = "",
    useOcr: bool = False,
    maxTokens: int = 1000,
    overlapTokens: int = 100,
) -> dict:
    """URL processing endpoint.

    POST body kwargs:
      url: str               — required; http(s) URL to fetch + parse
      useOcr: bool=False     — run OCR on PDFs (else PyMuPDF text extraction)
      maxTokens: int=1000    — chunking target tokens per chunk
      overlapTokens: int=100 — chunk overlap
    """
    _initialize()

    url = (url or "").strip()
    if not url:
        return {"error": "No url provided. Send {\"url\": \"...\"}"}

    use_ocr = bool(useOcr)
    max_tokens = int(maxTokens)
    overlap_tokens = int(overlapTokens)

    cache_key = _cache_key("extract_url", use_ocr, max_tokens, overlap_tokens, url)
    if cache_key in _result_cache:
        _result_cache.move_to_end(cache_key)
        cached = json.loads(_result_cache[cache_key])
        cached["cached"] = True
        return cached

    logger.info(f"extract_url: url={url} ocr={use_ocr} max_tokens={max_tokens}")
    t0 = time.time()
    with _inference_lock:
        result = _process_url_sync(url, use_ocr, max_tokens, overlap_tokens)
    elapsed = time.time() - t0
    logger.info(f"extract_url complete in {elapsed:.2f}s, statements={len(result.get('statements', []))}")

    payload = json.dumps(result)
    _cache_put(cache_key, payload)
    return result
