"""
RunPod Serverless Handler for Statement Extractor

This handler uses the corp-extractor library to extract structured statements from text.
Supports both direct text input and URL processing.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from typing import Optional

import runpod
from statement_extractor import (
    ExtractionOptions,
    PredicateComparisonConfig,
    PredicateTaxonomy,
    ScoringConfig,
    StatementExtractor,
)
from statement_extractor.document import (
    DocumentPipeline,
    DocumentPipelineConfig,
    URLLoaderConfig,
)
from statement_extractor.models.document import ChunkingConfig
from statement_extractor.pipeline import PipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = os.environ.get("MODEL_ID", "Corp-o-Rate-Community/statement-extractor")

# Cache configuration (default 10GB)
MAX_CACHE_SIZE_BYTES = int(os.environ.get("MAX_CACHE_SIZE_BYTES", 10 * 1024 * 1024 * 1024))

# Extraction options from environment
DEFAULT_NUM_BEAMS = int(os.environ.get("NUM_BEAMS", 4))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 2048))

# Concurrency settings
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", 4))

# Canonical predicates for corporate/ESG domain
CANONICAL_PREDICATES = [
    # Ownership & Corporate Structure
    "owned by", "parent organization", "ultimate parent", "has subsidiary",
    "acquired", "merged with", "spun off from", "controlling shareholder",
    "minority shareholder", "beneficial owner", "person of significant control",
    "nominee for", "succeeded by", "preceded by",
    # Investment & Finance
    "investor", "funded by", "creditor", "debtor", "vc backed by", "pe backed by",
    "ipo underwriter", "bond holder",
    # Leadership & Employment
    "chief executive officer", "chief financial officer", "chief operating officer",
    "founder", "board member", "chairperson", "company secretary", "employee",
    "former employee", "former director", "advisor", "consultant",
    # Organizational Structure
    "division of", "department of",
    # Supply Chain
    "supplier", "customer", "manufacturer", "distributor", "contractor",
    "outsources to", "subcontractor", "raw material source",
    # Geography & Jurisdiction
    "headquarters", "located in", "operates in", "facility in", "registered in",
    "tax residence", "offshore entity in", "branch in", "citizenship", "formed in", "residence",
    # Legal & Regulatory
    "sued", "sued by", "fined by", "regulated by", "licensed by", "sanctioned by",
    "investigated by", "settled with", "consent decree with", "debarred by",
    # Political
    "lobbies", "donated to", "endorsed by", "member of", "sponsored by",
    "lobbied by", "pac contribution", "revolving door",
    # Environmental & Social
    "polluted", "affected community", "displaced", "deforested", "benefited",
    "restored", "employed in", "invested in community", "violated rights",
    "emitted ghg", "water usage", "waste disposal",
    # Products & IP
    "brand of", "product of", "trademark of", "licensed from", "white label for",
    "recalls", "developer", "publisher",
    # Business Relationships
    "partner", "joint venture with", "franchisee of", "distributor for",
    "licensed to", "exclusive dealer", "operator",
    # Personal Relationships
    "spouse", "relative", "associate", "co-founder", "classmate", "club member",
    # Classification
    "industry", "competitor", "similar to", "same sector", "peer of", "instance of",
    # Events & Mentions
    "mentioned with", "accused of", "praised for", "criticized for",
    "announced", "rumored", "participant",
]

# Global state
extractor: Optional[StatementExtractor] = None
document_pipeline: Optional[DocumentPipeline] = None
result_cache: OrderedDict[str, str] = OrderedDict()
cache_size_bytes = 0
inference_lock: Optional[asyncio.Lock] = None
_initialized = False


def get_cache_key(text: str, output_format: str, use_canonical: bool, threshold: float) -> str:
    """Generate a cache key from input parameters."""
    key_str = f"{output_format}:canonical={use_canonical}:thresh={threshold}:{text}"
    return hashlib.sha256(key_str.encode()).hexdigest()


def get_entry_size(key: str, value: str) -> int:
    """Estimate memory size of a cache entry in bytes."""
    return len(key.encode()) + len(value.encode())


def evict_if_needed(new_entry_size: int):
    """Evict oldest entries if cache would exceed size limit."""
    global cache_size_bytes

    while result_cache and (cache_size_bytes + new_entry_size) > MAX_CACHE_SIZE_BYTES:
        oldest_key, oldest_value = result_cache.popitem(last=False)
        evicted_size = get_entry_size(oldest_key, oldest_value)
        cache_size_bytes -= evicted_size
        logger.info(f"Evicted cache entry: {oldest_key[:16]}... (freed {evicted_size} bytes)")


def cache_result(cache_key: str, result: str):
    """Store result in cache."""
    global cache_size_bytes
    entry_size = get_entry_size(cache_key, result)
    evict_if_needed(entry_size)
    result_cache[cache_key] = result
    cache_size_bytes += entry_size


def load_extractor():
    """Load the extractor if not already loaded."""
    global extractor

    if extractor is not None:
        return

    logger.info(f"Loading extractor with model: {MODEL_ID}")
    extractor = StatementExtractor(model_id=MODEL_ID)
    logger.info(f"Extractor loaded on device: {extractor.device}")


def extract_sync(
    text: str,
    use_canonical_predicates: bool,
    similarity_threshold: float,
) -> dict:
    """
    Run extraction using the corp-extractor library.

    Returns ExtractionResult as a dict (model_dump).
    """
    global extractor

    load_extractor()

    # Build extraction options
    predicate_config = PredicateComparisonConfig(
        similarity_threshold=similarity_threshold,
    )

    options = ExtractionOptions(
        num_beams=DEFAULT_NUM_BEAMS,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        merge_beams=True,
        embedding_dedup=True,
        use_gliner_extraction=True,
        scoring_config=ScoringConfig(),
        predicate_config=predicate_config,
    )

    if use_canonical_predicates:
        options.predicate_taxonomy = PredicateTaxonomy(predicates=CANONICAL_PREDICATES)

    result = extractor.extract(text, options)
    return result.model_dump()


def process_url_sync(
    url: str,
    use_ocr: bool = False,
    max_tokens: int = 1000,
    overlap_tokens: int = 100,
) -> dict:
    """
    Process a URL using the document pipeline.

    Returns a dict with statements and metadata.
    """
    global document_pipeline

    # Initialize document pipeline if needed
    if document_pipeline is None:
        logger.info("Initializing document pipeline...")
        pipeline_config = PipelineConfig(
            # Disable MNLI classifier for speed, use embedding classifier
            disabled_plugins={"mnli_taxonomy_classifier"},
        )
        doc_config = DocumentPipelineConfig(
            chunking=ChunkingConfig(
                target_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            ),
            generate_summary=True,
            deduplicate_across_chunks=True,
            pipeline_config=pipeline_config,
        )
        document_pipeline = DocumentPipeline(doc_config)
        logger.info("Document pipeline initialized")

    # Process the URL
    logger.info(f"Processing URL: {url}")
    loader_config = URLLoaderConfig(use_ocr=use_ocr)

    # Use asyncio to run the async method
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        ctx = loop.run_until_complete(
            document_pipeline.process_url(url, loader_config)
        )
    finally:
        loop.close()

    # Return standardized DocumentContext model_dump format
    # Exclude chunk_contexts (contains PipelineContexts with non-serializable fields)
    return ctx.model_dump(mode='json', exclude={'chunk_contexts'})


async def handler(job):
    """
    Async RunPod serverless handler function.

    Expected input format for text:
    {
        "input": {
            "text": "Your text here",
            "format": "json"  // optional: "json" (default) or "xml"
        }
    }

    Expected input format for URL:
    {
        "input": {
            "url": "https://example.com/article",
            "useOcr": false,  // optional: use OCR for PDF
            "maxTokens": 1000,  // optional: tokens per chunk
            "overlapTokens": 100  // optional: overlap between chunks
        }
    }
    """
    global inference_lock, _initialized

    # Initialize lock on first request
    if not _initialized:
        inference_lock = asyncio.Lock()
        _initialized = True

    logger.info(f"Received job: {job.get('id', 'unknown')}")

    job_input = job.get("input", {})

    # Check if this is a URL request
    url = job_input.get("url")
    if url:
        return await handle_url_request(job_input, url)

    # Get text from input
    text = job_input.get("text") or job_input.get("prompt", "")

    if not text:
        logger.error("No text or URL provided")
        return {"error": "No text or URL provided. Send {\"input\": {\"text\": \"...\"}} or {\"input\": {\"url\": \"...\"}}"}

    # Get options
    use_canonical_predicates = job_input.get("useCanonicalPredicates", False)
    similarity_threshold = float(job_input.get("similarityThreshold", 0.5))
    similarity_threshold = max(0.0, min(1.0, similarity_threshold))

    logger.info(f"Processing text of length: {len(text)}")

    # Check cache
    cache_key = get_cache_key(text, "json", use_canonical_predicates, similarity_threshold)
    if cache_key in result_cache:
        result_cache.move_to_end(cache_key)
        logger.info(f"Cache hit: {cache_key[:16]}...")
        cached_result = json.loads(result_cache[cache_key])
        cached_result["cached"] = True
        return cached_result

    try:
        start_time = time.time()

        # Run extraction with lock to prevent concurrent GPU access
        async with inference_lock:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                extract_sync,
                text,
                use_canonical_predicates,
                similarity_threshold,
            )

        elapsed = time.time() - start_time
        logger.info(f"Extraction complete in {elapsed:.2f}s")

        # Cache the result (as JSON string)
        cache_result(cache_key, json.dumps(result))

        return result

    except Exception as e:
        logger.exception("Error during extraction")
        return {"error": str(e)}


async def handle_url_request(job_input: dict, url: str):
    """Handle URL processing request."""
    global inference_lock

    # Get options
    use_ocr = job_input.get("useOcr", False)
    max_tokens = int(job_input.get("maxTokens", 1000))
    overlap_tokens = int(job_input.get("overlapTokens", 100))

    logger.info(f"Processing URL: {url}, OCR: {use_ocr}, maxTokens: {max_tokens}")

    # Check cache
    cache_key = get_cache_key(url, "url", use_ocr, max_tokens / 10000)
    if cache_key in result_cache:
        result_cache.move_to_end(cache_key)
        logger.info(f"Cache hit for URL: {cache_key[:16]}...")
        result = json.loads(result_cache[cache_key])
        result["cached"] = True
        return result

    try:
        start_time = time.time()

        # Run URL processing with lock
        async with inference_lock:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                process_url_sync,
                url,
                use_ocr,
                max_tokens,
                overlap_tokens,
            )

        elapsed = time.time() - start_time
        logger.info(f"URL processing complete in {elapsed:.2f}s, statements: {len(result.get('statements', []))}")

        # Cache the result
        cache_result(cache_key, json.dumps(result))

        return result

    except Exception as e:
        logger.exception(f"Error processing URL: {url}")
        return {"error": str(e)}


def concurrency_modifier(current_concurrency: int) -> int:
    """Return maximum concurrent jobs this worker can handle."""
    return MAX_CONCURRENCY


# Load extractor at startup for faster cold starts
load_extractor()

# Start the serverless handler
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": concurrency_modifier,
})
