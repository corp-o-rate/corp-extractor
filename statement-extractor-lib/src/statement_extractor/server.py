"""
Persistent local server for statement extraction.

Keeps T5-Gemma, GLiNER2, embedding models, and USearch indexes warm in memory
so repeated CLI invocations avoid the ~30s startup cost.

Usage:
    corp-extractor serve                    # Start on localhost:8111
    corp-extractor serve --port 9000        # Custom port
    corp-extractor serve --no-warmup        # Skip eager model loading
"""

import logging
import time
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class PipelineRequest(BaseModel):
    text: str
    config: dict[str, Any] = Field(default_factory=dict)


class SplitRequest(BaseModel):
    text: str
    options: dict[str, Any] = Field(default_factory=dict)


class DocumentRequest(BaseModel):
    text: str
    title: Optional[str] = None
    stages: str = "1-6"
    max_tokens: int = 1000
    overlap: int = 100
    no_summary: bool = False
    no_dedup: bool = False


# ---------------------------------------------------------------------------
# Globals populated at startup
# ---------------------------------------------------------------------------

_pipeline = None
_extractor = None
_plugins_loaded = False


def _ensure_plugins():
    global _plugins_loaded
    if _plugins_loaded:
        return
    try:
        from .plugins import splitters, extractors, qualifiers, labelers, taxonomy
        _ = splitters, extractors, qualifiers, labelers, taxonomy
    except ImportError as e:
        logger.debug(f"Some plugins failed to load: {e}")
    _plugins_loaded = True


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _ensure_plugins()
        from .pipeline import ExtractionPipeline
        _pipeline = ExtractionPipeline()
    return _pipeline


def _get_extractor():
    global _extractor
    if _extractor is None:
        from .extractor import StatementExtractor
        _extractor = StatementExtractor()
    return _extractor


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Corp Extractor Server",
    description="Persistent local server for statement extraction with warm models.",
)


@app.get("/")
def health():
    """Health check and status info."""
    import torch

    device = str(_extractor.device) if _extractor is not None else "not loaded"

    _ensure_plugins()
    from .pipeline.registry import PluginRegistry
    plugins = {
        "splitters": [p.name for p in PluginRegistry.get_splitters()],
        "extractors": [p.name for p in PluginRegistry.get_extractors()],
        "qualifiers": [p.name for p in PluginRegistry.get_qualifiers()],
        "labelers": [p.name for p in PluginRegistry.get_labelers()],
        "taxonomy": [p.name for p in PluginRegistry.get_taxonomy_classifiers()],
    }

    return {
        "status": "ok",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "models_loaded": {
            "extractor": _extractor is not None,
            "pipeline": _pipeline is not None,
        },
        "plugins": plugins,
    }


@app.post("/pipeline")
def pipeline_endpoint(req: PipelineRequest):
    """Run the full extraction pipeline."""
    from .pipeline import PipelineConfig

    config_dict = req.config
    config_kwargs: dict[str, Any] = {}

    # Parse enabled_stages
    if "enabled_stages" in config_dict:
        stages = config_dict["enabled_stages"]
        if isinstance(stages, str):
            config_kwargs["enabled_stages"] = _parse_stages(stages)
        elif isinstance(stages, list):
            config_kwargs["enabled_stages"] = set(stages)
        else:
            config_kwargs["enabled_stages"] = stages

    # Parse disabled_plugins
    if "disabled_plugins" in config_dict:
        dp = config_dict["disabled_plugins"]
        config_kwargs["disabled_plugins"] = set(dp) if isinstance(dp, list) else dp

    # Parse enabled_plugins
    if "enabled_plugins" in config_dict:
        ep = config_dict["enabled_plugins"]
        config_kwargs["enabled_plugins"] = set(ep) if isinstance(ep, list) else ep

    # Pass through option dicts
    for key in ("extractor_options", "splitter_options", "qualifier_options",
                "labeler_options", "taxonomy_options"):
        if key in config_dict:
            config_kwargs[key] = config_dict[key]

    config = PipelineConfig(**config_kwargs)

    _ensure_plugins()
    from .pipeline import ExtractionPipeline
    pipeline = ExtractionPipeline(config)
    ctx = pipeline.process(req.text)

    # Exclude non-JSON-serializable fields:
    # - taxonomy_results: dict with tuple keys (not JSON-compatible)
    # - classification_results: internal processing data with tuple values
    return ctx.model_dump(mode='json', exclude={'taxonomy_results', 'classification_results'})


@app.post("/split")
def split_endpoint(req: SplitRequest):
    """Run Stage 1 only (T5-Gemma extraction)."""
    from .models import ExtractionOptions

    opts_dict = req.options
    # Filter to fields ExtractionOptions actually accepts
    valid_fields = set(ExtractionOptions.model_fields.keys())
    filtered = {k: v for k, v in opts_dict.items() if k in valid_fields}
    options = ExtractionOptions(**filtered)

    extractor = _get_extractor()
    result = extractor.extract(req.text, options)
    return result.model_dump()


@app.post("/document")
def document_endpoint(req: DocumentRequest):
    """Run the document pipeline."""
    _ensure_plugins()
    from .document import DocumentPipeline, DocumentPipelineConfig, Document
    from .models.document import ChunkingConfig
    from .pipeline import PipelineConfig

    enabled_stages = _parse_stages(req.stages)

    chunking_config = ChunkingConfig(
        target_tokens=req.max_tokens,
        max_tokens=req.max_tokens * 2,
        overlap_tokens=req.overlap,
    )

    pipeline_config = PipelineConfig(enabled_stages=enabled_stages)

    doc_config = DocumentPipelineConfig(
        chunking=chunking_config,
        generate_summary=not req.no_summary,
        deduplicate_across_chunks=not req.no_dedup,
        pipeline_config=pipeline_config,
    )

    pipeline = DocumentPipeline(doc_config)
    document = Document.from_text(
        text=req.text,
        title=req.title or "Untitled",
        source_type="text",
    )
    ctx = pipeline.process(document)

    # Exclude chunk_contexts (contains PipelineContexts with non-serializable tuple-keyed fields)
    return ctx.model_dump(mode='json', exclude={'chunk_contexts'})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_stages(stages_str: str) -> set[int]:
    """Parse stage string like '1,2,3' or '1-3' into a set of ints."""
    result = set()
    for part in stages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            for i in range(int(start), int(end) + 1):
                result.add(i)
        else:
            result.add(int(part))
    return result


def warmup():
    """Eagerly load all models by running a tiny text through each path."""
    logger.info("Warming up models...")
    t0 = time.time()

    # Warm up the extractor (loads T5-Gemma + GLiNER2)
    t1 = time.time()
    extractor = _get_extractor()
    logger.info(f"  Extractor created on {extractor.device} ({time.time() - t1:.1f}s)")

    t1 = time.time()
    extractor.extract("Test warmup sentence.")
    logger.info(f"  Extractor warmup extraction done ({time.time() - t1:.1f}s)")

    # Warm up the pipeline (loads embedding models + USearch indexes)
    t1 = time.time()
    pipeline = _get_pipeline()
    pipeline.process("Test warmup sentence.")
    logger.info(f"  Pipeline warmup done ({time.time() - t1:.1f}s)")

    logger.info(f"Warmup complete in {time.time() - t0:.1f}s")


def run_server(host: str = "0.0.0.0", port: int = 8111, do_warmup: bool = True, verbose: bool = False):
    """Run the server with uvicorn."""
    import uvicorn

    log_level = "debug" if verbose else "info"

    if do_warmup:
        warmup()

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
