"""HTTP client for delegating to a running corp-extractor server."""

import logging
from typing import Any, Optional

import httpx

from .models import ExtractionResult, ExtractionOptions
from .pipeline.context import PipelineContext
from .document.context import DocumentContext

logger = logging.getLogger(__name__)

_TIMEOUT = 300  # 5 minutes — extraction can be slow


def server_split(
    server_url: str,
    text: str,
    options: Optional[ExtractionOptions] = None,
) -> ExtractionResult:
    """POST to /split, return ExtractionResult."""
    payload: dict[str, Any] = {"text": text}
    if options is not None:
        payload["options"] = options.model_dump()
    resp = httpx.post(
        f"{server_url.rstrip('/')}/split",
        json=payload,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return ExtractionResult.model_validate(resp.json())


def server_pipeline(
    server_url: str,
    text: str,
    config_dict: dict[str, Any],
) -> PipelineContext:
    """POST to /pipeline, return PipelineContext."""
    resp = httpx.post(
        f"{server_url.rstrip('/')}/pipeline",
        json={"text": text, "config": config_dict},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return PipelineContext.model_validate(resp.json())


def server_document(
    server_url: str,
    text: str,
    **kwargs: Any,
) -> DocumentContext:
    """POST to /document, return DocumentContext."""
    resp = httpx.post(
        f"{server_url.rstrip('/')}/document",
        json={"text": text, **kwargs},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return DocumentContext.model_validate(resp.json())
