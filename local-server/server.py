"""
Local API server for Statement Extractor model.

This server uses the corp-extractor library to extract structured statements from text.
Uses Diverse Beam Search (https://arxiv.org/abs/1610.02424) for high-quality extraction.

Usage:
    # Using .env file
    uv run python server.py

    # Using CLI arguments (overrides .env)
    uv run python server.py --model-path ../model --port 8000

Environment variables (.env):
    MODEL_PATH: Path to the model directory (or HuggingFace model ID)
    PORT: Port to run the server on (default: 8000)
    HOST: Host to bind to (default: 0.0.0.0)
    NUM_RETURN_SEQUENCES: Number of candidate sequences (default: 4)
    MIN_STATEMENT_RATIO: Minimum statements per sentence ratio (default: 1.0)
    MAX_EXTRACTION_ATTEMPTS: Max retry attempts (default: 3)
    MAX_CACHE_SIZE_BYTES: In-memory cache size limit (default: 1GB)
"""

import argparse
import hashlib
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from statement_extractor import StatementExtractor, ExtractionOptions

# Load .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Server configuration from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_path: str = Field(
        default="Corp-o-Rate-Community/statement-extractor",
        description="Path to model directory or HuggingFace model ID",
    )
    port: int = Field(
        default=8000,
        description="Port to run the server on",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind to",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    num_return_sequences: int = Field(
        default=4,
        description="Number of candidate sequences to generate",
    )
    min_statement_ratio: float = Field(
        default=1.0,
        description="Minimum statements per sentence ratio before retry",
    )
    max_extraction_attempts: int = Field(
        default=3,
        description="Maximum extraction retry attempts",
    )
    max_cache_size_bytes: int = Field(
        default=1 * 1024 * 1024 * 1024,  # 1GB default for local
        description="Maximum in-memory cache size in bytes",
    )


# Global settings instance
settings = Settings()

app = FastAPI(
    title="Statement Extractor API",
    description="Local API for T5-Gemma 2 statement extraction model (using corp-extractor library)",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global extractor
extractor: Optional[StatementExtractor] = None

# In-memory LRU cache for results
result_cache: OrderedDict[str, str] = OrderedDict()
cache_size_bytes: int = 0


class ExtractRequest(BaseModel):
    text: str


def get_cache_key(text: str) -> str:
    """Generate a cache key from input text."""
    return hashlib.sha256(text.encode()).hexdigest()


def get_entry_size(key: str, value: str) -> int:
    """Estimate memory size of a cache entry in bytes."""
    return len(key.encode()) + len(value.encode())


def evict_if_needed(new_entry_size: int) -> None:
    """Evict oldest entries if cache would exceed size limit."""
    global cache_size_bytes

    while result_cache and (cache_size_bytes + new_entry_size) > settings.max_cache_size_bytes:
        oldest_key, oldest_value = result_cache.popitem(last=False)
        evicted_size = get_entry_size(oldest_key, oldest_value)
        cache_size_bytes -= evicted_size
        logger.info(f"Evicted cache entry: {oldest_key[:16]}... (freed {evicted_size} bytes)")


def load_extractor(model_path: str) -> None:
    """Load the extractor."""
    global extractor

    logger.info(f"Loading extractor from {model_path}...")

    # Check if it's a local path
    path = Path(model_path)
    if path.exists():
        model_id = str(path)
    else:
        model_id = model_path  # Assume it's a HuggingFace model ID

    extractor = StatementExtractor(model_id=model_id)
    logger.info(f"Extractor loaded on device: {extractor.device}")


def extract_with_cache(text: str) -> tuple[dict, bool]:
    """
    Extract statements from text with caching.
    Returns (result_dict, cached) tuple where result_dict is ExtractionResult.model_dump().
    """
    import json as _json
    global cache_size_bytes

    # Check cache first
    cache_key = get_cache_key(text)
    if cache_key in result_cache:
        result_cache.move_to_end(cache_key)
        logger.info(f"Cache hit for key: {cache_key[:16]}...")
        return _json.loads(result_cache[cache_key]), True

    # Configure extraction options
    options = ExtractionOptions(
        num_beams=settings.num_return_sequences,
        min_statement_ratio=settings.min_statement_ratio,
        max_attempts=settings.max_extraction_attempts,
    )

    # Extract using the library
    result = extractor.extract(text, options)
    result_dict = result.model_dump()

    # Store in cache (as JSON string)
    result_json = _json.dumps(result_dict)
    entry_size = get_entry_size(cache_key, result_json)
    evict_if_needed(entry_size)
    result_cache[cache_key] = result_json
    cache_size_bytes += entry_size
    logger.info(f"Cached result for key: {cache_key[:16]}... (entries: {len(result_cache)}, size: {cache_size_bytes / 1024 / 1024:.1f}MB)")

    return result_dict, False


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": extractor is not None,
        "device": extractor.device if extractor else None,
        "model_path": settings.model_path,
        "cache_entries": len(result_cache),
        "cache_size_mb": round(cache_size_bytes / 1024 / 1024, 2),
    }


@app.post("/extract")
async def extract_statements_endpoint(request: ExtractRequest):
    """Extract statements from text. Returns ExtractionResult model_dump format."""
    if extractor is None:
        raise HTTPException(status_code=503, detail="Extractor not loaded")

    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty text")

        # Ensure text is wrapped in page tags
        if not text.startswith("<page>"):
            text = f"<page>{text}</page>"

        logger.info(f"Processing text of length {len(text)}")

        result_dict, cached = extract_with_cache(text)

        logger.info(f"Generated {len(result_dict.get('statements', []))} statements (cached: {cached})")

        return {**result_dict, "cached": cached}

    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache")
async def clear_cache():
    """Clear the in-memory cache."""
    global result_cache, cache_size_bytes
    count = len(result_cache)
    result_cache = OrderedDict()
    cache_size_bytes = 0
    logger.info(f"Cleared {count} cache entries")
    return {"cleared": count}


def main():
    """Main entry point."""
    import uvicorn

    parser = argparse.ArgumentParser(
        description="Statement Extractor API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables (can also be set in .env file):
  MODEL_PATH              Path to the model directory or HuggingFace model ID
  PORT                    Port to run the server on (default: 8000)
  HOST                    Host to bind to (default: 0.0.0.0)
  LOG_LEVEL               Logging level (default: INFO)
  NUM_RETURN_SEQUENCES    Candidates to generate (default: 4)
  MIN_STATEMENT_RATIO     Min statements/sentence ratio (default: 1.0)
  MAX_EXTRACTION_ATTEMPTS Max retry attempts (default: 3)
  MAX_CACHE_SIZE_BYTES    Cache size limit (default: 1GB)

Example .env file:
  MODEL_PATH=/path/to/model
  PORT=8000
  HOST=0.0.0.0
        """,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=f"Path to model directory or HuggingFace model ID (default: {settings.model_path})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port to run the server on (default: {settings.port})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help=f"Host to bind to (default: {settings.host})",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file (default: .env in current directory)",
    )

    args = parser.parse_args()

    # Reload settings if custom env file specified
    if args.env_file:
        load_dotenv(args.env_file, override=True)
        global settings
        settings = Settings()

    # CLI arguments override environment/settings
    model_path = args.model_path or settings.model_path
    port = args.port or settings.port
    host = args.host or settings.host

    # Set log level
    logging.getLogger().setLevel(settings.log_level.upper())

    logger.info("Configuration:")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Num return sequences: {settings.num_return_sequences}")
    logger.info(f"  Min statement ratio: {settings.min_statement_ratio}")
    logger.info(f"  Max extraction attempts: {settings.max_extraction_attempts}")
    logger.info(f"  Max cache size: {settings.max_cache_size_bytes / 1024 / 1024:.0f}MB")

    # Load extractor
    load_extractor(model_path)

    # Run server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
