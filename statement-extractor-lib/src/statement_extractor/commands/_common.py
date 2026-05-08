"""Shared utilities used across CLI command modules."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click

DEFAULT_SERVER_URL = "http://localhost:8111"


def _configure_logging(verbose: bool) -> None:
    """Configure logging for the extraction pipeline."""
    level = logging.DEBUG if verbose else logging.WARNING

    # Configure root logger for statement_extractor package
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    # Set level for all statement_extractor loggers
    for logger_name in [
        "statement_extractor",
        "statement_extractor.extractor",
        "statement_extractor.scoring",
        "statement_extractor.predicate_comparer",
        "statement_extractor.canonicalization",
        "statement_extractor.gliner_extraction",
        "statement_extractor.pipeline",
        "statement_extractor.plugins",
        "statement_extractor.plugins.extractors.gliner2",
        "statement_extractor.plugins.splitters",
        "statement_extractor.plugins.labelers",
        "statement_extractor.plugins.scrapers",
        "statement_extractor.plugins.scrapers.http",
        "statement_extractor.plugins.pdf",
        "statement_extractor.plugins.pdf.pypdf",
        "statement_extractor.plugins.pdf.glm_ocr",
        "statement_extractor.document",
        "statement_extractor.document.loader",
        "statement_extractor.document.html_extractor",
        "statement_extractor.document.pipeline",
        "statement_extractor.document.chunker",
    ]:
        logging.getLogger(logger_name).setLevel(level)

    # Suppress noisy third-party loggers
    for noisy_logger in [
        "httpcore",
        "httpcore.http11",
        "httpcore.connection",
        "httpx",
        "urllib3",
        "huggingface_hub",
        "asyncio",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def _server_request(server_url: str, endpoint: str, payload: dict, timeout: float = 300) -> dict:
    """Send a request to the server and return the JSON response."""
    import httpx
    url = f"{server_url.rstrip('/')}/{endpoint.lstrip('/')}"
    resp = httpx.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _resolve_db_path(db_path: Optional[str] = None) -> Path:
    """Resolve the database path from an explicit --db value or --db-version context.

    Checks v3 first, then falls back to v2 if the v3 file doesn't exist.
    """
    if db_path is not None:
        return Path(db_path)
    # Check for --db-version in the Click context chain
    try:
        ctx = click.get_current_context(silent=True)
        db_version = ctx.obj.get("db_version") if ctx and ctx.obj else None
    except RuntimeError:
        db_version = None
    if db_version is not None:
        from corp_entity_db.hub import DEFAULT_CACHE_DIR, db_filenames
        full_fn, _, _ = db_filenames(db_version)
        return DEFAULT_CACHE_DIR / full_fn
    # Default: try v3, fall back to v2
    from corp_entity_db.hub import DEFAULT_CACHE_DIR
    from corp_entity_db.store import DEFAULT_DB_PATH
    if DEFAULT_DB_PATH.exists():
        return DEFAULT_DB_PATH
    v2_path = DEFAULT_CACHE_DIR / "entities-v2.db"
    if v2_path.exists():
        return v2_path
    return DEFAULT_DB_PATH  # Return v3 path even if missing (for creation)


def _get_input_text(text: Optional[str], input_file: Optional[str]) -> Optional[str]:
    """Get input text from argument, file, or stdin."""
    if text == "-" or (text is None and input_file is None and not sys.stdin.isatty()):
        # Read from stdin
        return sys.stdin.read().strip()
    elif input_file:
        # Read from file
        with open(input_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    elif text:
        return text.strip()
    return None


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


def _load_all_plugins():
    """Load all plugins by importing their modules."""
    # Import all plugin modules to trigger registration
    try:
        from ..plugins import splitters, extractors, qualifiers, labelers, taxonomy
        # The @PluginRegistry decorators will register plugins on import
        _ = splitters, extractors, qualifiers, labelers, taxonomy  # Silence unused warnings
    except ImportError as e:
        logging.debug(f"Some plugins failed to load: {e}")


def _print_table(result, verbose: bool):
    """Print statements in a human-readable table format."""
    if not result.statements:
        click.echo("No statements extracted.")
        return

    click.echo(f"\nExtracted {len(result.statements)} statement(s):\n")
    click.echo("-" * 80)

    for i, stmt in enumerate(result.statements, 1):
        subject_type = f" ({stmt.subject.type.value})" if stmt.subject.type.value != "UNKNOWN" else ""
        object_type = f" ({stmt.object.type.value})" if stmt.object.type.value != "UNKNOWN" else ""

        click.echo(f"{i}. {stmt.subject.text}{subject_type}")
        click.echo(f"   --[{stmt.predicate}]-->")
        click.echo(f"   {stmt.object.text}{object_type}")

        if verbose:
            # Always show extraction method
            click.echo(f"   Method: {stmt.extraction_method.value}")

            if stmt.confidence_score is not None:
                click.echo(f"   Confidence: {stmt.confidence_score:.2f}")

            if stmt.canonical_predicate:
                click.echo(f"   Canonical: {stmt.canonical_predicate}")

            if stmt.was_reversed:
                click.echo(f"   (subject/object were swapped)")

            if stmt.source_text:
                source = stmt.source_text[:60] + "..." if len(stmt.source_text) > 60 else stmt.source_text
                click.echo(f"   Source: \"{source}\"")

        click.echo("-" * 80)
