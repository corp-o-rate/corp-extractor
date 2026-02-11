"""
Command-line interface for statement extraction.

Usage:
    corp-extractor split "Your text here"
    corp-extractor split -f input.txt
    corp-extractor pipeline "Your text here" --stages 1-5
    corp-extractor plugins list
"""

import json
import logging
import sys
from pathlib import Path
import queue
import threading
from typing import NamedTuple, Optional

import click


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


from . import __version__
from .models import (
    ExtractionOptions,
    PredicateComparisonConfig,
    PredicateTaxonomy,
    ScoringConfig,
)


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
        from .database.hub import DEFAULT_CACHE_DIR, db_filenames
        full_fn, _, _ = db_filenames(db_version)
        return DEFAULT_CACHE_DIR / full_fn
    # Default: try v3, fall back to v2
    from .database.hub import DEFAULT_CACHE_DIR
    from .database.store import DEFAULT_DB_PATH
    if DEFAULT_DB_PATH.exists():
        return DEFAULT_DB_PATH
    v2_path = DEFAULT_CACHE_DIR / "entities-v2.db"
    if v2_path.exists():
        return v2_path
    return DEFAULT_DB_PATH  # Return v3 path even if missing (for creation)


@click.group()
@click.version_option(version=__version__)
@click.option("--db-version", type=int, default=None, hidden=True, help="Database schema version for filenames (default: latest)")
@click.pass_context
def main(ctx: click.Context, db_version: int | None):
    """
    Extract structured statements from text.

    \b
    Commands:
        split      Extract sub-statements from text (simple, fast)
        pipeline   Run the full 6-stage extraction pipeline
        document   Process documents with chunking and citations
        plugins    List or inspect available plugins
        db         Manage entity/organization embedding database

    \b
    Examples:
        corp-extractor split "Apple announced a new iPhone."
        corp-extractor split -f article.txt --json
        corp-extractor pipeline "Apple CEO Tim Cook announced..." --stages 1-3
        corp-extractor document process report.txt --title "Annual Report"
        corp-extractor plugins list
    """
    ctx.ensure_object(dict)
    ctx.obj["db_version"] = db_version


# =============================================================================
# Split command (simple extraction)
# =============================================================================

@main.command("split")
@click.argument("text", required=False)
@click.option("-f", "--file", "input_file", type=click.Path(exists=True), help="Read input from file")
@click.option(
    "-o", "--output",
    type=click.Choice(["table", "json", "xml"], case_sensitive=False),
    default="table",
    help="Output format (default: table)"
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON (shortcut for -o json)")
@click.option("--xml", "output_xml", is_flag=True, help="Output as XML (shortcut for -o xml)")
# Beam search options
@click.option("-b", "--beams", type=int, default=4, help="Number of beams for diverse beam search (default: 4)")
@click.option("--diversity", type=float, default=1.0, help="Diversity penalty for beam search (default: 1.0)")
@click.option("--max-tokens", type=int, default=2048, help="Maximum tokens to generate (default: 2048)")
# Deduplication options
@click.option("--no-dedup", is_flag=True, help="Disable deduplication")
@click.option("--no-embeddings", is_flag=True, help="Disable embedding-based deduplication (faster)")
@click.option("--no-merge", is_flag=True, help="Disable beam merging (select single best beam)")
@click.option("--no-gliner", is_flag=True, help="Disable GLiNER2 extraction (use raw model output)")
@click.option("--predicates", type=str, help="Comma-separated list of predicate types for GLiNER2 relation extraction")
@click.option("--all-triples", is_flag=True, help="Keep all candidate triples instead of selecting best per source")
@click.option("--dedup-threshold", type=float, default=0.65, help="Similarity threshold for deduplication (default: 0.65)")
# Quality options
@click.option("--min-confidence", type=float, default=0.0, help="Minimum confidence threshold 0-1 (default: 0)")
# Taxonomy options
@click.option("--taxonomy", type=click.Path(exists=True), help="Load predicate taxonomy from file (one per line)")
@click.option("--taxonomy-threshold", type=float, default=0.5, help="Similarity threshold for taxonomy matching (default: 0.5)")
# Device options
@click.option("--device", type=click.Choice(["auto", "cuda", "mps", "cpu"]), default="auto", help="Device to use (default: auto)")
# Output options
@click.option("-v", "--verbose", is_flag=True, help="Show verbose output with confidence scores")
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress messages")
def split_cmd(
    text: Optional[str],
    input_file: Optional[str],
    output: str,
    output_json: bool,
    output_xml: bool,
    beams: int,
    diversity: float,
    max_tokens: int,
    no_dedup: bool,
    no_embeddings: bool,
    no_merge: bool,
    no_gliner: bool,
    predicates: Optional[str],
    all_triples: bool,
    dedup_threshold: float,
    min_confidence: float,
    taxonomy: Optional[str],
    taxonomy_threshold: float,
    device: str,
    verbose: bool,
    quiet: bool,
):
    """
    Extract sub-statements from text using T5-Gemma model.

    This command splits text into structured subject-predicate-object triples.
    It's fast and simple - use 'pipeline' for full entity resolution.

    \b
    Examples:
        corp-extractor split "Apple announced a new iPhone."
        corp-extractor split -f article.txt --json
        corp-extractor split -f article.txt -o json --beams 8
        cat article.txt | corp-extractor split -
        echo "Tim Cook is CEO of Apple." | corp-extractor split - --verbose

    \b
    Output formats:
        table  Human-readable table (default)
        json   JSON with full metadata
        xml    Raw XML from model
    """
    # Configure logging based on verbose flag
    _configure_logging(verbose)

    # Determine output format
    if output_json:
        output = "json"
    elif output_xml:
        output = "xml"

    # Get input text
    input_text = _get_input_text(text, input_file)
    if not input_text:
        raise click.UsageError("No input provided. Provide text argument or use -f file.txt")

    if not quiet:
        click.echo(f"Processing {len(input_text)} characters...", err=True)

    # Load taxonomy if provided
    predicate_taxonomy = None
    if taxonomy:
        predicate_taxonomy = PredicateTaxonomy.from_file(taxonomy)
        if not quiet:
            click.echo(f"Loaded taxonomy with {len(predicate_taxonomy.predicates)} predicates", err=True)

    # Configure predicate comparison
    predicate_config = PredicateComparisonConfig(
        similarity_threshold=taxonomy_threshold,
        dedup_threshold=dedup_threshold,
    )

    # Configure scoring
    scoring_config = ScoringConfig(min_confidence=min_confidence)

    # Parse predicates if provided
    predicate_list = None
    if predicates:
        predicate_list = [p.strip() for p in predicates.split(",") if p.strip()]
        if not quiet:
            click.echo(f"Using predicate list: {predicate_list}", err=True)

    # Configure extraction options
    options = ExtractionOptions(
        num_beams=beams,
        diversity_penalty=diversity,
        max_new_tokens=max_tokens,
        deduplicate=not no_dedup,
        embedding_dedup=not no_embeddings,
        merge_beams=not no_merge,
        use_gliner_extraction=not no_gliner,
        predicates=predicate_list,
        all_triples=all_triples,
        predicate_taxonomy=predicate_taxonomy,
        predicate_config=predicate_config,
        scoring_config=scoring_config,
        verbose=verbose,
    )

    # Import here to allow --help without loading torch
    from .extractor import StatementExtractor

    # Create extractor with specified device
    device_arg = None if device == "auto" else device
    extractor = StatementExtractor(device=device_arg)

    if not quiet:
        click.echo(f"Using device: {extractor.device}", err=True)

    # Run extraction
    try:
        if output == "xml":
            result = extractor.extract_as_xml(input_text, options)
            click.echo(result)
        elif output == "json":
            result = extractor.extract_as_json(input_text, options)
            click.echo(result)
        else:
            # Table format
            result = extractor.extract(input_text, options)
            _print_table(result, verbose)
    except Exception as e:
        logging.exception("Error extracting statements:")
        raise click.ClickException(f"Extraction failed: {e}")


# =============================================================================
# Pipeline command
# =============================================================================

@main.command("pipeline")
@click.argument("text", required=False)
@click.option("-f", "--file", "input_file", type=click.Path(exists=True), help="Read input from file")
@click.option(
    "--stages",
    type=str,
    default="1-6",
    help="Stages to run (e.g., '1,2,3' or '1-3' or '1-6')"
)
@click.option(
    "--skip-stages",
    type=str,
    default=None,
    help="Stages to skip (e.g., '4,5')"
)
@click.option(
    "--plugins",
    "enabled_plugins",
    type=str,
    default=None,
    help="Plugins to enable (comma-separated names)"
)
@click.option(
    "--disable-plugins",
    type=str,
    default=None,
    help="Plugins to disable (comma-separated names)"
)
@click.option(
    "--no-default-predicates",
    is_flag=True,
    help="Disable default predicate taxonomy (GLiNER2 will only use entity extraction)"
)
@click.option(
    "-o", "--output",
    type=click.Choice(["table", "json", "yaml", "triples"], case_sensitive=False),
    default="table",
    help="Output format (default: table)"
)
@click.option("-v", "--verbose", is_flag=True, help="Show verbose output")
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress messages")
def pipeline_cmd(
    text: Optional[str],
    input_file: Optional[str],
    stages: str,
    skip_stages: Optional[str],
    enabled_plugins: Optional[str],
    disable_plugins: Optional[str],
    no_default_predicates: bool,
    output: str,
    verbose: bool,
    quiet: bool,
):
    """
    Run the full 5-stage extraction pipeline.

    \b
    Stages:
        1. Splitting      - Text → Raw triples (T5-Gemma)
        2. Extraction     - Raw triples → Typed statements (GLiNER2)
        3. Qualification  - Add qualifiers and identifiers
        4. Canonicalization - Resolve to canonical forms
        5. Labeling       - Apply sentiment, relation type, confidence

    \b
    Examples:
        corp-extractor pipeline "Apple CEO Tim Cook announced..."
        corp-extractor pipeline -f article.txt --stages 1-3
        corp-extractor pipeline "..." --plugins gleif,companies_house
        corp-extractor pipeline "..." --disable-plugins sec_edgar
    """
    _configure_logging(verbose)

    # Get input text
    input_text = _get_input_text(text, input_file)
    if not input_text:
        raise click.UsageError("No input provided. Provide text argument or use -f file.txt")

    if not quiet:
        click.echo(f"Processing {len(input_text)} characters through pipeline...", err=True)

    # Import pipeline components (also loads plugins)
    from .pipeline import ExtractionPipeline, PipelineConfig
    _load_all_plugins()

    # Parse stages
    enabled_stages = _parse_stages(stages)
    if skip_stages:
        skip_set = _parse_stages(skip_stages)
        enabled_stages = enabled_stages - skip_set

    if not quiet:
        click.echo(f"Running stages: {sorted(enabled_stages)}", err=True)

    # Parse plugin selection
    enabled_plugin_set = None
    if enabled_plugins:
        enabled_plugin_set = {p.strip() for p in enabled_plugins.split(",") if p.strip()}

    disabled_plugin_set = None
    if disable_plugins:
        disabled_plugin_set = {p.strip() for p in disable_plugins.split(",") if p.strip()}

    # Build extractor options
    extractor_options = {}
    if no_default_predicates:
        extractor_options["use_default_predicates"] = False
        if not quiet:
            click.echo("Default predicates disabled - using entity extraction only", err=True)

    # Create config - only pass disabled_plugins if user explicitly specified, otherwise use defaults
    config_kwargs: dict = {
        "enabled_stages": enabled_stages,
        "enabled_plugins": enabled_plugin_set,
        "extractor_options": extractor_options,
    }
    if disabled_plugin_set is not None:
        config_kwargs["disabled_plugins"] = disabled_plugin_set
    config = PipelineConfig(**config_kwargs)

    # Run pipeline
    try:
        pipeline = ExtractionPipeline(config)
        ctx = pipeline.process(input_text)

        # Output results
        if output == "json":
            _print_pipeline_json(ctx)
        elif output == "yaml":
            _print_pipeline_yaml(ctx)
        elif output == "triples":
            _print_pipeline_triples(ctx)
        else:
            _print_pipeline_table(ctx, verbose)

        # Report errors/warnings
        if ctx.processing_errors and not quiet:
            click.echo(f"\nErrors: {len(ctx.processing_errors)}", err=True)
            for error in ctx.processing_errors:
                click.echo(f"  - {error}", err=True)

        if ctx.processing_warnings and verbose:
            click.echo(f"\nWarnings: {len(ctx.processing_warnings)}", err=True)
            for warning in ctx.processing_warnings:
                click.echo(f"  - {warning}", err=True)

    except Exception as e:
        logging.exception("Pipeline error:")
        raise click.ClickException(f"Pipeline failed: {e}")


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


def _print_pipeline_json(ctx):
    """Print pipeline results as JSON."""
    output = {
        "statement_count": ctx.statement_count,
        "split_sentences": [s.model_dump() for s in ctx.split_sentences],
        "statements": [s.model_dump() for s in ctx.statements],
        "labeled_statements": [stmt.as_dict() for stmt in ctx.labeled_statements],
        "timings": ctx.stage_timings,
        "warnings": ctx.processing_warnings,
        "errors": ctx.processing_errors,
    }
    click.echo(json.dumps(output, indent=2, default=str))


def _print_pipeline_yaml(ctx):
    """Print pipeline results as YAML."""
    try:
        import yaml
        output = {
            "statement_count": ctx.statement_count,
            "statements": [stmt.as_dict() for stmt in ctx.labeled_statements],
            "timings": ctx.stage_timings,
        }
        click.echo(yaml.dump(output, default_flow_style=False))
    except ImportError:
        click.echo("YAML output requires PyYAML: pip install pyyaml", err=True)
        _print_pipeline_json(ctx)


def _print_pipeline_triples(ctx):
    """Print pipeline results as simple triples."""
    if ctx.labeled_statements:
        for stmt in ctx.labeled_statements:
            click.echo(f"{stmt.subject_fqn}\t{stmt.statement.predicate}\t{stmt.object_fqn}")
    elif ctx.statements:
        for stmt in ctx.statements:
            click.echo(f"{stmt.subject.text}\t{stmt.predicate}\t{stmt.object.text}")
    elif ctx.split_sentences:
        # Stage 1 only output - just show the split sentences (no triples yet)
        for sentence in ctx.split_sentences:
            click.echo(sentence.text)


def _print_pipeline_table(ctx, verbose: bool):
    """Print pipeline results in table format."""
    # Try labeled statements first, then statements, then raw triples
    if ctx.labeled_statements:
        click.echo(f"\nExtracted {len(ctx.labeled_statements)} statement(s):\n")
        click.echo("-" * 80)

        for i, stmt in enumerate(ctx.labeled_statements, 1):
            click.echo(f"{i}. {stmt.subject_fqn}")
            click.echo(f"   --[{stmt.statement.predicate}]-->")
            click.echo(f"   {stmt.object_fqn}")

            # Show labels (always in recent versions, not just verbose)
            for label in stmt.labels:
                if isinstance(label.label_value, float):
                    click.echo(f"   {label.label_type}: {label.label_value:.3f}")
                else:
                    click.echo(f"   {label.label_type}: {label.label_value}")

            # Show top taxonomy results (sorted by confidence)
            if stmt.taxonomy_results:
                sorted_taxonomy = sorted(stmt.taxonomy_results, key=lambda t: t.confidence, reverse=True)
                top_taxonomy = sorted_taxonomy[:5]  # Show top 5
                taxonomy_strs = [f"{t.category}:{t.label} ({t.confidence:.2f})" for t in top_taxonomy]
                click.echo(f"   topics: {', '.join(taxonomy_strs)}")
                if len(sorted_taxonomy) > 5:
                    click.echo(f"   ... and {len(sorted_taxonomy) - 5} more topics")

            if verbose and stmt.statement.source_text:
                source = stmt.statement.source_text[:60] + "..." if len(stmt.statement.source_text) > 60 else stmt.statement.source_text
                click.echo(f"   Source: \"{source}\"")

            click.echo("-" * 80)

    elif ctx.statements:
        click.echo(f"\nExtracted {len(ctx.statements)} statement(s):\n")
        click.echo("-" * 80)

        for i, stmt in enumerate(ctx.statements, 1):
            subj_type = f" ({stmt.subject.type.value})" if stmt.subject.type.value != "UNKNOWN" else ""
            obj_type = f" ({stmt.object.type.value})" if stmt.object.type.value != "UNKNOWN" else ""

            click.echo(f"{i}. {stmt.subject.text}{subj_type}")
            click.echo(f"   --[{stmt.predicate}]-->")
            click.echo(f"   {stmt.object.text}{obj_type}")

            if verbose and stmt.confidence_score is not None:
                click.echo(f"   Confidence: {stmt.confidence_score:.2f}")

            click.echo("-" * 80)

    elif ctx.split_sentences:
        click.echo(f"\nSplit into {len(ctx.split_sentences)} atomic sentence(s):\n")
        click.echo("-" * 80)

        for i, sentence in enumerate(ctx.split_sentences, 1):
            text_preview = sentence.text[:100] + "..." if len(sentence.text) > 100 else sentence.text
            click.echo(f"{i}. {text_preview}")

            if verbose:
                click.echo(f"   Confidence: {sentence.confidence:.2f}")

            click.echo("-" * 80)

    else:
        click.echo("No statements extracted.")
        return

    # Show timings in verbose mode
    if verbose and ctx.stage_timings:
        click.echo("\nStage timings:")
        for stage, duration in ctx.stage_timings.items():
            click.echo(f"  {stage}: {duration:.3f}s")


# =============================================================================
# Plugins command
# =============================================================================

@main.command("plugins")
@click.argument("action", type=click.Choice(["list", "info"]))
@click.argument("plugin_name", required=False)
@click.option("--stage", type=int, help="Filter by stage number (1-5)")
def plugins_cmd(action: str, plugin_name: Optional[str], stage: Optional[int]):
    """
    List or inspect available plugins.

    \b
    Actions:
        list   List all available plugins
        info   Show details about a specific plugin

    \b
    Examples:
        corp-extractor plugins list
        corp-extractor plugins list --stage 3
        corp-extractor plugins info gleif_qualifier
    """
    # Import and load plugins
    _load_all_plugins()

    from .pipeline.registry import PluginRegistry

    if action == "list":
        plugins = PluginRegistry.list_plugins(stage=stage)
        if not plugins:
            click.echo("No plugins registered.")
            return

        # Group by stage
        by_stage: dict[int, list] = {}
        for plugin in plugins:
            stage_num = plugin["stage"]
            if stage_num not in by_stage:
                by_stage[stage_num] = []
            by_stage[stage_num].append(plugin)

        for stage_num in sorted(by_stage.keys()):
            stage_plugins = by_stage[stage_num]
            stage_name = stage_plugins[0]["stage_name"]
            click.echo(f"\nStage {stage_num}: {stage_name.title()}")
            click.echo("-" * 40)

            for p in stage_plugins:
                entity_types = p.get("entity_types", [])
                types_str = f" ({', '.join(entity_types)})" if entity_types else ""
                click.echo(f"  {p['name']}{types_str}  [priority: {p['priority']}]")

    elif action == "info":
        if not plugin_name:
            raise click.UsageError("Plugin name required for 'info' action")

        plugin = PluginRegistry.get_plugin(plugin_name)
        if not plugin:
            raise click.ClickException(f"Plugin not found: {plugin_name}")

        click.echo(f"\nPlugin: {plugin.name}")
        click.echo(f"Priority: {plugin.priority}")
        click.echo(f"Capabilities: {plugin.capabilities.name if plugin.capabilities else 'NONE'}")

        if plugin.description:
            click.echo(f"Description: {plugin.description}")

        if hasattr(plugin, "supported_entity_types"):
            types = [t.value for t in plugin.supported_entity_types]
            click.echo(f"Entity types: {', '.join(types)}")

        if hasattr(plugin, "label_type"):
            click.echo(f"Label type: {plugin.label_type}")

        if hasattr(plugin, "supported_identifier_types"):
            ids = plugin.supported_identifier_types
            if ids:
                click.echo(f"Supported identifiers: {', '.join(ids)}")

        if hasattr(plugin, "provided_identifier_types"):
            ids = plugin.provided_identifier_types
            if ids:
                click.echo(f"Provided identifiers: {', '.join(ids)}")


def _load_all_plugins():
    """Load all plugins by importing their modules."""
    # Import all plugin modules to trigger registration
    try:
        from .plugins import splitters, extractors, qualifiers, labelers, taxonomy
        # The @PluginRegistry decorators will register plugins on import
        _ = splitters, extractors, qualifiers, labelers, taxonomy  # Silence unused warnings
    except ImportError as e:
        logging.debug(f"Some plugins failed to load: {e}")


# =============================================================================
# Database commands
# =============================================================================

@main.group("db")
@click.pass_context
def db_cmd(ctx: click.Context):
    """
    Manage entity/organization embedding database.

    \b
    Commands:
        import-gleif           Import GLEIF LEI data (~3M records)
        import-sec             Import SEC Edgar bulk data (~100K+ filers)
        import-sec-officers    Import SEC Form 4 officers/directors
        import-ch-officers     Import UK Companies House officers (Prod195)
        import-companies-house Import UK Companies House (~5M records)
        import-wikidata        Import Wikidata organizations (SPARQL, may timeout)
        import-people          Import Wikidata notable people (SPARQL, may timeout)
        import-wikidata-dump   Import from Wikidata JSON dump (recommended)
        canonicalize           Link equivalent records across sources
        status                 Show database status
        search                 Search for an organization
        search-people          Search for a person
        download               Download database from HuggingFace
        upload                 Upload database with lite variant
        create-lite            Create lite version (no record data)

    \b
    Examples:
        corp-extractor db import-sec --download
        corp-extractor db import-sec-officers --start-year 2023 --limit 10000
        corp-extractor db import-gleif --download --limit 100000
        corp-extractor db import-wikidata-dump --download --limit 50000
        corp-extractor db canonicalize
        corp-extractor db status
        corp-extractor db search "Apple Inc"
        corp-extractor db search-people "Tim Cook"
        corp-extractor db upload entities.db
    """
    ctx.ensure_object(dict)


@db_cmd.command("gleif-info")
def db_gleif_info():
    """
    Show information about the latest available GLEIF data file.

    \b
    Examples:
        corp-extractor db gleif-info
    """
    from .database.importers import GleifImporter

    importer = GleifImporter()

    try:
        info = importer.get_latest_file_info()
        record_count = info.get('record_count')

        click.echo("\nLatest GLEIF Data File")
        click.echo("=" * 40)
        click.echo(f"File ID: {info['id']}")
        click.echo(f"Publish Date: {info['publish_date']}")
        click.echo(f"Record Count: {record_count:,}" if record_count else "Record Count: unknown")

        delta = info.get("delta_from_last_file", {})
        if delta:
            click.echo(f"\nChanges from previous file:")
            if delta.get('new'):
                click.echo(f"  New: {delta.get('new'):,}")
            if delta.get('updated'):
                click.echo(f"  Updated: {delta.get('updated'):,}")
            if delta.get('retired'):
                click.echo(f"  Retired: {delta.get('retired'):,}")

    except Exception as e:
        raise click.ClickException(f"Failed to get GLEIF info: {e}")


@db_cmd.command("import-gleif")
@click.argument("file_path", type=click.Path(exists=True), required=False)
@click.option("--download", is_flag=True, help="Download latest GLEIF file before importing")
@click.option("--force", is_flag=True, help="Force re-download even if cached")
@click.option("--db", "db_path", type=click.Path(), help="Database path (default: ~/.cache/corp-extractor/entities.db)")
@click.option("--limit", type=int, help="Limit number of records to import")
@click.option("--batch-size", type=int, default=50000, help="Batch size for commits (default: 50000)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_gleif(file_path: Optional[str], download: bool, force: bool, db_path: Optional[str], limit: Optional[int], batch_size: int, verbose: bool):
    """
    Import GLEIF LEI data into the entity database.

    If no file path is provided and --download is set, downloads the latest
    GLEIF data file automatically. Downloaded files are cached and reused
    unless --force is specified.

    \b
    Examples:
        corp-extractor db import-gleif /path/to/lei-records.xml
        corp-extractor db import-gleif --download
        corp-extractor db import-gleif --download --limit 10000
        corp-extractor db import-gleif --download --force  # Re-download
    """
    _configure_logging(verbose)

    from .database import OrganizationDatabase, CompanyEmbedder
    from .database.importers import GleifImporter

    importer = GleifImporter()

    # Handle file path
    if file_path is None:
        if not download:
            raise click.UsageError("Either provide a file path or use --download to fetch the latest GLEIF data")
        click.echo("Downloading latest GLEIF data...", err=True)
        file_path = str(importer.download_latest(force=force))
    elif download:
        click.echo("Downloading latest GLEIF data (ignoring provided file path)...", err=True)
        file_path = str(importer.download_latest(force=force))

    click.echo(f"Importing GLEIF data from {file_path}...", err=True)

    # Initialize components (readonly=False for import operations)
    db_path_obj = _resolve_db_path(db_path)
    embedder = CompanyEmbedder()
    database = OrganizationDatabase(db_path=db_path_obj, embedding_dim=embedder.embedding_dim, readonly=False)

    # Import records in batches
    records = []
    count = 0

    for record in importer.import_from_file(file_path, limit=limit):
        records.append(record)

        if len(records) >= batch_size:
            # Embed and insert batch (both float32 and int8)
            names = [r.name for r in records]
            embeddings, scalar_embeddings = embedder.embed_batch_and_quantize(names)
            database.insert_batch(records, embeddings, scalar_embeddings=scalar_embeddings)
            count += len(records)
            click.echo(f"Imported {count} records...", err=True)
            records = []

    # Final batch
    if records:
        names = [r.name for r in records]
        embeddings, scalar_embeddings = embedder.embed_batch_and_quantize(names)
        database.insert_batch(records, embeddings, scalar_embeddings=scalar_embeddings)
        count += len(records)

    click.echo(f"\nImported {count} GLEIF records successfully.", err=True)
    click.echo("Run `corp-extractor db post-import` to update search indexes.", err=True)
    database.close()


@db_cmd.command("import-sec")
@click.option("--download", is_flag=True, help="Download bulk submissions.zip (~500MB, ~100K+ filers)")
@click.option("--file", "file_path", type=click.Path(exists=True), help="Local file (submissions.zip or company_tickers.json)")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--limit", type=int, help="Limit number of records")
@click.option("--batch-size", type=int, default=10000, help="Batch size (default: 10000)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_sec(download: bool, file_path: Optional[str], db_path: Optional[str], limit: Optional[int], batch_size: int, verbose: bool):
    """
    Import SEC Edgar data into the entity database.

    By default, downloads the bulk submissions.zip file which contains
    ALL SEC filers (~100K+), not just companies with ticker symbols (~10K).

    \b
    Examples:
        corp-extractor db import-sec --download
        corp-extractor db import-sec --download --limit 50000
        corp-extractor db import-sec --file /path/to/submissions.zip
        corp-extractor db import-sec --file /path/to/company_tickers.json  # legacy
    """
    _configure_logging(verbose)

    from .database import OrganizationDatabase, CompanyEmbedder
    from .database.importers import SecEdgarImporter

    if not download and not file_path:
        raise click.UsageError("Either --download or --file is required")

    # Initialize components (readonly=False for import operations)
    db_path_obj = _resolve_db_path(db_path)
    embedder = CompanyEmbedder()
    database = OrganizationDatabase(db_path=db_path_obj, embedding_dim=embedder.embedding_dim, readonly=False)
    importer = SecEdgarImporter()

    # Get records
    if file_path:
        click.echo(f"Importing SEC Edgar data from {file_path}...", err=True)
        record_iter = importer.import_from_file(file_path, limit=limit)
    else:
        click.echo("Downloading SEC submissions.zip (~500MB)...", err=True)
        record_iter = importer.import_from_url(limit=limit)

    # Import records in batches
    records = []
    count = 0

    for record in record_iter:
        records.append(record)

        if len(records) >= batch_size:
            names = [r.name for r in records]
            embeddings, scalar_embeddings = embedder.embed_batch_and_quantize(names)
            database.insert_batch(records, embeddings, scalar_embeddings=scalar_embeddings)
            count += len(records)
            click.echo(f"Imported {count} records...", err=True)
            records = []

    # Final batch
    if records:
        names = [r.name for r in records]
        embeddings, scalar_embeddings = embedder.embed_batch_and_quantize(names)
        database.insert_batch(records, embeddings, scalar_embeddings=scalar_embeddings)
        count += len(records)

    click.echo(f"\nImported {count} SEC Edgar records successfully.", err=True)
    click.echo("Run `corp-extractor db post-import` to update search indexes.", err=True)
    database.close()


@db_cmd.command("import-sec-officers")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--start-year", type=int, default=2020, help="Start year (default: 2020)")
@click.option("--end-year", type=int, help="End year (default: current year)")
@click.option("--limit", type=int, help="Limit number of records")
@click.option("--batch-size", type=int, default=1000, help="Batch size for commits (default: 1000)")
@click.option("--resume", is_flag=True, help="Resume from saved progress")
@click.option("--skip-existing", is_flag=True, help="Skip records that already exist")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_sec_officers(db_path: Optional[str], start_year: int, end_year: Optional[int], limit: Optional[int], batch_size: int, resume: bool, skip_existing: bool, verbose: bool):
    """
    Import SEC Form 4 insider data into the people database.

    Downloads Form 4 filings from SEC EDGAR and extracts officers, directors,
    and significant investors (10%+ owners) from each company.

    Form 4 filings are submitted when insiders buy or sell company stock.
    They contain the person's name, role (officer/director), and company.

    Rate limited to 5 requests/second to comply with SEC guidelines.

    \b
    Examples:
        corp-extractor db import-sec-officers --limit 1000
        corp-extractor db import-sec-officers --start-year 2023
        corp-extractor db import-sec-officers --resume
        corp-extractor db import-sec-officers --skip-existing -v
    """
    _configure_logging(verbose)

    from .database.store import get_person_database, get_database
    from .database.embeddings import CompanyEmbedder
    from .database.importers.sec_form4 import SecForm4Importer

    # Default database path
    db_path_obj = _resolve_db_path(db_path)

    click.echo(f"Importing SEC Form 4 officers/directors to {db_path_obj}...", err=True)
    click.echo(f"Year range: {start_year} - {end_year or 'current'}", err=True)
    if resume:
        click.echo("Resuming from saved progress...", err=True)

    # Initialize components (readonly=False for import operations)
    database = get_person_database(db_path=db_path_obj, readonly=False)
    org_database = get_database(db_path=db_path_obj, readonly=False)
    embedder = CompanyEmbedder()
    importer = SecForm4Importer()

    # Import records in batches
    records = []
    count = 0
    skipped_existing = 0

    def progress_callback(year: int, quarter: int, filing_idx: int, accession: str, total: int) -> None:
        if verbose and filing_idx % 100 == 0:
            click.echo(f"  {year} Q{quarter}: {filing_idx} filings, {total} records", err=True)

    for record in importer.import_range(
        start_year=start_year,
        end_year=end_year,
        limit=limit,
        resume=resume,
        progress_callback=progress_callback,
    ):
        # Skip existing records if flag is set
        if skip_existing:
            existing = database.get_by_source_id(record.source, record.source_id)
            if existing is not None:
                skipped_existing += 1
                continue

        # Look up org ID by CIK if available
        issuer_cik = record.record.get("issuer_cik", "")
        if issuer_cik:
            org_id = org_database.get_id_by_source_id("sec_edgar", issuer_cik.zfill(10))
            if org_id is not None:
                record.known_for_org_id = org_id

        records.append(record)

        if len(records) >= batch_size:
            embedding_texts = [r.get_embedding_text() for r in records]
            embeddings, scalar_embeddings = embedder.embed_batch_and_quantize(embedding_texts)
            database.insert_batch(records, embeddings, scalar_embeddings=scalar_embeddings)
            count += len(records)
            click.echo(f"Imported {count} records...", err=True)
            records = []

    # Final batch
    if records:
        embedding_texts = [r.get_embedding_text() for r in records]
        embeddings, scalar_embeddings = embedder.embed_batch_and_quantize(embedding_texts)
        database.insert_batch(records, embeddings, scalar_embeddings=scalar_embeddings)
        count += len(records)

    if skip_existing and skipped_existing > 0:
        click.echo(f"\nImported {count} SEC officers/directors (skipped {skipped_existing} existing).", err=True)
    else:
        click.echo(f"\nImported {count} SEC officers/directors successfully.", err=True)
    click.echo("Run `corp-extractor db post-import` to update search indexes.", err=True)

    org_database.close()
    database.close()


@db_cmd.command("import-ch-officers")
@click.option("--file", "file_path", type=click.Path(exists=True), required=True, help="Path to CH officers zip file (Prod195)")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--limit", type=int, help="Limit number of records")
@click.option("--batch-size", type=int, default=1000, help="Batch size for commits (default: 1000)")
@click.option("--resume", is_flag=True, help="Resume from saved progress")
@click.option("--include-resigned", is_flag=True, help="Include resigned officers (default: current only)")
@click.option("--skip-existing", is_flag=True, help="Skip records that already exist")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_ch_officers(file_path: str, db_path: Optional[str], limit: Optional[int], batch_size: int, resume: bool, include_resigned: bool, skip_existing: bool, verbose: bool):
    """
    Import Companies House officers data into the people database.

    Requires the Prod195 bulk officers zip file from Companies House.
    Request access via BulkProducts@companieshouse.gov.uk.

    \b
    Examples:
        corp-extractor db import-ch-officers --file officers.zip --limit 10000
        corp-extractor db import-ch-officers --file officers.zip --resume
        corp-extractor db import-ch-officers --file officers.zip --include-resigned
    """
    _configure_logging(verbose)

    from .database.store import get_person_database, get_database
    from .database.embeddings import CompanyEmbedder
    from .database.importers.companies_house_officers import CompaniesHouseOfficersImporter

    # Default database path
    db_path_obj = _resolve_db_path(db_path)

    click.echo(f"Importing Companies House officers to {db_path_obj}...", err=True)
    if resume:
        click.echo("Resuming from saved progress...", err=True)

    # Initialize components (readonly=False for import operations)
    database = get_person_database(db_path=db_path_obj, readonly=False)
    org_database = get_database(db_path=db_path_obj, readonly=False)
    embedder = CompanyEmbedder()
    importer = CompaniesHouseOfficersImporter()

    # Import records in batches
    records = []
    count = 0
    skipped_existing = 0

    def progress_callback(file_idx: int, line_num: int, total: int) -> None:
        if verbose:
            click.echo(f"  File {file_idx}: line {line_num}, {total} records", err=True)

    for record in importer.import_from_zip(
        file_path,
        limit=limit,
        resume=resume,
        current_only=not include_resigned,
        progress_callback=progress_callback,
    ):
        # Skip existing records if flag is set
        if skip_existing:
            existing = database.get_by_source_id(record.source, record.source_id)
            if existing is not None:
                skipped_existing += 1
                continue

        # Look up org ID by company number if available
        company_number = record.record.get("company_number", "")
        if company_number:
            org_id = org_database.get_id_by_source_id("companies_house", company_number)
            if org_id is not None:
                record.known_for_org_id = org_id

        records.append(record)

        if len(records) >= batch_size:
            embedding_texts = [r.get_embedding_text() for r in records]
            embeddings, scalar_embeddings = embedder.embed_batch_and_quantize(embedding_texts)
            database.insert_batch(records, embeddings, scalar_embeddings=scalar_embeddings)
            count += len(records)
            click.echo(f"Imported {count} records...", err=True)
            records = []

    # Final batch
    if records:
        embedding_texts = [r.get_embedding_text() for r in records]
        embeddings, scalar_embeddings = embedder.embed_batch_and_quantize(embedding_texts)
        database.insert_batch(records, embeddings, scalar_embeddings=scalar_embeddings)
        count += len(records)

    if skip_existing and skipped_existing > 0:
        click.echo(f"\nImported {count} CH officers (skipped {skipped_existing} existing).", err=True)
    else:
        click.echo(f"\nImported {count} CH officers successfully.", err=True)
    click.echo("Run `corp-extractor db post-import` to update search indexes.", err=True)

    org_database.close()
    database.close()


@db_cmd.command("import-wikidata")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--limit", type=int, help="Limit number of records")
@click.option("--batch-size", type=int, default=1000, help="Batch size for commits (default: 1000)")
@click.option("--type", "query_type", type=click.Choice(["lei", "ticker", "public", "business", "organization", "nonprofit", "government"]), default="lei",
              help="Query type to use for fetching data")
@click.option("--all", "import_all", is_flag=True, help="Run all query types sequentially")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_wikidata(db_path: Optional[str], limit: Optional[int], batch_size: int, query_type: str, import_all: bool, verbose: bool):
    """
    Import organization data from Wikidata via SPARQL.

    Uses simplified SPARQL queries that avoid timeouts on Wikidata's endpoint.
    Query types target different organization categories.

    \b
    Query types:
        lei          Companies with LEI codes (fastest, most reliable)
        ticker       Companies listed on stock exchanges
        public       Direct instances of "public company" (Q891723)
        business     Direct instances of "business enterprise" (Q4830453)
        organization All organizations (Q43229) - NGOs, associations, etc.
        nonprofit    Non-profit organizations (Q163740)
        government   Government agencies (Q327333)

    \b
    Examples:
        corp-extractor db import-wikidata --limit 10
        corp-extractor db import-wikidata --type organization --limit 1000
        corp-extractor db import-wikidata --type nonprofit --limit 5000
        corp-extractor db import-wikidata --all --limit 10000
    """
    _configure_logging(verbose)

    from .database import OrganizationDatabase, CompanyEmbedder
    from .database.importers import WikidataImporter

    click.echo(f"Importing Wikidata organization data via SPARQL (type={query_type}, all={import_all})...", err=True)

    # Initialize components (readonly=False for import operations)
    db_path_obj = _resolve_db_path(db_path)
    embedder = CompanyEmbedder()
    database = OrganizationDatabase(db_path=db_path_obj, embedding_dim=embedder.embedding_dim, readonly=False)
    importer = WikidataImporter(batch_size=500)  # Smaller SPARQL batch size for reliability

    # Import records in batches
    records = []
    count = 0

    for record in importer.import_from_sparql(limit=limit, query_type=query_type, import_all=import_all):
        records.append(record)

        if len(records) >= batch_size:
            names = [r.name for r in records]
            embeddings, scalar_embeddings = embedder.embed_batch_and_quantize(names)
            database.insert_batch(records, embeddings, scalar_embeddings=scalar_embeddings)
            count += len(records)
            click.echo(f"Imported {count} records...", err=True)
            records = []

    # Final batch
    if records:
        names = [r.name for r in records]
        embeddings, scalar_embeddings = embedder.embed_batch_and_quantize(names)
        database.insert_batch(records, embeddings, scalar_embeddings=scalar_embeddings)
        count += len(records)

    click.echo(f"\nImported {count} Wikidata records successfully.", err=True)
    click.echo("Run `corp-extractor db post-import` to update search indexes.", err=True)
    database.close()


@db_cmd.command("import-people")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--limit", type=int, help="Limit number of records")
@click.option("--batch-size", type=int, default=1000, help="Batch size for commits (default: 1000)")
@click.option("--type", "query_type", type=click.Choice([
    "executive", "politician", "athlete", "artist",
    "academic", "scientist", "journalist", "entrepreneur", "activist"
]), default="executive", help="Person type to import")
@click.option("--all", "import_all", is_flag=True, help="Run all person type queries sequentially")
@click.option("--enrich", is_flag=True, help="Query individual people to get role/org data (slower, resumable)")
@click.option("--enrich-only", is_flag=True, help="Only enrich existing people (skip bulk import)")
@click.option("--enrich-dates", is_flag=True, help="Query individual people to get start/end dates (slower)")
@click.option("--skip-existing", is_flag=True, help="Skip records that already exist (default: update them)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_people(db_path: Optional[str], limit: Optional[int], batch_size: int, query_type: str, import_all: bool, enrich: bool, enrich_only: bool, enrich_dates: bool, skip_existing: bool, verbose: bool):
    """
    Import notable people data from Wikidata via SPARQL.

    Uses a two-phase approach for reliability:
    1. Bulk import: Fast fetch of QID, name, country (no timeouts)
    2. Enrich (optional): Per-person queries for role/org/dates

    Imports people with English Wikipedia articles (ensures notability).

    \b
    Examples:
        corp-extractor db import-people --type executive --limit 5000
        corp-extractor db import-people --all --limit 10000
        corp-extractor db import-people --type executive --enrich
        corp-extractor db import-people --enrich-only --limit 100
        corp-extractor db import-people --type politician -v
    """
    _configure_logging(verbose)

    from .database.store import get_person_database, get_database
    from .database.embeddings import CompanyEmbedder
    from .database.importers.wikidata_people import WikidataPeopleImporter

    # Default database path
    db_path_obj = _resolve_db_path(db_path)

    click.echo(f"Importing Wikidata people to {db_path_obj}...", err=True)

    # Initialize components (readonly=False for import operations)
    database = get_person_database(db_path=db_path_obj, readonly=False)
    org_database = get_database(db_path=db_path_obj, readonly=False)
    embedder = CompanyEmbedder()
    importer = WikidataPeopleImporter(batch_size=batch_size)

    count = 0

    # Phase 1: Bulk import (fast, minimal data) - skip if --enrich-only
    if not enrich_only:
        records = []
        skipped_existing = 0

        click.echo("Phase 1: Bulk import (QID, name, country)...", err=True)

        for record in importer.import_from_sparql(limit=limit, query_type=query_type, import_all=import_all):
            # Skip existing records if flag is set
            if skip_existing:
                existing = database.get_by_source_id(record.source, record.source_id)
                if existing is not None:
                    skipped_existing += 1
                    continue

            records.append(record)

            if len(records) >= batch_size:
                # Generate embeddings (both float32 and int8)
                embedding_texts = [r.get_embedding_text() for r in records]
                embeddings, scalar_embeddings = embedder.embed_batch_and_quantize(embedding_texts)
                database.insert_batch(records, embeddings, scalar_embeddings=scalar_embeddings)
                count += len(records)

                click.echo(f"  Imported {count} people...", err=True)
                records = []

        # Final batch
        if records:
            embedding_texts = [r.get_embedding_text() for r in records]
            embeddings, scalar_embeddings = embedder.embed_batch_and_quantize(embedding_texts)
            database.insert_batch(records, embeddings, scalar_embeddings=scalar_embeddings)
            count += len(records)

        if skip_existing and skipped_existing > 0:
            click.echo(f"\nPhase 1 complete: {count} people imported (skipped {skipped_existing} existing).", err=True)
        else:
            click.echo(f"\nPhase 1 complete: {count} people imported.", err=True)
    else:
        click.echo("Skipping Phase 1 (bulk import) - using existing database records.", err=True)
        # Enable enrich if enrich_only is set
        enrich = True

    # Phase 2: Enrich with role/org/dates (optional, slower but resumable)
    if enrich:
        click.echo("\nPhase 2: Enriching with role/org/dates (parallel queries)...", err=True)
        # Get all people without role/org
        people_to_enrich = []
        enriched_count = 0
        for record in database.iter_records():
            if not record.known_for_role and not record.known_for_org:
                people_to_enrich.append(record)
                enriched_count += 1
                # Apply limit if --enrich-only
                if enrich_only and limit and enriched_count >= limit:
                    break

        if people_to_enrich:
            click.echo(f"Found {len(people_to_enrich)} people to enrich...", err=True)
            importer.enrich_people_role_org_batch(people_to_enrich, delay_seconds=0.1, max_workers=5)

            # Persist the enriched data and re-generate embeddings
            updated = 0
            org_count = 0
            date_count = 0
            for person in people_to_enrich:
                if person.known_for_role or person.known_for_org:
                    # Look up org ID if we have org_qid
                    org_qid = person.record.get("org_qid", "")
                    if org_qid:
                        org_id = org_database.get_id_by_source_id("wikipedia", org_qid)
                        if org_id is not None:
                            person.known_for_org_id = org_id

                    # Update the record with new role/org/dates and re-embed
                    new_embedding_text = person.get_embedding_text()
                    new_embedding = embedder.embed(new_embedding_text)
                    if database.update_role_org(
                        person.source, person.source_id,
                        person.known_for_role, person.known_for_org,
                        person.known_for_org_id, new_embedding,
                        person.from_date, person.to_date,
                    ):
                        updated += 1
                        if person.known_for_org:
                            org_count += 1
                        if person.from_date or person.to_date:
                            date_count += 1
                        if verbose:
                            date_str = ""
                            if person.from_date or person.to_date:
                                date_str = f" ({person.from_date or '?'} - {person.to_date or '?'})"
                            click.echo(f"  {person.name}: {person.known_for_role} at {person.known_for_org}{date_str}", err=True)

            click.echo(f"Updated {updated} people ({org_count} with orgs, {date_count} with dates).", err=True)

    # Phase 3: Enrich with dates (optional, even slower)
    if enrich_dates:
        click.echo("\nPhase 3: Enriching with dates...", err=True)
        # Get all people without dates but with role (dates are associated with positions)
        people_to_enrich = []
        for record in database.iter_records():
            if not record.from_date and not record.to_date and record.known_for_role:
                people_to_enrich.append(record)

        if people_to_enrich:
            click.echo(f"Found {len(people_to_enrich)} people to enrich with dates...", err=True)
            enriched = importer.enrich_people_batch(people_to_enrich, delay_seconds=0.3)

            # Persist the enriched dates
            updated = 0
            for person in people_to_enrich:
                if person.from_date or person.to_date:
                    if database.update_dates(person.source, person.source_id, person.from_date, person.to_date):
                        updated += 1
                        if verbose:
                            click.echo(f"  {person.name}: {person.from_date or '?'} - {person.to_date or '?'}", err=True)

            click.echo(f"Updated {updated} people with dates.", err=True)

    click.echo("Run `corp-extractor db post-import` to update search indexes.", err=True)
    org_database.close()
    database.close()


class ImportBatch(NamedTuple):
    """A batch of records ready for embedding, produced by the reader thread."""
    record_type: str          # "people" or "org"
    records: list             # PersonRecord or CompanyRecord list
    embedding_texts: list[str]
    last_entity_index: int
    last_entity_id: str
    people_count: int         # cumulative people yielded so far
    orgs_count: int           # cumulative orgs yielded so far


def _reader_thread(
    importer,
    *,
    people: bool,
    orgs: bool,
    limit: Optional[int],
    require_enwiki: bool,
    skip_updates: bool,
    existing_people_ids: set[str],
    existing_org_ids: set[str],
    start_index: int,
    batch_size: int,
    embed_queue: queue.Queue,
    shutdown_event: threading.Event,
    thread_errors: list[Exception],
) -> None:
    """Reader thread: iterates the dump, accumulates batches, puts them on embed_queue."""
    import logging
    logger = logging.getLogger(__name__)

    people_records: list = []
    org_records: list = []
    last_entity_index = start_index
    last_entity_id = ""
    people_yielded = 0
    orgs_yielded = 0

    def progress_callback(entity_index: int, entity_id: str, ppl_count: int, org_count: int) -> None:
        nonlocal last_entity_index, last_entity_id
        last_entity_index = entity_index
        last_entity_id = entity_id

    def flush_people() -> None:
        nonlocal people_records, people_yielded
        if people_records and not shutdown_event.is_set():
            texts = [r.get_embedding_text() for r in people_records]
            people_yielded += len(people_records)
            batch = ImportBatch(
                record_type="people",
                records=list(people_records),
                embedding_texts=texts,
                last_entity_index=last_entity_index,
                last_entity_id=last_entity_id,
                people_count=people_yielded,
                orgs_count=orgs_yielded,
            )
            embed_queue.put(batch)
            people_records = []

    def flush_orgs() -> None:
        nonlocal org_records, orgs_yielded
        if org_records and not shutdown_event.is_set():
            texts = [r.name for r in org_records]
            orgs_yielded += len(org_records)
            batch = ImportBatch(
                record_type="org",
                records=list(org_records),
                embedding_texts=texts,
                last_entity_index=last_entity_index,
                last_entity_id=last_entity_id,
                people_count=people_yielded,
                orgs_count=orgs_yielded,
            )
            embed_queue.put(batch)
            org_records = []

    try:
        for record_type, record in importer.import_all(
            people_limit=limit if people else 0,
            orgs_limit=limit if orgs else 0,
            import_people=people,
            import_orgs=orgs,
            require_enwiki=require_enwiki,
            skip_people_ids=existing_people_ids if skip_updates else None,
            skip_org_ids=existing_org_ids if skip_updates else None,
            start_index=start_index,
            progress_callback=progress_callback,
        ):
            if shutdown_event.is_set():
                logger.info("Reader thread: shutdown requested, stopping")
                break

            if record_type == "person":
                people_records.append(record)
                if len(people_records) >= batch_size:
                    flush_people()
            else:  # org
                org_records.append(record)
                if len(org_records) >= batch_size:
                    flush_orgs()

        # Flush remaining partial batches
        flush_people()
        flush_orgs()

    except Exception as e:
        thread_errors.append(e)
    finally:
        # Sentinel: reader is done
        embed_queue.put(None)


def _embedder_thread(
    embedder,
    *,
    embed_queue: queue.Queue,
    result_queue: queue.Queue,
    shutdown_event: threading.Event,
    thread_errors: list[Exception],
) -> None:
    """Embedder thread: consumes ImportBatch, produces (batch, embeddings, scalar_embeddings)."""
    import logging
    logger = logging.getLogger(__name__)

    try:
        while True:
            batch = embed_queue.get()
            if batch is None:
                # Reader is done
                break
            if shutdown_event.is_set():
                logger.info("Embedder thread: shutdown requested, stopping")
                break

            embeddings, scalar_embeddings = embedder.embed_batch_and_quantize(batch.embedding_texts)
            result_queue.put((batch, embeddings, scalar_embeddings))

    except Exception as e:
        thread_errors.append(e)
    finally:
        # Sentinel: embedder is done
        result_queue.put(None)


@db_cmd.command("import-wikidata-dump")
@click.option("--dump", "dump_path", type=click.Path(exists=True), help="Path to Wikidata JSON dump file (.bz2 or .gz)")
@click.option("--download", is_flag=True, help="Download latest dump first (~100GB)")
@click.option("--force", is_flag=True, help="Force re-download even if cached")
@click.option("--no-aria2", is_flag=True, help="Don't use aria2c even if available (slower)")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--people/--no-people", default=True, help="Import people (default: yes)")
@click.option("--orgs/--no-orgs", default=True, help="Import organizations (default: yes)")
@click.option("--locations/--no-locations", default=False, help="Import locations (default: no)")
@click.option("--require-enwiki", is_flag=True, help="Only import orgs with English Wikipedia articles")
@click.option("--resume", is_flag=True, help="Resume from last position in dump file (tracks entity index)")
@click.option("--skip-updates", is_flag=True, help="Skip Q codes already in database (no updates)")
@click.option("--limit", type=int, help="Max records per type (people and/or orgs)")
@click.option("--batch-size", type=int, default=10000, help="Batch size for commits (default: 10000)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_wikidata_dump(
    dump_path: Optional[str],
    download: bool,
    force: bool,
    no_aria2: bool,
    db_path: Optional[str],
    people: bool,
    orgs: bool,
    locations: bool,
    require_enwiki: bool,
    resume: bool,
    skip_updates: bool,
    limit: Optional[int],
    batch_size: int,
    verbose: bool,
):
    """
    Import people, organizations, and locations from Wikidata JSON dump.

    This uses the full Wikidata JSON dump (~100GB compressed) to import
    all humans and organizations with English Wikipedia articles. This
    avoids SPARQL query timeouts that occur with large result sets.

    The dump is streamed line-by-line to minimize memory usage.

    \b
    Features:
    - No timeouts (processes locally)
    - Complete coverage (all notable people/orgs)
    - Resumable with --resume (tracks position in dump file)
    - Skip existing with --skip-updates (loads existing Q codes)
    - People like Andy Burnham are captured via occupation (P106)
    - Locations (countries, cities, regions) with parent hierarchy

    \b
    Resume options:
    - --resume: Resume from where the dump processing left off (tracks entity index).
                Progress is saved after each batch. Use this if import was interrupted.
    - --skip-updates: Skip Q codes already in database (no updates to existing records).
                      Use this to add new records without re-processing existing ones.

    \b
    Examples:
        corp-extractor db import-wikidata-dump --dump /path/to/dump.json.bz2 --limit 10000
        corp-extractor db import-wikidata-dump --download --people --no-orgs --limit 50000
        corp-extractor db import-wikidata-dump --dump dump.json.bz2 --orgs --no-people
        corp-extractor db import-wikidata-dump --dump dump.json.bz2 --locations --no-people --no-orgs  # Locations only
        corp-extractor db import-wikidata-dump --dump dump.json.bz2 --resume  # Resume interrupted import
        corp-extractor db import-wikidata-dump --dump dump.json.bz2 --skip-updates  # Skip existing Q codes
    """
    _configure_logging(verbose)

    from .database.store import get_person_database, get_database, get_locations_database
    from .database.embeddings import CompanyEmbedder
    from .database.importers.wikidata_dump import WikidataDumpImporter, DumpProgress

    if not dump_path and not download:
        raise click.UsageError("Either --dump path or --download is required")

    if not people and not orgs and not locations:
        raise click.UsageError("Must import at least one of --people, --orgs, or --locations")

    # Default database path
    db_path_obj = _resolve_db_path(db_path)

    click.echo(f"Importing Wikidata dump to {db_path_obj}...", err=True)

    # Initialize importer
    importer = WikidataDumpImporter(dump_path=dump_path)

    # Download if requested
    if download:
        import shutil
        dump_target = importer.get_dump_path()
        click.echo(f"Downloading Wikidata dump (~100GB) to:", err=True)
        click.echo(f"  {dump_target}", err=True)

        # Check for aria2c
        has_aria2 = shutil.which("aria2c") is not None
        use_aria2 = has_aria2 and not no_aria2

        if use_aria2:
            click.echo("  Using aria2c for fast parallel download (16 connections)", err=True)
            dump_file = importer.download_dump(force=force, use_aria2=True)
            click.echo(f"\nUsing dump: {dump_file}", err=True)
        else:
            if not has_aria2:
                click.echo("", err=True)
                click.echo("  TIP: Install aria2c for 10-20x faster downloads:", err=True)
                click.echo("       brew install aria2  (macOS)", err=True)
                click.echo("       apt install aria2   (Ubuntu/Debian)", err=True)
                click.echo("", err=True)

            # Use urllib to get content length first
            import urllib.request
            req = urllib.request.Request(
                "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2",
                headers={"User-Agent": "corp-extractor/1.0"},
                method="HEAD"
            )
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get("content-length", 0))

            if total_size:
                total_gb = total_size / (1024 ** 3)
                click.echo(f"  Size: {total_gb:.1f} GB", err=True)

            # Download with progress bar
            progress_bar = None

            def update_progress(downloaded: int, total: int) -> None:
                nonlocal progress_bar
                if progress_bar is None and total > 0:
                    progress_bar = click.progressbar(
                        length=total,
                        label="Downloading",
                        show_percent=True,
                        show_pos=True,
                        item_show_func=lambda x: f"{(x or 0) / (1024**3):.1f} GB" if x else "",
                    )
                    progress_bar.__enter__()
                if progress_bar:
                    # Update to absolute position
                    progress_bar.update(downloaded - progress_bar.pos)

            try:
                dump_file = importer.download_dump(force=force, use_aria2=False, progress_callback=update_progress)
            finally:
                if progress_bar:
                    progress_bar.__exit__(None, None, None)

            click.echo(f"\nUsing dump: {dump_file}", err=True)
    elif dump_path:
        click.echo(f"Using dump: {dump_path}", err=True)

    # Initialize embedder (loads model, may take time on first run)
    click.echo("Loading embedding model...", err=True)
    sys.stderr.flush()
    embedder = CompanyEmbedder()
    click.echo("Embedding model loaded.", err=True)
    sys.stderr.flush()

    # Load existing QID labels from database and seed the importer's cache
    # readonly=False because we also write new labels via persist_new_labels()
    database = get_person_database(db_path=db_path_obj, readonly=False)
    existing_labels = database.get_all_qid_labels()
    if existing_labels:
        click.echo(f"Loaded {len(existing_labels):,} existing QID labels from DB", err=True)
        importer.set_label_cache(existing_labels)
    known_qids_at_start = set(existing_labels.keys())

    # Load existing source_ids for skip_updates mode
    existing_people_ids: set[str] = set()
    existing_org_ids: set[str] = set()
    if skip_updates:
        click.echo("Loading existing records for --skip-updates...", err=True)
        if people:
            existing_people_ids = database.get_all_source_ids(source="wikidata")
            click.echo(f"  Found {len(existing_people_ids):,} existing people Q codes", err=True)
        if orgs:
            # readonly=False because we also write later via insert_batch
            org_database = get_database(db_path=db_path_obj, readonly=False)
            existing_org_ids = org_database.get_all_source_ids(source="wikipedia")
            click.echo(f"  Found {len(existing_org_ids):,} existing org Q codes", err=True)

    # Load progress for resume mode (position-based resume)
    progress: Optional[DumpProgress] = None
    start_index = 0
    if resume:
        progress = DumpProgress.load()
        if progress:
            # Verify the progress is for the same dump file
            actual_dump_path = importer._dump_path or Path(dump_path) if dump_path else importer.get_dump_path()
            if progress.matches_dump(actual_dump_path):
                start_index = progress.entity_index
                click.echo(f"Resuming from entity index {start_index:,}", err=True)
                click.echo(f"  Last entity: {progress.last_entity_id}", err=True)
                click.echo(f"  Last updated: {progress.last_updated}", err=True)
            else:
                click.echo("Warning: Progress file is for a different dump, starting from beginning", err=True)
                progress = None
        else:
            click.echo("No progress file found, starting from beginning", err=True)

    # Initialize progress tracking
    if progress is None:
        actual_dump_path = importer._dump_path or Path(dump_path) if dump_path else importer.get_dump_path()
        progress = DumpProgress(
            dump_path=str(actual_dump_path),
            dump_size=actual_dump_path.stat().st_size if actual_dump_path.exists() else 0,
        )

    # Helper to persist new labels after each batch
    def persist_new_labels() -> int:
        new_labels = importer.get_new_labels_since(known_qids_at_start)
        if new_labels:
            database.insert_qid_labels(new_labels)
            known_qids_at_start.update(new_labels.keys())
            return len(new_labels)
        return 0

    # ========================================
    # Location-only import (separate pass)
    # ========================================
    if locations and not people and not orgs:
        from .database.store import get_locations_database

        click.echo("\n=== Location Import ===", err=True)
        click.echo(f"  Locations: {'up to ' + str(limit) + ' records' if limit else 'unlimited'}", err=True)
        if require_enwiki:
            click.echo("    Filter: only locations with English Wikipedia articles", err=True)

        # Initialize locations database (readonly=False for import operations)
        locations_database = get_locations_database(db_path=db_path_obj, readonly=False)

        # Load existing location Q codes for skip_updates mode
        existing_location_ids: set[str] = set()
        if skip_updates:
            existing_location_ids = locations_database.get_all_source_ids(source="wikidata")
            click.echo(f"    Skip updates: {len(existing_location_ids):,} existing Q codes", err=True)

        if start_index > 0:
            click.echo(f"  Resuming from entity index {start_index:,}", err=True)

        location_records: list = []
        locations_count = 0
        last_entity_index = start_index
        last_entity_id = ""

        def location_progress_callback(entity_index: int, entity_id: str, loc_count: int) -> None:
            nonlocal last_entity_index, last_entity_id
            last_entity_index = entity_index
            last_entity_id = entity_id

        def save_location_progress() -> None:
            if progress:
                progress.entity_index = last_entity_index
                progress.last_entity_id = last_entity_id
                progress.save()

        def flush_location_batch() -> None:
            nonlocal location_records, locations_count
            if location_records:
                inserted = locations_database.insert_batch(location_records)
                locations_count += inserted
                location_records = []

        click.echo("Starting dump iteration...", err=True)
        sys.stderr.flush()

        try:
            if limit:
                # Use progress bar when we have limits
                with click.progressbar(
                    length=limit,
                    label="Processing dump",
                    show_percent=True,
                    show_pos=True,
                ) as pbar:
                    for record in importer.import_locations(
                        limit=limit,
                        require_enwiki=require_enwiki,
                        skip_ids=existing_location_ids if skip_updates else None,
                        start_index=start_index,
                        progress_callback=location_progress_callback,
                    ):
                        pbar.update(1)
                        location_records.append(record)
                        if len(location_records) >= batch_size:
                            flush_location_batch()
                            persist_new_labels()
                            save_location_progress()
            else:
                # No limit - show counter updates
                for record in importer.import_locations(
                    limit=None,
                    require_enwiki=require_enwiki,
                    skip_ids=existing_location_ids if skip_updates else None,
                    start_index=start_index,
                    progress_callback=location_progress_callback,
                ):
                    location_records.append(record)
                    if len(location_records) >= batch_size:
                        flush_location_batch()
                        persist_new_labels()
                        save_location_progress()
                        click.echo(f"\r  Progress: {locations_count:,} locations...", nl=False, err=True)
                        sys.stderr.flush()

                click.echo("", err=True)  # Newline after counter

            # Final batches
            flush_location_batch()
            persist_new_labels()
            save_location_progress()

        finally:
            # Ensure we save progress even on interrupt
            save_location_progress()

        click.echo(f"\nLocation import complete: {locations_count:,} locations", err=True)

        # Final label resolution
        click.echo("\n=== Final QID Label Resolution ===", err=True)
        all_labels = importer.get_label_cache()
        click.echo(f"  Total labels in cache: {len(all_labels):,}", err=True)

        # Final stats
        final_label_count = database.get_qid_labels_count()
        click.echo(f"  Total labels in DB: {final_label_count:,}", err=True)

        locations_database.close()
        database.close()
        click.echo("\nWikidata dump import complete!", err=True)
        return

    # Combined import - single pass through the dump for both people and orgs
    click.echo("\n=== Combined Import (single dump pass) ===", err=True)
    sys.stderr.flush()  # Ensure output is visible immediately
    if people:
        click.echo(f"  People: {'up to ' + str(limit) + ' records' if limit else 'unlimited'}", err=True)
        if skip_updates and existing_people_ids:
            click.echo(f"    Skip updates: {len(existing_people_ids):,} existing Q codes", err=True)
    if orgs:
        click.echo(f"  Orgs: {'up to ' + str(limit) + ' records' if limit else 'unlimited'}", err=True)
        if require_enwiki:
            click.echo("    Filter: only orgs with English Wikipedia articles", err=True)
        if skip_updates and existing_org_ids:
            click.echo(f"    Skip updates: {len(existing_org_ids):,} existing Q codes", err=True)
    if start_index > 0:
        click.echo(f"  Resuming from entity index {start_index:,}", err=True)

    # Initialize databases (readonly=False for import operations)
    person_database = get_person_database(db_path=db_path_obj, readonly=False)
    org_database = get_database(db_path=db_path_obj, readonly=False) if orgs else None

    # Pipeline: reader thread → embed_queue → embedder thread → result_queue → main thread (DB writes)
    embed_queue: queue.Queue = queue.Queue(maxsize=2)
    result_queue: queue.Queue = queue.Queue(maxsize=2)
    shutdown_event = threading.Event()
    thread_errors: list[Exception] = []

    people_count = 0
    orgs_count = 0
    last_entity_index = start_index
    last_entity_id = ""

    def save_progress() -> None:
        if progress:
            progress.entity_index = last_entity_index
            progress.last_entity_id = last_entity_id
            progress.people_yielded = people_count
            progress.orgs_yielded = orgs_count
            progress.save()

    click.echo("Starting parallel pipeline (reader → embedder → writer)...", err=True)
    sys.stderr.flush()

    reader = threading.Thread(
        target=_reader_thread,
        args=(importer,),
        kwargs=dict(
            people=people,
            orgs=orgs,
            limit=limit,
            require_enwiki=require_enwiki,
            skip_updates=skip_updates,
            existing_people_ids=existing_people_ids,
            existing_org_ids=existing_org_ids,
            start_index=start_index,
            batch_size=batch_size,
            embed_queue=embed_queue,
            shutdown_event=shutdown_event,
            thread_errors=thread_errors,
        ),
        daemon=True,
        name="wikidata-reader",
    )
    embedder_thread = threading.Thread(
        target=_embedder_thread,
        args=(embedder,),
        kwargs=dict(
            embed_queue=embed_queue,
            result_queue=result_queue,
            shutdown_event=shutdown_event,
            thread_errors=thread_errors,
        ),
        daemon=True,
        name="wikidata-embedder",
    )

    reader.start()
    embedder_thread.start()

    try:
        while True:
            # Check for thread errors before blocking
            if thread_errors:
                raise thread_errors[0]

            result = result_queue.get()
            if result is None:
                # Embedder is done — all batches processed
                break

            batch, embeddings, scalar_embeddings = result

            # Insert into database (main thread owns SQLite writes)
            if batch.record_type == "people":
                person_database.insert_batch(batch.records, embeddings, scalar_embeddings=scalar_embeddings)
            elif batch.record_type == "org" and org_database:
                org_database.insert_batch(batch.records, embeddings, scalar_embeddings=scalar_embeddings)

            # Update progress from batch metadata
            people_count = batch.people_count
            orgs_count = batch.orgs_count
            last_entity_index = batch.last_entity_index
            last_entity_id = batch.last_entity_id

            persist_new_labels()
            save_progress()

            click.echo(f"\r  Progress: {people_count:,} people, {orgs_count:,} orgs...", nl=False, err=True)
            sys.stderr.flush()

        click.echo("", err=True)  # Newline after counter

        # Check for any errors that occurred after the sentinel
        if thread_errors:
            raise thread_errors[0]

    except KeyboardInterrupt:
        click.echo("\n  Interrupted! Saving progress...", err=True)
        shutdown_event.set()
        # Drain any completed batches from result_queue so we don't lose work
        while True:
            try:
                result = result_queue.get_nowait()
                if result is None:
                    break
                batch, embeddings, scalar_embeddings = result
                if batch.record_type == "people":
                    person_database.insert_batch(batch.records, embeddings, scalar_embeddings=scalar_embeddings)
                elif batch.record_type == "org" and org_database:
                    org_database.insert_batch(batch.records, embeddings, scalar_embeddings=scalar_embeddings)
                people_count = batch.people_count
                orgs_count = batch.orgs_count
                last_entity_index = batch.last_entity_index
                last_entity_id = batch.last_entity_id
            except queue.Empty:
                break
        persist_new_labels()
        save_progress()
        click.echo(f"  Saved: {people_count:,} people, {orgs_count:,} orgs", err=True)
        raise
    finally:
        save_progress()
        reader.join(timeout=5)
        embedder_thread.join(timeout=5)

    click.echo(f"Import complete: {people_count:,} people, {orgs_count:,} orgs", err=True)

    # Backfill known_for_org from reverse org→person mappings
    # (P169 CEO, P488 chairperson, P112 founder, P1037 director, P3320 board member)
    reverse_map = importer.get_reverse_person_orgs()
    if reverse_map and people:
        # Resolve org QIDs to labels before backfilling
        label_cache = importer.get_label_cache()
        resolved_map: dict[str, list[tuple[str, str, str | None, str | None]]] = {}
        for person_qid, entries in reverse_map.items():
            resolved_entries: list[tuple[str, str, str | None, str | None]] = []
            for org_qid, role, start_date, end_date in entries:
                org_label = label_cache.get(org_qid, "")
                if org_label:
                    resolved_entries.append((org_label, role, start_date, end_date))
            if resolved_entries:
                resolved_map[person_qid] = resolved_entries
        if resolved_map:
            backfilled = person_database.backfill_known_for_org(resolved_map)
            click.echo(
                f"Backfilled known_for_org from org executive properties: "
                f"{backfilled:,} updated ({len(resolved_map):,} reverse mappings)",
                err=True,
            )

    # Keep references for final label resolution
    database = person_database
    if org_database:
        org_database.close()

    # Final label resolution pass for any remaining unresolved QIDs
    click.echo("\n=== Final QID Label Resolution ===", err=True)

    # Get the full label cache (includes labels from DB + new ones from import)
    all_labels = importer.get_label_cache()
    click.echo(f"  Total labels in cache: {len(all_labels):,}", err=True)

    # Check for any remaining unresolved QIDs in the database
    people_unresolved = database.get_unresolved_qids()
    click.echo(f"  Unresolved QIDs in people: {len(people_unresolved):,}", err=True)

    org_unresolved: set[str] = set()
    if orgs:
        # readonly=False because we also call resolve_qid_labels later
        org_database = get_database(db_path=db_path_obj, readonly=False)
        org_unresolved = org_database.get_unresolved_qids()
        click.echo(f"  Unresolved QIDs in orgs: {len(org_unresolved):,}", err=True)

    all_unresolved = people_unresolved | org_unresolved
    need_sparql = all_unresolved - set(all_labels.keys())

    if need_sparql:
        click.echo(f"  Resolving {len(need_sparql):,} remaining QIDs via SPARQL...", err=True)
        sparql_resolved = importer.resolve_qids_via_sparql(need_sparql)
        all_labels.update(sparql_resolved)
        # Persist newly resolved labels
        if sparql_resolved:
            database.insert_qid_labels(sparql_resolved)
            click.echo(f"  SPARQL resolved and stored: {len(sparql_resolved):,}", err=True)

    # Update records with any newly resolved labels
    if all_labels:
        updates, deletes = database.resolve_qid_labels(all_labels)
        if updates or deletes:
            click.echo(f"  People: {updates:,} updated, {deletes:,} duplicates deleted", err=True)

        if orgs:
            org_database = get_database(db_path=db_path_obj, readonly=False)
            org_updates, org_deletes = org_database.resolve_qid_labels(all_labels)
            if org_updates or org_deletes:
                click.echo(f"  Orgs: {org_updates:,} updated, {org_deletes:,} duplicates deleted", err=True)
            org_database.close()

    # Final stats
    final_label_count = database.get_qid_labels_count()
    click.echo(f"  Total labels in DB: {final_label_count:,}", err=True)

    # Run canonicalization to link equivalent records across sources
    click.echo("\n=== Canonicalization ===", err=True)
    if people:
        people_result = person_database.canonicalize()
        click.echo(
            f"  People: {people_result['canonical_groups']:,} groups, "
            f"{people_result['matched_by_org']:,} by org, "
            f"{people_result['matched_by_date']:,} by date",
            err=True,
        )
    if orgs:
        canon_org_database = get_database(db_path=db_path_obj, readonly=False)
        org_result = canon_org_database.canonicalize()
        click.echo(
            f"  Orgs: {org_result.get('groups_found', 0):,} groups",
            err=True,
        )
        canon_org_database.close()

    database.close()

    click.echo("\nWikidata dump import complete!", err=True)
    click.echo("Run `corp-extractor db post-import` to update search indexes.", err=True)


@db_cmd.command("search-people")
@click.argument("query")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--top-k", type=int, default=10, help="Number of results")
@click.option("--hybrid", is_flag=True, help="Use hybrid text + embeddings search (default is embeddings-only)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_search_people(query: str, db_path: Optional[str], top_k: int, hybrid: bool, verbose: bool):
    """
    Search for a person in the database.

    \b
    Examples:
        corp-extractor db search-people "Tim Cook"
        corp-extractor db search-people "Elon Musk" --top-k 5
        corp-extractor db search-people "Elon Musk" --hybrid
    """
    _configure_logging(verbose)

    from .database.store import get_person_database
    from .database.embeddings import CompanyEmbedder

    # Default database path
    db_path_obj = _resolve_db_path(db_path)

    mode = "hybrid (text + embeddings)" if hybrid else "embeddings-only"
    click.echo(f"Searching for '{query}' in {db_path_obj} [{mode}]...", err=True)

    # Initialize components
    database = get_person_database(db_path=db_path_obj)
    embedder = CompanyEmbedder()

    # Embed query and search
    query_embedding = embedder.embed(query)
    query_text = query if hybrid else None
    results = database.search(query_embedding, top_k=top_k, query_text=query_text)

    if not results:
        click.echo("No results found.", err=True)
        return

    click.echo(f"\nFound {len(results)} results:\n")
    for i, (record, similarity) in enumerate(results, 1):
        role_str = f" ({record.known_for_role})" if record.known_for_role else ""
        org_str = f" at {record.known_for_org}" if record.known_for_org else ""
        country_str = f" [{record.country}]" if record.country else ""
        click.echo(f"  {i}. {record.name}{role_str}{org_str}{country_str}")
        click.echo(f"     Source: wikidata:{record.source_id}, Type: {record.person_type.value}, Score: {similarity:.3f}")
        click.echo()

    database.close()


@db_cmd.command("search-people-perf-test")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--top-k", type=int, default=5, help="Number of results per query")
@click.option("--hybrid", is_flag=True, help="Use hybrid text + embeddings search")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_search_people_perf_test(db_path: Optional[str], top_k: int, hybrid: bool, verbose: bool):
    """
    Run 20 sequential person searches to measure warm-up and steady-state latency.

    \b
    Examples:
        corp-extractor db search-people-perf-test
        corp-extractor db search-people-perf-test --hybrid
    """
    import time as _time

    _configure_logging(verbose)

    from .database.store import get_person_database
    from .database.embeddings import CompanyEmbedder

    test_queries = [
        "Tim Cook, CEO, Apple",
        "Satya Nadella, CEO, Microsoft",
        "Andy Jassy, CEO, Amazon",
        "Sundar Pichai, CEO, Google",
        "Jensen Huang, CEO, NVIDIA",
        "Mark Zuckerberg, CEO, Meta",
        "Jamie Dimon, CEO, JPMorgan Chase",
        "Warren Buffett, CEO, Berkshire Hathaway",
        "Elon Musk, CEO, Tesla",
        "Lisa Su, CEO, AMD",
        "Mary Barra, CEO, General Motors",
        "David Solomon, CEO, Goldman Sachs",
        "Brian Moynihan, CEO, Bank of America",
        "Arvind Krishna, CEO, IBM",
        "Pat Gelsinger, CEO, Intel",
        "Chuck Robbins, CEO, Cisco",
        "Safra Catz, CEO, Oracle",
        "Shantanu Narayen, CEO, Adobe",
        "Marc Benioff, CEO, Salesforce",
        "Darius Adamczyk, CEO, Honeywell",
    ]

    db_path_obj = _resolve_db_path(db_path)

    mode = "hybrid" if hybrid else "embeddings-only"
    click.echo(f"Person search perf test [{mode}] — {len(test_queries)} queries, top_k={top_k}", err=True)
    click.echo(f"Database: {db_path_obj}", err=True)
    click.echo("-" * 70, err=True)

    database = get_person_database(db_path=db_path_obj)
    embedder = CompanyEmbedder()

    timings: list[tuple[str, float, int]] = []

    for i, query in enumerate(test_queries, 1):
        t0 = _time.perf_counter()
        query_embedding = embedder.embed(query)
        embed_elapsed = _time.perf_counter() - t0

        t1 = _time.perf_counter()
        query_text = query if hybrid else None
        results = database.search(query_embedding, top_k=top_k, query_text=query_text)
        search_elapsed = _time.perf_counter() - t1

        total_elapsed = _time.perf_counter() - t0
        n_results = len(results)
        timings.append((query, total_elapsed, n_results))

        top_match = results[0][0].name if results else "—"
        top_score = f"{results[0][1]:.3f}" if results else "—"
        click.echo(
            f"  {i:2d}. {total_elapsed * 1000:7.1f}ms  "
            f"(embed {embed_elapsed * 1000:.0f}ms + search {search_elapsed * 1000:.0f}ms)  "
            f"hits={n_results:<3d}  "
            f"top: {top_match} ({top_score})"
        )

    click.echo("-" * 70, err=True)
    durations = [t for _, t, _ in timings]
    click.echo(
        f"Total: {sum(durations):.2f}s  |  "
        f"Mean: {sum(durations) / len(durations) * 1000:.1f}ms  |  "
        f"Min: {min(durations) * 1000:.1f}ms  |  "
        f"Max: {max(durations) * 1000:.1f}ms",
        err=True,
    )

    database.close()


@db_cmd.command("import-companies-house")
@click.option("--download", is_flag=True, help="Download bulk data file (free, no API key needed)")
@click.option("--force", is_flag=True, help="Force re-download even if cached")
@click.option("--file", "file_path", type=click.Path(exists=True), help="Local Companies House CSV/JSON file")
@click.option("--search", "search_terms", type=str, help="Comma-separated search terms (requires API key)")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--limit", type=int, help="Limit number of records")
@click.option("--batch-size", type=int, default=50000, help="Batch size for commits (default: 50000)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_companies_house(
    download: bool,
    force: bool,
    file_path: Optional[str],
    search_terms: Optional[str],
    db_path: Optional[str],
    limit: Optional[int],
    batch_size: int,
    verbose: bool,
):
    """
    Import UK Companies House data into the entity database.

    \b
    Options:
    --download    Download free bulk data (all UK companies, ~5M records)
    --file        Import from local CSV/JSON file
    --search      Search via API (requires COMPANIES_HOUSE_API_KEY)

    \b
    Examples:
        corp-extractor db import-companies-house --download
        corp-extractor db import-companies-house --download --limit 100000
        corp-extractor db import-companies-house --file /path/to/companies.csv
        corp-extractor db import-companies-house --search "bank,insurance"
    """
    _configure_logging(verbose)

    from .database import OrganizationDatabase, CompanyEmbedder
    from .database.importers import CompaniesHouseImporter

    if not file_path and not search_terms and not download:
        raise click.UsageError("Either --download, --file, or --search is required")

    click.echo("Importing Companies House data...", err=True)

    # Initialize components (readonly=False for import operations)
    db_path_obj = _resolve_db_path(db_path)
    embedder = CompanyEmbedder()
    database = OrganizationDatabase(db_path=db_path_obj, embedding_dim=embedder.embedding_dim, readonly=False)
    importer = CompaniesHouseImporter()

    # Get records
    if download:
        # Download bulk data file
        csv_path = importer.download_bulk_data(force=force)
        click.echo(f"Using bulk data file: {csv_path}", err=True)
        record_iter = importer.import_from_file(csv_path, limit=limit)
    elif file_path:
        record_iter = importer.import_from_file(file_path, limit=limit)
    else:
        terms = [t.strip() for t in search_terms.split(",") if t.strip()]
        click.echo(f"Searching for: {terms}", err=True)
        record_iter = importer.import_from_search(
            search_terms=terms,
            limit_per_term=limit or 100,
            total_limit=limit,
        )

    # Import records in batches
    records = []
    count = 0

    for record in record_iter:
        records.append(record)

        if len(records) >= batch_size:
            names = [r.name for r in records]
            embeddings, scalar_embeddings = embedder.embed_batch_and_quantize(names)
            database.insert_batch(records, embeddings, scalar_embeddings=scalar_embeddings)
            count += len(records)
            click.echo(f"Imported {count} records...", err=True)
            records = []

    # Final batch
    if records:
        names = [r.name for r in records]
        embeddings, scalar_embeddings = embedder.embed_batch_and_quantize(names)
        database.insert_batch(records, embeddings, scalar_embeddings=scalar_embeddings)
        count += len(records)

    click.echo(f"\nImported {count} Companies House records successfully.", err=True)
    click.echo("Run `corp-extractor db post-import` to update search indexes.", err=True)
    database.close()


@db_cmd.command("status")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--for-llm", is_flag=True, help="Output schema and type details for LLM documentation")
def db_status(db_path: Optional[str], for_llm: bool):
    """
    Show database status and statistics.

    \b
    Examples:
        corp-extractor db status
        corp-extractor db status --for-llm
        corp-extractor db status --db /path/to/entities.db
    """
    import sqlite3

    from .database import OrganizationDatabase
    from .database.hub import DEFAULT_DB_FILENAME, DEFAULT_DB_FULL_FILENAME, DEFAULT_DB_LITE_FILENAME
    from .database.store import get_locations_database, get_person_database, get_roles_database

    db_path_obj = _resolve_db_path(db_path)

    try:
        database = OrganizationDatabase(db_path=db_path_obj)
        stats = database.get_stats()

        click.echo("\nEntity Database Status")
        click.echo("=" * 40)
        click.echo(f"Total organizations: {stats.total_records:,}")
        click.echo(f"Embedding dimension: {stats.embedding_dimension}")
        click.echo(f"Database size: {stats.database_size_bytes / 1024 / 1024:.2f} MB")

        # Get person stats
        person_db = get_person_database(db_path=db_path_obj)
        person_stats = person_db.get_stats()
        click.echo(f"Total people: {person_stats.get('total_records', 0):,}")

        if stats.by_source:
            click.echo("\n=== Organizations by Source ===")
            click.echo(f"{'Source':<20} {'Records':>15}")
            click.echo("-" * 36)
            for source, count in sorted(stats.by_source.items(), key=lambda x: -x[1]):
                click.echo(f"{source:<20} {count:>15,}")

        # People by source
        if person_stats.get("by_source"):
            click.echo("\n=== People by Source ===")
            click.echo(f"{'Source':<20} {'Records':>15}")
            click.echo("-" * 36)
            for source, count in sorted(person_stats["by_source"].items(), key=lambda x: -x[1]):
                click.echo(f"{source:<20} {count:>15,}")

        # Roles and Locations counts
        try:
            roles_db = get_roles_database(db_path=db_path_obj)
            roles_stats = roles_db.get_stats()
            locations_db = get_locations_database(db_path=db_path_obj)
            locations_stats = locations_db.get_stats()

            click.echo("\n=== Other Tables ===")
            click.echo(f"{'Table':<20} {'Records':>15}")
            click.echo("-" * 36)
            click.echo(f"{'roles':<20} {roles_stats['total_roles']:>15,}")
            click.echo(f"{'locations':<20} {locations_stats['total_locations']:>15,}")
        except Exception:
            pass  # Tables may not exist in older databases

        # For LLM mode: output enum tables and schema details
        if for_llm:
            db_file = str(db_path_obj)
            conn = sqlite3.connect(f"file:{db_file}?immutable=1", uri=True)
            conn.row_factory = sqlite3.Row

            click.echo("\n" + "=" * 60)
            click.echo("LLM DOCUMENTATION DETAILS")
            click.echo("=" * 60)

            # Database file variants
            click.echo("\n=== Database File Variants ===")
            click.echo(f"Default filename: {DEFAULT_DB_FILENAME}")
            click.echo(f"Full database: {DEFAULT_DB_FULL_FILENAME}")
            click.echo(f"Lite database: {DEFAULT_DB_LITE_FILENAME}")
            click.echo(f"Default path: {_resolve_db_path()}")

            # Source types
            click.echo("\n=== source_types (Data Sources) ===")
            click.echo(f"{'ID':<5} {'Name':<20}")
            click.echo("-" * 25)
            cursor = conn.execute("SELECT id, name FROM source_types ORDER BY id")
            for row in cursor:
                click.echo(f"{row['id']:<5} {row['name']:<20}")

            # Organization types
            click.echo("\n=== organization_types (EntityType) ===")
            click.echo(f"{'ID':<5} {'Name':<25}")
            click.echo("-" * 30)
            cursor = conn.execute("SELECT id, name FROM organization_types ORDER BY id")
            for row in cursor:
                click.echo(f"{row['id']:<5} {row['name']:<25}")

            # People types
            click.echo("\n=== people_types (PersonType) ===")
            click.echo(f"{'ID':<5} {'Name':<20}")
            click.echo("-" * 25)
            cursor = conn.execute("SELECT id, name FROM people_types ORDER BY id")
            for row in cursor:
                click.echo(f"{row['id']:<5} {row['name']:<20}")

            # Simplified location types
            click.echo("\n=== simplified_location_types ===")
            click.echo(f"{'ID':<5} {'Name':<20}")
            click.echo("-" * 25)
            cursor = conn.execute("SELECT id, name FROM simplified_location_types ORDER BY id")
            for row in cursor:
                click.echo(f"{row['id']:<5} {row['name']:<20}")

            # Location types (sample)
            click.echo("\n=== location_types (Sample - Wikidata QID mappings) ===")
            click.echo(f"{'ID':<5} {'QID':<12} {'Name':<30} {'Simplified':<15}")
            click.echo("-" * 65)
            cursor = conn.execute("""
                SELECT lt.id, lt.qid, lt.name, slt.name as simplified
                FROM location_types lt
                JOIN simplified_location_types slt ON lt.simplified_id = slt.id
                ORDER BY lt.id
                LIMIT 20
            """)
            for row in cursor:
                qid = f"Q{row['qid']}" if row["qid"] else ""
                click.echo(f"{row['id']:<5} {qid:<12} {row['name']:<30} {row['simplified']:<15}")
            click.echo("... (showing first 20 of many)")

            # Table schemas
            click.echo("\n=== Table Schemas ===")
            for table in ["organizations", "people", "roles", "locations"]:
                click.echo(f"\n{table}:")
                cursor = conn.execute(f"PRAGMA table_info({table})")
                for row in cursor:
                    nullable = "" if row["notnull"] else "NULL"
                    pk = "PK" if row["pk"] else ""
                    click.echo(f"  {row['name']:<25} {row['type']:<15} {pk:<3} {nullable}")

            conn.close()

        database.close()

    except Exception as e:
        raise click.ClickException(f"Failed to read database: {e}")


@db_cmd.command("canonicalize")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--batch-size", type=int, default=10000, help="Batch size for updates (default: 10000)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_canonicalize(db_path: Optional[str], batch_size: int, verbose: bool):
    """
    Canonicalize organizations by linking equivalent records across sources.

    Records are considered equivalent if they share:
    - Same LEI (globally unique legal entity identifier)
    - Same ticker symbol
    - Same CIK (SEC identifier)
    - Same normalized name (after lowercasing, removing dots)
    - Same name with suffix expansion (Ltd -> Limited, etc.)

    For each group, the highest-priority source becomes canonical:
    gleif > sec_edgar > companies_house > wikipedia

    Canonicalization enables better search re-ranking by boosting results
    that have records from multiple authoritative sources.

    \b
    Examples:
        corp-extractor db canonicalize
        corp-extractor db canonicalize -v
        corp-extractor db canonicalize --db /path/to/entities.db
    """
    _configure_logging(verbose)

    from .database import OrganizationDatabase
    from .database.store import get_person_database

    db_path_obj = _resolve_db_path(db_path)

    try:
        # Canonicalize organizations (readonly=False for write operations)
        database = OrganizationDatabase(db_path=db_path_obj, readonly=False)
        click.echo("Running organization canonicalization...", err=True)

        result = database.canonicalize(batch_size=batch_size)

        click.echo("\nOrganization Canonicalization Results")
        click.echo("=" * 40)
        click.echo(f"Total records processed: {result['total_records']:,}")
        click.echo(f"Equivalence groups found: {result['groups_found']:,}")
        click.echo(f"Multi-record groups: {result['multi_record_groups']:,}")
        click.echo(f"Records updated: {result['records_updated']:,}")

        database.close()

        # Canonicalize people (readonly=False for write operations)
        person_db = get_person_database(db_path=db_path_obj, readonly=False)
        click.echo("\nRunning people canonicalization...", err=True)

        people_result = person_db.canonicalize(batch_size=batch_size)

        click.echo("\nPeople Canonicalization Results")
        click.echo("=" * 40)
        click.echo(f"Total records processed: {people_result['total_records']:,}")
        click.echo(f"Matched by organization: {people_result['matched_by_org']:,}")
        click.echo(f"Matched by date overlap: {people_result['matched_by_date']:,}")
        click.echo(f"Canonical groups: {people_result['canonical_groups']:,}")
        click.echo(f"Records in multi-record groups: {people_result['records_in_groups']:,}")

        person_db.close()

    except Exception as e:
        raise click.ClickException(f"Canonicalization failed: {e}")


@db_cmd.command("search")
@click.argument("query")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--top-k", type=int, default=10, help="Number of results")
@click.option("--source", type=click.Choice(["gleif", "sec_edgar", "companies_house", "wikipedia"]), help="Filter by source")
@click.option("--hybrid", is_flag=True, help="Use hybrid text + embeddings search (default is embeddings-only)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_search(query: str, db_path: Optional[str], top_k: int, source: Optional[str], hybrid: bool, verbose: bool):
    """
    Search for an organization in the database.

    \b
    Examples:
        corp-extractor db search "Apple Inc"
        corp-extractor db search "Microsoft" --source sec_edgar
        corp-extractor db search "Apple" --hybrid
    """
    _configure_logging(verbose)

    from .database import OrganizationDatabase, CompanyEmbedder

    db_path_obj = _resolve_db_path(db_path)
    embedder = CompanyEmbedder()
    database = OrganizationDatabase(db_path=db_path_obj)

    mode = "hybrid (text + embeddings)" if hybrid else "embeddings-only"
    click.echo(f"Searching for '{query}' [{mode}]...", err=True)

    # Embed query
    query_embedding = embedder.embed(query)

    # Search
    query_text = query if hybrid else None
    results = database.search(query_embedding, top_k=top_k, source_filter=source, query_text=query_text)

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"\nTop {len(results)} matches:")
    click.echo("-" * 60)

    for i, (record, similarity) in enumerate(results, 1):
        click.echo(f"{i}. {record.legal_name}")
        click.echo(f"   Source: {record.source} | ID: {record.source_id}")
        click.echo(f"   Canonical ID: {record.canonical_id}")
        click.echo(f"   Similarity: {similarity:.4f}")
        if verbose and record.record:
            if record.record.get("ticker"):
                click.echo(f"   Ticker: {record.record['ticker']}")
            if record.record.get("jurisdiction"):
                click.echo(f"   Jurisdiction: {record.record['jurisdiction']}")
        click.echo()

    database.close()


@db_cmd.command("download")
@click.option("--repo", type=str, default="Corp-o-Rate-Community/entity-references", help="HuggingFace repo ID")
@click.option("--db", "db_path", type=click.Path(), help="Output path for database")
@click.option("--full", is_flag=True, help="Download full version (larger, includes record metadata)")
@click.option("--force", is_flag=True, help="Force re-download")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_download(repo: str, db_path: Optional[str], full: bool, force: bool, verbose: bool):
    """
    Download entity database from HuggingFace Hub.

    By default downloads the lite version (smaller, without record metadata).
    Use --full for the complete database with all source record data.

    \b
    Examples:
        corp-extractor db download
        corp-extractor db download --full
        corp-extractor db download --repo my-org/my-entity-db
    """
    _configure_logging(verbose)
    from .database.hub import download_database, db_filenames, USEARCH_INDEX_FILES

    ctx = click.get_current_context(silent=True)
    db_version = ctx.obj.get("db_version") if ctx and ctx.obj else None
    full_fn, lite_fn, _ = db_filenames(db_version)
    filename = full_fn if full else lite_fn
    click.echo(f"Downloading {'full ' if full else 'lite '}database from {repo}...", err=True)

    try:
        path = download_database(
            repo_id=repo,
            filename=filename,
            force_download=force,
        )
        click.echo(f"Database downloaded to: {path}")

        # Create v2 symlink for backwards compatibility
        db_dir = path.parent
        v2_link = db_dir / "entities-v2.db"
        if not v2_link.exists():
            v2_link.symlink_to(path.name)
            click.echo(f"  Symlink: entities-v2.db -> {path.name}")

        # Report USearch index file status
        for idx_name in USEARCH_INDEX_FILES:
            idx_path = db_dir / idx_name
            if idx_path.exists():
                click.echo(f"  Index: {idx_name} ({idx_path.stat().st_size / 1024**2:.0f} MB)")
            else:
                click.echo(f"  Index: {idx_name} (not found — run: corp-extractor db build-index)")
    except Exception as e:
        raise click.ClickException(f"Download failed: {e}")


@db_cmd.command("upload")
@click.argument("db_path", type=click.Path(exists=True), required=False)
@click.option("--repo", type=str, default="Corp-o-Rate-Community/entity-references", help="HuggingFace repo ID")
@click.option("--message", type=str, default="Update entity database", help="Commit message")
@click.option("--no-lite", is_flag=True, help="Skip creating lite version (without record data)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_upload(db_path: Optional[str], repo: str, message: str, no_lite: bool, verbose: bool):
    """
    Upload entity database to HuggingFace Hub.

    First VACUUMs the database, then creates and uploads:
    - Full database + lite variant (without record data or embeddings)
    - USearch HNSW index files (.bin)

    If no path is provided, uploads from the default cache location.
    Requires HF_TOKEN environment variable to be set.

    \b
    Examples:
        corp-extractor db upload
        corp-extractor db upload /path/to/entities.db
        corp-extractor db upload --no-lite
        corp-extractor db upload --repo my-org/my-entity-db
    """
    _configure_logging(verbose)
    from .database.hub import upload_database_with_variants, DEFAULT_CACHE_DIR, db_filenames

    ctx = click.get_current_context(silent=True)
    db_version = ctx.obj.get("db_version") if ctx and ctx.obj else None
    full_fn, _, _ = db_filenames(db_version)

    # Use default cache location if no path provided
    if db_path is None:
        db_path = str(DEFAULT_CACHE_DIR / full_fn)
        if not Path(db_path).exists():
            raise click.ClickException(
                f"Database not found at default location: {db_path}\n"
                "Build the database first with import commands, or specify a path."
            )

    click.echo(f"Uploading {db_path} to {repo}...", err=True)
    click.echo("  - Running VACUUM to optimize database", err=True)
    if not no_lite:
        click.echo("  - Creating lite version (without record data)", err=True)

    try:
        results = upload_database_with_variants(
            db_path=db_path,
            repo_id=repo,
            commit_message=message,
            include_lite=not no_lite,
            version=db_version,
        )
        click.echo(f"\nUploaded {len(results)} file(s) successfully:")
        for filename, url in results.items():
            click.echo(f"  - {filename}")
    except Exception as e:
        raise click.ClickException(f"Upload failed: {e}")


@db_cmd.command("create-lite")
@click.argument("db_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output path (default: adds -lite suffix)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_create_lite(db_path: str, output: Optional[str], verbose: bool):
    """
    Create a lite version of the database without record data or embeddings.

    The lite version strips the `record` column and drops all embedding
    tables. Search uses USearch HNSW index files (.bin) instead.

    \b
    Examples:
        corp-extractor db create-lite entities.db
        corp-extractor db create-lite entities.db -o entities-lite.db
    """
    _configure_logging(verbose)
    from .database.hub import create_lite_database

    click.echo(f"Creating lite database from {db_path}...", err=True)

    try:
        lite_path = create_lite_database(db_path, output)
        click.echo(f"Lite database created: {lite_path}")
    except Exception as e:
        raise click.ClickException(f"Failed to create lite database: {e}")


@db_cmd.command("repair-embeddings")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--batch-size", type=int, default=1000, help="Batch size for embedding generation (default: 1000)")
@click.option("--source", type=str, help="Only repair specific source (gleif, sec_edgar, etc.)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_repair_embeddings(db_path: Optional[str], batch_size: int, source: Optional[str], verbose: bool):
    """
    Generate missing embeddings for organizations in the database.

    This repairs databases where organizations were imported without embeddings
    being properly stored in the organization_embeddings table.

    \b
    Examples:
        corp-extractor db repair-embeddings
        corp-extractor db repair-embeddings --source wikipedia
        corp-extractor db repair-embeddings --batch-size 500
    """
    _configure_logging(verbose)

    from .database import OrganizationDatabase, CompanyEmbedder

    # readonly=False for write operations (embedding repair)
    db_path_obj = _resolve_db_path(db_path)
    database = OrganizationDatabase(db_path=db_path_obj, readonly=False)
    embedder = CompanyEmbedder()

    # Check how many need repair
    missing_count = database.get_missing_embedding_count()
    if missing_count == 0:
        click.echo("All organizations have embeddings. Nothing to repair.")
        database.close()
        return

    click.echo(f"Found {missing_count:,} organizations without embeddings.", err=True)
    click.echo("Generating embeddings...", err=True)

    # Process in batches
    org_ids = []
    names = []
    count = 0

    for org_id, name in database.get_organizations_without_embeddings(batch_size=batch_size, source=source):
        org_ids.append(org_id)
        names.append(name)

        if len(names) >= batch_size:
            # Generate both float32 and int8 embeddings
            embeddings, scalar_embeddings = embedder.embed_batch_and_quantize(names)
            database.insert_both_embeddings_batch(org_ids, embeddings, scalar_embeddings)
            count += len(names)
            click.echo(f"Repaired {count:,} / {missing_count:,} embeddings...", err=True)
            org_ids = []
            names = []

    # Final batch
    if names:
        embeddings, scalar_embeddings = embedder.embed_batch_and_quantize(names)
        database.insert_both_embeddings_batch(org_ids, embeddings, scalar_embeddings)
        count += len(names)

    click.echo(f"\nRepaired {count:,} embeddings successfully.", err=True)
    database.close()


@db_cmd.command("rebuild-vec")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_rebuild_vec(db_path: Optional[str], verbose: bool):
    """
    Rebuild vec0 embedding tables with distance_metric=cosine.

    Required once for databases created before indexed KNN support was added.
    This enables fast MATCH-based vector search instead of brute-force scans.

    \b
    Examples:
        corp-extractor db rebuild-vec
        corp-extractor db rebuild-vec --db /path/to/entities.db
    """
    _configure_logging(verbose)

    from .database.store import (
        get_database,
        get_person_database,
    )

    db_path_obj = _resolve_db_path(db_path)

    click.echo(f"Rebuilding vec0 tables in {db_path_obj}...", err=True)

    # Rebuild organization embedding tables
    click.echo("Rebuilding organization embedding tables...", err=True)
    org_db = get_database(db_path=db_path_obj, readonly=False)
    org_stats = org_db.rebuild_vec_tables()
    for table, count in org_stats.items():
        click.echo(f"  {table}: {count} rows", err=True)
    org_db.close()

    # Rebuild person embedding tables
    click.echo("Rebuilding person embedding tables...", err=True)
    person_db = get_person_database(db_path=db_path_obj, readonly=False)
    person_stats = person_db.rebuild_vec_tables()
    for table, count in person_stats.items():
        click.echo(f"  {table}: {count} rows", err=True)
    person_db.close()

    total = sum(org_stats.values()) + sum(person_stats.values())
    click.echo(f"\nDone. Rebuilt {total} total embeddings with cosine distance metric.", err=True)


@db_cmd.command("backfill-scalar")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--batch-size", type=int, default=10000, help="Batch size for processing (default: 10000)")
@click.option("--embed-batch-size", type=int, default=64, help="Batch size for embedding generation (default: 64)")
@click.option("--skip-generate", is_flag=True, help="Skip generating missing float32 embeddings (only quantize existing)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_backfill_scalar(db_path: Optional[str], batch_size: int, embed_batch_size: int, skip_generate: bool, verbose: bool):
    """
    Backfill scalar (int8) embeddings for the entity database.

    This command handles two cases:
    1. Records with float32 but missing scalar → quantize existing
    2. Records missing both embeddings → generate both from scratch

    Scalar embeddings provide 75% storage reduction with ~92% recall at top-100.

    \b
    Examples:
        corp-extractor db backfill-scalar
        corp-extractor db backfill-scalar --batch-size 5000 -v
        corp-extractor db backfill-scalar --skip-generate  # Only quantize existing
    """
    _configure_logging(verbose)
    import numpy as np

    from .database import OrganizationDatabase, CompanyEmbedder
    from .database.store import get_person_database

    db_path_obj = _resolve_db_path(db_path)
    embedder = None  # Lazy load only if needed

    # Process organizations (readonly=False for write operations)
    org_db = OrganizationDatabase(db_path=db_path_obj, readonly=False)

    # Phase 1: Quantize existing float32 embeddings to scalar
    org_quantized = 0
    click.echo("Phase 1: Quantizing existing float32 embeddings to scalar...", err=True)
    for batch_ids in org_db.get_missing_scalar_embedding_ids(batch_size=batch_size):
        fp32_map = org_db.get_embeddings_by_ids(batch_ids)
        if not fp32_map:
            continue

        ids = list(fp32_map.keys())
        int8_embeddings = np.array([
            np.clip(np.round(fp32_map[i] * 127), -127, 127).astype(np.int8)
            for i in ids
        ])

        org_db.insert_scalar_embeddings_batch(ids, int8_embeddings)
        org_quantized += len(ids)
        click.echo(f"  Quantized {org_quantized:,} organization embeddings...", err=True)

    click.echo(f"Quantized {org_quantized:,} organization embeddings.", err=True)

    # Phase 2: Generate embeddings for records missing both
    org_generated = 0
    if not skip_generate:
        click.echo("\nPhase 2: Generating embeddings for organizations missing both...", err=True)

        for batch in org_db.get_missing_all_embedding_ids(batch_size=batch_size):
            if not batch:
                continue

            # Lazy load embedder
            if embedder is None:
                click.echo("  Loading embedding model...", err=True)
                embedder = CompanyEmbedder()

            # Process in smaller batches for embedding generation
            for i in range(0, len(batch), embed_batch_size):
                sub_batch = batch[i:i + embed_batch_size]
                ids = [item[0] for item in sub_batch]
                names = [item[1] for item in sub_batch]

                # Generate both float32 and int8 embeddings
                fp32_batch, int8_batch = embedder.embed_batch_and_quantize(names, batch_size=embed_batch_size)

                # Insert both
                org_db.insert_both_embeddings_batch(ids, fp32_batch, int8_batch)
                org_generated += len(ids)

                if org_generated % 10000 == 0:
                    click.echo(f"  Generated {org_generated:,} organization embeddings...", err=True)

        click.echo(f"Generated {org_generated:,} organization embeddings.", err=True)

    # Process people (readonly=False for write operations)
    person_db = get_person_database(db_path=db_path_obj, readonly=False)

    # Phase 1: Quantize existing float32 embeddings to scalar
    person_quantized = 0
    click.echo("\nPhase 1: Quantizing existing float32 person embeddings to scalar...", err=True)
    for batch_ids in person_db.get_missing_scalar_embedding_ids(batch_size=batch_size):
        fp32_map = person_db.get_embeddings_by_ids(batch_ids)
        if not fp32_map:
            continue

        ids = list(fp32_map.keys())
        int8_embeddings = np.array([
            np.clip(np.round(fp32_map[i] * 127), -127, 127).astype(np.int8)
            for i in ids
        ])

        person_db.insert_scalar_embeddings_batch(ids, int8_embeddings)
        person_quantized += len(ids)
        click.echo(f"  Quantized {person_quantized:,} person embeddings...", err=True)

    click.echo(f"Quantized {person_quantized:,} person embeddings.", err=True)

    # Phase 2: Generate embeddings for records missing both
    person_generated = 0
    if not skip_generate:
        click.echo("\nPhase 2: Generating embeddings for people missing both...", err=True)

        for batch in person_db.get_missing_all_embedding_ids(batch_size=batch_size):
            if not batch:
                continue

            # Lazy load embedder
            if embedder is None:
                click.echo("  Loading embedding model...", err=True)
                embedder = CompanyEmbedder()

            # Process in smaller batches for embedding generation
            for i in range(0, len(batch), embed_batch_size):
                sub_batch = batch[i:i + embed_batch_size]
                ids = [item[0] for item in sub_batch]
                names = [item[1] for item in sub_batch]

                # Generate both float32 and int8 embeddings
                fp32_batch, int8_batch = embedder.embed_batch_and_quantize(names, batch_size=embed_batch_size)

                # Insert both
                person_db.insert_both_embeddings_batch(ids, fp32_batch, int8_batch)
                person_generated += len(ids)

                if person_generated % 10000 == 0:
                    click.echo(f"  Generated {person_generated:,} person embeddings...", err=True)

        click.echo(f"Generated {person_generated:,} person embeddings.", err=True)

    # Summary
    click.echo(f"\nSummary:", err=True)
    click.echo(f"  Organizations: {org_quantized:,} quantized, {org_generated:,} generated", err=True)
    click.echo(f"  People: {person_quantized:,} quantized, {person_generated:,} generated", err=True)
    click.echo(f"  Total: {org_quantized + org_generated + person_quantized + person_generated:,} embeddings processed", err=True)


@db_cmd.command("build-index")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--M", type=int, default=32, help="Number of connections per node (default 32)")
@click.option("--ef-construction", type=int, default=200, help="Construction quality parameter (default 200)")
@click.option("--ef-search", type=int, default=200, help="Search quality parameter (default 200)")
@click.option("--people/--no-people", default=True, help="Build USearch index for people")
@click.option("--orgs/--no-orgs", default=True, help="Build USearch index for organizations")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_build_index(db_path: Optional[str], m: int, ef_construction: int, ef_search: int, people: bool, orgs: bool, verbose: bool):
    """
    Build USearch index for fast approximate nearest neighbor search.

    USearch uses HNSW algorithm (Hierarchical Navigable Small World) and provides
    sub-millisecond query times on millions of vectors. Much faster than GPU or PCA approaches.

    \b
    Parameters:
    - M: Higher = better quality, more memory (16-64, default 32)
    - ef_construction: Higher = better quality, slower build (100-500, default 200)
    - ef_search: Higher = better quality, slower search (50-500, default 200)

    \b
    Examples:
        corp-extractor db build-index
        corp-extractor db build-index --M 16              # Faster build, less memory
        corp-extractor db build-index --M 64 --ef-construction 400   # Highest quality
        corp-extractor db build-index --no-orgs           # People only
    """
    _configure_logging(verbose)

    from .database.store import (
        _get_shared_connection,
        build_hnsw_index,
    )

    db_path_obj = _resolve_db_path(db_path)

    if not db_path_obj.exists():
        raise click.ClickException(f"Database not found: {db_path_obj}")

    click.echo(f"Database: {db_path_obj}", err=True)
    click.echo(f"Parameters: M={m}, ef_construction={ef_construction}, ef_search={ef_search}", err=True)

    # Open read-only connection (we only read from scalar table)
    conn = _get_shared_connection(db_path_obj, readonly=True)

    if people:
        click.echo("\n--- Building USearch index for people ---", err=True)

        def people_progress(done: int, total: int) -> None:
            click.echo(f"  Loaded {done:,}/{total:,} vectors...", err=True)

        try:
            count = build_hnsw_index(
                conn, "people",
                M=m,
                ef_construction=ef_construction,
                ef_search=ef_search,
                progress_callback=people_progress,
            )
            click.echo(f"People USearch index: {count:,} vectors indexed", err=True)
        except Exception as e:
            raise click.ClickException(f"People USearch index build failed: {e}")

    if orgs:
        click.echo("\n--- Building USearch index for organizations ---", err=True)

        def orgs_progress(done: int, total: int) -> None:
            click.echo(f"  Loaded {done:,}/{total:,} vectors...", err=True)

        try:
            count = build_hnsw_index(
                conn, "organizations",
                M=m,
                ef_construction=ef_construction,
                ef_search=ef_search,
                progress_callback=orgs_progress,
            )
            click.echo(f"Organizations USearch index: {count:,} vectors indexed", err=True)
        except Exception as e:
            raise click.ClickException(f"Organizations USearch index build failed: {e}")

    click.echo("\nUSearch index build complete!", err=True)


def _run_post_import(db_path_obj: Path, people: bool = True, orgs: bool = True, batch_size: int = 10000, embed_batch_size: int = 64) -> None:
    """
    Standard post-import steps: generate embeddings, build USearch indexes, VACUUM.

    Called automatically after imports or manually via `db post-import`.
    """
    import sqlite3

    import numpy as np

    from .database import OrganizationDatabase, CompanyEmbedder
    from .database.store import (
        get_person_database,
        _get_shared_connection,
        build_hnsw_index,
    )

    embedder = None  # Lazy load

    # --- Step 1: Generate embeddings for new records ---
    click.echo("\n=== Step 1: Generate embeddings for new records ===", err=True)

    if orgs:
        org_db = OrganizationDatabase(db_path=db_path_obj, readonly=False)

        # Quantize existing float32 → int8
        org_quantized = 0
        for batch_ids in org_db.get_missing_scalar_embedding_ids(batch_size=batch_size):
            fp32_map = org_db.get_embeddings_by_ids(batch_ids)
            if not fp32_map:
                continue
            ids = list(fp32_map.keys())
            int8_embeddings = np.array([
                np.clip(np.round(fp32_map[i] * 127), -127, 127).astype(np.int8)
                for i in ids
            ])
            org_db.insert_scalar_embeddings_batch(ids, int8_embeddings)
            org_quantized += len(ids)
        if org_quantized:
            click.echo(f"  Quantized {org_quantized:,} org embeddings (float32 → int8)", err=True)

        # Generate both for records missing entirely
        org_generated = 0
        for batch in org_db.get_missing_all_embedding_ids(batch_size=batch_size):
            if not batch:
                continue
            if embedder is None:
                click.echo("  Loading embedding model...", err=True)
                embedder = CompanyEmbedder()
            for i in range(0, len(batch), embed_batch_size):
                sub_batch = batch[i:i + embed_batch_size]
                ids = [item[0] for item in sub_batch]
                names = [item[1] for item in sub_batch]
                fp32_batch, int8_batch = embedder.embed_batch_and_quantize(names, batch_size=embed_batch_size)
                org_db.insert_both_embeddings_batch(ids, fp32_batch, int8_batch)
                org_generated += len(ids)
                if org_generated % 10000 == 0:
                    click.echo(f"  Generated {org_generated:,} org embeddings...", err=True)
        if org_generated:
            click.echo(f"  Generated {org_generated:,} org embeddings", err=True)

        if not org_quantized and not org_generated:
            click.echo("  Organizations: all embeddings up to date", err=True)
        org_db.close()

    if people:
        person_db = get_person_database(db_path=db_path_obj, readonly=False)

        person_quantized = 0
        for batch_ids in person_db.get_missing_scalar_embedding_ids(batch_size=batch_size):
            fp32_map = person_db.get_embeddings_by_ids(batch_ids)
            if not fp32_map:
                continue
            ids = list(fp32_map.keys())
            int8_embeddings = np.array([
                np.clip(np.round(fp32_map[i] * 127), -127, 127).astype(np.int8)
                for i in ids
            ])
            person_db.insert_scalar_embeddings_batch(ids, int8_embeddings)
            person_quantized += len(ids)
        if person_quantized:
            click.echo(f"  Quantized {person_quantized:,} person embeddings (float32 → int8)", err=True)

        person_generated = 0
        for batch in person_db.get_missing_all_embedding_ids(batch_size=batch_size):
            if not batch:
                continue
            if embedder is None:
                click.echo("  Loading embedding model...", err=True)
                embedder = CompanyEmbedder()
            for i in range(0, len(batch), embed_batch_size):
                sub_batch = batch[i:i + embed_batch_size]
                ids = [item[0] for item in sub_batch]
                names = [item[1] for item in sub_batch]
                fp32_batch, int8_batch = embedder.embed_batch_and_quantize(names, batch_size=embed_batch_size)
                person_db.insert_both_embeddings_batch(ids, fp32_batch, int8_batch)
                person_generated += len(ids)
                if person_generated % 10000 == 0:
                    click.echo(f"  Generated {person_generated:,} person embeddings...", err=True)
        if person_generated:
            click.echo(f"  Generated {person_generated:,} person embeddings", err=True)

        if not person_quantized and not person_generated:
            click.echo("  People: all embeddings up to date", err=True)
        person_db.close()

    # --- Step 2: Rebuild USearch indexes ---
    click.echo("\n=== Step 2: Rebuild USearch indexes ===", err=True)

    conn = _get_shared_connection(db_path_obj, readonly=True)

    if people:
        count = build_hnsw_index(conn, "people")
        click.echo(f"  People USearch index: {count:,} vectors", err=True)

    if orgs:
        count = build_hnsw_index(conn, "organizations")
        click.echo(f"  Organizations USearch index: {count:,} vectors", err=True)

    # --- Step 3: VACUUM ---
    click.echo("\n=== Step 3: VACUUM database ===", err=True)
    vacuum_conn = sqlite3.connect(str(db_path_obj))
    vacuum_conn.execute("VACUUM")
    vacuum_conn.close()
    db_size_mb = db_path_obj.stat().st_size / 1024**2
    click.echo(f"  Database size after VACUUM: {db_size_mb:,.0f} MB", err=True)

    click.echo("\nPost-import complete!", err=True)


@db_cmd.command("post-import")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--people/--no-people", default=True, help="Process people")
@click.option("--orgs/--no-orgs", default=True, help="Process organizations")
@click.option("--batch-size", type=int, default=10000, help="Batch size (default: 10000)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_post_import(db_path: Optional[str], people: bool, orgs: bool, batch_size: int, verbose: bool):
    """
    Run standard post-import steps: embeddings, USearch indexes, VACUUM.

    Run this after any import command to ensure search indexes are up to date.

    \b
    Steps:
    1. Generate float32 + int8 embeddings for new records
    2. Rebuild USearch HNSW indexes for fast search
    3. VACUUM database to reclaim space

    \b
    Examples:
        corp-extractor db post-import
        corp-extractor db post-import --no-orgs     # People only
        corp-extractor db post-import -v             # Verbose logging
    """
    _configure_logging(verbose)

    db_path_obj = _resolve_db_path(db_path)

    if not db_path_obj.exists():
        raise click.ClickException(f"Database not found: {db_path_obj}")

    click.echo(f"Database: {db_path_obj}", err=True)
    _run_post_import(db_path_obj, people=people, orgs=orgs, batch_size=batch_size)


@db_cmd.command("migrate")
@click.argument("db_path", type=click.Path(exists=True))
@click.option("--rename-file", is_flag=True, help="Also rename companies.db to entities.db")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_migrate(db_path: str, rename_file: bool, yes: bool, verbose: bool):
    """
    Migrate database from legacy schema to new schema.

    Migrates from old naming (companies/company_embeddings tables)
    to new naming (organizations/organization_embeddings tables).

    \b
    What this does:
    - Renames 'companies' table to 'organizations'
    - Renames 'company_embeddings' table to 'organization_embeddings'
    - Updates all indexes

    \b
    Examples:
        corp-extractor db migrate companies.db
        corp-extractor db migrate companies.db --rename-file
        corp-extractor db migrate ~/.cache/corp-extractor/companies.db --yes
    """
    _configure_logging(verbose)

    from pathlib import Path
    from .database import OrganizationDatabase

    db_path_obj = Path(db_path)

    if not yes:
        click.confirm(
            f"This will migrate {db_path} from legacy schema (companies) to new schema (organizations).\n"
            "This operation cannot be undone. Continue?",
            abort=True
        )

    try:
        # readonly=False for schema migrations
        database = OrganizationDatabase(db_path=db_path, readonly=False)
        migrations = database.migrate_from_legacy_schema()
        database.close()

        if migrations:
            click.echo("Migration completed:")
            for table, action in migrations.items():
                click.echo(f"  {table}: {action}")
        else:
            click.echo("No migration needed. Database already uses new schema.")

        # Optionally rename the file
        if rename_file and db_path_obj.name.startswith("companies"):
            new_name = db_path_obj.name.replace("companies", "entities")
            new_path = db_path_obj.parent / new_name
            db_path_obj.rename(new_path)
            click.echo(f"Renamed file: {db_path} -> {new_path}")

    except Exception as e:
        raise click.ClickException(f"Migration failed: {e}")


@db_cmd.command("migrate-v2")
@click.argument("source_db", type=click.Path(exists=True))
@click.argument("target_db", type=click.Path())
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option("--resume", is_flag=True, help="Resume from last completed step")
def db_migrate_v2(source_db: str, target_db: str, verbose: bool, resume: bool):
    """
    Migrate database from v1 schema to v2 normalized schema.

    Creates a NEW database file with the v2 normalized schema.
    The original database is preserved unchanged.

    Use --resume to continue a migration that was interrupted.

    \b
    V2 changes:
    - TEXT enum fields replaced with INTEGER foreign keys
    - New enum lookup tables (source_types, people_types, etc.)
    - New roles and locations tables
    - QIDs stored as integers (Q prefix stripped)
    - Human-readable views for queries

    \b
    Examples:
        corp-extractor db migrate-v2 entities.db entities-v2.db
        corp-extractor db migrate-v2 entities.db entities-v2.db --resume
        corp-extractor db migrate-v2 ~/.cache/corp-extractor/entities.db ./entities-v2.db -v
    """
    _configure_logging(verbose)

    from pathlib import Path
    from .database.migrate_v2 import migrate_database

    source_path = Path(source_db)
    target_path = Path(target_db)

    if target_path.exists() and not resume:
        raise click.ClickException(
            f"Target database already exists: {target_path}\n"
            "Use --resume to continue an interrupted migration."
        )

    if resume:
        click.echo(f"Resuming migration from {source_path} to {target_path}...")
    else:
        click.echo(f"Migrating {source_path} to {target_path}...")

    try:
        stats = migrate_database(source_path, target_path, resume=resume)

        click.echo("\nMigration complete:")
        for key, value in stats.items():
            click.echo(f"  {key}: {value:,}")

    except Exception as e:
        raise click.ClickException(f"Migration failed: {e}")


@db_cmd.command("search-roles")
@click.argument("query")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--limit", default=10, help="Maximum results to return")
def db_search_roles(query: str, db_path: Optional[str], limit: int):
    """
    Search for roles by name.

    \b
    Examples:
        corp-extractor db search-roles "CEO"
        corp-extractor db search-roles "Chief Executive" --limit 5
    """
    from .database.store import get_roles_database

    roles_db = get_roles_database(db_path)
    results = roles_db.search(query, top_k=limit)

    if not results:
        click.echo(f"No roles found matching '{query}'")
        return

    click.echo(f"Found {len(results)} role(s) matching '{query}':")
    for role_id, name, score in results:
        click.echo(f"  [{role_id}] {name} (score: {score:.2f})")


@db_cmd.command("search-locations")
@click.argument("query")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--type", "location_type", type=str, help="Filter by simplified type (country, city, etc.)")
@click.option("--limit", default=10, help="Maximum results to return")
def db_search_locations(query: str, db_path: Optional[str], location_type: Optional[str], limit: int):
    """
    Search for locations by name.

    \b
    Examples:
        corp-extractor db search-locations "California"
        corp-extractor db search-locations "Paris" --type city
        corp-extractor db search-locations "Germany" --type country
    """
    from .database.store import get_locations_database

    locations_db = get_locations_database(db_path)
    results = locations_db.search(query, top_k=limit, simplified_type=location_type)

    if not results:
        click.echo(f"No locations found matching '{query}'")
        return

    click.echo(f"Found {len(results)} location(s) matching '{query}':")
    for loc_id, name, score in results:
        click.echo(f"  [{loc_id}] {name} (score: {score:.2f})")


@db_cmd.command("import-locations")
@click.option("--from-pycountry", is_flag=True, help="Import countries from pycountry")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_locations(from_pycountry: bool, db_path: Optional[str], verbose: bool):
    """
    Import locations into the database.

    \b
    Examples:
        corp-extractor db import-locations --from-pycountry
    """
    _configure_logging(verbose)

    if not from_pycountry:
        raise click.UsageError("Must specify --from-pycountry")

    from .database.store import get_locations_database

    # readonly=False for import operations
    locations_db = get_locations_database(db_path, readonly=False)
    count = locations_db.import_from_pycountry()

    click.echo(f"Imported {count:,} locations from pycountry")
    click.echo("Run `corp-extractor db post-import` to update search indexes.", err=True)


# =============================================================================
# Document commands
# =============================================================================

@main.group("document")
def document_cmd():
    """
    Process documents with chunking, deduplication, and citations.

    \b
    Commands:
        process    Process a document through the full pipeline
        chunk      Preview chunking without extraction

    \b
    Examples:
        corp-extractor document process article.txt
        corp-extractor document process report.pdf --no-summary
        corp-extractor document chunk article.txt --max-tokens 500
    """
    pass


@document_cmd.command("process")
@click.argument("input_source")  # Can be file path or URL
@click.option("--title", type=str, help="Document title (for citations)")
@click.option("--author", "authors", type=str, multiple=True, help="Document author(s)")
@click.option("--year", type=int, help="Publication year")
@click.option("--max-tokens", type=int, default=1000, help="Target tokens per chunk (default: 1000)")
@click.option("--overlap", type=int, default=100, help="Token overlap between chunks (default: 100)")
@click.option("--no-summary", is_flag=True, help="Skip document summarization")
@click.option("--no-dedup", is_flag=True, help="Skip deduplication across chunks")
@click.option("--use-ocr", is_flag=True, help="Force OCR for PDF parsing")
@click.option("--pdf-parser", type=str, default=None, help="PDF parser plugin name (e.g., glm_ocr_parser)")
@click.option(
    "--stages",
    type=str,
    default="1-6",
    help="Pipeline stages to run (e.g., '1-3' or '1,2,5')"
)
@click.option(
    "-o", "--output",
    type=click.Choice(["table", "json", "triples"], case_sensitive=False),
    default="table",
    help="Output format (default: table)"
)
@click.option("-v", "--verbose", is_flag=True, help="Show verbose output")
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress messages")
def document_process(
    input_source: str,
    title: Optional[str],
    authors: tuple[str, ...],
    year: Optional[int],
    max_tokens: int,
    overlap: int,
    no_summary: bool,
    no_dedup: bool,
    use_ocr: bool,
    pdf_parser: Optional[str],
    stages: str,
    output: str,
    verbose: bool,
    quiet: bool,
):
    """
    Process a document or URL through the extraction pipeline with chunking.

    Supports text files, PDFs, and URLs (web pages and PDFs).

    \b
    Examples:
        corp-extractor document process article.txt
        corp-extractor document process report.pdf --pdf-parser glm_ocr_parser
        corp-extractor document process report.txt --title "Annual Report" --year 2024
        corp-extractor document process https://example.com/article
        corp-extractor document process https://example.com/report.pdf --use-ocr
        corp-extractor document process doc.txt --no-summary --stages 1-3
        corp-extractor document process doc.txt -o json
    """
    _configure_logging(verbose)

    # Import document pipeline
    from .document import DocumentPipeline, DocumentPipelineConfig, Document
    from .models.document import ChunkingConfig
    from .pipeline import PipelineConfig
    _load_all_plugins()

    # Parse stages
    enabled_stages = _parse_stages(stages)

    # Build configs
    chunking_config = ChunkingConfig(
        target_tokens=max_tokens,
        max_tokens=max_tokens * 2,
        overlap_tokens=overlap,
    )

    pipeline_config = PipelineConfig(
        enabled_stages=enabled_stages,
    )

    doc_config = DocumentPipelineConfig(
        chunking=chunking_config,
        generate_summary=not no_summary,
        deduplicate_across_chunks=not no_dedup,
        pipeline_config=pipeline_config,
    )

    # Create pipeline
    pipeline = DocumentPipeline(doc_config)

    # Detect if input is a URL
    is_url = input_source.startswith(("http://", "https://"))

    # Process
    try:
        if is_url:
            # Process URL
            from .document import URLLoaderConfig

            if not quiet:
                click.echo(f"Fetching URL: {input_source}", err=True)

            loader_config = URLLoaderConfig(
                use_ocr=use_ocr,
                pdf_parser_plugin=pdf_parser,
            )
            ctx = pipeline.process_url_sync(input_source, loader_config)

            if not quiet:
                click.echo(f"Processed: {ctx.document.metadata.title or 'Untitled'}", err=True)

        else:
            # Process file
            from pathlib import Path
            import os

            if not os.path.exists(input_source):
                raise click.ClickException(f"File not found: {input_source}")

            file_path = Path(input_source)

            if file_path.suffix.lower() == ".pdf":
                # Local PDF file — read as binary and run through PDF parser
                pdf_bytes = file_path.read_bytes()
                if not pdf_bytes:
                    raise click.ClickException("PDF file is empty")

                if not quiet:
                    click.echo(
                        f"Processing PDF: {input_source} ({len(pdf_bytes)} bytes)", err=True
                    )

                from .pipeline.registry import PluginRegistry

                parsers = PluginRegistry.get_pdf_parsers()
                if not parsers:
                    raise click.ClickException("No PDF parser plugins registered")

                parser = None
                if pdf_parser:
                    for p in parsers:
                        if p.name == pdf_parser:
                            parser = p
                            break
                    if parser is None:
                        available = ", ".join(p.name for p in parsers)
                        raise click.ClickException(
                            f"PDF parser not found: {pdf_parser}. Available: {available}"
                        )
                else:
                    parser = parsers[0]

                if not quiet:
                    click.echo(f"Using PDF parser: {parser.name}", err=True)

                parse_result = parser.parse(pdf_bytes, use_ocr=use_ocr)
                if not parse_result.ok:
                    raise click.ClickException(f"PDF parsing failed: {parse_result.error}")

                doc_title = title or parse_result.metadata.get("title") or file_path.stem
                doc_authors = list(authors)
                if not doc_authors and parse_result.metadata.get("author"):
                    doc_authors = [parse_result.metadata["author"]]

                document = Document.from_pages(
                    pages=parse_result.pages,
                    title=doc_title,
                    source_type="pdf",
                    authors=doc_authors,
                    year=year,
                )
            else:
                # Text file
                with open(input_source, "r", encoding="utf-8") as f:
                    text = f.read()

                if not text.strip():
                    raise click.ClickException("Input file is empty")

                if not quiet:
                    click.echo(
                        f"Processing document: {input_source} ({len(text)} chars)", err=True
                    )

                doc_title = title or file_path.stem
                document = Document.from_text(
                    text=text,
                    title=doc_title,
                    source_type="text",
                    authors=list(authors),
                    year=year,
                )

            ctx = pipeline.process(document)

        # Output results
        if output == "json":
            _print_document_json(ctx)
        elif output == "triples":
            _print_document_triples(ctx)
        else:
            _print_document_table(ctx, verbose)

        # Report stats
        if not quiet:
            click.echo(f"\nChunks: {ctx.chunk_count}", err=True)
            click.echo(f"Statements: {ctx.statement_count}", err=True)
            if ctx.duplicates_removed > 0:
                click.echo(f"Duplicates removed: {ctx.duplicates_removed}", err=True)

            if ctx.processing_errors:
                click.echo(f"\nErrors: {len(ctx.processing_errors)}", err=True)
                for error in ctx.processing_errors:
                    click.echo(f"  - {error}", err=True)

    except Exception as e:
        logging.exception("Document processing error:")
        raise click.ClickException(f"Processing failed: {e}")


@document_cmd.command("chunk")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--max-tokens", type=int, default=1000, help="Target tokens per chunk (default: 1000)")
@click.option("--overlap", type=int, default=100, help="Token overlap between chunks (default: 100)")
@click.option("-o", "--output", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.option("-v", "--verbose", is_flag=True, help="Show verbose output")
def document_chunk(
    input_path: str,
    max_tokens: int,
    overlap: int,
    output: str,
    verbose: bool,
):
    """
    Preview document chunking without running extraction.

    Shows how a document would be split into chunks for processing.

    \b
    Examples:
        corp-extractor document chunk article.txt
        corp-extractor document chunk article.txt --max-tokens 500
        corp-extractor document chunk article.txt -o json
    """
    _configure_logging(verbose)

    # Read input file
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        raise click.ClickException("Input file is empty")

    click.echo(f"Chunking document: {input_path} ({len(text)} chars)", err=True)

    from .document import DocumentChunker, Document
    from .models.document import ChunkingConfig

    config = ChunkingConfig(
        target_tokens=max_tokens,
        max_tokens=max_tokens * 2,
        overlap_tokens=overlap,
    )

    from pathlib import Path
    document = Document.from_text(text, title=Path(input_path).stem)
    chunker = DocumentChunker(config)
    chunks = chunker.chunk_document(document)

    if output == "json":
        import json
        chunk_data = [
            {
                "index": c.chunk_index,
                "tokens": c.token_count,
                "chars": len(c.text),
                "pages": c.page_numbers,
                "overlap": c.overlap_chars,
                "preview": c.text[:100] + "..." if len(c.text) > 100 else c.text,
            }
            for c in chunks
        ]
        click.echo(json.dumps({"chunks": chunk_data, "total": len(chunks)}, indent=2))
    else:
        click.echo(f"\nCreated {len(chunks)} chunk(s):\n")
        click.echo("-" * 80)

        for chunk in chunks:
            click.echo(f"Chunk {chunk.chunk_index + 1}:")
            click.echo(f"  Tokens: {chunk.token_count}")
            click.echo(f"  Characters: {len(chunk.text)}")
            if chunk.page_numbers:
                click.echo(f"  Pages: {chunk.page_numbers}")
            if chunk.overlap_chars > 0:
                click.echo(f"  Overlap: {chunk.overlap_chars} chars")

            preview = chunk.text[:200].replace("\n", " ")
            if len(chunk.text) > 200:
                preview += "..."
            click.echo(f"  Preview: {preview}")
            click.echo("-" * 80)


def _print_document_json(ctx):
    """Print document context as JSON."""
    import json
    click.echo(json.dumps(ctx.as_dict(), indent=2, default=str))


def _print_document_triples(ctx):
    """Print document statements as triples."""
    for stmt in ctx.labeled_statements:
        parts = [stmt.subject_fqn, stmt.statement.predicate, stmt.object_fqn]
        if stmt.page_number:
            parts.append(f"p.{stmt.page_number}")
        click.echo("\t".join(parts))


def _print_document_table(ctx, verbose: bool):
    """Print document context in table format."""
    # Show summary if available
    if ctx.document.summary:
        click.echo("\nDocument Summary:")
        click.echo("-" * 40)
        click.echo(ctx.document.summary)
        click.echo("-" * 40)

    if not ctx.labeled_statements:
        click.echo("\nNo statements extracted.")
        return

    click.echo(f"\nExtracted {len(ctx.labeled_statements)} statement(s):\n")
    click.echo("-" * 80)

    for i, stmt in enumerate(ctx.labeled_statements, 1):
        click.echo(f"{i}. {stmt.subject_fqn}")
        click.echo(f"   --[{stmt.statement.predicate}]-->")
        click.echo(f"   {stmt.object_fqn}")

        # Show citation
        if stmt.citation:
            click.echo(f"   Citation: {stmt.citation}")
        elif stmt.page_number:
            click.echo(f"   Page: {stmt.page_number}")

        # Show labels
        for label in stmt.labels:
            if isinstance(label.label_value, float):
                click.echo(f"   {label.label_type}: {label.label_value:.3f}")
            else:
                click.echo(f"   {label.label_type}: {label.label_value}")

        # Show taxonomy (top 3)
        if stmt.taxonomy_results:
            sorted_taxonomy = sorted(stmt.taxonomy_results, key=lambda t: t.confidence, reverse=True)[:3]
            taxonomy_strs = [f"{t.category}:{t.label}" for t in sorted_taxonomy]
            click.echo(f"   Topics: {', '.join(taxonomy_strs)}")

        if verbose and stmt.statement.source_text:
            source = stmt.statement.source_text[:60] + "..." if len(stmt.statement.source_text) > 60 else stmt.statement.source_text
            click.echo(f"   Source: \"{source}\"")

        click.echo("-" * 80)

    # Show timings in verbose mode
    if verbose and ctx.stage_timings:
        click.echo("\nStage timings:")
        for stage, duration in ctx.stage_timings.items():
            click.echo(f"  {stage}: {duration:.3f}s")


# =============================================================================
# Helper functions
# =============================================================================

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


if __name__ == "__main__":
    main()
