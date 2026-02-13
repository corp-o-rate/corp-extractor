"""Pipeline command — full 5-stage extraction pipeline."""

import json
import logging
from typing import Optional

import click

from ._common import (
    _configure_logging,
    _get_input_text,
    _load_all_plugins,
    _parse_stages,
    _server_request,
)


@click.command("pipeline")
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

    # Check for server mode
    server = click.get_current_context().obj.get("server")
    if server:
        if not quiet:
            click.echo(f"Using server: {server}", err=True)
        try:
            # Build config dict for server
            config_dict: dict = {"enabled_stages": stages}
            if skip_stages:
                skip_set = _parse_stages(skip_stages)
                remaining = _parse_stages(stages) - skip_set
                config_dict["enabled_stages"] = sorted(remaining)
            if enabled_plugins:
                config_dict["enabled_plugins"] = [p.strip() for p in enabled_plugins.split(",") if p.strip()]
            if disable_plugins:
                config_dict["disabled_plugins"] = [p.strip() for p in disable_plugins.split(",") if p.strip()]
            if no_default_predicates:
                config_dict["extractor_options"] = {"use_default_predicates": False}

            result_data = _server_request(server, "/pipeline", {"text": input_text, "config": config_dict})
            # Reconstruct PipelineContext from server JSON and reuse local formatters
            from ..pipeline.context import PipelineContext
            ctx = PipelineContext.model_validate(result_data)
            if output == "json":
                _print_pipeline_json(ctx)
            elif output == "yaml":
                _print_pipeline_yaml(ctx)
            elif output == "triples":
                _print_pipeline_triples(ctx)
            else:
                _print_pipeline_table(ctx, verbose)
        except Exception as e:
            raise click.ClickException(f"Server request failed: {e}")
        return

    # Import pipeline components (also loads plugins)
    from ..pipeline import ExtractionPipeline, PipelineConfig
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
