"""Document commands — process documents with chunking and citations."""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import click

from ._common import (
    _configure_logging,
    _load_all_plugins,
    _parse_stages,
    _server_request,
)


@click.group("document")
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

    # Check for server mode — server handles text files only (not URLs/PDFs)
    is_url = input_source.startswith(("http://", "https://"))
    is_pdf = not is_url and input_source.lower().endswith(".pdf")
    server = click.get_current_context().obj.get("server")
    if server and not is_url and not is_pdf:
        # Read text file locally and send text to server
        if not os.path.exists(input_source):
            raise click.ClickException(f"File not found: {input_source}")
        with open(input_source, "r", encoding="utf-8") as f:
            doc_text = f.read()
        if not doc_text.strip():
            raise click.ClickException("Input file is empty")
        if not quiet:
            click.echo(f"Using server: {server}", err=True)
            click.echo(f"Processing document: {input_source} ({len(doc_text)} chars)", err=True)
        try:
            payload = {
                "text": doc_text,
                "title": title or Path(input_source).stem,
                "stages": stages,
                "max_tokens": max_tokens,
                "overlap": overlap,
                "no_summary": no_summary,
                "no_dedup": no_dedup,
            }
            result_data = _server_request(server, "/document", payload)
            # Reconstruct DocumentContext from server JSON and reuse local formatters
            from ..document.context import DocumentContext
            ctx = DocumentContext.model_validate(result_data)
            if output == "json":
                _print_document_json(ctx)
            elif output == "triples":
                _print_document_triples(ctx)
            else:
                _print_document_table(ctx, verbose)
            if not quiet:
                click.echo(f"\nChunks: {ctx.chunk_count}", err=True)
                click.echo(f"Statements: {ctx.statement_count}", err=True)
                if ctx.duplicates_removed > 0:
                    click.echo(f"Duplicates removed: {ctx.duplicates_removed}", err=True)
        except Exception as e:
            raise click.ClickException(f"Server request failed: {e}")
        return

    # Import document pipeline
    from ..document import DocumentPipeline, DocumentPipelineConfig, Document
    from ..models.document import ChunkingConfig
    from ..pipeline import PipelineConfig
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

    # Process
    try:
        if is_url:
            # Process URL
            from ..document import URLLoaderConfig

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

                from ..pipeline.registry import PluginRegistry

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

    from ..document import DocumentChunker, Document
    from ..models.document import ChunkingConfig

    config = ChunkingConfig(
        target_tokens=max_tokens,
        max_tokens=max_tokens * 2,
        overlap_tokens=overlap,
    )

    document = Document.from_text(text, title=Path(input_path).stem)
    chunker = DocumentChunker(config)
    chunks = chunker.chunk_document(document)

    if output == "json":
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
