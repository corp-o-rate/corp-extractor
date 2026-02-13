"""
DocumentPipeline - Orchestrates document-level extraction with chunking and citations.

Wraps ExtractionPipeline to provide document-level features:
- Text chunking with page awareness
- Batch processing through pipeline stages
- Statement deduplication across chunks
- Citation generation from document metadata
"""

import logging
import time
from typing import Optional

from pydantic import BaseModel, Field

from ..models.document import ChunkingConfig, Document
from ..pipeline import ExtractionPipeline, PipelineConfig
from .chunker import DocumentChunker
from .context import DocumentContext
from .deduplicator import StatementDeduplicator
from .summarizer import DocumentSummarizer

logger = logging.getLogger(__name__)


class DocumentPipelineConfig(BaseModel):
    """Configuration for document pipeline processing."""

    chunking: ChunkingConfig = Field(
        default_factory=ChunkingConfig,
        description="Configuration for text chunking"
    )
    generate_summary: bool = Field(
        default=True,
        description="Whether to generate a document summary"
    )
    deduplicate_across_chunks: bool = Field(
        default=True,
        description="Whether to deduplicate statements across chunks"
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        description="Number of items to process in each batch"
    )
    pipeline_config: Optional[PipelineConfig] = Field(
        default=None,
        description="Configuration for the underlying extraction pipeline"
    )


class DocumentPipeline:
    """
    Document-level extraction pipeline.

    Processes documents through:
    1. Summary generation (optional)
    2. Chunking with page awareness
    3. Batch extraction through all pipeline stages
    4. Deduplication across chunks
    5. Citation generation

    Example:
        >>> pipeline = DocumentPipeline()
        >>> document = Document.from_text("Long document text...", title="Report")
        >>> ctx = pipeline.process(document)
        >>> for stmt in ctx.labeled_statements:
        ...     print(f"{stmt}: {stmt.citation}")
    """

    def __init__(
        self,
        config: Optional[DocumentPipelineConfig] = None,
        server_url: Optional[str] = None,
    ):
        """
        Initialize the document pipeline.

        Args:
            config: Document pipeline configuration
            server_url: If provided, delegate processing to a running server
        """
        self.config = config or DocumentPipelineConfig()
        self._server_url = server_url

        if not server_url:
            # Only initialize local components when processing locally
            self._chunker = DocumentChunker(self.config.chunking)
            self._deduplicator = StatementDeduplicator()
            self._summarizer = DocumentSummarizer() if self.config.generate_summary else None
            self._pipeline = ExtractionPipeline(self.config.pipeline_config)

    def process(self, document: Document) -> DocumentContext:
        """
        Process a document through the pipeline.

        Args:
            document: Document to process

        Returns:
            DocumentContext with all extraction results
        """
        if self._server_url:
            from ..client import server_document
            kwargs = {
                "title": document.metadata.title,
                "stages": ",".join(str(s) for s in sorted(self.config.pipeline_config.enabled_stages)) if self.config.pipeline_config else "1-6",
                "max_tokens": self.config.chunking.target_tokens,
                "overlap": self.config.chunking.overlap_tokens,
                "no_summary": not self.config.generate_summary,
                "no_dedup": not self.config.deduplicate_across_chunks,
            }
            return server_document(self._server_url, document.full_text, **kwargs)

        logger.info(f"Starting document pipeline: {document.document_id}")
        start_time = time.time()

        ctx = DocumentContext(document=document)

        try:
            # Step 1: Generate summary (if enabled)
            if self.config.generate_summary and self._summarizer:
                self._generate_summary(document, ctx)

            # Step 2: Chunk the document
            chunks = self._chunker.chunk_document(document)
            ctx.chunks = chunks
            logger.info(f"Created {len(chunks)} chunks")

            if not chunks:
                logger.warning("No chunks created from document")
                return ctx

            # Step 3: Process all chunks through Stage 1 (Splitting)
            self._process_stage1(ctx)

            # Step 4: Deduplicate raw triples
            if self.config.deduplicate_across_chunks:
                self._deduplicate_triples(ctx)

            # Step 5: Process through remaining stages (2-6)
            self._process_remaining_stages(ctx)

            # Step 6: Add citations to statements
            self._add_citations(ctx)

        except Exception as e:
            logger.exception("Document pipeline failed")
            ctx.add_error(f"Pipeline error: {str(e)}")
            raise

        total_time = time.time() - start_time
        ctx.record_timing("total", total_time)

        logger.info(
            f"Document pipeline complete: {ctx.statement_count} statements, "
            f"{ctx.duplicates_removed} duplicates removed, "
            f"{total_time:.2f}s"
        )

        return ctx

    def _generate_summary(self, document: Document, ctx: DocumentContext) -> None:
        """Generate document summary."""
        logger.info("Generating document summary")
        start_time = time.time()

        try:
            summary = self._summarizer.summarize(document)
            document.summary = summary
            logger.info(f"Generated summary: {len(summary)} chars")
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            ctx.add_warning(f"Summary generation failed: {e}")

        ctx.record_timing("summarization", time.time() - start_time)

    def _process_stage1(self, ctx: DocumentContext) -> None:
        """Process all chunks through Stage 1 (Splitting) using batch processing."""
        from ..pipeline.registry import PluginRegistry
        from ..plugins.base import PluginCapability

        logger.info(f"Processing {len(ctx.chunks)} chunks through Stage 1 (batch mode)")
        start_time = time.time()

        # Get the splitter plugin
        splitters = PluginRegistry.get_splitters()
        if not splitters:
            logger.warning("No splitter plugins registered")
            return

        # Use first enabled splitter
        splitter = None
        for s in splitters:
            plugin_enabled = (
                self.config.pipeline_config is None or
                self.config.pipeline_config.is_plugin_enabled(s.name)
            )
            if plugin_enabled:
                splitter = s
                break

        if not splitter:
            logger.warning("No enabled splitter plugin found")
            return

        # Extract all chunk texts
        chunk_texts = [chunk.text for chunk in ctx.chunks]

        # Create a dummy context for the splitter
        from ..pipeline.context import PipelineContext
        dummy_ctx = PipelineContext(
            source_text="",  # Not used for batch splitting
            source_metadata=self.config.pipeline_config.model_dump() if self.config.pipeline_config else {},
        )

        all_triples = []

        # Require batch processing capability
        if PluginCapability.BATCH_PROCESSING not in splitter.capabilities:
            raise RuntimeError(
                f"Splitter plugin '{splitter.name}' does not support batch processing. "
                "Document pipeline requires BATCH_PROCESSING capability for efficient GPU utilization."
            )

        logger.info(f"Using batch splitting with {splitter.name}")
        batch_results = splitter.split_batch(chunk_texts, dummy_ctx)

        # Annotate triples with document/chunk info
        for chunk, triples in zip(ctx.chunks, batch_results):
            for triple in triples:
                triple.document_id = ctx.document.document_id
                triple.chunk_index = chunk.chunk_index
                triple.page_number = chunk.primary_page
                all_triples.append(triple)

        ctx.raw_triples = all_triples
        ctx.pre_dedup_count = len(all_triples)

        ctx.record_timing("stage1_batch", time.time() - start_time)
        logger.info(f"Stage 1 produced {len(all_triples)} raw triples from {len(ctx.chunks)} chunks")

    def _deduplicate_triples(self, ctx: DocumentContext) -> None:
        """Deduplicate raw triples across chunks."""
        logger.info(f"Deduplicating {len(ctx.raw_triples)} triples")
        start_time = time.time()

        original_count = len(ctx.raw_triples)
        ctx.raw_triples = self._deduplicator.deduplicate_batch(ctx.raw_triples)
        ctx.post_dedup_count = len(ctx.raw_triples)

        removed = original_count - len(ctx.raw_triples)
        ctx.record_timing("deduplication", time.time() - start_time)
        logger.info(f"Removed {removed} duplicate triples")

    def _process_remaining_stages(self, ctx: DocumentContext) -> None:
        """Process through stages 2-6."""
        logger.info(f"Processing {len(ctx.raw_triples)} triples through stages 2-6")
        start_time = time.time()

        # Create a pipeline config for stages 2-6
        # Exclude enabled_stages from base config to avoid duplicate keyword argument
        base_config = {}
        if self.config.pipeline_config:
            base_config = self.config.pipeline_config.model_dump(exclude={"enabled_stages"})
        stages_config = PipelineConfig(
            enabled_stages={2, 3, 4, 5, 6},
            **base_config
        )

        # Create a combined context with all raw triples
        from ..pipeline.context import PipelineContext

        combined_ctx = PipelineContext(
            source_text=ctx.document.full_text,
            source_metadata={
                "document_id": ctx.document.document_id,
                "title": ctx.document.metadata.title,
            },
        )
        combined_ctx.raw_triples = ctx.raw_triples

        # Run stages 2-6
        pipeline = ExtractionPipeline(stages_config)

        # We need to manually run stages since we're providing pre-existing triples
        # Stage 2: Extraction
        if stages_config.is_stage_enabled(2):
            combined_ctx = pipeline._run_extraction(combined_ctx)

        # Propagate document info to statements
        for stmt in combined_ctx.statements:
            # Find the source triple to get document info
            for triple in ctx.raw_triples:
                if triple.source_sentence in stmt.source_text:
                    stmt.document_id = triple.document_id
                    stmt.chunk_index = triple.chunk_index
                    stmt.page_number = triple.page_number
                    break

        # Stage 3: Qualification
        if stages_config.is_stage_enabled(3):
            combined_ctx = pipeline._run_qualification(combined_ctx)

        # Stage 4: Canonicalization
        if stages_config.is_stage_enabled(4):
            combined_ctx = pipeline._run_canonicalization(combined_ctx)

        # Stage 5: Labeling
        if stages_config.is_stage_enabled(5):
            combined_ctx = pipeline._run_labeling(combined_ctx)

        # Stage 6: Taxonomy
        if stages_config.is_stage_enabled(6):
            combined_ctx = pipeline._run_taxonomy(combined_ctx)

        # Propagate document info to labeled statements
        for labeled_stmt in combined_ctx.labeled_statements:
            labeled_stmt.document_id = labeled_stmt.statement.document_id
            labeled_stmt.page_number = labeled_stmt.statement.page_number

        ctx.statements = combined_ctx.statements
        ctx.labeled_statements = combined_ctx.labeled_statements

        # Merge timings
        for stage, duration in combined_ctx.stage_timings.items():
            ctx.record_timing(stage, duration)

        ctx.record_timing("stages_2_6_batch", time.time() - start_time)
        logger.info(f"Stages 2-6 produced {len(ctx.labeled_statements)} labeled statements")

    def _add_citations(self, ctx: DocumentContext) -> None:
        """Add citations to all labeled statements."""
        logger.info("Adding citations to statements")

        for stmt in ctx.labeled_statements:
            citation = ctx.document.metadata.format_citation(stmt.page_number)
            stmt.citation = citation if citation else None

    def process_text(
        self,
        text: str,
        title: Optional[str] = None,
        **metadata_kwargs,
    ) -> DocumentContext:
        """
        Process plain text through the document pipeline.

        Convenience method that creates a Document from text.

        Args:
            text: Text to process
            title: Optional document title
            **metadata_kwargs: Additional document metadata

        Returns:
            DocumentContext with extraction results
        """
        document = Document.from_text(text, title=title, **metadata_kwargs)
        return self.process(document)

    async def process_url(
        self,
        url: str,
        loader_config: Optional["URLLoaderConfig"] = None,
    ) -> DocumentContext:
        """
        Process a URL through the document pipeline.

        Fetches the URL, extracts content (HTML or PDF), and processes it.

        Args:
            url: URL to process
            loader_config: Optional loader configuration

        Returns:
            DocumentContext with extraction results
        """
        from .loader import URLLoader, URLLoaderConfig

        loader = URLLoader(loader_config or URLLoaderConfig())
        document = await loader.load(url)
        return self.process(document)

    def process_url_sync(
        self,
        url: str,
        loader_config: Optional["URLLoaderConfig"] = None,
    ) -> DocumentContext:
        """
        Process a URL through the document pipeline (synchronous).

        Fetches the URL, extracts content (HTML or PDF), and processes it.

        Args:
            url: URL to process
            loader_config: Optional loader configuration

        Returns:
            DocumentContext with extraction results
        """
        import asyncio
        return asyncio.run(self.process_url(url, loader_config))
