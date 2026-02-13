"""
ExtractionPipeline - Main orchestrator for the 5-stage extraction pipeline.

Coordinates the flow of data through all pipeline stages:
1. Splitting: Text → SplitSentence (atomic sentences)
2. Extraction: SplitSentence → PipelineStatement (subject-predicate-object triples)
3. Qualification: Entity → CanonicalEntity
4. Labeling: Statement → LabeledStatement
5. Taxonomy: Statement → TaxonomyResult
"""

import logging
import time
from typing import Any, Optional

from .context import PipelineContext
from .config import PipelineConfig, get_stage_name
from .registry import PluginRegistry
from ..models import (
    QualifiedEntity,
    CanonicalEntity,
    LabeledStatement,
    TaxonomyResult,
)

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    Main pipeline orchestrator.

    Coordinates the flow of data through all 5 stages:
    1. Splitting: Text → SplitSentence (using splitter plugins)
    2. Extraction: SplitSentence → PipelineStatement (using extractor plugins)
    3. Qualification: Entity → CanonicalEntity (using qualifier + canonicalizer plugins)
    4. Labeling: Statement → LabeledStatement (using labeler plugins)
    5. Taxonomy: Statement → TaxonomyResult (using taxonomy plugins)
    """

    def __init__(self, config: Optional[PipelineConfig] = None, server_url: Optional[str] = None):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration (uses defaults if not provided)
            server_url: If provided, delegate processing to a running server
        """
        self.config = config or PipelineConfig.default()
        self._server_url = server_url

    def process(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PipelineContext:
        """
        Process text through the extraction pipeline.

        Args:
            text: Input text to process
            metadata: Optional metadata about the source

        Returns:
            PipelineContext with accumulated results from all stages
        """
        if self._server_url:
            from ..client import server_pipeline
            config_dict = self._config_to_dict()
            return server_pipeline(self._server_url, text, config_dict)

        # Merge config options into metadata for plugins
        combined_metadata = metadata.copy() if metadata else {}

        # Pass extractor options from config to context
        if self.config.extractor_options:
            existing_extractor_opts = combined_metadata.get("extractor_options", {})
            combined_metadata["extractor_options"] = {
                **self.config.extractor_options,
                **existing_extractor_opts,  # Allow explicit metadata to override config
            }

        ctx = PipelineContext(
            source_text=text,
            source_metadata=combined_metadata,
        )

        logger.info(f"Starting pipeline processing: {len(text)} chars")

        try:
            # Stage 1: Splitting
            if self.config.is_stage_enabled(1):
                ctx = self._run_splitting(ctx)

            # Stage 2: Extraction
            if self.config.is_stage_enabled(2):
                ctx = self._run_extraction(ctx)

            # Stage 3: Qualification (runs qualifiers + canonicalizers)
            if self.config.is_stage_enabled(3):
                ctx = self._run_qualification(ctx)

            # Stage 4: Labeling
            if self.config.is_stage_enabled(4):
                ctx = self._run_labeling(ctx)

            # Stage 5: Taxonomy classification
            if self.config.is_stage_enabled(5):
                ctx = self._run_taxonomy(ctx)

        except Exception as e:
            logger.exception("Pipeline processing failed")
            ctx.add_error(f"Pipeline error: {str(e)}")
            if self.config.fail_fast:
                raise

        logger.info(
            f"Pipeline complete: {ctx.statement_count} statements, "
            f"{len(ctx.processing_errors)} errors"
        )

        return ctx

    def _config_to_dict(self) -> dict[str, Any]:
        """Serialize PipelineConfig to the dict format the server expects."""
        d: dict[str, Any] = {}
        if self.config.enabled_stages:
            d["enabled_stages"] = sorted(self.config.enabled_stages)
        if self.config.disabled_plugins:
            d["disabled_plugins"] = sorted(self.config.disabled_plugins)
        if self.config.enabled_plugins:
            d["enabled_plugins"] = sorted(self.config.enabled_plugins)
        for key in ("extractor_options", "splitter_options", "qualifier_options",
                    "labeler_options", "taxonomy_options"):
            val = getattr(self.config, key, None)
            if val:
                d[key] = val
        return d

    def _run_splitting(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 1: Split text into atomic sentences."""
        stage_name = get_stage_name(1)
        logger.debug(f"Running {stage_name} stage")
        start_time = time.time()

        splitters = PluginRegistry.get_splitters()
        if not splitters:
            ctx.add_warning("No splitter plugins registered")
            return ctx

        # Use first enabled splitter (highest priority)
        for splitter in splitters:
            if not self.config.is_plugin_enabled(splitter.name):
                continue

            logger.debug(f"Using splitter: {splitter.name}")
            try:
                split_sentences = splitter.split(ctx.source_text, ctx)
                ctx.split_sentences = split_sentences
                logger.info(f"Splitting produced {len(split_sentences)} sentences")
                break
            except Exception as e:
                logger.exception(f"Splitter {splitter.name} failed")
                ctx.add_error(f"Splitter {splitter.name} failed: {str(e)}")
                if self.config.fail_fast:
                    raise

        ctx.record_timing(stage_name, time.time() - start_time)
        return ctx

    def _run_extraction(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 2: Extract subject-predicate-object triples from split sentences."""
        stage_name = get_stage_name(2)
        logger.debug(f"Running {stage_name} stage")
        start_time = time.time()

        if not ctx.split_sentences:
            logger.debug("No split sentences to extract from")
            return ctx

        extractors = PluginRegistry.get_extractors()
        if not extractors:
            ctx.add_warning("No extractor plugins registered")
            return ctx

        # Collect classification schemas from labelers for the extractor
        classification_schemas = self._collect_classification_schemas()
        if classification_schemas:
            logger.debug(f"Collected {len(classification_schemas)} classification schemas from labelers")

        # Use first enabled extractor (highest priority)
        for extractor in extractors:
            if not self.config.is_plugin_enabled(extractor.name):
                continue

            # Pass classification schemas to extractor if it supports them
            if classification_schemas and hasattr(extractor, 'add_classification_schema'):
                for schema in classification_schemas:
                    extractor.add_classification_schema(schema)

            logger.debug(f"Using extractor: {extractor.name}")
            try:
                statements = extractor.extract(ctx.split_sentences, ctx)
                ctx.statements = statements
                logger.info(f"Extraction produced {len(statements)} statements")
                break
            except Exception as e:
                logger.exception(f"Extractor {extractor.name} failed")
                ctx.add_error(f"Extractor {extractor.name} failed: {str(e)}")
                if self.config.fail_fast:
                    raise

        ctx.record_timing(stage_name, time.time() - start_time)
        return ctx

    def _collect_classification_schemas(self) -> list:
        """Collect classification schemas from enabled labelers."""
        schemas = []
        labelers = PluginRegistry.get_labelers()

        for labeler in labelers:
            if not self.config.is_plugin_enabled(labeler.name):
                continue

            # Check for classification schema (simple multi-choice)
            if hasattr(labeler, 'classification_schema') and labeler.classification_schema:
                schemas.append(labeler.classification_schema)
                logger.debug(
                    f"Labeler {labeler.name} provides classification schema: "
                    f"{labeler.classification_schema}"
                )

        return schemas

    def _run_qualification(self, ctx: PipelineContext) -> PipelineContext:
        """
        Stage 3: Qualify entities with identifiers, canonical names, and FQNs.

        Runs qualifier plugins for each entity type. Qualifier plugins now return
        CanonicalEntity directly (with qualifiers, canonical match, and FQN).
        """
        stage_name = get_stage_name(3)
        logger.debug(f"Running {stage_name} stage")
        start_time = time.time()

        if not ctx.statements:
            logger.debug("No statements to qualify")
            return ctx

        # Collect all unique entities from statements
        entities_to_qualify = {}
        for stmt in ctx.statements:
            for entity in [stmt.subject, stmt.object]:
                if entity.entity_ref not in entities_to_qualify:
                    entities_to_qualify[entity.entity_ref] = entity

        logger.info(f"Stage 3: Qualifying {len(entities_to_qualify)} unique entities")

        # Process each entity through qualifier plugins
        entities_list = list(entities_to_qualify.items())
        for idx, (entity_ref, entity) in enumerate(entities_list, 1):
            logger.info(f"  [{idx}/{len(entities_list)}] Qualifying '{entity.text}' ({entity.type.value})")

            # Run qualifier plugins - first one to return a result wins
            canonical = None
            type_qualifiers = PluginRegistry.get_qualifiers_for_type(entity.type)

            for qualifier_plugin in type_qualifiers:
                if not self.config.is_plugin_enabled(qualifier_plugin.name):
                    continue

                try:
                    result = qualifier_plugin.qualify(entity, ctx)
                    if result is not None:
                        canonical = result
                        logger.info(f"    Qualified by {qualifier_plugin.name}: {canonical.fqn}")
                        break  # Use first successful match
                except Exception as e:
                    logger.error(f"Qualifier {qualifier_plugin.name} failed for {entity.text}: {e}")
                    ctx.add_error(f"Qualifier {qualifier_plugin.name} failed: {str(e)}")
                    if self.config.fail_fast:
                        raise

            # Create fallback CanonicalEntity if no plugin matched
            if canonical is None:
                qualified = QualifiedEntity(
                    entity_ref=entity_ref,
                    original_text=entity.text,
                    entity_type=entity.type,
                )
                canonical = CanonicalEntity.from_qualified(qualified=qualified)
                logger.debug(f"    No qualification found, using original text")

            ctx.canonical_entities[entity_ref] = canonical

        logger.info(f"Qualified {len(ctx.canonical_entities)} entities")
        ctx.record_timing(stage_name, time.time() - start_time)
        return ctx

    def _run_labeling(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 4: Apply labels to statements."""
        stage_name = get_stage_name(4)
        logger.debug(f"Running {stage_name} stage")
        start_time = time.time()

        if not ctx.statements:
            logger.debug("No statements to label")
            return ctx

        # Ensure canonical entities exist (run qualification if skipped)
        if not ctx.canonical_entities:
            self._run_qualification(ctx)

        labelers = PluginRegistry.get_labelers()

        for stmt in ctx.statements:
            # Get canonical entities
            subj_canonical = ctx.canonical_entities.get(stmt.subject.entity_ref)
            obj_canonical = ctx.canonical_entities.get(stmt.object.entity_ref)

            if not subj_canonical or not obj_canonical:
                # Create fallback canonical entities
                if not subj_canonical:
                    subj_qualified = ctx.qualified_entities.get(
                        stmt.subject.entity_ref,
                        QualifiedEntity(
                            entity_ref=stmt.subject.entity_ref,
                            original_text=stmt.subject.text,
                            entity_type=stmt.subject.type,
                        )
                    )
                    subj_canonical = CanonicalEntity.from_qualified(subj_qualified)

                if not obj_canonical:
                    obj_qualified = ctx.qualified_entities.get(
                        stmt.object.entity_ref,
                        QualifiedEntity(
                            entity_ref=stmt.object.entity_ref,
                            original_text=stmt.object.text,
                            entity_type=stmt.object.type,
                        )
                    )
                    obj_canonical = CanonicalEntity.from_qualified(obj_qualified)

            # Create labeled statement
            labeled = LabeledStatement(
                statement=stmt,
                subject_canonical=subj_canonical,
                object_canonical=obj_canonical,
            )

            # Apply all labelers
            for labeler in labelers:
                if not self.config.is_plugin_enabled(labeler.name):
                    continue

                try:
                    label = labeler.label(stmt, subj_canonical, obj_canonical, ctx)
                    if label:
                        labeled.add_label(label)
                except Exception as e:
                    logger.error(f"Labeler {labeler.name} failed: {e}")
                    ctx.add_error(f"Labeler {labeler.name} failed: {str(e)}")
                    if self.config.fail_fast:
                        raise

            ctx.labeled_statements.append(labeled)

        logger.info(f"Labeled {len(ctx.labeled_statements)} statements")
        ctx.record_timing(stage_name, time.time() - start_time)
        return ctx

    def _run_taxonomy(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 5: Classify statements against taxonomies."""
        from ..plugins.base import PluginCapability

        stage_name = get_stage_name(5)
        logger.debug(f"Running {stage_name} stage")
        start_time = time.time()

        if not ctx.labeled_statements:
            logger.debug("No labeled statements to classify")
            return ctx

        taxonomy_classifiers = PluginRegistry.get_taxonomy_classifiers()
        if not taxonomy_classifiers:
            logger.debug("No taxonomy classifiers registered")
            return ctx

        total_results = 0

        # Prepare batch items: list of (statement, subject_canonical, object_canonical)
        batch_items = [
            (labeled_stmt.statement, labeled_stmt.subject_canonical, labeled_stmt.object_canonical)
            for labeled_stmt in ctx.labeled_statements
        ]

        # Apply all taxonomy classifiers
        for classifier in taxonomy_classifiers:
            if not self.config.is_plugin_enabled(classifier.name):
                continue

            try:
                # Require batch processing capability
                if PluginCapability.BATCH_PROCESSING not in classifier.capabilities:
                    raise RuntimeError(
                        f"Taxonomy classifier '{classifier.name}' does not support batch processing. "
                        "Pipeline requires BATCH_PROCESSING capability for efficient GPU utilization."
                    )

                logger.debug(f"Using batch classification for {classifier.name} ({len(batch_items)} items)")
                batch_results = classifier.classify_batch(batch_items, ctx)

                # Apply results to each labeled statement
                for labeled_stmt, results in zip(ctx.labeled_statements, batch_results):
                    if results:
                        stmt = labeled_stmt.statement
                        key = (stmt.source_text, classifier.taxonomy_name)
                        if key not in ctx.taxonomy_results:
                            ctx.taxonomy_results[key] = []
                        ctx.taxonomy_results[key].extend(results)
                        total_results += len(results)
                        labeled_stmt.taxonomy_results.extend(results)

                        for result in results:
                            logger.debug(
                                f"Taxonomy {classifier.name}: {result.full_label} "
                                f"(confidence={result.confidence:.2f})"
                            )

            except Exception as e:
                logger.error(f"Taxonomy classifier {classifier.name} failed: {e}")
                ctx.add_error(f"Taxonomy classifier {classifier.name} failed: {str(e)}")
                if self.config.fail_fast:
                    raise

        logger.info(f"Taxonomy produced {total_results} labels across {len(ctx.taxonomy_results)} statement-taxonomy pairs")
        ctx.record_timing(stage_name, time.time() - start_time)
        return ctx
