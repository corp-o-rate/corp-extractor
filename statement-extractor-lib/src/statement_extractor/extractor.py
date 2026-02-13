"""
Statement Extractor - Extract structured statements from text using T5-Gemma 2.

This module uses Diverse Beam Search (Vijayakumar et al., 2016) to generate
multiple candidate extractions and selects/merges the best results using
quality scoring.

Paper: https://arxiv.org/abs/1610.02424
"""

import logging
import re
import xml.etree.ElementTree as ET
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from .models import (
    Entity,
    EntityType,
    ExtractionMethod,
    ExtractionOptions,
    ExtractionResult,
    PredicateComparisonConfig,
    PredicateTaxonomy,
    ScoringConfig,
    Statement,
)

logger = logging.getLogger(__name__)


class StopOnSequence(StoppingCriteria):
    """
    Stop generation when a specific multi-token sequence is generated.

    Decodes the generated tokens and checks if the stop sequence appears.
    Works with sequences that span multiple tokens (e.g., "</statements>").
    """

    def __init__(self, tokenizer, stop_sequence: str, input_length: int):
        self.tokenizer = tokenizer
        self.stop_sequence = stop_sequence
        self.input_length = input_length
        # Track which beams have stopped (for batch generation)
        self.stopped = set()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check each sequence in the batch
        for idx, seq in enumerate(input_ids):
            if idx in self.stopped:
                continue
            # Only decode the generated portion (after input)
            generated = seq[self.input_length:]
            decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
            if self.stop_sequence in decoded:
                self.stopped.add(idx)

        # Stop when ALL sequences have the stop sequence
        return len(self.stopped) >= len(input_ids)


def repair_xml(xml_string: str) -> tuple[str, list[str]]:
    """
    Attempt to repair common XML syntax errors.

    Returns:
        Tuple of (repaired_xml, list_of_repairs_made)
    """
    repairs = []
    original = xml_string

    # 1. Fix unescaped ampersands (but not already escaped entities)
    # Match & not followed by amp; lt; gt; quot; apos; or #
    ampersand_pattern = r'&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)'
    if re.search(ampersand_pattern, xml_string):
        xml_string = re.sub(ampersand_pattern, '&amp;', xml_string)
        repairs.append("escaped unescaped ampersands")

    # 2. Fix unescaped < and > inside text content (not tags)
    # This is tricky - we need to be careful not to break actual tags
    # For now, just handle the most common case: < followed by space or lowercase
    less_than_pattern = r'<(?=\s|[a-z]{2,}[^a-z/>])'
    if re.search(less_than_pattern, xml_string):
        xml_string = re.sub(less_than_pattern, '&lt;', xml_string)
        repairs.append("escaped unescaped less-than signs")

    # 3. Fix truncated closing tags (e.g., "</statemen" -> try to complete)
    truncated_patterns = [
        (r'</statement[^s>]*$', '</statements>'),
        (r'</stm[^t>]*$', '</stmt>'),
        (r'</subjec[^t>]*$', '</subject>'),
        (r'</objec[^t>]*$', '</object>'),
        (r'</predica[^t>]*$', '</predicate>'),
        (r'</tex[^t>]*$', '</text>'),
    ]
    for pattern, replacement in truncated_patterns:
        if re.search(pattern, xml_string):
            xml_string = re.sub(pattern, replacement, xml_string)
            repairs.append(f"completed truncated tag: {replacement}")

    # 4. Add missing </statements> if we have <statements> but no closing
    if '<statements>' in xml_string and '</statements>' not in xml_string:
        # Try to find a good place to add it
        # Look for the last complete </stmt> and add after it
        last_stmt = xml_string.rfind('</stmt>')
        if last_stmt != -1:
            insert_pos = last_stmt + len('</stmt>')
            xml_string = xml_string[:insert_pos] + '</statements>'
            repairs.append("added missing </statements> after last </stmt>")
        else:
            xml_string = xml_string + '</statements>'
            repairs.append("added missing </statements> at end")

    # 5. Fix unclosed <stmt> tags - find <stmt> without matching </stmt>
    # Count opens and closes
    open_stmts = len(re.findall(r'<stmt>', xml_string))
    close_stmts = len(re.findall(r'</stmt>', xml_string))
    if open_stmts > close_stmts:
        # Find incomplete statement blocks and try to close them
        # Look for patterns like <stmt>...<subject>...</subject> without </stmt>
        # This is complex, so just add closing tags before </statements>
        missing = open_stmts - close_stmts
        if '</statements>' in xml_string:
            xml_string = xml_string.replace('</statements>', '</stmt>' * missing + '</statements>')
            repairs.append(f"added {missing} missing </stmt> tag(s)")

    # 6. Remove any content after </statements>
    end_pos = xml_string.find('</statements>')
    if end_pos != -1:
        end_pos += len('</statements>')
        if end_pos < len(xml_string):
            xml_string = xml_string[:end_pos]
            repairs.append("removed content after </statements>")

    if xml_string != original:
        return xml_string, repairs
    return xml_string, []

# Default model
DEFAULT_MODEL_ID = "Corp-o-Rate-Community/statement-extractor"


class StatementExtractor:
    """
    Extract structured statements from unstructured text.

    Uses the T5-Gemma 2 statement extraction model with Diverse Beam Search
    to generate high-quality subject-predicate-object triples.

    Features:
    - Quality-based beam scoring (not just longest output)
    - Beam merging for better coverage
    - Embedding-based predicate comparison for smart deduplication
    - Configurable precision/recall tradeoff

    Example:
        >>> extractor = StatementExtractor()
        >>> result = extractor.extract("Apple Inc. announced a new iPhone today.")
        >>> for stmt in result:
        ...     print(stmt)
        Apple Inc. -- announced --> a new iPhone
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        predicate_taxonomy: Optional[PredicateTaxonomy] = None,
        predicate_config: Optional[PredicateComparisonConfig] = None,
        scoring_config: Optional[ScoringConfig] = None,
    ):
        """
        Initialize the statement extractor.

        Args:
            model_id: HuggingFace model ID or local path
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            torch_dtype: Torch dtype (default: bfloat16 on GPU, float32 on CPU)
            predicate_taxonomy: Optional taxonomy for predicate normalization
            predicate_config: Configuration for predicate comparison
            scoring_config: Configuration for quality scoring
        """
        self.model_id = model_id
        self._model: Optional[AutoModelForSeq2SeqLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Auto-detect dtype (bfloat16 only for CUDA, float32 for MPS/CPU)
        if torch_dtype is None:
            self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        else:
            self.torch_dtype = torch_dtype

        # Scoring and comparison config
        self._predicate_taxonomy = predicate_taxonomy
        self._predicate_config = predicate_config
        self._scoring_config = scoring_config

        # Lazy-loaded components
        self._beam_scorer = None
        self._predicate_comparer = None

    def _load_model(self) -> None:
        """Load model and tokenizer if not already loaded."""
        if self._model is not None:
            return

        logger.info(f"Loading model: {self.model_id}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        if self.device == "cuda":
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                device_map="auto",
            )
        else:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            self._model = self._model.to(self.device)

        logger.info(f"Model loaded on {self.device}")

    def _get_beam_scorer(self, options: ExtractionOptions):
        """Get or create beam scorer with current config."""
        from .scoring import BeamScorer

        config = options.scoring_config or self._scoring_config or ScoringConfig()
        return BeamScorer(config=config)

    def _get_predicate_comparer(self, options: ExtractionOptions):
        """Get or create predicate comparer if embeddings enabled."""
        if not options.embedding_dedup:
            return None

        from .predicate_comparer import PredicateComparer

        taxonomy = options.predicate_taxonomy or self._predicate_taxonomy
        config = options.predicate_config or self._predicate_config or PredicateComparisonConfig()
        return PredicateComparer(taxonomy=taxonomy, config=config, device=self.device)

    @property
    def model(self) -> AutoModelForSeq2SeqLM:
        """Get the model, loading it if necessary."""
        self._load_model()
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer, loading it if necessary."""
        self._load_model()
        return self._tokenizer

    def extract(
        self,
        text: str,
        options: Optional[ExtractionOptions] = None,
    ) -> ExtractionResult:
        """
        Extract statements from text.

        Args:
            text: Input text to extract statements from
            options: Extraction options (uses defaults if not provided)

        Returns:
            ExtractionResult containing the extracted statements
        """
        if options is None:
            options = ExtractionOptions()

        logger.debug("=" * 60)
        logger.debug("EXTRACTION STARTED")
        logger.debug("=" * 60)
        logger.debug(f"Input text length: {len(text)} chars")
        logger.debug(f"Options: num_beams={options.num_beams}, diversity={options.diversity_penalty}")
        logger.debug(f"  merge_beams={options.merge_beams}, embedding_dedup={options.embedding_dedup}")
        logger.debug(f"  deduplicate={options.deduplicate}, max_new_tokens={options.max_new_tokens}")

        # Store original text for scoring
        original_text = text

        # Wrap text in page tags if not already wrapped
        if not text.startswith("<page>"):
            text = f"<page>{text}</page>"

        # Run extraction with retry logic
        statements = self._extract_with_scoring(text, original_text, options)

        logger.debug("=" * 60)
        logger.debug(f"EXTRACTION COMPLETE: {len(statements)} statements")
        logger.debug("=" * 60)

        return ExtractionResult(
            statements=statements,
            source_text=original_text,
        )

    def extract_as_xml(
        self,
        text: str,
        options: Optional[ExtractionOptions] = None,
    ) -> str:
        """
        Extract statements and return raw XML output.

        Note: This bypasses the new scoring/merging logic for backward compatibility.
        Use extract() for full quality scoring.

        Args:
            text: Input text to extract statements from
            options: Extraction options

        Returns:
            XML string with <statements> containing <stmt> elements
        """
        if options is None:
            options = ExtractionOptions()

        if not text.startswith("<page>"):
            text = f"<page>{text}</page>"

        return self._extract_raw_xml(text, options)

    def extract_as_json(
        self,
        text: str,
        options: Optional[ExtractionOptions] = None,
        indent: Optional[int] = 2,
    ) -> str:
        """
        Extract statements and return JSON string.

        Args:
            text: Input text to extract statements from
            options: Extraction options
            indent: JSON indentation (None for compact)

        Returns:
            JSON string representation of the extraction result
        """
        result = self.extract(text, options)
        return result.model_dump_json(indent=indent)

    def extract_as_dict(
        self,
        text: str,
        options: Optional[ExtractionOptions] = None,
    ) -> dict:
        """
        Extract statements and return as dictionary.

        Args:
            text: Input text to extract statements from
            options: Extraction options

        Returns:
            Dictionary representation of the extraction result
        """
        result = self.extract(text, options)
        return result.model_dump()

    def _extract_with_scoring(
        self,
        text: str,
        original_text: str,
        options: ExtractionOptions,
    ) -> list[Statement]:
        """
        Extract statements with quality scoring and beam merging.

        This is the new extraction pipeline that:
        1. Generates multiple candidates via DBS
        2. Parses each to statements
        3. Scores each triple for quality (semantic + entity)
        4. Merges top beams or selects best beam
        5. Deduplicates using embeddings (if enabled)
        """
        logger.debug("-" * 40)
        logger.debug("PHASE 1: Tokenization")
        logger.debug("-" * 40)

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=4096,
            truncation=True,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        logger.debug(f"Tokenized: {input_ids.shape[1]} tokens")

        # Count sentences for quality check
        num_sentences = self._count_sentences(text)
        min_expected = int(num_sentences * options.min_statement_ratio)

        logger.debug(f"Input has ~{num_sentences} sentences, min expected: {min_expected}")

        # Get beam scorer
        beam_scorer = self._get_beam_scorer(options)

        logger.debug("-" * 40)
        logger.debug("PHASE 2: Diverse Beam Search Generation")
        logger.debug("-" * 40)

        all_candidates: list[list[Statement]] = []

        for attempt in range(options.max_attempts):
            logger.debug(f"Attempt {attempt + 1}/{options.max_attempts}: Generating {options.num_beams} beams...")

            # Generate candidate beams
            candidates = self._generate_candidate_beams(inputs, options)
            logger.debug(f"  Generated {len(candidates)} valid XML outputs")

            # Parse each candidate to statements
            parsed_candidates = []
            for i, xml_output in enumerate(candidates):
                statements = self._parse_xml_to_statements(xml_output, options)
                if statements:
                    parsed_candidates.append(statements)
                    logger.debug(f"  Beam {i}: {len(statements)} statements parsed")
                else:
                    logger.warning(f"  Beam {i}: 0 statements (parse failed)")
                    logger.warning(f"  Beam {i} XML output:\n{xml_output}")

            all_candidates.extend(parsed_candidates)

            # Check if we have enough statements
            total_stmts = sum(len(c) for c in parsed_candidates)
            logger.debug(f"  Total: {len(parsed_candidates)} beams, {total_stmts} statements")

            if total_stmts >= min_expected:
                logger.debug(f"  Sufficient statements ({total_stmts} >= {min_expected}), stopping")
                break

        if not all_candidates:
            logger.debug("No valid candidates generated, returning empty result")
            return []

        logger.debug("-" * 40)
        logger.debug("PHASE 3: Beam Selection/Merging")
        logger.debug("-" * 40)

        # Select or merge beams
        if options.merge_beams:
            logger.debug(f"Merging {len(all_candidates)} beams...")
            statements = beam_scorer.merge_beams(all_candidates, original_text)
            logger.debug(f"  After merge: {len(statements)} statements")
        else:
            logger.debug(f"Selecting best beam from {len(all_candidates)} candidates...")
            statements = beam_scorer.select_best_beam(all_candidates, original_text)
            logger.debug(f"  Selected beam has {len(statements)} statements")

        logger.debug("-" * 40)
        logger.debug("PHASE 4: Deduplication")
        logger.debug("-" * 40)

        # Apply embedding-based deduplication if enabled
        if options.embedding_dedup and options.deduplicate:
            logger.debug("Using embedding-based deduplication...")
            pre_dedup_count = len(statements)
            try:
                comparer = self._get_predicate_comparer(options)
                if comparer:
                    statements = comparer.deduplicate_statements(
                        statements,
                        entity_canonicalizer=options.entity_canonicalizer
                    )
                    logger.debug(f"  After embedding dedup: {len(statements)} statements (removed {pre_dedup_count - len(statements)})")

                    # Also normalize predicates if taxonomy provided
                    if options.predicate_taxonomy or self._predicate_taxonomy:
                        logger.debug("Normalizing predicates to taxonomy...")
                        statements = comparer.normalize_predicates(statements)
            except Exception as e:
                logger.warning(f"Embedding deduplication failed, falling back to exact match: {e}")
                statements = self._deduplicate_statements_exact(statements, options)
                logger.debug(f"  After exact dedup: {len(statements)} statements")
        elif options.deduplicate:
            logger.debug("Using exact text deduplication...")
            pre_dedup_count = len(statements)
            statements = self._deduplicate_statements_exact(statements, options)
            logger.debug(f"  After exact dedup: {len(statements)} statements (removed {pre_dedup_count - len(statements)})")
        else:
            logger.debug("Deduplication disabled")

        # Select best triple per source text (unless all_triples enabled)
        if not options.all_triples:
            logger.debug("-" * 40)
            logger.debug("PHASE 5: Best Triple Selection")
            logger.debug("-" * 40)
            pre_select_count = len(statements)
            statements = self._select_best_per_source(statements)
            logger.debug(f"  Selected best per source: {len(statements)} statements (from {pre_select_count})")

        # Log final statements
        logger.debug("-" * 40)
        logger.debug("FINAL STATEMENTS:")
        logger.debug("-" * 40)
        for i, stmt in enumerate(statements):
            conf = f" (conf={stmt.confidence_score:.2f})" if stmt.confidence_score else ""
            canonical = f" -> {stmt.canonical_predicate}" if stmt.canonical_predicate else ""
            logger.debug(f"  {i+1}. {stmt.subject.text} --[{stmt.predicate}{canonical}]--> {stmt.object.text}{conf}")

        return statements

    def _generate_candidate_beams(
        self,
        inputs,
        options: ExtractionOptions,
    ) -> list[str]:
        """Generate multiple candidate beams using diverse beam search."""
        num_seqs = options.num_beams

        # Create stopping criteria to stop when </statements> is generated
        input_length = inputs["input_ids"].shape[1]
        stop_criteria = StopOnSequence(
            tokenizer=self.tokenizer,
            stop_sequence="</statements>",
            input_length=input_length,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=options.max_new_tokens,
                max_length=None,  # Override model default, use max_new_tokens only
                num_beams=num_seqs,
                num_beam_groups=num_seqs,
                num_return_sequences=num_seqs,
                diversity_penalty=options.diversity_penalty,
                do_sample=False,
                top_p=None,  # Override model config to suppress warning
                top_k=None,  # Override model config to suppress warning
                trust_remote_code=True,
                custom_generate="transformers-community/group-beam-search",
                stopping_criteria=StoppingCriteriaList([stop_criteria]),
            )

        # Decode and process candidates
        end_tag = "</statements>"
        candidates: list[str] = []

        for i, output in enumerate(outputs):
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            output_len = len(output)

            # Truncate at </statements>
            if end_tag in decoded:
                end_pos = decoded.find(end_tag) + len(end_tag)
                decoded = decoded[:end_pos]
                candidates.append(decoded)
                logger.debug(f"Beam {i}: {output_len} tokens, found end tag, {len(decoded)} chars")
            else:
                # Log the issue - likely truncated
                logger.warning(f"Beam {i}: {output_len} tokens, NO end tag found (truncated?)")
                logger.warning(f"Beam {i} full output ({len(decoded)} chars):\n{decoded}")

        # Include fallback if no valid candidates
        if not candidates and len(outputs) > 0:
            fallback = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.warning(f"Using fallback beam (no valid candidates found), {len(fallback)} chars")
            candidates.append(fallback)

        return candidates

    def _extract_raw_xml(
        self,
        text: str,
        options: ExtractionOptions,
    ) -> str:
        """
        Extract and return raw XML (legacy method for backward compatibility).

        Uses length-based selection like the original implementation.
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=4096,
            truncation=True,
        ).to(self.device)

        num_sentences = self._count_sentences(text)
        min_expected = int(num_sentences * options.min_statement_ratio)

        all_results: list[tuple[str, int]] = []

        for attempt in range(options.max_attempts):
            candidates = self._generate_candidate_beams(inputs, options)

            for candidate in candidates:
                if options.deduplicate:
                    candidate = self._deduplicate_xml(candidate)
                num_stmts = self._count_statements(candidate)
                all_results.append((candidate, num_stmts))

            best_so_far = max(all_results, key=lambda x: x[1])[1] if all_results else 0
            if best_so_far >= min_expected:
                break

        if not all_results:
            return "<statements></statements>"

        # Select best result (longest, for backward compatibility)
        return max(all_results, key=lambda x: len(x[0]))[0]

    def _deduplicate_statements_exact(
        self,
        statements: list[Statement],
        options: ExtractionOptions,
    ) -> list[Statement]:
        """Deduplicate statements using exact text matching."""
        from .canonicalization import deduplicate_statements_exact
        return deduplicate_statements_exact(
            statements,
            entity_canonicalizer=options.entity_canonicalizer
        )

    def _select_best_per_source(
        self,
        statements: list[Statement],
    ) -> list[Statement]:
        """
        Select the highest-scoring triple for each unique source text.

        Groups statements by source_text and keeps only the one with
        the highest confidence_score from each group.

        Statements without source_text are kept as-is.
        """
        if not statements:
            return statements

        # Group by source_text
        from collections import defaultdict
        groups: dict[str | None, list[Statement]] = defaultdict(list)

        for stmt in statements:
            groups[stmt.source_text].append(stmt)

        # Select best from each group
        result: list[Statement] = []

        for source_text, group in groups.items():
            if source_text is None or len(group) == 1:
                # No source text or only one statement - keep as-is
                result.extend(group)
            else:
                # Multiple candidates for same source - select best
                best = max(
                    group,
                    key=lambda s: s.confidence_score if s.confidence_score is not None else 0.0
                )
                logger.debug(
                    f"  Selected best for source '{source_text[:40]}...': "
                    f"'{best.subject.text}' --[{best.predicate}]--> '{best.object.text}' "
                    f"(score={best.confidence_score:.2f}, method={best.extraction_method.value})"
                )
                result.append(best)

        return result

    def _deduplicate_xml(self, xml_output: str) -> str:
        """Remove duplicate <stmt> blocks from XML output (legacy method)."""
        try:
            root = ET.fromstring(xml_output)
        except ET.ParseError:
            return xml_output

        if root.tag != 'statements':
            return xml_output

        seen: set[tuple[str, str, str]] = set()
        unique_stmts: list[ET.Element] = []

        for stmt in root.findall('stmt'):
            subject = stmt.findtext('subject', '').strip().lower()
            predicate = stmt.findtext('predicate', '').strip().lower()
            obj = stmt.findtext('object', '').strip().lower()
            key = (subject, predicate, obj)

            if key not in seen:
                seen.add(key)
                unique_stmts.append(stmt)

        new_root = ET.Element('statements')
        for stmt in unique_stmts:
            new_root.append(stmt)

        return ET.tostring(new_root, encoding='unicode')

    def _parse_xml_to_statements(
        self,
        xml_output: str,
        options: Optional[ExtractionOptions] = None,
    ) -> list[Statement]:
        """
        Parse XML output into Statement objects.

        Uses model for subject, object, entity types, and source_text.
        Always uses GLiNER2 for predicate extraction (model predicates are unreliable).

        Produces two candidates for each statement:
        1. Hybrid: model subject/object + GLiNER2 predicate
        2. GLiNER2-only: all components from GLiNER2

        Both go into the candidate pool; scoring/dedup picks the best.
        """
        statements: list[Statement] = []
        use_gliner_extraction = options.use_gliner_extraction if options else True
        predicates = options.predicates if options else None

        try:
            root = ET.fromstring(xml_output)
        except ET.ParseError as e:
            # Log full output for debugging
            logger.debug(f"Initial XML parse failed: {e}")
            logger.debug(f"Raw XML output ({len(xml_output)} chars):\n{xml_output}")

            # Try to repair the XML
            repaired_xml, repairs = repair_xml(xml_output)
            if repairs:
                logger.debug(f"Attempted XML repairs: {', '.join(repairs)}")
                try:
                    root = ET.fromstring(repaired_xml)
                    logger.info(f"XML repair successful, parsing repaired output")
                except ET.ParseError as e2:
                    logger.warning(f"XML repair failed, still cannot parse: {e2}")
                    logger.warning(f"Repaired XML ({len(repaired_xml)} chars):\n{repaired_xml}")
                    return statements
            else:
                logger.warning(f"No repairs possible for XML output")
                logger.warning(f"Full XML output ({len(xml_output)} chars):\n{xml_output}")
                return statements

        if root.tag != 'statements':
            logger.warning(f"Root tag is '{root.tag}', expected 'statements'")
            return statements

        for stmt_elem in root.findall('stmt'):
            try:
                # Parse subject from model
                subject_elem = stmt_elem.find('subject')
                subject_text = subject_elem.text.strip() if subject_elem is not None and subject_elem.text else ""
                subject_type = self._parse_entity_type(subject_elem.get('type') if subject_elem is not None else None)

                # Parse object from model
                object_elem = stmt_elem.find('object')
                object_text = object_elem.text.strip() if object_elem is not None and object_elem.text else ""
                object_type = self._parse_entity_type(object_elem.get('type') if object_elem is not None else None)

                # Parse source text from model
                text_elem = stmt_elem.find('text')
                source_text = text_elem.text.strip() if text_elem is not None and text_elem.text else None

                # Skip if missing required components from model
                if not subject_text or not object_text:
                    logger.debug(f"Skipping statement: missing subject or object from model")
                    continue

                if use_gliner_extraction and source_text:
                    try:
                        from .gliner_extraction import extract_triple_from_text

                        # Get model predicate for fallback/refinement
                        predicate_elem = stmt_elem.find('predicate')
                        model_predicate = predicate_elem.text.strip() if predicate_elem is not None and predicate_elem.text else ""

                        gliner_result = extract_triple_from_text(
                            source_text=source_text,
                            model_subject=subject_text,
                            model_object=object_text,
                            model_predicate=model_predicate,
                            predicates=predicates,
                        )
                        if gliner_result:
                            gliner_subj, gliner_pred, gliner_obj = gliner_result

                            if gliner_pred:
                                # Candidate 1: Hybrid (model subject/object + GLiNER2 predicate)
                                logger.debug(
                                    f"Adding hybrid candidate: '{subject_text}' --[{gliner_pred}]--> '{object_text}'"
                                )
                                statements.append(Statement(
                                    subject=Entity(text=subject_text, type=subject_type),
                                    predicate=gliner_pred,
                                    object=Entity(text=object_text, type=object_type),
                                    source_text=source_text,
                                    extraction_method=ExtractionMethod.HYBRID,
                                ))

                                # Candidate 2: GLiNER2-only (if different from hybrid)
                                if gliner_subj and gliner_obj:
                                    is_different = (gliner_subj != subject_text or gliner_obj != object_text)
                                    if is_different:
                                        logger.debug(
                                            f"Adding GLiNER2-only candidate: '{gliner_subj}' --[{gliner_pred}]--> '{gliner_obj}'"
                                        )
                                        statements.append(Statement(
                                            subject=Entity(text=gliner_subj, type=subject_type),
                                            predicate=gliner_pred,
                                            object=Entity(text=gliner_obj, type=object_type),
                                            source_text=source_text,
                                            extraction_method=ExtractionMethod.GLINER,
                                        ))
                            else:
                                logger.debug(
                                    f"GLiNER2 found no predicate for: '{subject_text}' --> '{object_text}'"
                                )
                    except Exception as e:
                        logger.debug(f"GLiNER2 extraction failed: {e}")
                else:
                    # GLiNER2 disabled - fall back to model predicate
                    predicate_elem = stmt_elem.find('predicate')
                    model_predicate = predicate_elem.text.strip() if predicate_elem is not None and predicate_elem.text else ""

                    if model_predicate:
                        statements.append(Statement(
                            subject=Entity(text=subject_text, type=subject_type),
                            predicate=model_predicate,
                            object=Entity(text=object_text, type=object_type),
                            source_text=source_text,
                            extraction_method=ExtractionMethod.MODEL,
                        ))
                    else:
                        logger.debug(
                            f"Skipping statement (no predicate, spaCy disabled): "
                            f"'{subject_text}' --> '{object_text}'"
                        )
            except Exception as e:
                logger.warning(f"Failed to parse statement: {e}")
                continue

        return statements

    def _parse_entity_type(self, type_str: Optional[str]) -> EntityType:
        """Parse entity type string to EntityType enum."""
        if type_str is None:
            return EntityType.UNKNOWN
        try:
            return EntityType(type_str.upper())
        except ValueError:
            return EntityType.UNKNOWN

    @staticmethod
    def _count_sentences(text: str) -> int:
        """Count approximate number of sentences in text."""
        clean_text = re.sub(r'<[^>]+>', '', text)
        sentences = re.split(r'[.!?]+', clean_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return max(1, len(sentences))

    @staticmethod
    def _count_statements(xml_output: str) -> int:
        """Count number of <stmt> tags in output."""
        return len(re.findall(r'<stmt>', xml_output))


def _result_to_xml(result: ExtractionResult) -> str:
    """Convert an ExtractionResult to XML format."""
    import xml.etree.ElementTree as ET

    root = ET.Element("statements")
    for stmt in result.statements:
        stmt_el = ET.SubElement(root, "stmt")
        subj = ET.SubElement(stmt_el, "subject")
        subj.text = stmt.subject.text
        if stmt.subject.type.value != "UNKNOWN":
            subj.set("type", stmt.subject.type.value)
        obj = ET.SubElement(stmt_el, "object")
        obj.text = stmt.object.text
        if stmt.object.type.value != "UNKNOWN":
            obj.set("type", stmt.object.type.value)
        pred = ET.SubElement(stmt_el, "predicate")
        pred.text = stmt.predicate
        if stmt.source_text:
            text_el = ET.SubElement(stmt_el, "text")
            text_el.text = stmt.source_text
    return ET.tostring(root, encoding="unicode")


# Convenience functions for simple usage

_default_extractor: Optional[StatementExtractor] = None


def _get_default_extractor() -> StatementExtractor:
    """Get or create the default extractor instance."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = StatementExtractor()
    return _default_extractor


def extract_statements(
    text: str,
    options: Optional[ExtractionOptions] = None,
    server_url: Optional[str] = None,
    **kwargs,
) -> ExtractionResult:
    """
    Extract structured statements from text.

    This is a convenience function that uses a default StatementExtractor instance.
    For more control, create your own StatementExtractor.

    By default, uses embedding-based deduplication and beam merging for
    high-quality extraction. Requires sentence-transformers package.

    Args:
        text: Input text to extract statements from
        options: Extraction options (or pass individual options as kwargs)
        server_url: If provided, delegate to a running corp-extractor server
        **kwargs: Individual option overrides (num_beams, diversity_penalty, etc.)

    Returns:
        ExtractionResult containing Statement objects

    Example:
        >>> result = extract_statements("Apple announced a new product.")
        >>> for stmt in result:
        ...     print(f"{stmt.subject.text} -> {stmt.predicate} -> {stmt.object.text}")

        >>> # Delegate to a running server
        >>> result = extract_statements("text", server_url="http://localhost:8111")
    """
    if server_url:
        from .client import server_split
        opts = options or (ExtractionOptions(**kwargs) if kwargs else None)
        return server_split(server_url, text, opts)
    if options is None and kwargs:
        options = ExtractionOptions(**kwargs)
    return _get_default_extractor().extract(text, options)


def extract_statements_as_xml(
    text: str,
    options: Optional[ExtractionOptions] = None,
    server_url: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Extract statements and return raw XML output.

    Args:
        text: Input text to extract statements from
        options: Extraction options
        server_url: If provided, delegate to server then convert to XML
        **kwargs: Individual option overrides

    Returns:
        XML string with <statements> containing <stmt> elements
    """
    if server_url:
        result = extract_statements(text, options, server_url=server_url, **kwargs)
        return _result_to_xml(result)
    if options is None and kwargs:
        options = ExtractionOptions(**kwargs)
    return _get_default_extractor().extract_as_xml(text, options)


def extract_statements_as_json(
    text: str,
    options: Optional[ExtractionOptions] = None,
    indent: Optional[int] = 2,
    server_url: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Extract statements and return JSON string.

    Args:
        text: Input text to extract statements from
        options: Extraction options
        indent: JSON indentation (None for compact)
        server_url: If provided, delegate to server
        **kwargs: Individual option overrides

    Returns:
        JSON string representation of the extraction result
    """
    if server_url:
        result = extract_statements(text, options, server_url=server_url, **kwargs)
        return result.model_dump_json(indent=indent)
    if options is None and kwargs:
        options = ExtractionOptions(**kwargs)
    return _get_default_extractor().extract_as_json(text, options, indent)


def extract_statements_as_dict(
    text: str,
    options: Optional[ExtractionOptions] = None,
    server_url: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Extract statements and return as dictionary.

    Args:
        text: Input text to extract statements from
        options: Extraction options
        server_url: If provided, delegate to server
        **kwargs: Individual option overrides

    Returns:
        Dictionary representation of the extraction result
    """
    if server_url:
        result = extract_statements(text, options, server_url=server_url, **kwargs)
        return result.model_dump()
    if options is None and kwargs:
        options = ExtractionOptions(**kwargs)
    return _get_default_extractor().extract_as_dict(text, options)
