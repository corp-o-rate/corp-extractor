"""
Statement Extractor - Extract structured statements from text using T5-Gemma 2.

A Python library for extracting subject-predicate-object triples from unstructured text.
Uses Diverse Beam Search (Vijayakumar et al., 2016) for high-quality extraction.

Paper: https://arxiv.org/abs/1610.02424

Features:
- Quality-based beam scoring and merging
- Embedding-based predicate comparison for smart deduplication
- Configurable precision/recall tradeoff
- Support for predicate taxonomies

Example:
    >>> from statement_extractor import extract_statements
    >>> result = extract_statements("Apple Inc. announced a new iPhone today.")
    >>> for stmt in result:
    ...     print(f"{stmt.subject.text} -> {stmt.predicate} -> {stmt.object.text}")
    Apple Inc. -> announced -> a new iPhone

    >>> # Access confidence scores
    >>> for stmt in result:
    ...     print(f"{stmt} (confidence: {stmt.confidence_score:.2f})")

    >>> # Get as different formats
    >>> xml = extract_statements_as_xml("Some text...")
    >>> json_str = extract_statements_as_json("Some text...")
    >>> data = extract_statements_as_dict("Some text...")
"""

__version__ = "0.10.1"

# Core models
from .models import (
    Entity,
    EntityType,
    ExtractionMethod,
    ExtractionOptions,
    ExtractionResult,
    Statement,
    # New in 0.2.0
    PredicateMatch,
    PredicateTaxonomy,
    PredicateComparisonConfig,
    ScoringConfig,
)

# Main extractor
from .extractor import (
    StatementExtractor,
    extract_statements,
    extract_statements_as_dict,
    extract_statements_as_json,
    extract_statements_as_xml,
)

# Scoring utilities
from .scoring import (
    BeamScorer,
    TripleScorer,
)

__all__ = [
    # Version
    "__version__",
    # Core models
    "Entity",
    "EntityType",
    "ExtractionMethod",
    "ExtractionOptions",
    "ExtractionResult",
    "Statement",
    # Configuration models (new in 0.2.0)
    "PredicateMatch",
    "PredicateTaxonomy",
    "PredicateComparisonConfig",
    "ScoringConfig",
    # Extractor class
    "StatementExtractor",
    # Convenience functions
    "extract_statements",
    "extract_statements_as_dict",
    "extract_statements_as_json",
    "extract_statements_as_xml",
    # Scoring
    "BeamScorer",
    "TripleScorer",
    # LLM (lazy import)
    "LLM",
    "get_llm",
]


# Lazy imports for optional dependencies
def __getattr__(name: str):
    """Lazy import for optional modules."""
    if name == "PredicateComparer":
        from .predicate_comparer import PredicateComparer
        return PredicateComparer
    if name == "EmbeddingDependencyError":
        from .predicate_comparer import EmbeddingDependencyError
        return EmbeddingDependencyError
    if name == "LLM":
        from .llm import LLM
        return LLM
    if name == "get_llm":
        from .llm import get_llm
        return get_llm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
