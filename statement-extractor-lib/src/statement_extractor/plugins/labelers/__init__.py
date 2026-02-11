"""
Labeler plugins for Stage 5 (Labeling).

Applies labels to statements (sentiment, relation type, confidence).

Note: Taxonomy classification is handled in Stage 6 (Taxonomy) via
plugins/taxonomy/ modules, not here. The TaxonomyLabeler classes are
provided for backward compatibility but are NOT auto-registered.
"""

from .base import BaseLabelerPlugin
from .sentiment import SentimentLabeler
from .relation_type import RelationTypeLabeler
from .confidence import ConfidenceLabeler

# Taxonomy labelers - exported for backward compatibility only
# NOT auto-registered as Stage 5 labelers (use Stage 6 taxonomy plugins instead)
from .taxonomy import TaxonomyLabeler
from .taxonomy_embedding import EmbeddingTaxonomyLabeler
from .hf_classifier import HuggingFaceClassifierLabeler

__all__ = [
    "BaseLabelerPlugin",
    "SentimentLabeler",
    "RelationTypeLabeler",
    "ConfidenceLabeler",
    # Taxonomy labelers (not auto-registered - for manual use only)
    "TaxonomyLabeler",
    "EmbeddingTaxonomyLabeler",
    # HuggingFace classifier (not auto-registered - requires constructor args)
    "HuggingFaceClassifierLabeler",
]
