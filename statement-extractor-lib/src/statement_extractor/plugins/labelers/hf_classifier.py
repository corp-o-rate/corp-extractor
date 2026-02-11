"""
HuggingFaceClassifierLabeler - General-purpose labeler using HuggingFace sequence classification models.

Loads any fine-tuned AutoModelForSequenceClassification model and produces
StatementLabel results. Supports both single and batch processing.
"""

import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..base import BaseLabelerPlugin, PluginCapability
from ...pipeline.context import PipelineContext
from ...models import PipelineStatement, CanonicalEntity, StatementLabel

logger = logging.getLogger(__name__)


def _detect_device(override: str | None = None) -> torch.device:
    """Detect the best available device: MPS > CUDA > CPU."""
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class HuggingFaceClassifierLabeler(BaseLabelerPlugin):
    """
    Labeler that uses a fine-tuned HuggingFace sequence classification model.

    Lazily loads the model on first use. Supports batch processing for
    efficient GPU utilization.
    """

    def __init__(
        self,
        model_path: str,
        label_type: str,
        device: str | None = None,
        priority: int = 50,
    ) -> None:
        """
        Args:
            model_path: Path to directory containing the fine-tuned HF model.
            label_type: Type of label this labeler produces (e.g. "statement_category", "tense").
            device: Optional device override (e.g. "cpu", "cuda", "mps").
            priority: Plugin priority (lower runs first).
        """
        self._model_path = model_path
        self._label_type = label_type
        self._device_override = device
        self._priority = priority

        # Lazily loaded
        self._model: AutoModelForSequenceClassification | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._label_mapping: dict[int, str] | None = None
        self._device: torch.device | None = None

    def _ensure_loaded(self) -> None:
        """Load model, tokenizer, and label mapping on first use."""
        if self._model is not None:
            return

        model_dir = Path(self._model_path)
        logger.info("Loading HF classifier model from %s", model_dir)

        self._device = _detect_device(self._device_override)
        logger.info("Using device: %s", self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self._model.to(self._device)
        self._model.eval()

        # Try label_mapping.json first, fall back to model config
        mapping_file = model_dir / "label_mapping.json"
        if mapping_file.exists():
            with open(mapping_file) as f:
                raw = json.load(f)
            # Keys in JSON are strings, convert to int
            self._label_mapping = {int(k): v for k, v in raw.items()}
            logger.info("Loaded label mapping from label_mapping.json: %d labels", len(self._label_mapping))
        else:
            self._label_mapping = dict(self._model.config.id2label)
            logger.info("Using model config id2label: %d labels", len(self._label_mapping))

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"hf_classifier_{self._label_type}"

    @property
    def label_type(self) -> str:
        return self._label_type

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def capabilities(self) -> PluginCapability:
        return PluginCapability.BATCH_PROCESSING

    @property
    def description(self) -> str:
        return f"HuggingFace classifier for '{self._label_type}' using model at {self._model_path}"

    @property
    def model_vram_gb(self) -> float:
        return 0.5

    @property
    def per_item_vram_gb(self) -> float:
        return 0.02

    # ── Single-item labeling ────────────────────────────────────────────

    def label(
        self,
        statement: PipelineStatement,
        subject_canonical: CanonicalEntity,
        object_canonical: CanonicalEntity,
        context: PipelineContext,
    ) -> StatementLabel | None:
        """Classify a single statement using the HF model."""
        self._ensure_loaded()

        inputs = self._tokenizer(
            statement.source_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)
        max_prob, predicted_idx = torch.max(probs, dim=-1)

        predicted_label = self._label_mapping[predicted_idx.item()]
        confidence = max_prob.item()

        return StatementLabel(
            label_type=self._label_type,
            label_value=predicted_label,
            confidence=confidence,
            labeler=self.name,
        )

    # ── Batch labeling ──────────────────────────────────────────────────

    def label_batch(
        self,
        items: list[tuple[PipelineStatement, CanonicalEntity, CanonicalEntity]],
        context: PipelineContext,
    ) -> list[StatementLabel | None]:
        """Classify a batch of statements efficiently using the HF model."""
        if not items:
            return []

        self._ensure_loaded()

        texts = [stmt.source_text for stmt, _, _ in items]
        batch_size = self.get_optimal_batch_size()
        results: list[StatementLabel | None] = []

        logger.info("Running batch classification: %d items, batch_size=%d", len(texts), batch_size)

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]

            inputs = self._tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**inputs).logits

            probs = torch.softmax(logits, dim=-1)
            max_probs, predicted_indices = torch.max(probs, dim=-1)

            for prob, idx in zip(max_probs, predicted_indices):
                predicted_label = self._label_mapping[idx.item()]
                results.append(
                    StatementLabel(
                        label_type=self._label_type,
                        label_value=predicted_label,
                        confidence=prob.item(),
                        labeler=self.name,
                    )
                )

        return results
