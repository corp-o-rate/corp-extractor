"""Regression tests for EmbeddingCompanyQualifier.

Covers the LLM-as-arbiter behaviour added after the
"Electric Future Initiative" → "ELECTRIC FUTURE LIMITED" mismatch:
- A NONE verdict from the LLM must produce no canonical match.
- An LLM exception must produce no canonical match (fail closed).
- The prompt must include the GLiNER-assigned entity type and the new
  "not a company" rejection rule.
"""

from typing import Any, Optional

from statement_extractor.models import EntityType
from statement_extractor.models.entity import ExtractedEntity
from statement_extractor.pipeline.context import PipelineContext
from statement_extractor.plugins.qualifiers.embedding_company import (
    EmbeddingCompanyQualifier,
)

SOURCE_TEXT = (
    "The federal government launched the Electric Future Initiative last "
    "year to accelerate domestic EV production. Critics say the program "
    "duplicates work already funded under the Inflation Reduction Act."
)


class _FakeRecord:
    """Minimal stand-in for corp_entity_db.models.CompanyRecord."""

    def __init__(
        self,
        name: str = "ELECTRIC FUTURE LIMITED",
        source: str = "companies_house",
        source_id: str = "12345678",
        region: str = "UK",
    ):
        self.name = name
        self.source = source
        self.source_id = source_id
        self.region = region
        self.record: dict[str, Any] = {"jurisdiction": "UK", "country": "UK"}


class _FakeDatabase:
    def __init__(self, results: list[tuple[_FakeRecord, float]]):
        self._results = results

    def search(self, _embedding, top_k: int = 20):
        return list(self._results[:top_k])


class _FakeEmbedder:
    def embed(self, _text: str):
        return [0.0, 0.0, 0.0]


class _FakeLLM:
    """LLM stub that records every prompt and returns a fixed response."""

    def __init__(self, response: str = "NONE", raise_exc: Optional[Exception] = None):
        self._response = response
        self._raise = raise_exc
        self.calls: list[str] = []

    def generate(self, prompt: str, max_tokens: int = 10, stop=None) -> str:
        self.calls.append(prompt)
        if self._raise is not None:
            raise self._raise
        return self._response


def _make_qualifier(
    candidates: list[tuple[_FakeRecord, float]],
    llm: _FakeLLM,
) -> EmbeddingCompanyQualifier:
    qualifier = EmbeddingCompanyQualifier(use_llm_confirmation=True, auto_download_db=False)
    qualifier._database = _FakeDatabase(candidates)
    qualifier._embedder = _FakeEmbedder()
    qualifier._llm = llm
    return qualifier


def _make_entity_and_context(
    text: str = "Electric Future Initiative",
) -> tuple[ExtractedEntity, PipelineContext]:
    span_start = SOURCE_TEXT.index(text) if text in SOURCE_TEXT else None
    span = (span_start, span_start + len(text)) if span_start is not None else None
    entity = ExtractedEntity(text=text, type=EntityType.ORG, span=span)
    context = PipelineContext(source_text=SOURCE_TEXT)
    return entity, context


def test_llm_none_verdict_rejects_match():
    """When the LLM responds NONE, qualify() must return None — no canonical match stamped."""
    llm = _FakeLLM(response="NONE")
    candidates = [(_FakeRecord(), 0.812)]
    qualifier = _make_qualifier(candidates, llm)

    entity, context = _make_entity_and_context()
    result = qualifier.qualify(entity, context)

    assert result is None
    assert len(llm.calls) == 1, "LLM should be consulted exactly once"
    prompt = llm.calls[0]
    assert "Extracted type: ORG" in prompt
    # The new rejection rule must appear in the prompt
    assert "do NOT refer to a company at all" in prompt or "NOT a company" in prompt
    assert "Initiative" in prompt  # rule explicitly mentions Initiative as a non-company signal


def test_llm_exception_fails_closed():
    """When the LLM raises, qualify() returns None — no silent fallback to top embedding."""
    llm = _FakeLLM(raise_exc=RuntimeError("simulated LLM unavailable"))
    candidates = [(_FakeRecord(), 0.95)]  # even a very high similarity must not bypass
    qualifier = _make_qualifier(candidates, llm)

    entity, context = _make_entity_and_context()
    result = qualifier.qualify(entity, context)

    assert result is None
    assert len(llm.calls) == 1, "LLM was called once before raising"


def test_llm_unparseable_response_fails_closed():
    """Unparseable LLM output yields None — no fallback to top embedding."""
    llm = _FakeLLM(response="maybe candidate 2?")
    candidates = [(_FakeRecord(), 0.95)]
    qualifier = _make_qualifier(candidates, llm)

    entity, context = _make_entity_and_context()
    result = qualifier.qualify(entity, context)

    assert result is None


def test_llm_out_of_range_index_fails_closed():
    """LLM picking an index outside [1, N] yields None."""
    llm = _FakeLLM(response="7")
    candidates = [(_FakeRecord(), 0.95), (_FakeRecord(name="ELECTRIC FUTURE GROUP"), 0.7)]
    qualifier = _make_qualifier(candidates, llm)

    entity, context = _make_entity_and_context()
    result = qualifier.qualify(entity, context)

    assert result is None


def test_llm_valid_pick_returns_canonical():
    """A valid in-range LLM pick produces a CanonicalEntity."""
    llm = _FakeLLM(response="1")
    record = _FakeRecord(name="Apple Inc.", source="sec_edgar", source_id="320193", region="US")
    candidates = [(record, 0.97)]
    qualifier = _make_qualifier(candidates, llm)

    entity = ExtractedEntity(text="Apple", type=EntityType.ORG)
    context = PipelineContext(source_text="Apple announced a new iPhone today.")
    result = qualifier.qualify(entity, context)

    assert result is not None
    assert result.canonical_match is not None
    assert result.canonical_match.canonical_name == "Apple Inc."
