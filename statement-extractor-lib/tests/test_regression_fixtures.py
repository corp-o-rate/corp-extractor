"""Fixture-driven regression tests for the full ExtractionPipeline.

Each YAML file under ``tests/fixtures/regression/`` is parameterised into a
separate test. Marked ``@pytest.mark.slow`` because real T5-Gemma + GLiNER2 +
Gemma-3-12B disambiguator + embedding DB are loaded. The local pre-push hook
(``.githooks/pre-push``) runs this whole suite.
"""

import logging
from pathlib import Path

import pytest

from tests._regression import (
    RegressionFixture,
    assert_against,
    discover_fixtures,
)

logger = logging.getLogger(__name__)


try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from gliner2 import GLiNER2  # noqa: F401
    HAS_GLINER = True
except ImportError:
    HAS_GLINER = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
requires_gliner = pytest.mark.skipif(not HAS_GLINER, reason="gliner2 not installed")


@pytest.fixture(scope="module")
def pipeline():
    """Load the full extraction pipeline once for every fixture in this module.

    Cold-start is heavy (~30 s for model load + first-time DB/embedder download).
    Module scope keeps that cost paid exactly once per pytest invocation.
    """
    from statement_extractor.pipeline import ExtractionPipeline, PipelineConfig

    logger.info("Loading ExtractionPipeline (module-scoped)...")
    return ExtractionPipeline(PipelineConfig())


FIXTURE_PATHS: list[Path] = discover_fixtures()


@pytest.mark.slow
@requires_torch
@requires_gliner
@pytest.mark.parametrize(
    "fixture_path",
    FIXTURE_PATHS,
    ids=[p.stem for p in FIXTURE_PATHS] or ["no_fixtures"],
)
def test_regression_fixture(pipeline, fixture_path: Path):
    if not FIXTURE_PATHS:
        pytest.skip("No regression fixtures found in tests/fixtures/regression/")

    fixture = RegressionFixture.from_yaml(fixture_path)
    logger.info("Running regression fixture: %s", fixture.name)

    ctx = pipeline.process(fixture.text)

    logger.info(
        "Pipeline produced %d canonical entities and %d labeled statements for fixture '%s'",
        len(ctx.canonical_entities),
        len(ctx.labeled_statements),
        fixture.name,
    )

    assert_against(fixture, ctx)
