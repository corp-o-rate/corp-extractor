"""Helpers for fixture-driven regression tests.

Each fixture is a YAML file under ``tests/fixtures/regression/`` containing the
input text and a list of sparse must / must-not assertions that the final
pipeline output (``PipelineContext``) must satisfy. See README in that directory
(or the in-tree fixture) for the schema.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from statement_extractor.models.canonical import CanonicalEntity
from statement_extractor.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "regression"


class EntityExpectation(BaseModel):
    """One assertion row about how a surface-text entity must be canonicalised."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="Surface text of the entity (case-insensitive match)")
    must_resolve: bool = Field(
        default=False,
        description="If true, at least one CanonicalEntity for this surface text must have a non-null canonical_match.canonical_id",
    )
    canonical_id_matches: Optional[str] = Field(
        default=None,
        description="Regex; at least one matching CanonicalEntity must satisfy",
    )
    canonical_name_matches: Optional[str] = Field(
        default=None,
        description="Regex; at least one matching CanonicalEntity must satisfy",
    )
    canonical_id_must_not_match: Optional[str] = Field(
        default=None,
        description="Regex; every CanonicalEntity for this surface text must NOT match (null is OK)",
    )
    canonical_name_must_not_match: Optional[str] = Field(
        default=None,
        description="Regex; every CanonicalEntity for this surface text must NOT match (null is OK)",
    )
    reason: Optional[str] = Field(default=None, description="Human note explaining the rule")


class FixtureExpect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_statements: int = Field(default=0, ge=0)
    entities: list[EntityExpectation] = Field(default_factory=list)


class RegressionFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[str] = None
    text: str
    expect: FixtureExpect

    @classmethod
    def from_yaml(cls, path: Path) -> "RegressionFixture":
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


def discover_fixtures() -> list[Path]:
    """Return every ``*.yaml`` fixture under ``tests/fixtures/regression/``."""
    if not FIXTURES_DIR.is_dir():
        return []
    return sorted(FIXTURES_DIR.glob("*.yaml"))


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _matching_entities(ctx: PipelineContext, surface_text: str) -> list[CanonicalEntity]:
    target = _norm(surface_text)
    return [
        ce
        for ce in ctx.canonical_entities.values()
        if _norm(ce.qualified_entity.original_text) == target
    ]


def _format_entity(ce: CanonicalEntity) -> str:
    cm = ce.canonical_match
    cid = cm.canonical_id if cm else None
    cname = cm.canonical_name if cm else None
    return f"<text={ce.qualified_entity.original_text!r} canonical_id={cid!r} canonical_name={cname!r}>"


def assert_against(fixture: RegressionFixture, ctx: PipelineContext) -> None:
    """Assert ``ctx`` satisfies every rule in ``fixture.expect``. Raises ``AssertionError`` on first violation."""
    failures: list[str] = []

    if len(ctx.labeled_statements) < fixture.expect.min_statements:
        failures.append(
            f"min_statements: expected >= {fixture.expect.min_statements}, "
            f"got {len(ctx.labeled_statements)}"
        )

    for rule in fixture.expect.entities:
        matches = _matching_entities(ctx, rule.text)

        if not matches:
            failures.append(
                f"entity {rule.text!r}: no CanonicalEntity with this surface text appeared in ctx.canonical_entities "
                f"({len(ctx.canonical_entities)} canonical entities present)"
            )
            continue

        # Positive rules — at least one match must satisfy each.
        if rule.must_resolve:
            ok = any(ce.canonical_match and ce.canonical_match.canonical_id for ce in matches)
            if not ok:
                failures.append(
                    f"entity {rule.text!r}: must_resolve=true but none of "
                    f"{len(matches)} canonical entities have a canonical_id. "
                    f"Observed: {[_format_entity(ce) for ce in matches]}"
                )

        if rule.canonical_id_matches:
            pat = re.compile(rule.canonical_id_matches)
            ok = any(
                ce.canonical_match
                and ce.canonical_match.canonical_id
                and pat.search(ce.canonical_match.canonical_id)
                for ce in matches
            )
            if not ok:
                failures.append(
                    f"entity {rule.text!r}: canonical_id_matches /{rule.canonical_id_matches}/ "
                    f"did not match any canonical_id. "
                    f"Observed: {[_format_entity(ce) for ce in matches]}"
                )

        if rule.canonical_name_matches:
            pat = re.compile(rule.canonical_name_matches)
            ok = any(
                ce.canonical_match
                and ce.canonical_match.canonical_name
                and pat.search(ce.canonical_match.canonical_name)
                for ce in matches
            )
            if not ok:
                failures.append(
                    f"entity {rule.text!r}: canonical_name_matches /{rule.canonical_name_matches}/ "
                    f"did not match any canonical_name. "
                    f"Observed: {[_format_entity(ce) for ce in matches]}"
                )

        # Negative rules — every match must satisfy.
        if rule.canonical_id_must_not_match:
            pat = re.compile(rule.canonical_id_must_not_match)
            bad = [
                ce
                for ce in matches
                if ce.canonical_match
                and ce.canonical_match.canonical_id
                and pat.search(ce.canonical_match.canonical_id)
            ]
            if bad:
                reason = f" (reason: {rule.reason})" if rule.reason else ""
                failures.append(
                    f"entity {rule.text!r}: canonical_id_must_not_match /{rule.canonical_id_must_not_match}/ "
                    f"violated by {[_format_entity(ce) for ce in bad]}{reason}"
                )

        if rule.canonical_name_must_not_match:
            pat = re.compile(rule.canonical_name_must_not_match)
            bad = [
                ce
                for ce in matches
                if ce.canonical_match
                and ce.canonical_match.canonical_name
                and pat.search(ce.canonical_match.canonical_name)
            ]
            if bad:
                reason = f" (reason: {rule.reason})" if rule.reason else ""
                failures.append(
                    f"entity {rule.text!r}: canonical_name_must_not_match /{rule.canonical_name_must_not_match}/ "
                    f"violated by {[_format_entity(ce) for ce in bad]}{reason}"
                )

    if failures:
        header = f"Regression fixture '{fixture.name}' failed {len(failures)} rule(s):"
        body = "\n  - " + "\n  - ".join(failures)
        raise AssertionError(header + body)

    logger.info(
        "Regression fixture '%s' passed (%d entity rules, %d statements observed)",
        fixture.name,
        len(fixture.expect.entities),
        len(ctx.labeled_statements),
    )
