"""
PersonQualifierPlugin - Qualifies PERSON entities with role, organization, and canonical ID.

Uses Gemma3 12B (instruction-tuned) to extract:
- role: Job title/position (e.g., "CEO", "President")
- org: Organization/employer (e.g., "Apple Inc", "Microsoft")

Then searches the person database to find canonical matches for notable people
(those in Wikipedia/Wikidata), using extracted role/org to help disambiguate.
"""

import logging
import re
from typing import Optional

from ..base import BaseQualifierPlugin, PluginCapability
from ...pipeline.context import PipelineContext
from ...pipeline.registry import PluginRegistry
from ...models import (
    ExtractedEntity,
    EntityQualifiers,
    EntityType,
    QualifiedEntity,
    CanonicalEntity,
    CanonicalMatch,
    ResolvedRole,
    ResolvedOrganization,
)
from ...llm import LLM

logger = logging.getLogger(__name__)


# LLM prompt template for person matching confirmation.
#
# The LLM is the final arbiter: it must VERIFY that a candidate refers to the
# same individual person. Reject when the extracted name isn't a person at all
# (an organisation, a role/title, a location, a group description).
PERSON_MATCH_PROMPT = """You are verifying whether any candidate refers to the SAME individual person as a name extracted from text.

Extracted name: "{query_name}"
Context from text: {context_info}
Source text: "{source_preview}"

Candidates from database (with Wikipedia info):
{candidates}

Task: Select the candidate that refers to the SAME individual person as "{query_name}", or respond "NONE" if no candidate is a verified match.

IMPORTANT RULES (apply in order):
1. Reject (respond "NONE") if "{query_name}" does NOT refer to an individual person. Organisations, role/title strings ("CEO", "President"), locations, and generic group descriptions ("Community leaders", "Business owners", "Investors", "Customers", "Officials") are NOT matches for a specific person.
2. The candidate name must closely match the extracted name "{query_name}".
3. Similar-sounding names are NOT matches (e.g., "Andy Vassies" does NOT match "Andy Jassy").
4. If no candidate has a name that matches "{query_name}", respond "NONE".
5. Consider role and organization context only AFTER confirming the name refers to the same individual.
6. When in doubt, prefer "NONE" over a wrong match.

Respond with ONLY the number of the best match (1, 2, 3, etc.) or "NONE".
"""


@PluginRegistry.qualifier
class PersonQualifierPlugin(BaseQualifierPlugin):
    """
    Qualifier plugin for PERSON entities.

    Uses Gemma3 12B to extract role and organization from context.
    Then searches the person database to find canonical matches for notable people.
    Falls back to pattern matching if LLM is not available.
    """

    # Common role patterns for fallback
    ROLE_PATTERNS = [
        r"\b(CEO|CFO|CTO|COO|CMO|CIO|CISO|CSO)\b",
        r"\b(Chief\s+\w+\s+Officer)\b",
        r"\b(President|Chairman|Director|Manager|Executive|Founder|Co-Founder)\b",
        r"\b(Vice\s+President|VP)\b",
        r"\b(Head\s+of\s+\w+)\b",
        r"\b(Senior\s+\w+|Lead\s+\w+|Principal\s+\w+)\b",
    ]

    def __init__(
        self,
        model_id: str = "google/gemma-3-12b-it-qat-q4_0-gguf",
        gguf_file: Optional[str] = None,
        use_llm: bool = True,
        use_4bit: bool = True,
        use_database: bool = True,
        db_path: Optional[str] = None,
        top_k: int = 10,
        min_similarity: float = 0.5,
        auto_download_db: bool = True,
    ):
        """
        Initialize the person qualifier.

        Args:
            model_id: HuggingFace model ID for LLM qualification
            gguf_file: GGUF filename for quantized models (auto-detected if model_id ends with -gguf)
            use_llm: Whether to use LLM
            use_4bit: Use 4-bit quantization (requires bitsandbytes, ignored for GGUF)
            use_database: Whether to use person database for canonical matching
            db_path: Path to database (auto-detects if None)
            top_k: Number of candidates to retrieve from database
            min_similarity: Minimum similarity threshold for database matches
            auto_download_db: Whether to auto-download database from HuggingFace
        """
        self._use_llm = use_llm
        self._use_database = use_database
        self._db_path = db_path
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._auto_download_db = auto_download_db

        self._llm: Optional[LLM] = None
        if use_llm:
            self._llm = LLM(
                model_id=model_id,
                gguf_file=gguf_file,
                use_4bit=use_4bit,
            )

        # Lazy-loaded components
        self._database = None
        self._embedder = None
        self._cache: dict[str, Optional[CanonicalEntity]] = {}

    @property
    def name(self) -> str:
        return "person_qualifier"

    @property
    def priority(self) -> int:
        return 10  # High priority for PERSON entities

    @property
    def capabilities(self) -> PluginCapability:
        caps = PluginCapability.CACHING
        if self._use_llm:
            caps |= PluginCapability.LLM_REQUIRED
        return caps

    @property
    def description(self) -> str:
        return "Extracts role and organization for PERSON entities, with optional database lookup for notable people"

    @property
    def supported_entity_types(self) -> set[EntityType]:
        return {EntityType.PERSON}

    @property
    def provided_identifier_types(self) -> list[str]:
        return ["wikidata_id"]

    def _get_database(self):
        """Get or initialize the person database."""
        if self._database is not None:
            return self._database

        if not self._use_database:
            return None

        try:
            from ...database.store import get_person_database
            from ...database.hub import get_database_path

            # Find database path
            db_path = self._db_path
            if db_path is None:
                db_path = get_database_path(auto_download=self._auto_download_db)

            if db_path is None:
                logger.warning("Person database not available. Skipping database qualification.")
                return None

            # Use singleton to ensure database is only loaded once
            self._database = get_person_database(db_path=db_path)
            logger.info(f"Loaded person database from {db_path}")
            return self._database

        except Exception as e:
            logger.warning(f"Failed to load person database: {e}")
            return None

    def _get_embedder(self):
        """Get or initialize the embedder."""
        if self._embedder is not None:
            return self._embedder

        try:
            from ...database import CompanyEmbedder
            self._embedder = CompanyEmbedder()
            return self._embedder
        except Exception as e:
            logger.warning(f"Failed to load embedder: {e}")
            return None

    def _get_org_resolver(self):
        """Get or initialize the organization resolver."""
        if not hasattr(self, '_org_resolver'):
            self._org_resolver = None

        if self._org_resolver is not None:
            return self._org_resolver

        try:
            from ...database.resolver import get_organization_resolver
            self._org_resolver = get_organization_resolver(
                db_path=self._db_path,
                auto_download_db=self._auto_download_db,
            )
            return self._org_resolver
        except Exception as e:
            logger.warning(f"Failed to initialize organization resolver: {e}")
            return None

    def _resolve_organization(self, org_name: str) -> Optional[ResolvedOrganization]:
        """
        Resolve an organization name against the organization database.

        Uses the shared OrganizationResolver utility.

        Args:
            org_name: Organization name to resolve

        Returns:
            ResolvedOrganization if found, None otherwise
        """
        resolver = self._get_org_resolver()
        if resolver is None:
            return None

        return resolver.resolve(org_name)

    def qualify(
        self,
        entity: ExtractedEntity,
        context: PipelineContext,
    ) -> Optional[CanonicalEntity]:
        """
        Qualify a PERSON entity with role, organization, and optionally canonical ID.

        Args:
            entity: The PERSON entity to qualify
            context: Pipeline context for accessing source text

        Returns:
            CanonicalEntity with role/org qualifiers and FQN, or None if nothing found
        """
        if entity.type != EntityType.PERSON:
            return None

        # Check cache
        cache_key = entity.text.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Use the full source text for LLM qualification
        full_text = context.source_text

        # Step 1: Extract role and org using LLM or patterns
        qualifiers: Optional[EntityQualifiers] = None
        if self._llm is not None:
            result = self._extract_with_llm(entity.text, full_text)
            if result and (result.role or result.org):
                qualifiers = result

        # Fallback to pattern matching (only if LLM extraction returned nothing)
        if qualifiers is None:
            qualifiers = self._extract_with_patterns(entity.text, full_text)

        # Step 2: Search database for canonical match (if database is available)
        canonical_match = None
        if self._use_database:
            canonical_match = self._search_database(
                entity.text,
                qualifiers.role if qualifiers else None,
                qualifiers.org if qualifiers else None,
                context,
            )

        # If no qualifiers found and no database match, return None
        if qualifiers is None and canonical_match is None:
            self._cache[cache_key] = None
            return None

        # Step 3: Build CanonicalEntity
        result = self._build_canonical_entity(entity, qualifiers, canonical_match)
        self._cache[cache_key] = result
        return result

    def _search_database(
        self,
        person_name: str,
        extracted_role: Optional[str],
        extracted_org: Optional[str],
        context: PipelineContext,
    ) -> Optional[CanonicalMatch]:
        """
        Search the person database for a canonical match.

        Uses embedding similarity + role/org matching for disambiguation.

        Args:
            person_name: Name of the person
            extracted_role: Role extracted from text (e.g., "CEO")
            extracted_org: Organization extracted from text (e.g., "Apple Inc")
            context: Pipeline context

        Returns:
            CanonicalMatch if a confident match is found, None otherwise
        """
        database = self._get_database()
        if database is None:
            return None

        embedder = self._get_embedder()
        if embedder is None:
            return None

        # Log extracted context
        logger.debug(f"    Person search context:")
        logger.debug(f"      Name: '{person_name}'")
        logger.debug(f"      Extracted role: {extracted_role or '(none)'}")
        logger.debug(f"      Extracted org: {extracted_org or '(none)'}")

        # Build query text with context for better embedding match
        # This matches how PersonRecord.get_embedding_text() builds embedding text
        query_parts = [person_name]
        if extracted_role:
            query_parts.append(extracted_role)
        if extracted_org:
            query_parts.append(extracted_org)
        query_text = " | ".join(query_parts)

        logger.debug(f"    Embedding query: '{query_text}'")
        query_embedding = embedder.embed(query_text)

        # Search database with embeddings-only (fast KNN MATCH path)
        logger.debug(f"    Searching person database...")
        results = database.search(
            query_embedding,
            top_k=self._top_k * 3,  # Fetch more to allow for org filtering
        )

        logger.debug(f"    Database returned {len(results)} raw results")

        # If org was extracted, boost candidates that match the org
        if extracted_org:
            # Re-score with org preference
            org_matched = []
            org_unmatched = []
            for record, sim in results:
                if record.known_for_org_name and self._org_matches(extracted_org, record.known_for_org_name):
                    logger.debug(f"      Org match: {record.name} at {record.known_for_org_name}")
                    org_matched.append((record, sim))
                else:
                    org_unmatched.append((record, sim))
            # Prioritize org matches
            if org_matched:
                logger.info(f"    Found {len(org_matched)} candidates matching org '{extracted_org}'")
                results = org_matched + org_unmatched
            else:
                logger.debug(f"    No candidates match org '{extracted_org}'")

        # Filter by minimum similarity
        results = [(r, s) for r, s in results if s >= self._min_similarity]
        logger.debug(f"    After min_similarity filter ({self._min_similarity}): {len(results)} results")

        if not results:
            logger.debug(f"    No person matches found above threshold {self._min_similarity}")
            return None

        # Boost scores based on name/role/org matching
        scored_results = []
        for record, similarity in results:
            boosted_score = self._compute_match_score(
                record, similarity, extracted_role, extracted_org, query_name=person_name
            )
            scored_results.append((record, similarity, boosted_score))

        # Sort by boosted score
        scored_results.sort(key=lambda x: x[2], reverse=True)

        # Log top candidates with detailed context
        logger.info(f"    Found {len(scored_results)} candidates for '{person_name}':")
        for i, (record, sim, boosted) in enumerate(scored_results[:5], 1):
            role_str = f" ({record.known_for_role})" if record.known_for_role else ""
            org_str = f" at {record.known_for_org_name}" if record.known_for_org_name else ""
            boost_delta = boosted - sim
            boost_info = f" [+{boost_delta:.3f} boost]" if boost_delta > 0 else ""
            logger.info(f"      {i}. {record.name}{role_str}{org_str} (sim={sim:.3f}, boosted={boosted:.3f}{boost_info})")

        # Select best match using LLM if available
        logger.info(f"    Selecting best match (LLM={self._llm is not None})...")
        best_match = self._select_best_match(person_name, scored_results, extracted_role, extracted_org, context)

        if best_match is None:
            logger.info(f"    No confident match for '{person_name}'")
            return None

        record, similarity, boosted = best_match
        logger.info(f"    Matched: '{record.name}' (wikidata:{record.source_id}, similarity={similarity:.3f})")

        # Build canonical match
        return CanonicalMatch(
            canonical_id=f"wikidata:{record.source_id}",
            canonical_name=record.name,
            match_method="embedding",
            match_confidence=min(max(boosted, 0.0), 1.0),
            match_details={
                "source": "wikidata",
                "source_id": record.source_id,
                "similarity": similarity,
                "known_for_role": record.known_for_role,
                "known_for_org_name": record.known_for_org_name,
                "birth_date": record.birth_date,
                "death_date": record.death_date,
                "is_historic": record.is_historic,
            },
        )

    def _compute_match_score(
        self,
        record,
        embedding_similarity: float,
        extracted_role: Optional[str],
        extracted_org: Optional[str],
        query_name: Optional[str] = None,
    ) -> float:
        """
        Compute boosted match score using role/org context.

        Boosts similarity score if extracted role/org matches database record.
        """
        score = embedding_similarity

        # Major boost for exact name match (normalized)
        if query_name:
            query_norm = self._normalize_person_name(query_name)
            record_norm = self._normalize_person_name(record.name)
            if query_norm == record_norm:
                score += 0.25  # +25% boost for exact name match
                logger.debug(f"      Exact name match boost: '{query_name}' == '{record.name}'")

        # Boost if role matches (fuzzy)
        if extracted_role and record.known_for_role:
            if self._role_matches(extracted_role, record.known_for_role):
                score += 0.1  # +10% boost
                logger.debug(f"      Role match boost: {extracted_role} ~ {record.known_for_role}")

        # Boost if org matches (fuzzy)
        if extracted_org and record.known_for_org_name:
            if self._org_matches(extracted_org, record.known_for_org_name):
                score += 0.15  # +15% boost (org is stronger signal)
                logger.debug(f"      Org match boost: {extracted_org} ~ {record.known_for_org_name}")

        return min(score, 1.0)  # Cap at 1.0

    def _role_matches(self, extracted: str, known: str) -> bool:
        """Fuzzy role matching."""
        extracted_lower = extracted.lower().strip()
        known_lower = known.lower().strip()

        # Exact match
        if extracted_lower == known_lower:
            return True

        # CEO variants
        ceo_variants = {"ceo", "chief executive", "chief executive officer"}
        if extracted_lower in ceo_variants and known_lower in ceo_variants:
            return True

        # CFO variants
        cfo_variants = {"cfo", "chief financial officer"}
        if extracted_lower in cfo_variants and known_lower in cfo_variants:
            return True

        # President variants
        president_variants = {"president", "chairman", "chairman and ceo"}
        if extracted_lower in president_variants and known_lower in president_variants:
            return True

        # Founder variants
        founder_variants = {"founder", "co-founder", "cofounder", "founding member"}
        if extracted_lower in founder_variants and known_lower in founder_variants:
            return True

        # Contains check for partial matches
        if extracted_lower in known_lower or known_lower in extracted_lower:
            return True

        return False

    def _org_matches(self, extracted: str, known: str) -> bool:
        """Fuzzy org matching using simple normalization."""
        # Normalize both
        extracted_norm = self._normalize_org_name(extracted)
        known_norm = self._normalize_org_name(known)

        # Exact normalized match
        if extracted_norm == known_norm:
            return True

        # Check if one contains the other (e.g., "Apple" in "Apple Inc")
        if extracted_norm in known_norm or known_norm in extracted_norm:
            return True

        return False

    def _normalize_person_name(self, name: str) -> str:
        """Normalize person name for comparison."""
        # Lowercase and strip
        normalized = name.lower().strip()
        # Remove common titles
        for title in ["dr.", "dr ", "mr.", "mr ", "mrs.", "mrs ", "ms.", "ms ", "prof.", "prof "]:
            if normalized.startswith(title):
                normalized = normalized[len(title):]
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        return normalized

    def _normalize_org_name(self, name: str) -> str:
        """Simple org name normalization."""
        # Lowercase
        normalized = name.lower().strip()

        # Remove common suffixes
        suffixes = [
            " inc.", " inc", " corp.", " corp", " corporation",
            " ltd.", " ltd", " limited", " llc", " plc",
            " co.", " co", " company",
        ]
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]

        return normalized.strip()

    def _select_best_match(
        self,
        query_name: str,
        candidates: list[tuple],
        extracted_role: Optional[str],
        extracted_org: Optional[str],
        context: PipelineContext,
    ) -> Optional[tuple]:
        """
        Select the best match from candidates.

        When the LLM is available it is the sole arbiter: an exception or
        unparseable response yields ``None``. The top-boosted-score fallback
        is only used when no LLM is configured (test/dev path).
        """
        if not candidates:
            return None

        # LLM is the arbiter when configured.
        if self._llm is not None:
            try:
                return self._llm_select_match(query_name, candidates, extracted_role, extracted_org, context)
            except Exception as e:
                logger.warning(f"    LLM confirmation failed: {e}; rejecting match (fail closed)")
                return None

        # No-LLM path: use top match if boosted score is high enough.
        top_record, top_similarity, top_boosted = candidates[0]
        if top_boosted >= 0.85:
            logger.info(f"    No LLM, using top match: '{top_record.name}' (boosted={top_boosted:.3f})")
            return candidates[0]

        logger.info(f"    No confident match (top boosted={top_boosted:.3f} < 0.85)")
        return None

    def _llm_select_match(
        self,
        query_name: str,
        candidates: list[tuple],
        extracted_role: Optional[str],
        extracted_org: Optional[str],
        context: PipelineContext,
    ) -> Optional[tuple]:
        """Use LLM to select the best match."""
        # Format candidates for prompt
        candidate_lines = []
        for i, (record, similarity, boosted) in enumerate(candidates[:10], 1):
            role_str = f", {record.known_for_role}" if record.known_for_role else ""
            org_str = f" at {record.known_for_org_name}" if record.known_for_org_name else ""
            country_str = f", {record.country}" if record.country else ""
            # Include life dates for context (helps identify historic figures)
            dates_parts = []
            if record.birth_date:
                dates_parts.append(f"b. {record.birth_date[:4]}")  # Just year
            if record.death_date:
                dates_parts.append(f"d. {record.death_date[:4]}")  # Just year
            dates_str = f" [{' - '.join(dates_parts)}]" if dates_parts else ""
            candidate_lines.append(
                f"{i}. {record.name}{role_str}{org_str}{country_str}{dates_str} (score: {boosted:.2f})"
            )

        # Build context info from extracted role/org
        context_parts = []
        if extracted_role:
            context_parts.append(f"role={extracted_role}")
        if extracted_org:
            context_parts.append(f"org={extracted_org}")
        context_info = ", ".join(context_parts) if context_parts else "no role/org extracted"

        # Source text preview
        source_preview = ""
        if context.source_text:
            source_preview = context.source_text[:300] + "..." if len(context.source_text) > 300 else context.source_text

        prompt = PERSON_MATCH_PROMPT.format(
            query_name=query_name,
            context_info=context_info,
            source_preview=source_preview,
            candidates="\n".join(candidate_lines),
        )

        # Get LLM response
        assert self._llm is not None
        response = self._llm.generate(prompt, max_tokens=10, stop=["\n"])
        response = response.strip()

        logger.info(f"    LLM response for '{query_name}': {response}")

        # Parse response
        if response.upper() == "NONE":
            logger.info(f"    LLM chose: NONE (no match)")
            return None

        try:
            idx = int(response) - 1
        except ValueError:
            logger.warning(f"    LLM response '{response}' could not be parsed as number; rejecting match")
            return None

        if 0 <= idx < len(candidates):
            chosen = candidates[idx]
            logger.info(f"    LLM chose: #{idx + 1} '{chosen[0].name}' (boosted={chosen[2]:.3f})")
            return chosen

        logger.warning(f"    LLM picked index {idx + 1} outside range [1, {len(candidates)}]; rejecting match")
        return None

    def _extract_with_llm(
        self,
        person_name: str,
        context_text: str,
    ) -> Optional[EntityQualifiers]:
        """Extract role and org using Gemma3 with simple line-based output."""
        if self._llm is None:
            return None

        try:
            prompt = f"""Extract info about "{person_name}" from the text below.
Reply with exactly 3 lines:
NAME: the person's full name
ROLE: their job title (CEO, President, etc.) or NONE
ORG: the company/organization name or NONE

Text: {context_text[:500]}

NAME:"""

            logger.debug(f"LLM extraction prompt for '{person_name}'")
            response = self._llm.generate(prompt, max_tokens=100, stop=["\n\n", "</s>"])
            logger.debug(f"LLM response: {response}")

            # Parse line-based response
            lines = response.strip().split("\n")
            name = None
            role = None
            org = None

            for line in lines:
                line = line.strip()
                if line.startswith("NAME:"):
                    name = line[5:].strip()
                elif line.startswith("ROLE:"):
                    val = line[5:].strip()
                    if val.upper() != "NONE":
                        role = val
                elif line.startswith("ORG:"):
                    val = line[4:].strip()
                    if val.upper() != "NONE":
                        org = val
                # Handle case where first line is just the name (after our "NAME:" in prompt)
                elif not name and line and not line.startswith(("ROLE", "ORG")):
                    name = line

            logger.debug(f"LLM extracted: name={name!r}, role={role!r}, org={org!r}")

            if role or org:
                return EntityQualifiers(role=role, org=org)

        except Exception as e:
            logger.exception(f"LLM extraction failed: {e}")
            raise e

        return None

    def _extract_with_patterns(
        self,
        person_name: str,
        context_text: str,
    ) -> Optional[EntityQualifiers]:
        """Extract role and org using pattern matching."""
        role = None
        org = None

        # Look for role patterns
        for pattern in self.ROLE_PATTERNS:
            match = re.search(pattern, context_text, re.IGNORECASE)
            if match:
                role = match.group(1)
                break

        # Look for "of [Organization]" or "at [Organization]" patterns
        org_patterns = [
            rf'{re.escape(person_name)}[^.]*?\bof\s+([A-Z][A-Za-z\s&]+(?:Inc|Corp|Ltd|LLC|Company|Co)?\.?)',
            rf'{re.escape(person_name)}[^.]*?\bat\s+([A-Z][A-Za-z\s&]+(?:Inc|Corp|Ltd|LLC|Company|Co)?\.?)',
            rf'([A-Z][A-Za-z\s&]+(?:Inc|Corp|Ltd|LLC|Company|Co)?\.?)\s*(?:\'s|s)?\s*{re.escape(person_name)}',
        ]

        for pattern in org_patterns:
            match = re.search(pattern, context_text)
            if match:
                org = match.group(1).strip()
                # Clean up trailing punctuation
                org = org.rstrip('.,;')
                break

        if role or org:
            return EntityQualifiers(role=role, org=org)

        return None

    def _build_canonical_entity(
        self,
        entity: ExtractedEntity,
        qualifiers: Optional[EntityQualifiers],
        canonical_match: Optional[CanonicalMatch],
    ) -> CanonicalEntity:
        """Build CanonicalEntity from qualifiers and optional canonical match."""
        # Ensure qualifiers is not None
        if qualifiers is None:
            qualifiers = EntityQualifiers()

        # If we have a canonical match, add wikidata ID to identifiers
        identifiers: dict[str, str] = dict(qualifiers.identifiers) if qualifiers.identifiers else {}
        resolved_role: Optional[ResolvedRole] = None
        resolved_org: Optional[ResolvedOrganization] = None

        if canonical_match:
            match_details = canonical_match.match_details or {}
            source_id = str(match_details.get("source_id", ""))
            if source_id:
                identifiers["wikidata_id"] = source_id
            if canonical_match.canonical_id:
                identifiers["canonical_id"] = canonical_match.canonical_id

            # Extract role and org from database match
            known_role = str(match_details.get("known_for_role", "") or "")
            known_org = str(match_details.get("known_for_org_name", "") or "")

            # Create ResolvedRole from database match
            if known_role:
                resolved_role = ResolvedRole(
                    canonical_name=known_role,
                    canonical_id=None,  # Role ID would need separate lookup
                    source="wikidata",
                    source_id=source_id if source_id else None,
                )

            # Update qualifiers with info from database if not already set
            if not qualifiers.role and known_role:
                final_role = known_role
                final_org = qualifiers.org or known_org or None
            else:
                final_role = qualifiers.role
                final_org = qualifiers.org
        else:
            final_role = qualifiers.role
            final_org = qualifiers.org

        # Resolve organization against the organization database
        org_to_resolve = final_org
        if org_to_resolve:
            logger.debug(f"    Resolving organization: '{org_to_resolve}'")
            resolved_org = self._resolve_organization(org_to_resolve)
            if resolved_org:
                logger.info(f"    Resolved org: '{org_to_resolve}' -> '{resolved_org.canonical_name}' ({resolved_org.canonical_id})")

        # Build the final qualifiers with resolved info
        qualifiers = EntityQualifiers(
            role=final_role,
            org=final_org,
            identifiers=identifiers,
            resolved_role=resolved_role,
            resolved_org=resolved_org,
        )

        # Create QualifiedEntity
        qualified = QualifiedEntity(
            entity_ref=entity.entity_ref,
            original_text=entity.text,
            entity_type=entity.type,
            qualifiers=qualifiers,
            qualification_sources=[self.name],
        )

        # Build FQN - prefer resolved names when available
        if canonical_match and canonical_match.canonical_name:
            # Use canonical person name from database
            fqn_parts: list[str] = [canonical_match.canonical_name]
            if qualifiers.role:
                fqn_parts.append(f"({qualifiers.role})")
            # Use resolved org name if available
            if resolved_org:
                fqn_parts.append(f"at {resolved_org.canonical_name}")
            elif qualifiers.org:
                fqn_parts.append(f"at {qualifiers.org}")
            fqn = " ".join(fqn_parts)
        else:
            # Build FQN: "Person Name (Role, Org)" or "Person Name (Role)" or "Person Name (Org)"
            fqn_parts_for_display: list[str] = []
            if qualifiers.role:
                fqn_parts_for_display.append(qualifiers.role)
            # Use resolved org name if available
            if resolved_org:
                fqn_parts_for_display.append(resolved_org.canonical_name)
            elif qualifiers.org:
                fqn_parts_for_display.append(qualifiers.org)

            if fqn_parts_for_display:
                fqn = f"{entity.text} ({', '.join(fqn_parts_for_display)})"
            else:
                fqn = entity.text

        return CanonicalEntity(
            entity_ref=entity.entity_ref,
            qualified_entity=qualified,
            canonical_match=canonical_match,
            fqn=fqn,
        )


# Allow importing without decorator for testing
PersonQualifierPluginClass = PersonQualifierPlugin
