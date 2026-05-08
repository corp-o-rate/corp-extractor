"""
Qualifier models for the extraction pipeline.

EntityQualifiers: Semantic qualifiers and external identifiers
QualifiedEntity: Entity with qualification information from Stage 3
ResolvedRole: Canonical role information from database
ResolvedOrganization: Canonical organization information from database (re-exported from corp_entity_db)
"""

from typing import Optional

from pydantic import BaseModel, Field

# ResolvedOrganization is owned by corp-entity-db (the OrganizationResolver
# constructs and returns instances of it). Re-export here so EntityQualifiers
# uses the same class identity that the resolver returns — otherwise Pydantic
# V2 would reject a corp_entity_db.ResolvedOrganization instance assigned to
# a field typed against a separately-defined local class with the same shape.
from corp_entity_db.models import ResolvedOrganization

from .entity import EntityType


class ResolvedRole(BaseModel):
    """
    Resolved/canonical role information for a person.

    Populated when matching a person against the database,
    capturing the canonical role from Wikidata or other sources.
    """
    canonical_name: str = Field(..., description="Canonical role name (e.g., 'Chief Executive Officer')")
    canonical_id: Optional[str] = Field(None, description="Full canonical ID (e.g., 'wikidata:Q484876')")
    source: str = Field(..., description="Source of resolution (e.g., 'wikidata')")
    source_id: Optional[str] = Field(None, description="ID in the source (e.g., 'Q484876' for Wikidata)")


__all__ = ["ResolvedRole", "ResolvedOrganization", "EntityQualifiers", "QualifiedEntity"]


class EntityQualifiers(BaseModel):
    """
    Qualifiers that provide context and identifiers for an entity.

    Populated by Stage 3 (Qualification) plugins such as:
    - PersonQualifierPlugin: Adds role, org for PERSON entities
    - GLEIFQualifierPlugin: Adds LEI for ORG entities
    - CompaniesHouseQualifierPlugin: Adds UK company number
    - SECEdgarQualifierPlugin: Adds SEC CIK, ticker
    """
    # Canonical name from database (for ORG entities)
    legal_name: Optional[str] = Field(None, description="Canonical legal name from database")

    # Semantic qualifiers (for PERSON entities)
    org: Optional[str] = Field(None, description="Organization/employer name")
    role: Optional[str] = Field(None, description="Job title/position/role")

    # Location qualifiers
    region: Optional[str] = Field(None, description="State/province/region")
    country: Optional[str] = Field(None, description="Country name or ISO code")
    city: Optional[str] = Field(None, description="City name")
    jurisdiction: Optional[str] = Field(None, description="Legal jurisdiction (e.g., 'UK', 'US-DE')")

    # External identifiers (keyed by identifier type)
    identifiers: dict[str, str] = Field(
        default_factory=dict,
        description="External identifiers: lei, ch_number, sec_cik, ticker, wikidata_qid, etc."
    )

    # Resolved canonical information (for PERSON entities)
    resolved_role: Optional[ResolvedRole] = Field(
        None,
        description="Canonical role information from database lookup"
    )
    resolved_org: Optional[ResolvedOrganization] = Field(
        None,
        description="Canonical organization information from database lookup"
    )

    def has_any_qualifier(self) -> bool:
        """Check if any qualifier or identifier is set."""
        return bool(
            self.legal_name or self.org or self.role or self.region or self.country or
            self.city or self.jurisdiction or self.identifiers or
            self.resolved_role or self.resolved_org
        )

    def merge_with(self, other: "EntityQualifiers") -> "EntityQualifiers":
        """
        Merge qualifiers from another instance, preferring non-None values.

        Returns a new EntityQualifiers with merged values.
        """
        merged_identifiers = {**self.identifiers, **other.identifiers}
        return EntityQualifiers(
            legal_name=other.legal_name or self.legal_name,
            org=other.org or self.org,
            role=other.role or self.role,
            region=other.region or self.region,
            country=other.country or self.country,
            city=other.city or self.city,
            jurisdiction=other.jurisdiction or self.jurisdiction,
            identifiers=merged_identifiers,
            resolved_role=other.resolved_role or self.resolved_role,
            resolved_org=other.resolved_org or self.resolved_org,
        )


class QualifiedEntity(BaseModel):
    """
    An entity with qualification information from Stage 3.

    Links back to the original ExtractedEntity via entity_ref and
    adds qualifiers from various qualification plugins.
    """
    entity_ref: str = Field(..., description="Reference to the original ExtractedEntity")
    original_text: str = Field(..., description="Original entity text")
    entity_type: EntityType = Field(..., description="Entity type")
    qualifiers: EntityQualifiers = Field(
        default_factory=EntityQualifiers,
        description="Qualifiers and identifiers for this entity"
    )
    qualification_sources: list[str] = Field(
        default_factory=list,
        description="List of plugins that contributed qualifiers"
    )

    def add_qualifier_source(self, source: str) -> None:
        """Add a qualification source to the list."""
        if source not in self.qualification_sources:
            self.qualification_sources.append(source)

    class Config:
        frozen = False  # Allow modification during pipeline stages
