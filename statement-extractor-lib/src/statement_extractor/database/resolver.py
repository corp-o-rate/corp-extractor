"""Re-export shim — resolver module now lives in corp-entity-db."""

from corp_entity_db.resolver import *  # noqa: F401, F403
from corp_entity_db.resolver import OrganizationResolver, get_organization_resolver
