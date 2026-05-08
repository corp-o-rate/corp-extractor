"""Re-export shim — store module now lives in corp-entity-db.

corp_entity_db splits its public API across `store` (DB classes + factories)
and `models` (Pydantic shapes). We pull each name from its real home so a
plain import error at module load doesn't disable downstream callers (e.g.
PersonQualifierPlugin's lazy person-DB load, which silently skipped database
qualification when this shim mis-imported CompanyMatch from store).
"""

from corp_entity_db.store import (  # noqa: F401
    OrganizationDatabase,
    PersonDatabase,
    RolesDatabase,
    LocationsDatabase,
    get_database,
    get_person_database,
    get_roles_database,
    get_locations_database,
    DEFAULT_DB_PATH,
)
from corp_entity_db.models import (  # noqa: F401
    CompanyMatch,
    PersonMatch,
    DatabaseStats,
)
