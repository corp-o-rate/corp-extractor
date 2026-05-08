"""Re-export shim — embeddings module now lives in corp-entity-db."""

from corp_entity_db.embeddings import *  # noqa: F401, F403
from corp_entity_db.embeddings import CompanyEmbedder, get_embedder
