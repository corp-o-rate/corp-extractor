# Entity Database

The entity database is a separate project,
[corp-entity-db](https://corp-entity-db.vercel.app/), as of corp-extractor
v0.10.0. It provides organizations, people, roles, and locations with
embedding-based semantic search, sourced from GLEIF, SEC Edgar, Companies
House, and Wikidata.

## Why a separate project?

- Independent release cadence — schema migrations, importer fixes, and
  USearch index rebuilds happen without coupling to corp-extractor.
- Reusable from anything that needs entity search, not just the
  extraction pipeline.
- Smaller `corp-extractor` install footprint when entity qualification
  isn't needed.

## How `corp-extractor` uses it

`corp-extractor` depends on `corp-entity-db>=0.1.0` and consumes it
through the qualifier plugins (Stage 3 of the pipeline):

- `embedding_company_qualifier` — looks up organizations by embedding
  similarity, attaching canonical IDs (LEI, CIK, UK CH number, Wikidata
  QID) to extracted ORG entities.
- `person_qualifier` — looks up notable people, optionally using a local
  LLM (Gemma-3-12B GGUF) to disambiguate when multiple candidates match.

Both pull the database via `corp_entity_db.hub.get_database_path()` /
`download_database()`. Thin re-export shims under
`statement_extractor.database` (`OrganizationDatabase`, `PersonDatabase`,
`get_database`, etc.) keep older code working without changes.

## Where to look for what

| Need | Source |
|------|--------|
| CLI reference, search/download/build commands | <https://corp-entity-db.vercel.app/> |
| Schema, sizes, source coverage, EntityType/PersonType classifications | <https://corp-entity-db.vercel.app/> |
| Python API (`OrganizationDatabase`, `CompanyEmbedder`, `OrganizationResolver`) | `corp_entity_db` package docs |
| How qualification is wired into the pipeline | `statement-extractor-lib/src/statement_extractor/plugins/qualifiers/` |

## Cerebrium deployment

The Cerebrium app for `corp-extractor` deploys into the same Cerebrium
project as the corp-entity-db app, so both share `/persistent-storage`.
The database files, USearch indexes, and embedding model
(`google/embeddinggemma-300m`) are downloaded once by whichever app boots
first and reused by the other. See `cerebrium/main.py` for the volume
probe that points `corp_entity_db.hub.DEFAULT_CACHE_DIR` at the volume.
