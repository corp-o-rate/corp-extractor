"""Database commands package — db group and command registration."""

import click


@click.group("db")
@click.pass_context
def db_cmd(ctx: click.Context):
    """
    Manage entity/organization embedding database.

    \b
    Commands:
        import-gleif           Import GLEIF LEI data (~3M records)
        import-sec             Import SEC Edgar bulk data (~100K+ filers)
        import-sec-officers    Import SEC Form 4 officers/directors
        import-ch-officers     Import UK Companies House officers (Prod195)
        import-companies-house Import UK Companies House (~5M records)
        import-wikidata        Import Wikidata organizations (SPARQL, may timeout)
        import-people          Import Wikidata notable people (SPARQL, may timeout)
        import-wikidata-dump   Import from Wikidata JSON dump (recommended)
        canonicalize           Link equivalent records across sources
        status                 Show database status
        search                 Search for an organization
        search-people          Search for a person
        download               Download database from HuggingFace
        upload                 Upload database with lite variant
        create-lite            Create lite version (no record data)
        repair-resume          Backfill people from org executive data (DB only)
        fix-resume             Backfill people by re-scanning Wikidata dump

    \b
    Examples:
        corp-extractor db import-sec --download
        corp-extractor db import-sec-officers --start-year 2023 --limit 10000
        corp-extractor db import-gleif --download --limit 100000
        corp-extractor db import-wikidata-dump --download --limit 50000
        corp-extractor db canonicalize
        corp-extractor db status
        corp-extractor db search "Apple Inc"
        corp-extractor db search-people "Tim Cook"
        corp-extractor db upload entities.db
    """
    ctx.ensure_object(dict)


# Register all db sub-commands
from .imports import (
    db_gleif_info,
    db_import_gleif,
    db_import_sec,
    db_import_sec_officers,
    db_import_ch_officers,
    db_import_wikidata,
    db_import_people,
    db_import_companies_house,
    db_import_locations,
)

db_cmd.add_command(db_gleif_info)
db_cmd.add_command(db_import_gleif)
db_cmd.add_command(db_import_sec)
db_cmd.add_command(db_import_sec_officers)
db_cmd.add_command(db_import_ch_officers)
db_cmd.add_command(db_import_wikidata)
db_cmd.add_command(db_import_people)
db_cmd.add_command(db_import_companies_house)
db_cmd.add_command(db_import_locations)

from .wikidata_dump import db_import_wikidata_dump

db_cmd.add_command(db_import_wikidata_dump)

from .search import (
    db_search,
    db_search_people,
    db_search_people_perf_test,
    db_search_roles,
    db_search_locations,
)

db_cmd.add_command(db_search)
db_cmd.add_command(db_search_people)
db_cmd.add_command(db_search_people_perf_test)
db_cmd.add_command(db_search_roles)
db_cmd.add_command(db_search_locations)

from .management import (
    db_status,
    db_canonicalize,
    db_download,
    db_upload,
    db_create_lite,
    db_repair_embeddings,
    db_rebuild_vec,
    db_backfill_scalar,
    db_build_index,
    db_post_import,
    db_migrate,
    db_migrate_v2,
)

db_cmd.add_command(db_status)
db_cmd.add_command(db_canonicalize)
db_cmd.add_command(db_download)
db_cmd.add_command(db_upload)
db_cmd.add_command(db_create_lite)
db_cmd.add_command(db_repair_embeddings)
db_cmd.add_command(db_rebuild_vec)
db_cmd.add_command(db_backfill_scalar)
db_cmd.add_command(db_build_index)
db_cmd.add_command(db_post_import)
db_cmd.add_command(db_migrate)
db_cmd.add_command(db_migrate_v2)

from .repair import db_repair_resume, db_fix_resume

db_cmd.add_command(db_repair_resume)
db_cmd.add_command(db_fix_resume)
