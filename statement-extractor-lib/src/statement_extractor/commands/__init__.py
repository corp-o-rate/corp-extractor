"""CLI commands package — main click group and command registration."""

import os

import click

from .. import __version__
from ._common import DEFAULT_SERVER_URL


@click.group()
@click.version_option(version=__version__)
@click.option("--db-version", type=int, default=None, hidden=True, help="Database schema version for filenames (default: latest)")
@click.option("--server", "use_server", is_flag=True, help=f"Use local server at {DEFAULT_SERVER_URL}")
@click.option("--server-url", type=str, default=None, help="Use server at custom URL")
@click.pass_context
def main(ctx: click.Context, db_version: int | None, use_server: bool, server_url: str | None):
    """
    Extract structured statements from text.

    \b
    Commands:
        split      Extract sub-statements from text (simple, fast)
        pipeline   Run the full 6-stage extraction pipeline
        document   Process documents with chunking and citations
        serve      Start persistent local server (keeps models warm)
        plugins    List or inspect available plugins

    \b
    Entity database management has moved to the corp-entity-db package.
    Install it with: pip install corp-entity-db

    \b
    Examples:
        corp-extractor split "Apple announced a new iPhone."
        corp-extractor split -f article.txt --json
        corp-extractor pipeline "Apple CEO Tim Cook announced..." --stages 1-3
        corp-extractor document process report.txt --title "Annual Report"
        corp-extractor serve --port 8111
        corp-extractor --server pipeline "Apple announced..."
        corp-extractor plugins list
    """
    ctx.ensure_object(dict)
    ctx.obj["db_version"] = db_version
    # Resolve server URL: --server-url > --server flag > CORP_EXTRACTOR_SERVER env var
    server = server_url or (DEFAULT_SERVER_URL if use_server else None) or os.environ.get("CORP_EXTRACTOR_SERVER")
    ctx.obj["server"] = server


# Register top-level commands
from .split import split_cmd
from .pipeline import pipeline_cmd
from .plugins import plugins_cmd
from .serve import serve_cmd
from .document import document_cmd

main.add_command(split_cmd)
main.add_command(pipeline_cmd)
main.add_command(plugins_cmd)
main.add_command(serve_cmd)
main.add_command(document_cmd)
