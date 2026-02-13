"""Serve command — persistent local server."""

import click

from ._common import _configure_logging


@click.command("serve")
@click.option("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
@click.option("--port", type=int, default=8111, help="Port to listen on (default: 8111)")
@click.option("--no-warmup", is_flag=True, help="Skip model warmup on startup (load on first request)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def serve_cmd(host: str, port: int, no_warmup: bool, verbose: bool):
    """
    Start a persistent local server that keeps models warm in memory.

    Avoids the ~30s model loading cost on every CLI invocation.
    Once running, use --server on other commands to delegate to it.

    \b
    Examples:
        corp-extractor serve
        corp-extractor serve --port 9000
        corp-extractor serve --no-warmup
        corp-extractor --server pipeline "Apple CEO Tim Cook..."
    """
    _configure_logging(verbose)
    from ..server import run_server
    run_server(host=host, port=port, do_warmup=not no_warmup, verbose=verbose)
