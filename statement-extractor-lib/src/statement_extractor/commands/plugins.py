"""Plugins command — list and inspect available plugins."""

from typing import Optional

import click

from ._common import _load_all_plugins


@click.command("plugins")
@click.argument("action", type=click.Choice(["list", "info"]))
@click.argument("plugin_name", required=False)
@click.option("--stage", type=int, help="Filter by stage number (1-5)")
def plugins_cmd(action: str, plugin_name: Optional[str], stage: Optional[int]):
    """
    List or inspect available plugins.

    \b
    Actions:
        list   List all available plugins
        info   Show details about a specific plugin

    \b
    Examples:
        corp-extractor plugins list
        corp-extractor plugins list --stage 3
        corp-extractor plugins info gleif_qualifier
    """
    # Import and load plugins
    _load_all_plugins()

    from ..pipeline.registry import PluginRegistry

    if action == "list":
        plugins = PluginRegistry.list_plugins(stage=stage)
        if not plugins:
            click.echo("No plugins registered.")
            return

        # Group by stage
        by_stage: dict[int, list] = {}
        for plugin in plugins:
            stage_num = plugin["stage"]
            if stage_num not in by_stage:
                by_stage[stage_num] = []
            by_stage[stage_num].append(plugin)

        for stage_num in sorted(by_stage.keys()):
            stage_plugins = by_stage[stage_num]
            stage_name = stage_plugins[0]["stage_name"]
            click.echo(f"\nStage {stage_num}: {stage_name.title()}")
            click.echo("-" * 40)

            for p in stage_plugins:
                entity_types = p.get("entity_types", [])
                types_str = f" ({', '.join(entity_types)})" if entity_types else ""
                click.echo(f"  {p['name']}{types_str}  [priority: {p['priority']}]")

    elif action == "info":
        if not plugin_name:
            raise click.UsageError("Plugin name required for 'info' action")

        plugin = PluginRegistry.get_plugin(plugin_name)
        if not plugin:
            raise click.ClickException(f"Plugin not found: {plugin_name}")

        click.echo(f"\nPlugin: {plugin.name}")
        click.echo(f"Priority: {plugin.priority}")
        click.echo(f"Capabilities: {plugin.capabilities.name if plugin.capabilities else 'NONE'}")

        if plugin.description:
            click.echo(f"Description: {plugin.description}")

        if hasattr(plugin, "supported_entity_types"):
            types = [t.value for t in plugin.supported_entity_types]
            click.echo(f"Entity types: {', '.join(types)}")

        if hasattr(plugin, "label_type"):
            click.echo(f"Label type: {plugin.label_type}")

        if hasattr(plugin, "supported_identifier_types"):
            ids = plugin.supported_identifier_types
            if ids:
                click.echo(f"Supported identifiers: {', '.join(ids)}")

        if hasattr(plugin, "provided_identifier_types"):
            ids = plugin.provided_identifier_types
            if ids:
                click.echo(f"Provided identifiers: {', '.join(ids)}")
