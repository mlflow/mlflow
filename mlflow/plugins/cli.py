"""
CLI Commands for Genesis-Flow Plugin Management

Provides command-line interface for managing plugins:
- List available plugins
- Enable/disable plugins
- Get plugin information
- Plugin statistics
"""

import logging
import click
from typing import Optional

from mlflow.plugins import get_plugin_manager
from mlflow.plugins.base import PluginType, PluginState

logger = logging.getLogger(__name__)

@click.group(name="plugins")
def plugins_cli():
    """Manage Genesis-Flow plugins."""
    pass

@plugins_cli.command()
@click.option("--type", "-t", "plugin_type", type=click.Choice([t.value for t in PluginType]), 
              help="Filter by plugin type")
@click.option("--state", "-s", type=click.Choice([s.value for s in PluginState]),
              help="Filter by plugin state")
@click.option("--detailed", "-d", is_flag=True, help="Show detailed plugin information")
def list(plugin_type: Optional[str], state: Optional[str], detailed: bool):
    """List available plugins."""
    try:
        manager = get_plugin_manager()
        manager.initialize()
        
        # Convert string filters to enums
        type_filter = PluginType(plugin_type) if plugin_type else None
        state_filter = PluginState(state) if state else None
        
        plugins = manager.list_plugins(plugin_type=type_filter, state=state_filter)
        
        if not plugins:
            click.echo("No plugins found matching criteria.")
            return
        
        if detailed:
            _show_detailed_plugin_list(plugins)
        else:
            _show_simple_plugin_list(plugins)
            
    except Exception as e:
        click.echo(f"Error listing plugins: {e}", err=True)

@plugins_cli.command()
@click.argument("name")
def info(name: str):
    """Show detailed information about a plugin."""
    try:
        manager = get_plugin_manager()
        manager.initialize()
        
        plugin = manager.get_plugin(name)
        if plugin:
            _show_plugin_info(plugin.get_info())
        else:
            # Check if plugin is available but not enabled
            if manager.is_plugin_available(name):
                click.echo(f"Plugin '{name}' is available but not enabled.")
                click.echo(f"Use 'genesis-flow plugins enable {name}' to enable it.")
            else:
                click.echo(f"Plugin '{name}' not found.")
                
    except Exception as e:
        click.echo(f"Error getting plugin info: {e}", err=True)

@plugins_cli.command()
@click.argument("name")
@click.option("--force", "-f", is_flag=True, help="Force enable even if dependencies are missing")
def enable(name: str, force: bool):
    """Enable a plugin."""
    try:
        manager = get_plugin_manager()
        manager.initialize()
        
        if not manager.is_plugin_available(name):
            click.echo(f"Plugin '{name}' not found.")
            return
        
        if manager.is_plugin_enabled(name):
            click.echo(f"Plugin '{name}' is already enabled.")
            return
        
        # Check dependencies unless forced
        if not force:
            plugin_class = manager._registry.get_plugin_class(name)
            if plugin_class:
                # Create temporary instance to check dependencies
                metadata = manager._get_plugin_metadata(name, plugin_class)
                temp_instance = plugin_class(metadata)
                
                if not temp_instance.check_dependencies():
                    click.echo(f"Plugin '{name}' has unmet dependencies.")
                    click.echo("Use --force to enable anyway, or install missing dependencies.")
                    return
        
        if manager.enable_plugin(name):
            click.echo(f"Plugin '{name}' enabled successfully.")
        else:
            click.echo(f"Failed to enable plugin '{name}'.")
            
    except Exception as e:
        click.echo(f"Error enabling plugin: {e}", err=True)

@plugins_cli.command()
@click.argument("name")
def disable(name: str):
    """Disable a plugin."""
    try:
        manager = get_plugin_manager()
        manager.initialize()
        
        if not manager.is_plugin_enabled(name):
            click.echo(f"Plugin '{name}' is not enabled.")
            return
        
        if manager.disable_plugin(name):
            click.echo(f"Plugin '{name}' disabled successfully.")
        else:
            click.echo(f"Failed to disable plugin '{name}'.")
            
    except Exception as e:
        click.echo(f"Error disabling plugin: {e}", err=True)

@plugins_cli.command()
@click.argument("name")
def reload(name: str):
    """Reload a plugin."""
    try:
        manager = get_plugin_manager()
        manager.initialize()
        
        if not manager.is_plugin_available(name):
            click.echo(f"Plugin '{name}' not found.")
            return
        
        if manager.reload_plugin(name):
            click.echo(f"Plugin '{name}' reloaded successfully.")
        else:
            click.echo(f"Failed to reload plugin '{name}'.")
            
    except Exception as e:
        click.echo(f"Error reloading plugin: {e}", err=True)

@plugins_cli.command()
def stats():
    """Show plugin statistics."""
    try:
        manager = get_plugin_manager()
        manager.initialize()
        
        stats = manager.get_stats()
        
        click.echo("Plugin Statistics:")
        click.echo("=" * 50)
        click.echo(f"Total Registered: {stats['total_registered']}")
        click.echo(f"Total Instances:  {stats['total_instances']}")
        click.echo(f"Enabled:          {stats['enabled']}")
        click.echo(f"Disabled:         {stats['disabled']}")
        click.echo(f"Failed:           {stats['failed']}")
        
        if stats['by_type']:
            click.echo("\nBy Type:")
            for plugin_type, count in stats['by_type'].items():
                click.echo(f"  {plugin_type:15} {count}")
                
    except Exception as e:
        click.echo(f"Error getting plugin stats: {e}", err=True)

@plugins_cli.command()
def discover():
    """Rediscover available plugins."""
    try:
        from mlflow.plugins.discovery import refresh_discovery
        
        plugins = refresh_discovery()
        click.echo(f"Discovered {len(plugins)} plugins:")
        
        for name in sorted(plugins.keys()):
            click.echo(f"  - {name}")
            
    except Exception as e:
        click.echo(f"Error discovering plugins: {e}", err=True)

@plugins_cli.command()
@click.option("--type", "-t", "plugin_type", type=click.Choice([t.value for t in PluginType]),
              help="Enable all plugins of this type")
def enable_all(plugin_type: Optional[str]):
    """Enable all available plugins or all plugins of a specific type."""
    try:
        manager = get_plugin_manager()
        manager.initialize()
        
        type_filter = PluginType(plugin_type) if plugin_type else None
        plugins = manager.list_plugins(plugin_type=type_filter, state=PluginState.DISCOVERED)
        
        if not plugins:
            click.echo("No plugins available to enable.")
            return
        
        enabled_count = 0
        for plugin_info in plugins:
            name = plugin_info['name']
            if manager.enable_plugin(name):
                click.echo(f"✓ Enabled {name}")
                enabled_count += 1
            else:
                click.echo(f"✗ Failed to enable {name}")
        
        click.echo(f"\nEnabled {enabled_count}/{len(plugins)} plugins.")
        
    except Exception as e:
        click.echo(f"Error enabling plugins: {e}", err=True)

@plugins_cli.command()
@click.option("--type", "-t", "plugin_type", type=click.Choice([t.value for t in PluginType]),
              help="Disable all plugins of this type")
def disable_all(plugin_type: Optional[str]):
    """Disable all enabled plugins or all plugins of a specific type."""
    try:
        manager = get_plugin_manager()
        manager.initialize()
        
        type_filter = PluginType(plugin_type) if plugin_type else None
        plugins = manager.list_plugins(plugin_type=type_filter, state=PluginState.ENABLED)
        
        if not plugins:
            click.echo("No plugins enabled to disable.")
            return
        
        disabled_count = 0
        for plugin_info in plugins:
            name = plugin_info['name']
            if manager.disable_plugin(name):
                click.echo(f"✓ Disabled {name}")
                disabled_count += 1
            else:
                click.echo(f"✗ Failed to disable {name}")
        
        click.echo(f"\nDisabled {disabled_count}/{len(plugins)} plugins.")
        
    except Exception as e:
        click.echo(f"Error disabling plugins: {e}", err=True)

def _show_simple_plugin_list(plugins):
    """Show simple plugin list."""
    click.echo(f"Found {len(plugins)} plugins:")
    click.echo()
    
    # Group by state
    by_state = {}
    for plugin in plugins:
        state = plugin['state']
        if state not in by_state:
            by_state[state] = []
        by_state[state].append(plugin)
    
    for state in ['enabled', 'disabled', 'loaded', 'discovered', 'failed']:
        if state in by_state:
            click.echo(f"{state.upper()}:")
            for plugin in by_state[state]:
                type_info = f" ({plugin['type']})" if 'type' in plugin else ""
                click.echo(f"  ✓ {plugin['name']}{type_info}")
            click.echo()

def _show_detailed_plugin_list(plugins):
    """Show detailed plugin list."""
    click.echo(f"Found {len(plugins)} plugins:")
    click.echo("=" * 80)
    
    for plugin in plugins:
        click.echo(f"Name:        {plugin['name']}")
        click.echo(f"Version:     {plugin.get('version', 'Unknown')}")
        click.echo(f"Type:        {plugin.get('type', 'Unknown')}")
        click.echo(f"State:       {plugin['state']}")
        click.echo(f"Description: {plugin.get('description', 'No description')}")
        
        if plugin.get('dependencies'):
            click.echo(f"Dependencies: {', '.join(plugin['dependencies'])}")
        
        if plugin.get('tags'):
            click.echo(f"Tags:        {', '.join(plugin['tags'])}")
        
        click.echo("-" * 80)

def _show_plugin_info(plugin_info):
    """Show detailed information about a single plugin."""
    click.echo("Plugin Information:")
    click.echo("=" * 50)
    
    for key, value in plugin_info.items():
        if isinstance(value, list):
            if value:
                click.echo(f"{key.capitalize()}: {', '.join(value)}")
            else:
                click.echo(f"{key.capitalize()}: None")
        else:
            click.echo(f"{key.capitalize()}: {value}")

# Add to mlflow CLI
def register_plugin_cli():
    """Register plugin CLI commands with MLflow CLI."""
    try:
        import mlflow.cli
        
        # Add plugins command group to mlflow CLI
        if hasattr(mlflow.cli, 'cli'):
            mlflow.cli.cli.add_command(plugins_cli)
            
    except Exception as e:
        logger.warning(f"Failed to register plugin CLI: {e}")