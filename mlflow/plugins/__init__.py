"""
Genesis-Flow Plugin Architecture

This module provides a comprehensive plugin system for ML frameworks and extensions.
Plugins can be dynamically discovered, loaded, enabled/disabled, and managed.

The plugin system supports:
- Framework plugins (PyTorch, TensorFlow, XGBoost, etc.)
- Custom logging plugins 
- Model registry plugins
- Deployment plugins
- UI extensions

Example usage:
    from mlflow.plugins import PluginManager
    
    # Get plugin manager
    manager = PluginManager()
    
    # List available plugins
    plugins = manager.list_plugins()
    
    # Enable a plugin
    manager.enable_plugin("pytorch")
    
    # Use the plugin
    import mlflow.pytorch
"""

from mlflow.plugins.manager import PluginManager
from mlflow.plugins.registry import PluginRegistry
from mlflow.plugins.base import BasePlugin, FrameworkPlugin
from mlflow.plugins.discovery import discover_plugins, scan_entrypoints

__all__ = [
    "PluginManager",
    "PluginRegistry", 
    "BasePlugin",
    "FrameworkPlugin",
    "discover_plugins",
    "scan_entrypoints",
]

# Global plugin manager instance
_plugin_manager = None

def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager

def list_plugins():
    """List all available plugins."""
    return get_plugin_manager().list_plugins()

def enable_plugin(name: str):
    """Enable a plugin by name."""
    return get_plugin_manager().enable_plugin(name)

def disable_plugin(name: str):
    """Disable a plugin by name."""
    return get_plugin_manager().disable_plugin(name)

def is_plugin_enabled(name: str) -> bool:
    """Check if a plugin is enabled."""
    return get_plugin_manager().is_plugin_enabled(name)

def get_plugin(name: str):
    """Get a plugin instance by name."""
    return get_plugin_manager().get_plugin(name)