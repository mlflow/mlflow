"""
Plugin Discovery System for Genesis-Flow

Automatically discovers plugins from:
1. Built-in plugins (shipped with Genesis-Flow)
2. Installed packages with entry points
3. Local plugin directories
4. Environment variables
"""

import os
import logging
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Type
from pkg_resources import iter_entry_points

from mlflow.plugins.base import BasePlugin, PluginMetadata, PluginType

logger = logging.getLogger(__name__)

class PluginDiscovery:
    """Plugin discovery and loading system."""
    
    # Entry point group names for different plugin types
    ENTRY_POINT_GROUPS = {
        PluginType.FRAMEWORK: "genesis_flow.frameworks",
        PluginType.LOGGING: "genesis_flow.logging",
        PluginType.MODEL_REGISTRY: "genesis_flow.registries",
        PluginType.DEPLOYMENT: "genesis_flow.deployment",
        PluginType.ARTIFACT: "genesis_flow.artifacts",
        PluginType.UI: "genesis_flow.ui",
        PluginType.CUSTOM: "genesis_flow.plugins",
    }
    
    def __init__(self):
        self.discovered_plugins: Dict[str, Type[BasePlugin]] = {}
        self.built_in_plugins: Dict[str, Type[BasePlugin]] = {}
        
    def discover_all_plugins(self) -> Dict[str, Type[BasePlugin]]:
        """
        Discover all available plugins from all sources.
        
        Returns:
            Dict mapping plugin names to plugin classes
        """
        plugins = {}
        
        # Discover built-in plugins
        plugins.update(self.discover_builtin_plugins())
        
        # Discover entry point plugins
        plugins.update(self.discover_entrypoint_plugins())
        
        # Discover local plugins
        plugins.update(self.discover_local_plugins())
        
        # Discover environment plugins
        plugins.update(self.discover_environment_plugins())
        
        self.discovered_plugins = plugins
        logger.info(f"Discovered {len(plugins)} plugins total")
        return plugins
    
    def discover_builtin_plugins(self) -> Dict[str, Type[BasePlugin]]:
        """
        Discover built-in plugins shipped with Genesis-Flow.
        
        Returns:
            Dict mapping plugin names to plugin classes
        """
        plugins = {}
        
        try:
            # Import built-in framework plugins
            from mlflow.plugins.builtin import BUILTIN_PLUGINS
            
            for name, plugin_class in BUILTIN_PLUGINS.items():
                if self._validate_plugin_class(plugin_class):
                    plugins[name] = plugin_class
                    logger.debug(f"Discovered built-in plugin: {name}")
                else:
                    logger.warning(f"Invalid built-in plugin: {name}")
            
            self.built_in_plugins = plugins
            logger.info(f"Discovered {len(plugins)} built-in plugins")
            
        except ImportError:
            logger.debug("No built-in plugins module found")
        except Exception as e:
            logger.error(f"Error discovering built-in plugins: {e}")
        
        return plugins
    
    def discover_entrypoint_plugins(self) -> Dict[str, Type[BasePlugin]]:
        """
        Discover plugins registered via setuptools entry points.
        
        Returns:
            Dict mapping plugin names to plugin classes
        """
        plugins = {}
        
        try:
            # Scan all entry point groups
            for plugin_type, group_name in self.ENTRY_POINT_GROUPS.items():
                for entry_point in iter_entry_points(group=group_name):
                    try:
                        plugin_class = entry_point.load()
                        
                        if self._validate_plugin_class(plugin_class):
                            plugins[entry_point.name] = plugin_class
                            logger.debug(f"Discovered entry point plugin: {entry_point.name}")
                        else:
                            logger.warning(f"Invalid entry point plugin: {entry_point.name}")
                            
                    except Exception as e:
                        logger.error(f"Failed to load entry point plugin {entry_point.name}: {e}")
            
            logger.info(f"Discovered {len(plugins)} entry point plugins")
            
        except Exception as e:
            logger.error(f"Error discovering entry point plugins: {e}")
        
        return plugins
    
    def discover_local_plugins(self) -> Dict[str, Type[BasePlugin]]:
        """
        Discover plugins from local directories.
        
        Searches for plugins in:
        - ~/.genesis-flow/plugins/
        - ./plugins/
        - $GENESIS_FLOW_PLUGINS_PATH
        
        Returns:
            Dict mapping plugin names to plugin classes
        """
        plugins = {}
        
        # Default plugin directories
        plugin_dirs = [
            Path.home() / ".genesis-flow" / "plugins",
            Path.cwd() / "plugins",
        ]
        
        # Add custom plugin path from environment
        custom_path = os.getenv("GENESIS_FLOW_PLUGINS_PATH")
        if custom_path:
            plugin_dirs.extend([Path(p) for p in custom_path.split(os.pathsep)])
        
        for plugin_dir in plugin_dirs:
            if plugin_dir.exists() and plugin_dir.is_dir():
                plugins.update(self._scan_directory(plugin_dir))
        
        logger.info(f"Discovered {len(plugins)} local plugins")
        return plugins
    
    def discover_environment_plugins(self) -> Dict[str, Type[BasePlugin]]:
        """
        Discover plugins specified via environment variables.
        
        Supports:
        - GENESIS_FLOW_PLUGINS: Comma-separated list of plugin module paths
        
        Returns:
            Dict mapping plugin names to plugin classes
        """
        plugins = {}
        
        env_plugins = os.getenv("GENESIS_FLOW_PLUGINS")
        if env_plugins:
            for plugin_module in env_plugins.split(","):
                plugin_module = plugin_module.strip()
                if plugin_module:
                    try:
                        module = importlib.import_module(plugin_module)
                        
                        # Look for plugin classes in the module
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (isinstance(attr, type) and 
                                issubclass(attr, BasePlugin) and 
                                attr != BasePlugin):
                                
                                if self._validate_plugin_class(attr):
                                    # Use the plugin's metadata name if available
                                    name = getattr(attr, 'metadata', {}).get('name', attr_name.lower())
                                    plugins[name] = attr
                                    logger.debug(f"Discovered environment plugin: {name}")
                                    
                    except Exception as e:
                        logger.error(f"Failed to load environment plugin {plugin_module}: {e}")
        
        logger.info(f"Discovered {len(plugins)} environment plugins")
        return plugins
    
    def _scan_directory(self, directory: Path) -> Dict[str, Type[BasePlugin]]:
        """
        Scan a directory for plugin files.
        
        Args:
            directory: Directory to scan
            
        Returns:
            Dict mapping plugin names to plugin classes
        """
        plugins = {}
        
        try:
            for plugin_file in directory.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue  # Skip private files
                
                try:
                    spec = importlib.util.spec_from_file_location(
                        plugin_file.stem, plugin_file
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Look for plugin classes
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (isinstance(attr, type) and 
                                issubclass(attr, BasePlugin) and 
                                attr != BasePlugin):
                                
                                if self._validate_plugin_class(attr):
                                    # Use the plugin's metadata name if available
                                    name = getattr(attr, 'metadata', {}).get('name', attr_name.lower())
                                    plugins[name] = attr
                                    logger.debug(f"Discovered local plugin: {name} from {plugin_file}")
                                    
                except Exception as e:
                    logger.error(f"Failed to load plugin from {plugin_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
        
        return plugins
    
    def _validate_plugin_class(self, plugin_class: Type) -> bool:
        """
        Validate that a class is a proper plugin.
        
        Args:
            plugin_class: Class to validate
            
        Returns:
            True if valid plugin class, False otherwise
        """
        try:
            # Must be a subclass of BasePlugin
            if not issubclass(plugin_class, BasePlugin):
                return False
            
            # Must not be BasePlugin itself
            if plugin_class == BasePlugin:
                return False
            
            # Must have required methods
            required_methods = ['load', 'enable', 'disable', 'unload']
            for method in required_methods:
                if not hasattr(plugin_class, method):
                    logger.warning(f"Plugin {plugin_class} missing required method: {method}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating plugin class {plugin_class}: {e}")
            return False
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict]:
        """
        Get information about a discovered plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin information dict or None if not found
        """
        if plugin_name in self.discovered_plugins:
            plugin_class = self.discovered_plugins[plugin_name]
            
            # Try to create a temporary instance to get metadata
            try:
                # This assumes plugins have a default metadata
                if hasattr(plugin_class, 'get_default_metadata'):
                    metadata = plugin_class.get_default_metadata()
                    return {
                        "name": metadata.name,
                        "version": metadata.version,
                        "description": metadata.description,
                        "author": metadata.author,
                        "type": metadata.plugin_type.value,
                        "dependencies": metadata.dependencies,
                        "optional_dependencies": metadata.optional_dependencies,
                        "tags": metadata.tags,
                    }
                else:
                    return {
                        "name": plugin_name,
                        "class": plugin_class.__name__,
                        "module": plugin_class.__module__,
                    }
                    
            except Exception as e:
                logger.error(f"Error getting info for plugin {plugin_name}: {e}")
                return None
        
        return None

# Global discovery instance
_discovery = None

def get_discovery() -> PluginDiscovery:
    """Get the global plugin discovery instance."""
    global _discovery
    if _discovery is None:
        _discovery = PluginDiscovery()
    return _discovery

def discover_plugins() -> Dict[str, Type[BasePlugin]]:
    """Discover all available plugins."""
    return get_discovery().discover_all_plugins()

def scan_entrypoints() -> Dict[str, Type[BasePlugin]]:
    """Scan for plugins via entry points only."""
    return get_discovery().discover_entrypoint_plugins()

def refresh_discovery():
    """Refresh plugin discovery (clear cache and rediscover)."""
    global _discovery
    _discovery = None
    return discover_plugins()