"""
Plugin Registry for Genesis-Flow

Central registry for tracking discovered and registered plugins.
Provides thread-safe registration, lookup, and management of plugin classes.
"""

import logging
import threading
from typing import Dict, List, Optional, Type, Set

from mlflow.plugins.base import BasePlugin, PluginType

logger = logging.getLogger(__name__)

class PluginRegistry:
    """
    Thread-safe registry for plugin classes.
    
    Maintains a central catalog of all discovered plugin classes,
    their types, and metadata.
    """
    
    def __init__(self):
        self._plugins: Dict[str, Type[BasePlugin]] = {}
        self._plugins_by_type: Dict[PluginType, Set[str]] = {}
        self._lock = threading.RLock()
        
        # Initialize type sets
        for plugin_type in PluginType:
            self._plugins_by_type[plugin_type] = set()
    
    def register_plugin_class(self, name: str, plugin_class: Type[BasePlugin]) -> bool:
        """
        Register a plugin class.
        
        Args:
            name: Plugin name
            plugin_class: Plugin class
            
        Returns:
            True if registered successfully, False otherwise
        """
        with self._lock:
            try:
                # Validate plugin class
                if not self._validate_plugin_class(plugin_class):
                    logger.error(f"Invalid plugin class for {name}")
                    return False
                
                # Check for conflicts
                if name in self._plugins:
                    existing_class = self._plugins[name]
                    if existing_class != plugin_class:
                        logger.warning(f"Plugin {name} already registered with different class")
                        # Allow override but log it
                    else:
                        logger.debug(f"Plugin {name} already registered with same class")
                        return True
                
                # Register the plugin
                self._plugins[name] = plugin_class
                
                # Add to type index
                plugin_type = self._get_plugin_type(plugin_class)
                if plugin_type:
                    self._plugins_by_type[plugin_type].add(name)
                
                logger.debug(f"Registered plugin: {name}")
                return True
                
            except Exception as e:
                logger.error(f"Error registering plugin {name}: {e}")
                return False
    
    def unregister_plugin(self, name: str) -> bool:
        """
        Unregister a plugin class.
        
        Args:
            name: Plugin name
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        with self._lock:
            try:
                if name not in self._plugins:
                    logger.warning(f"Plugin {name} not registered")
                    return True  # Already unregistered
                
                plugin_class = self._plugins[name]
                
                # Remove from main registry
                del self._plugins[name]
                
                # Remove from type index
                plugin_type = self._get_plugin_type(plugin_class)
                if plugin_type and name in self._plugins_by_type[plugin_type]:
                    self._plugins_by_type[plugin_type].remove(name)
                
                logger.debug(f"Unregistered plugin: {name}")
                return True
                
            except Exception as e:
                logger.error(f"Error unregistering plugin {name}: {e}")
                return False
    
    def get_plugin_class(self, name: str) -> Optional[Type[BasePlugin]]:
        """
        Get a plugin class by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin class or None if not found
        """
        with self._lock:
            return self._plugins.get(name)
    
    def is_plugin_registered(self, name: str) -> bool:
        """
        Check if a plugin is registered.
        
        Args:
            name: Plugin name
            
        Returns:
            True if registered, False otherwise
        """
        with self._lock:
            return name in self._plugins
    
    def list_plugins(self, plugin_type: Optional[PluginType] = None) -> List[str]:
        """
        List registered plugin names.
        
        Args:
            plugin_type: Optional filter by plugin type
            
        Returns:
            List of plugin names
        """
        with self._lock:
            if plugin_type:
                return list(self._plugins_by_type.get(plugin_type, set()))
            else:
                return list(self._plugins.keys())
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> Dict[str, Type[BasePlugin]]:
        """
        Get all plugins of a specific type.
        
        Args:
            plugin_type: Plugin type
            
        Returns:
            Dict mapping plugin names to classes
        """
        with self._lock:
            result = {}
            plugin_names = self._plugins_by_type.get(plugin_type, set())
            for name in plugin_names:
                if name in self._plugins:
                    result[name] = self._plugins[name]
            return result
    
    def get_plugin_types(self) -> Dict[str, PluginType]:
        """
        Get plugin types for all registered plugins.
        
        Returns:
            Dict mapping plugin names to types
        """
        with self._lock:
            result = {}
            for name, plugin_class in self._plugins.items():
                plugin_type = self._get_plugin_type(plugin_class)
                if plugin_type:
                    result[name] = plugin_type
            return result
    
    def clear(self):
        """Clear all registered plugins."""
        with self._lock:
            self._plugins.clear()
            for plugin_type in PluginType:
                self._plugins_by_type[plugin_type].clear()
            logger.info("Plugin registry cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get registry statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            stats = {
                "total": len(self._plugins),
            }
            
            # Count by type
            for plugin_type in PluginType:
                count = len(self._plugins_by_type[plugin_type])
                stats[plugin_type.value] = count
            
            return stats
    
    def _validate_plugin_class(self, plugin_class: Type) -> bool:
        """
        Validate a plugin class.
        
        Args:
            plugin_class: Class to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Must be a class
            if not isinstance(plugin_class, type):
                return False
            
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
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating plugin class: {e}")
            return False
    
    def _get_plugin_type(self, plugin_class: Type[BasePlugin]) -> Optional[PluginType]:
        """
        Determine the type of a plugin class.
        
        Args:
            plugin_class: Plugin class
            
        Returns:
            Plugin type or None if cannot determine
        """
        try:
            # Try to get from class metadata
            if hasattr(plugin_class, 'get_default_metadata'):
                metadata = plugin_class.get_default_metadata()
                return metadata.plugin_type
            
            # Try to infer from class hierarchy
            from mlflow.plugins.base import (
                FrameworkPlugin, LoggingPlugin, ModelRegistryPlugin,
                DeploymentPlugin, ArtifactPlugin
            )
            
            if issubclass(plugin_class, FrameworkPlugin):
                return PluginType.FRAMEWORK
            elif issubclass(plugin_class, LoggingPlugin):
                return PluginType.LOGGING
            elif issubclass(plugin_class, ModelRegistryPlugin):
                return PluginType.MODEL_REGISTRY
            elif issubclass(plugin_class, DeploymentPlugin):
                return PluginType.DEPLOYMENT
            elif issubclass(plugin_class, ArtifactPlugin):
                return PluginType.ARTIFACT
            else:
                return PluginType.CUSTOM
                
        except Exception as e:
            logger.error(f"Error determining plugin type for {plugin_class}: {e}")
            return PluginType.CUSTOM
    
    def export_registry(self) -> Dict[str, Dict]:
        """
        Export registry contents for serialization.
        
        Returns:
            Serializable registry data
        """
        with self._lock:
            result = {}
            for name, plugin_class in self._plugins.items():
                result[name] = {
                    "class_name": plugin_class.__name__,
                    "module": plugin_class.__module__,
                    "type": self._get_plugin_type(plugin_class).value if self._get_plugin_type(plugin_class) else None,
                }
            return result
    
    def find_plugins_by_pattern(self, pattern: str) -> List[str]:
        """
        Find plugins matching a name pattern.
        
        Args:
            pattern: Pattern to match (supports * wildcard)
            
        Returns:
            List of matching plugin names
        """
        import fnmatch
        
        with self._lock:
            return [name for name in self._plugins.keys() 
                   if fnmatch.fnmatch(name, pattern)]