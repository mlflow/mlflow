"""
Plugin Manager for Genesis-Flow

Central management system for all plugins including loading, enabling,
disabling, and lifecycle management.
"""

import logging
import threading
from typing import Dict, List, Optional, Any, Type
from contextlib import contextmanager

from mlflow.plugins.base import BasePlugin, PluginState, PluginType
from mlflow.plugins.discovery import PluginDiscovery, discover_plugins
from mlflow.plugins.registry import PluginRegistry
from mlflow.version import VERSION as GENESIS_FLOW_VERSION

logger = logging.getLogger(__name__)

class PluginManager:
    """
    Central plugin management system for Genesis-Flow.
    
    Handles plugin discovery, loading, enabling/disabling, and lifecycle management.
    Thread-safe operations with proper error handling and state management.
    """
    
    def __init__(self):
        self._registry = PluginRegistry()
        self._discovery = PluginDiscovery()
        self._instances: Dict[str, BasePlugin] = {}
        self._lock = threading.RLock()
        self._initialized = False
        
    def initialize(self, auto_discover: bool = True, auto_enable_builtin: bool = True):
        """
        Initialize the plugin manager.
        
        Args:
            auto_discover: Whether to automatically discover all plugins
            auto_enable_builtin: Whether to automatically enable built-in plugins
        """
        with self._lock:
            if self._initialized:
                return
            
            try:
                logger.info("Initializing Genesis-Flow plugin manager")
                
                if auto_discover:
                    self._discover_plugins()
                
                if auto_enable_builtin:
                    self._enable_builtin_plugins()
                
                self._initialized = True
                logger.info(f"Plugin manager initialized with {len(self._instances)} plugins")
                
            except Exception as e:
                logger.error(f"Failed to initialize plugin manager: {e}")
                raise
    
    def _discover_plugins(self):
        """Discover all available plugins."""
        discovered = self._discovery.discover_all_plugins()
        
        for name, plugin_class in discovered.items():
            try:
                self._registry.register_plugin_class(name, plugin_class)
                logger.debug(f"Registered plugin class: {name}")
            except Exception as e:
                logger.error(f"Failed to register plugin {name}: {e}")
    
    def _enable_builtin_plugins(self):
        """Enable all built-in plugins that are compatible."""
        builtin_plugins = self._discovery.discover_builtin_plugins()
        
        for name, plugin_class in builtin_plugins.items():
            try:
                if self._is_plugin_compatible(name):
                    self.enable_plugin(name)
                    logger.debug(f"Auto-enabled built-in plugin: {name}")
                else:
                    logger.warning(f"Built-in plugin {name} is not compatible")
            except Exception as e:
                logger.error(f"Failed to auto-enable built-in plugin {name}: {e}")
    
    def list_plugins(self, plugin_type: Optional[PluginType] = None, 
                    state: Optional[PluginState] = None) -> List[Dict[str, Any]]:
        """
        List all plugins with optional filtering.
        
        Args:
            plugin_type: Filter by plugin type
            state: Filter by plugin state
            
        Returns:
            List of plugin information dictionaries
        """
        with self._lock:
            plugins = []
            
            # Include registered but not instantiated plugins
            for name in self._registry.list_plugins():
                plugin_info = self._get_plugin_info(name)
                if plugin_info:
                    # Apply filters
                    if plugin_type and plugin_info.get('type') != plugin_type.value:
                        continue
                    if state and plugin_info.get('state') != state.value:
                        continue
                    
                    plugins.append(plugin_info)
            
            return plugins
    
    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """
        Get a plugin instance by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None if not found/enabled
        """
        with self._lock:
            return self._instances.get(name)
    
    def enable_plugin(self, name: str) -> bool:
        """
        Enable a plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            True if successfully enabled, False otherwise
        """
        with self._lock:
            try:
                # Check if already enabled
                if name in self._instances:
                    instance = self._instances[name]
                    if instance.state == PluginState.ENABLED:
                        logger.debug(f"Plugin {name} already enabled")
                        return True
                
                # Get plugin class from registry
                plugin_class = self._registry.get_plugin_class(name)
                if not plugin_class:
                    logger.error(f"Plugin {name} not found in registry")
                    return False
                
                # Create instance if not exists
                if name not in self._instances:
                    try:
                        # Get metadata for the plugin
                        metadata = self._get_plugin_metadata(name, plugin_class)
                        instance = plugin_class(metadata)
                        self._instances[name] = instance
                    except Exception as e:
                        logger.error(f"Failed to create instance of plugin {name}: {e}")
                        return False
                
                instance = self._instances[name]
                
                # Check compatibility
                if not instance.is_compatible(GENESIS_FLOW_VERSION):
                    logger.error(f"Plugin {name} is not compatible with Genesis-Flow {GENESIS_FLOW_VERSION}")
                    return False
                
                # Check dependencies
                if not instance.check_dependencies():
                    logger.error(f"Plugin {name} dependencies not satisfied")
                    return False
                
                # Load if not loaded
                if instance.state == PluginState.DISCOVERED:
                    if not instance.load():
                        logger.error(f"Failed to load plugin {name}")
                        return False
                
                # Enable the plugin
                if instance.enable():
                    logger.info(f"Plugin {name} enabled successfully")
                    return True
                else:
                    logger.error(f"Failed to enable plugin {name}")
                    return False
                
            except Exception as e:
                logger.error(f"Error enabling plugin {name}: {e}")
                return False
    
    def disable_plugin(self, name: str) -> bool:
        """
        Disable a plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            True if successfully disabled, False otherwise
        """
        with self._lock:
            try:
                if name not in self._instances:
                    logger.warning(f"Plugin {name} not found or not enabled")
                    return True  # Already disabled
                
                instance = self._instances[name]
                
                if instance.disable():
                    logger.info(f"Plugin {name} disabled successfully")
                    return True
                else:
                    logger.error(f"Failed to disable plugin {name}")
                    return False
                
            except Exception as e:
                logger.error(f"Error disabling plugin {name}: {e}")
                return False
    
    def unload_plugin(self, name: str) -> bool:
        """
        Unload a plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            True if successfully unloaded, False otherwise
        """
        with self._lock:
            try:
                if name not in self._instances:
                    logger.warning(f"Plugin {name} not found")
                    return True  # Already unloaded
                
                instance = self._instances[name]
                
                if instance.unload():
                    del self._instances[name]
                    logger.info(f"Plugin {name} unloaded successfully")
                    return True
                else:
                    logger.error(f"Failed to unload plugin {name}")
                    return False
                
            except Exception as e:
                logger.error(f"Error unloading plugin {name}: {e}")
                return False
    
    def reload_plugin(self, name: str) -> bool:
        """
        Reload a plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            True if successfully reloaded, False otherwise
        """
        with self._lock:
            try:
                # Unload first
                if name in self._instances:
                    if not self.unload_plugin(name):
                        return False
                
                # Re-enable
                return self.enable_plugin(name)
                
            except Exception as e:
                logger.error(f"Error reloading plugin {name}: {e}")
                return False
    
    def is_plugin_enabled(self, name: str) -> bool:
        """
        Check if a plugin is enabled.
        
        Args:
            name: Plugin name
            
        Returns:
            True if enabled, False otherwise
        """
        with self._lock:
            if name in self._instances:
                return self._instances[name].state == PluginState.ENABLED
            return False
    
    def is_plugin_available(self, name: str) -> bool:
        """
        Check if a plugin is available (registered).
        
        Args:
            name: Plugin name
            
        Returns:
            True if available, False otherwise
        """
        return self._registry.is_plugin_registered(name)
    
    def _is_plugin_compatible(self, name: str) -> bool:
        """Check if a plugin is compatible with current environment."""
        try:
            plugin_class = self._registry.get_plugin_class(name)
            if not plugin_class:
                return False
            
            # Create temporary instance to check compatibility
            metadata = self._get_plugin_metadata(name, plugin_class)
            temp_instance = plugin_class(metadata)
            
            return (temp_instance.is_compatible(GENESIS_FLOW_VERSION) and 
                   temp_instance.check_dependencies())
            
        except Exception as e:
            logger.error(f"Error checking compatibility for plugin {name}: {e}")
            return False
    
    def _get_plugin_metadata(self, name: str, plugin_class: Type[BasePlugin]):
        """Get or create metadata for a plugin."""
        try:
            # Try to get metadata from plugin class
            if hasattr(plugin_class, 'get_default_metadata'):
                return plugin_class.get_default_metadata()
            
            # Create basic metadata
            from mlflow.plugins.base import PluginMetadata, PluginType
            return PluginMetadata(
                name=name,
                version="1.0.0",
                description=f"Plugin {name}",
                author="Genesis-Flow",
                plugin_type=PluginType.CUSTOM,
                dependencies=[],
                optional_dependencies=[],
                min_genesis_flow_version="1.0.0",
            )
            
        except Exception as e:
            logger.error(f"Error getting metadata for plugin {name}: {e}")
            raise
    
    def _get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a plugin."""
        try:
            if name in self._instances:
                return self._instances[name].get_info()
            
            # Get info from plugin class
            plugin_class = self._registry.get_plugin_class(name)
            if plugin_class:
                try:
                    metadata = self._get_plugin_metadata(name, plugin_class)
                    return {
                        "name": metadata.name,
                        "version": metadata.version,
                        "description": metadata.description,
                        "author": metadata.author,
                        "type": metadata.plugin_type.value,
                        "state": PluginState.DISCOVERED.value,
                        "dependencies": metadata.dependencies,
                        "optional_dependencies": metadata.optional_dependencies,
                        "tags": metadata.tags,
                    }
                except Exception:
                    return {
                        "name": name,
                        "class": plugin_class.__name__,
                        "state": PluginState.DISCOVERED.value,
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting info for plugin {name}: {e}")
            return None
    
    @contextmanager
    def plugin_context(self, name: str):
        """
        Context manager for temporarily enabling a plugin.
        
        Args:
            name: Plugin name
        """
        was_enabled = self.is_plugin_enabled(name)
        
        try:
            if not was_enabled:
                if not self.enable_plugin(name):
                    raise RuntimeError(f"Failed to enable plugin {name}")
            
            yield self.get_plugin(name)
            
        finally:
            if not was_enabled and self.is_plugin_enabled(name):
                self.disable_plugin(name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """
        Get all enabled plugins of a specific type.
        
        Args:
            plugin_type: Type of plugins to get
            
        Returns:
            List of enabled plugin instances
        """
        with self._lock:
            plugins = []
            for instance in self._instances.values():
                if (instance.state == PluginState.ENABLED and 
                    instance.metadata.plugin_type == plugin_type):
                    plugins.append(instance)
            return plugins
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get plugin manager statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            stats = {
                "total_registered": len(self._registry.list_plugins()),
                "total_instances": len(self._instances),
                "enabled": sum(1 for p in self._instances.values() 
                             if p.state == PluginState.ENABLED),
                "disabled": sum(1 for p in self._instances.values() 
                              if p.state == PluginState.DISABLED),
                "failed": sum(1 for p in self._instances.values() 
                            if p.state == PluginState.FAILED),
                "by_type": {},
            }
            
            # Count by type
            for instance in self._instances.values():
                plugin_type = instance.metadata.plugin_type.value
                if plugin_type not in stats["by_type"]:
                    stats["by_type"][plugin_type] = 0
                stats["by_type"][plugin_type] += 1
            
            return stats
    
    def shutdown(self):
        """Shutdown the plugin manager and unload all plugins."""
        with self._lock:
            logger.info("Shutting down plugin manager")
            
            # Unload all plugins
            for name in list(self._instances.keys()):
                try:
                    self.unload_plugin(name)
                except Exception as e:
                    logger.error(f"Error unloading plugin {name} during shutdown: {e}")
            
            self._instances.clear()
            self._initialized = False
            logger.info("Plugin manager shutdown complete")