"""
Base Plugin Classes for Genesis-Flow Plugin System

Defines the interface and base classes that all plugins must implement.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PluginType(Enum):
    """Types of plugins supported by Genesis-Flow."""
    FRAMEWORK = "framework"      # ML framework integration (PyTorch, TensorFlow, etc.)
    LOGGING = "logging"         # Custom logging backends
    MODEL_REGISTRY = "registry" # Model registry integrations
    DEPLOYMENT = "deployment"   # Deployment platforms
    ARTIFACT = "artifact"       # Artifact storage backends
    UI = "ui"                   # UI extensions
    CUSTOM = "custom"           # Custom extensions

@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str]
    optional_dependencies: List[str]
    min_genesis_flow_version: str
    max_genesis_flow_version: Optional[str] = None
    homepage: Optional[str] = None
    documentation: Optional[str] = None
    license: Optional[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class PluginState(Enum):
    """States a plugin can be in."""
    DISCOVERED = "discovered"
    LOADED = "loaded"
    ENABLED = "enabled" 
    DISABLED = "disabled"
    FAILED = "failed"

class BasePlugin(ABC):
    """
    Base class for all Genesis-Flow plugins.
    
    All plugins must inherit from this class and implement the required methods.
    """
    
    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self.state = PluginState.DISCOVERED
        self._logger = logging.getLogger(f"mlflow.plugins.{metadata.name}")
        self._hooks = {}
        
    @abstractmethod
    def load(self) -> bool:
        """
        Load the plugin and perform any necessary initialization.
        
        Returns:
            True if loading was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def enable(self) -> bool:
        """
        Enable the plugin for use.
        
        Returns:
            True if enabling was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disable(self) -> bool:
        """
        Disable the plugin.
        
        Returns:
            True if disabling was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def unload(self) -> bool:
        """
        Unload the plugin and clean up resources.
        
        Returns:
            True if unloading was successful, False otherwise
        """
        pass
    
    def is_compatible(self, genesis_flow_version: str) -> bool:
        """
        Check if the plugin is compatible with the current Genesis-Flow version.
        
        Args:
            genesis_flow_version: Current Genesis-Flow version
            
        Returns:
            True if compatible, False otherwise
        """
        from packaging.version import Version
        
        try:
            current = Version(genesis_flow_version)
            min_version = Version(self.metadata.min_genesis_flow_version)
            
            if current < min_version:
                return False
                
            if self.metadata.max_genesis_flow_version:
                max_version = Version(self.metadata.max_genesis_flow_version)
                if current > max_version:
                    return False
                    
            return True
            
        except Exception as e:
            self._logger.warning(f"Version compatibility check failed: {e}")
            return False
    
    def check_dependencies(self) -> bool:
        """
        Check if all required dependencies are available.
        
        Returns:
            True if all dependencies are satisfied, False otherwise
        """
        try:
            import importlib
            
            for dep in self.metadata.dependencies:
                try:
                    importlib.import_module(dep)
                except ImportError:
                    self._logger.error(f"Required dependency '{dep}' not found")
                    return False
            
            # Check optional dependencies and log warnings
            for dep in self.metadata.optional_dependencies:
                try:
                    importlib.import_module(dep)
                except ImportError:
                    self._logger.warning(f"Optional dependency '{dep}' not found")
            
            return True
            
        except Exception as e:
            self._logger.error(f"Dependency check failed: {e}")
            return False
    
    def register_hook(self, event: str, callback: Callable):
        """Register a callback for a specific event."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)
    
    def trigger_hook(self, event: str, *args, **kwargs):
        """Trigger all callbacks for a specific event."""
        if event in self._hooks:
            for callback in self._hooks[event]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    self._logger.error(f"Hook callback failed for {event}: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "description": self.metadata.description,
            "author": self.metadata.author,
            "type": self.metadata.plugin_type.value,
            "state": self.state.value,
            "dependencies": self.metadata.dependencies,
            "optional_dependencies": self.metadata.optional_dependencies,
            "tags": self.metadata.tags,
        }

class FrameworkPlugin(BasePlugin):
    """
    Base class for ML framework plugins.
    
    Framework plugins provide integration with ML libraries like PyTorch, TensorFlow, etc.
    """
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self._module_path = None
        self._lazy_loader = None
    
    @abstractmethod
    def get_module_path(self) -> str:
        """
        Get the module path for the framework integration.
        
        Returns:
            Module path (e.g., "mlflow.pytorch")
        """
        pass
    
    @abstractmethod
    def get_autolog_functions(self) -> Dict[str, Callable]:
        """
        Get autologging functions provided by this framework.
        
        Returns:
            Dict mapping function names to callables
        """
        pass
    
    @abstractmethod
    def get_save_functions(self) -> Dict[str, Callable]:
        """
        Get model saving functions provided by this framework.
        
        Returns:
            Dict mapping function names to callables
        """
        pass
    
    @abstractmethod
    def get_load_functions(self) -> Dict[str, Callable]:
        """
        Get model loading functions provided by this framework.
        
        Returns:
            Dict mapping function names to callables
        """
        pass
    
    def load(self) -> bool:
        """Load the framework plugin."""
        try:
            if not self.check_dependencies():
                self.state = PluginState.FAILED
                return False
            
            self._module_path = self.get_module_path()
            
            # Create lazy loader for the framework
            from mlflow.utils.lazy_load import LazyLoader
            import mlflow
            
            self._lazy_loader = LazyLoader(
                self._module_path,
                mlflow.__dict__,
                self._module_path
            )
            
            self.state = PluginState.LOADED
            self._logger.info(f"Framework plugin {self.metadata.name} loaded")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to load framework plugin {self.metadata.name}: {e}")
            self.state = PluginState.FAILED
            return False
    
    def enable(self) -> bool:
        """Enable the framework plugin."""
        try:
            if self.state != PluginState.LOADED:
                if not self.load():
                    return False
            
            # Register the lazy loader in mlflow namespace
            import mlflow
            setattr(mlflow, self.metadata.name, self._lazy_loader)
            
            self.state = PluginState.ENABLED
            self._logger.info(f"Framework plugin {self.metadata.name} enabled")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to enable framework plugin {self.metadata.name}: {e}")
            self.state = PluginState.FAILED
            return False
    
    def disable(self) -> bool:
        """Disable the framework plugin."""
        try:
            import mlflow
            
            if hasattr(mlflow, self.metadata.name):
                delattr(mlflow, self.metadata.name)
            
            self.state = PluginState.LOADED
            self._logger.info(f"Framework plugin {self.metadata.name} disabled")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to disable framework plugin {self.metadata.name}: {e}")
            return False
    
    def unload(self) -> bool:
        """Unload the framework plugin."""
        try:
            self.disable()
            
            self._lazy_loader = None
            self._module_path = None
            
            self.state = PluginState.DISCOVERED
            self._logger.info(f"Framework plugin {self.metadata.name} unloaded")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to unload framework plugin {self.metadata.name}: {e}")
            return False

class LoggingPlugin(BasePlugin):
    """Base class for custom logging backend plugins."""
    
    @abstractmethod
    def get_logging_backend(self):
        """Get the logging backend implementation."""
        pass

class ModelRegistryPlugin(BasePlugin):
    """Base class for model registry integration plugins."""
    
    @abstractmethod
    def get_registry_backend(self):
        """Get the model registry backend implementation."""
        pass

class DeploymentPlugin(BasePlugin):
    """Base class for deployment platform plugins."""
    
    @abstractmethod
    def get_deployment_backend(self):
        """Get the deployment backend implementation."""
        pass

class ArtifactPlugin(BasePlugin):
    """Base class for artifact storage plugins."""
    
    @abstractmethod
    def get_artifact_repository(self):
        """Get the artifact repository implementation."""
        pass