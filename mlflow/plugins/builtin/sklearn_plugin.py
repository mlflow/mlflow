"""
Scikit-learn Plugin for Genesis-Flow

Provides scikit-learn integration as a modular plugin.
"""

import logging
from typing import Dict, Callable

from mlflow.plugins.base import FrameworkPlugin, PluginMetadata, PluginType

logger = logging.getLogger(__name__)

class SklearnPlugin(FrameworkPlugin):
    """
    Scikit-learn framework plugin for Genesis-Flow.
    
    Provides scikit-learn model logging, autologging, and deployment capabilities.
    """
    
    @classmethod
    def get_default_metadata(cls) -> PluginMetadata:
        """Get default metadata for scikit-learn plugin."""
        return PluginMetadata(
            name="sklearn",
            version="1.0.0",
            description="Scikit-learn machine learning framework integration",
            author="Genesis-Flow Team",
            plugin_type=PluginType.FRAMEWORK,
            dependencies=["scikit-learn"],
            optional_dependencies=["numpy", "scipy", "pandas"],
            min_genesis_flow_version="1.0.0",
            homepage="https://scikit-learn.org",
            documentation="https://mlflow.org/docs/latest/models.html#scikit-learn-sklearn",
            license="Apache 2.0",
            tags=["machine-learning", "scikit-learn", "sklearn"],
        )
    
    def get_module_path(self) -> str:
        """Get the module path for scikit-learn integration."""
        return "mlflow.sklearn"
    
    def get_autolog_functions(self) -> Dict[str, Callable]:
        """Get scikit-learn autologging functions."""
        try:
            from mlflow.sklearn import autolog
            return {
                "autolog": autolog,
            }
        except ImportError:
            logger.warning("Scikit-learn autolog not available")
            return {}
    
    def get_save_functions(self) -> Dict[str, Callable]:
        """Get scikit-learn model saving functions."""
        try:
            from mlflow.sklearn import log_model, save_model
            return {
                "log_model": log_model,
                "save_model": save_model,
            }
        except ImportError:
            logger.warning("Scikit-learn save functions not available")
            return {}
    
    def get_load_functions(self) -> Dict[str, Callable]:
        """Get scikit-learn model loading functions."""
        try:
            from mlflow.sklearn import load_model
            return {
                "load_model": load_model,
            }
        except ImportError:
            logger.warning("Scikit-learn load functions not available")
            return {}
    
    def enable(self) -> bool:
        """Enable scikit-learn plugin with enhanced functionality."""
        try:
            # Call parent enable
            if not super().enable():
                return False
            
            # Additional scikit-learn-specific setup
            self._setup_sklearn_environment()
            
            self._logger.info("Scikit-learn plugin enabled with full functionality")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to enable scikit-learn plugin: {e}")
            return False
    
    def _setup_sklearn_environment(self):
        """Setup scikit-learn-specific environment and configurations."""
        try:
            import sklearn
            
            # Log scikit-learn version
            self._logger.info(f"Scikit-learn version: {sklearn.__version__}")
            
            # Setup hooks for automatic model artifact logging
            self.register_hook("model_save", self._on_model_save)
            self.register_hook("model_fit", self._on_model_fit)
            
        except Exception as e:
            self._logger.warning(f"Scikit-learn environment setup warning: {e}")
    
    def _on_model_save(self, model, *args, **kwargs):
        """Hook called when a model is saved."""
        try:
            import sklearn.base
            
            if isinstance(model, sklearn.base.BaseEstimator):
                self._logger.debug("Scikit-learn model save detected")
                # Additional logging or processing can be added here
                
        except Exception as e:
            self._logger.warning(f"Scikit-learn model save hook error: {e}")
    
    def _on_model_fit(self, model, *args, **kwargs):
        """Hook called when a model is fitted."""
        try:
            import sklearn.base
            
            if isinstance(model, sklearn.base.BaseEstimator):
                self._logger.debug("Scikit-learn model fit detected")
                # Could automatically log model parameters here
                
        except Exception as e:
            self._logger.warning(f"Scikit-learn model fit hook error: {e}")
    
    def check_dependencies(self) -> bool:
        """Enhanced dependency checking for scikit-learn."""
        # Check basic dependencies first
        if not super().check_dependencies():
            return False
        
        try:
            import sklearn
            
            # Check minimum scikit-learn version
            from packaging.version import Version
            min_sklearn_version = "1.0.0"
            
            if Version(sklearn.__version__) < Version(min_sklearn_version):
                self._logger.error(f"Scikit-learn {min_sklearn_version}+ required, found {sklearn.__version__}")
                return False
            
            self._logger.debug(f"Scikit-learn {sklearn.__version__} dependency satisfied")
            return True
            
        except ImportError:
            self._logger.error("Scikit-learn not installed")
            return False
        except Exception as e:
            self._logger.error(f"Scikit-learn dependency check failed: {e}")
            return False