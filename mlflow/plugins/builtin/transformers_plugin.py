"""
Transformers Plugin for Genesis-Flow

Provides Hugging Face Transformers integration as a modular plugin.
"""

import logging
from typing import Dict, Callable

from mlflow.plugins.base import FrameworkPlugin, PluginMetadata, PluginType

logger = logging.getLogger(__name__)

class TransformersPlugin(FrameworkPlugin):
    """
    Hugging Face Transformers framework plugin for Genesis-Flow.
    
    Provides transformers model logging, autologging, and deployment capabilities.
    """
    
    @classmethod
    def get_default_metadata(cls) -> PluginMetadata:
        """Get default metadata for transformers plugin."""
        return PluginMetadata(
            name="transformers",
            version="1.0.0",
            description="Hugging Face Transformers NLP framework integration",
            author="Genesis-Flow Team",
            plugin_type=PluginType.FRAMEWORK,
            dependencies=["transformers"],
            optional_dependencies=["torch", "tensorflow", "sentencepiece", "tokenizers"],
            min_genesis_flow_version="1.0.0",
            homepage="https://huggingface.co/transformers",
            documentation="https://mlflow.org/docs/latest/models.html#transformers-transformers",
            license="Apache 2.0",
            tags=["nlp", "transformers", "hugging-face", "llm"],
        )
    
    def get_module_path(self) -> str:
        """Get the module path for transformers integration."""
        return "mlflow.transformers"
    
    def get_autolog_functions(self) -> Dict[str, Callable]:
        """Get transformers autologging functions."""
        try:
            from mlflow.transformers import autolog
            return {
                "autolog": autolog,
            }
        except ImportError:
            logger.warning("Transformers autolog not available")
            return {}
    
    def get_save_functions(self) -> Dict[str, Callable]:
        """Get transformers model saving functions."""
        try:
            from mlflow.transformers import log_model, save_model
            return {
                "log_model": log_model,
                "save_model": save_model,
            }
        except ImportError:
            logger.warning("Transformers save functions not available")
            return {}
    
    def get_load_functions(self) -> Dict[str, Callable]:
        """Get transformers model loading functions."""
        try:
            from mlflow.transformers import load_model
            return {
                "load_model": load_model,
            }
        except ImportError:
            logger.warning("Transformers load functions not available")
            return {}
    
    def enable(self) -> bool:
        """Enable transformers plugin with enhanced functionality."""
        try:
            # Call parent enable
            if not super().enable():
                return False
            
            # Additional transformers-specific setup
            self._setup_transformers_environment()
            
            self._logger.info("Transformers plugin enabled with full functionality")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to enable transformers plugin: {e}")
            return False
    
    def _setup_transformers_environment(self):
        """Setup transformers-specific environment and configurations."""
        try:
            import transformers
            
            # Log transformers version
            self._logger.info(f"Transformers version: {transformers.__version__}")
            
            # Check for available backends
            try:
                import torch
                self._logger.info(f"PyTorch backend available: {torch.__version__}")
            except ImportError:
                pass
            
            try:
                import tensorflow as tf
                self._logger.info(f"TensorFlow backend available: {tf.__version__}")
            except ImportError:
                pass
            
            # Setup hooks for automatic model artifact logging
            self.register_hook("model_save", self._on_model_save)
            self.register_hook("model_load", self._on_model_load)
            
        except Exception as e:
            self._logger.warning(f"Transformers environment setup warning: {e}")
    
    def _on_model_save(self, model, *args, **kwargs):
        """Hook called when a model is saved."""
        try:
            # Check if it's a transformers model
            if hasattr(model, 'config') and hasattr(model, 'save_pretrained'):
                self._logger.debug("Transformers model save detected")
                # Additional logging or processing can be added here
                
        except Exception as e:
            self._logger.warning(f"Transformers model save hook error: {e}")
    
    def _on_model_load(self, model_path, *args, **kwargs):
        """Hook called when a model is loaded."""
        try:
            self._logger.debug(f"Transformers model load detected: {model_path}")
            # Additional processing can be added here
            
        except Exception as e:
            self._logger.warning(f"Transformers model load hook error: {e}")
    
    def check_dependencies(self) -> bool:
        """Enhanced dependency checking for transformers."""
        # Check basic dependencies first
        if not super().check_dependencies():
            return False
        
        try:
            import transformers
            
            # Check minimum transformers version
            from packaging.version import Version
            min_transformers_version = "4.20.0"
            
            if Version(transformers.__version__) < Version(min_transformers_version):
                self._logger.error(f"Transformers {min_transformers_version}+ required, found {transformers.__version__}")
                return False
            
            # Check that at least one backend is available
            has_torch = False
            has_tf = False
            
            try:
                import torch
                has_torch = True
            except ImportError:
                pass
            
            try:
                import tensorflow
                has_tf = True
            except ImportError:
                pass
            
            if not (has_torch or has_tf):
                self._logger.warning("Neither PyTorch nor TensorFlow backend found for Transformers")
                # Don't fail, but warn
            
            self._logger.debug(f"Transformers {transformers.__version__} dependency satisfied")
            return True
            
        except ImportError:
            self._logger.error("Transformers not installed")
            return False
        except Exception as e:
            self._logger.error(f"Transformers dependency check failed: {e}")
            return False