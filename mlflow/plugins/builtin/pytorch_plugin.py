"""
PyTorch Plugin for Genesis-Flow

Provides PyTorch integration as a modular plugin that can be enabled/disabled.
"""

import logging
from typing import Dict, Callable

from mlflow.plugins.base import FrameworkPlugin, PluginMetadata, PluginType

logger = logging.getLogger(__name__)

class PyTorchPlugin(FrameworkPlugin):
    """
    PyTorch framework plugin for Genesis-Flow.
    
    Provides PyTorch model logging, autologging, and deployment capabilities.
    """
    
    @classmethod
    def get_default_metadata(cls) -> PluginMetadata:
        """Get default metadata for PyTorch plugin."""
        return PluginMetadata(
            name="pytorch",
            version="1.0.0",
            description="PyTorch deep learning framework integration",
            author="Genesis-Flow Team",
            plugin_type=PluginType.FRAMEWORK,
            dependencies=["torch"],
            optional_dependencies=["torchvision", "torchaudio"],
            min_genesis_flow_version="1.0.0",
            homepage="https://pytorch.org",
            documentation="https://mlflow.org/docs/latest/models.html#pytorch-pytorch",
            license="Apache 2.0",
            tags=["deep-learning", "neural-networks", "pytorch"],
        )
    
    def get_module_path(self) -> str:
        """Get the module path for PyTorch integration."""
        return "mlflow.pytorch"
    
    def get_autolog_functions(self) -> Dict[str, Callable]:
        """Get PyTorch autologging functions."""
        try:
            from mlflow.pytorch import autolog
            return {
                "autolog": autolog,
            }
        except ImportError:
            logger.warning("PyTorch autolog not available")
            return {}
    
    def get_save_functions(self) -> Dict[str, Callable]:
        """Get PyTorch model saving functions."""
        try:
            from mlflow.pytorch import log_model, save_model
            return {
                "log_model": log_model,
                "save_model": save_model,
            }
        except ImportError:
            logger.warning("PyTorch save functions not available")
            return {}
    
    def get_load_functions(self) -> Dict[str, Callable]:
        """Get PyTorch model loading functions."""
        try:
            from mlflow.pytorch import load_model
            return {
                "load_model": load_model,
            }
        except ImportError:
            logger.warning("PyTorch load functions not available")
            return {}
    
    def enable(self) -> bool:
        """Enable PyTorch plugin with enhanced functionality."""
        try:
            # Call parent enable
            if not super().enable():
                return False
            
            # Additional PyTorch-specific setup
            self._setup_pytorch_environment()
            
            self._logger.info("PyTorch plugin enabled with full functionality")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to enable PyTorch plugin: {e}")
            return False
    
    def _setup_pytorch_environment(self):
        """Setup PyTorch-specific environment and configurations."""
        try:
            import torch
            
            # Log PyTorch version and CUDA availability
            self._logger.info(f"PyTorch version: {torch.__version__}")
            self._logger.info(f"CUDA available: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                self._logger.info(f"CUDA version: {torch.version.cuda}")
                self._logger.info(f"GPU count: {torch.cuda.device_count()}")
            
            # Setup hook for automatic model artifact logging
            self.register_hook("model_save", self._on_model_save)
            
        except Exception as e:
            self._logger.warning(f"PyTorch environment setup warning: {e}")
    
    def _on_model_save(self, model, *args, **kwargs):
        """Hook called when a model is saved."""
        try:
            import torch
            
            if isinstance(model, torch.nn.Module):
                self._logger.debug("PyTorch model save detected")
                # Additional logging or processing can be added here
                
        except Exception as e:
            self._logger.warning(f"PyTorch model save hook error: {e}")
    
    def check_dependencies(self) -> bool:
        """Enhanced dependency checking for PyTorch."""
        # Check basic dependencies first
        if not super().check_dependencies():
            return False
        
        try:
            import torch
            
            # Check minimum PyTorch version
            from packaging.version import Version
            min_pytorch_version = "1.9.0"
            
            if Version(torch.__version__) < Version(min_pytorch_version):
                self._logger.error(f"PyTorch {min_pytorch_version}+ required, found {torch.__version__}")
                return False
            
            self._logger.debug(f"PyTorch {torch.__version__} dependency satisfied")
            return True
            
        except ImportError:
            self._logger.error("PyTorch not installed")
            return False
        except Exception as e:
            self._logger.error(f"PyTorch dependency check failed: {e}")
            return False