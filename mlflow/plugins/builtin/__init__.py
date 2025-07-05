"""
Built-in Plugins for Genesis-Flow

This module contains all built-in plugins that ship with Genesis-Flow.
These plugins provide integration with essential ML frameworks.
"""

from mlflow.plugins.builtin.pytorch_plugin import PyTorchPlugin
from mlflow.plugins.builtin.sklearn_plugin import SklearnPlugin
from mlflow.plugins.builtin.transformers_plugin import TransformersPlugin

# Registry of all built-in plugins
BUILTIN_PLUGINS = {
    "pytorch": PyTorchPlugin,
    "sklearn": SklearnPlugin,
    "transformers": TransformersPlugin,
}

__all__ = [
    "BUILTIN_PLUGINS",
    "PyTorchPlugin",
    "SklearnPlugin", 
    "TransformersPlugin",
]