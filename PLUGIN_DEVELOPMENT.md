# Genesis-Flow Plugin Development Guide

This guide explains how to develop, register, and use plugins with the Genesis-Flow plugin architecture.

## Plugin Architecture Overview

Genesis-Flow uses a modular plugin system that allows:

- **Dynamic Discovery**: Plugins are automatically discovered from multiple sources
- **Lazy Loading**: Framework integrations load only when needed
- **Lifecycle Management**: Plugins can be enabled, disabled, and reloaded
- **Type Safety**: Strong typing and validation for plugin interfaces
- **Extensibility**: Easy addition of new ML frameworks and capabilities

## Plugin Types

### Framework Plugins
Integrate ML frameworks (PyTorch, TensorFlow, XGBoost, etc.)
- Provide model logging and loading
- Enable autologging capabilities
- Handle framework-specific configurations

### Logging Plugins
Custom logging backends and integrations
- Alternative storage systems
- Custom metric formats
- Enhanced monitoring

### Model Registry Plugins
Model registry integrations
- External registry systems
- Custom model metadata
- Version management

### Deployment Plugins
Deployment platform integrations
- Cloud platforms
- Container orchestration
- Serving frameworks

### Artifact Plugins
Custom artifact storage backends
- Alternative storage systems
- Custom serialization formats
- Enhanced security

## Creating a Framework Plugin

### 1. Use the Template

Start with the framework plugin template:

```bash
cp mlflow/plugins/templates/framework_plugin_template.py my_framework_plugin.py
```

### 2. Customize the Plugin

```python
from mlflow.plugins.base import FrameworkPlugin, PluginMetadata, PluginType

class TensorFlowPlugin(FrameworkPlugin):
    @classmethod
    def get_default_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="tensorflow",
            version="1.0.0",
            description="TensorFlow deep learning framework integration",
            author="Your Organization",
            plugin_type=PluginType.FRAMEWORK,
            dependencies=["tensorflow"],
            optional_dependencies=["tensorflow-gpu", "tensorboard"],
            min_genesis_flow_version="1.0.0",
            homepage="https://tensorflow.org",
            tags=["deep-learning", "tensorflow", "neural-networks"],
        )
    
    def get_module_path(self) -> str:
        return "mlflow.tensorflow"
    
    def get_autolog_functions(self) -> Dict[str, Callable]:
        try:
            from mlflow.tensorflow import autolog
            return {"autolog": autolog}
        except ImportError:
            return {}
    
    # Implement other required methods...
```

### 3. Implement Required Methods

All framework plugins must implement:

- `get_module_path()`: Return the MLflow integration module path
- `get_autolog_functions()`: Return autologging functions
- `get_save_functions()`: Return model saving functions
- `get_load_functions()`: Return model loading functions
- `check_dependencies()`: Validate framework installation

### 4. Add Framework-Specific Logic

```python
def _setup_framework_environment(self):
    """Setup TensorFlow-specific environment."""
    import tensorflow as tf
    
    # Log TensorFlow information
    self._logger.info(f"TensorFlow version: {tf.__version__}")
    self._logger.info(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Configure TensorFlow for optimal MLflow integration
    tf.config.set_soft_device_placement(True)
    
    # Register hooks for automatic logging
    self.register_hook("model_save", self._on_model_save)

def _on_model_save(self, model, *args, **kwargs):
    """Handle TensorFlow model save events."""
    if hasattr(model, 'summary'):
        # Log model architecture
        self._logger.debug("TensorFlow Keras model detected")
```

## Plugin Registration

### Method 1: Built-in Plugins

Add to `mlflow/plugins/builtin/__init__.py`:

```python
from mlflow.plugins.builtin.tensorflow_plugin import TensorFlowPlugin

BUILTIN_PLUGINS = {
    # existing plugins...
    "tensorflow": TensorFlowPlugin,
}
```

### Method 2: Entry Points

In your package's `setup.py` or `pyproject.toml`:

```python
# setup.py
entry_points={
    "genesis_flow.frameworks": [
        "tensorflow = mypackage.tensorflow_plugin:TensorFlowPlugin",
    ],
}
```

```toml
# pyproject.toml
[project.entry-points."genesis_flow.frameworks"]
tensorflow = "mypackage.tensorflow_plugin:TensorFlowPlugin"
```

### Method 3: Local Plugins

Place plugin files in:
- `~/.genesis-flow/plugins/`
- `./plugins/`
- Custom path via `GENESIS_FLOW_PLUGINS_PATH`

### Method 4: Environment Variable

```bash
export GENESIS_FLOW_PLUGINS="mypackage.tensorflow_plugin,another.plugin"
```

## Using Plugins

### Programmatic Usage

```python
from mlflow.plugins import get_plugin_manager

# Initialize plugin manager
manager = get_plugin_manager()
manager.initialize()

# List available plugins
plugins = manager.list_plugins()
print(f"Available plugins: {[p['name'] for p in plugins]}")

# Enable a plugin
manager.enable_plugin("tensorflow")

# Use the plugin through MLflow
import mlflow
import mlflow.tensorflow  # Now available

# Disable when done
manager.disable_plugin("tensorflow")
```

### CLI Usage

```bash
# List all plugins
genesis-flow plugins list

# Show detailed plugin information
genesis-flow plugins info tensorflow

# Enable a plugin
genesis-flow plugins enable tensorflow

# Disable a plugin
genesis-flow plugins disable tensorflow

# Show plugin statistics
genesis-flow plugins stats

# Enable all framework plugins
genesis-flow plugins enable-all --type framework
```

### Context Manager

```python
from mlflow.plugins import get_plugin_manager

manager = get_plugin_manager()

# Temporarily enable a plugin
with manager.plugin_context("tensorflow") as tf_plugin:
    # Plugin is enabled only within this context
    import mlflow.tensorflow
    # Use TensorFlow integration
    pass
# Plugin is automatically disabled here
```

## Plugin Development Best Practices

### 1. Error Handling

```python
def enable(self) -> bool:
    try:
        if not super().enable():
            return False
        
        # Framework-specific setup
        self._setup_framework()
        return True
        
    except Exception as e:
        self._logger.error(f"Failed to enable {self.metadata.name}: {e}")
        return False
```

### 2. Dependency Management

```python
def check_dependencies(self) -> bool:
    if not super().check_dependencies():
        return False
    
    try:
        import myframework
        from packaging.version import Version
        
        if Version(myframework.__version__) < Version("2.0.0"):
            self._logger.error("MyFramework 2.0+ required")
            return False
            
        return True
    except ImportError:
        self._logger.error("MyFramework not installed")
        return False
```

### 3. Graceful Degradation

```python
def get_autolog_functions(self) -> Dict[str, Callable]:
    try:
        from mlflow.myframework import autolog
        return {"autolog": autolog}
    except ImportError:
        # Return empty dict if integration not available
        self._logger.warning("MyFramework autolog not available")
        return {}
```

### 4. Configuration Management

```python
def _setup_framework_environment(self):
    # Check for framework-specific environment variables
    gpu_memory_growth = os.getenv("MYFRAMEWORK_GPU_MEMORY_GROWTH", "true")
    
    # Apply configuration
    if gpu_memory_growth.lower() == "true":
        self._configure_gpu_memory_growth()
```

### 5. Hooks and Events

```python
def enable(self) -> bool:
    if not super().enable():
        return False
    
    # Register for framework events
    self.register_hook("model_save", self._on_model_save)
    self.register_hook("training_start", self._on_training_start)
    
    return True

def _on_model_save(self, model, path, **kwargs):
    """Automatically log model metadata."""
    if self._is_my_framework_model(model):
        self._log_model_architecture(model)
```

## Testing Plugins

### Unit Tests

```python
import pytest
from mlflow.plugins.base import PluginState
from mypackage.tensorflow_plugin import TensorFlowPlugin

def test_tensorflow_plugin_metadata():
    metadata = TensorFlowPlugin.get_default_metadata()
    assert metadata.name == "tensorflow"
    assert metadata.plugin_type == PluginType.FRAMEWORK

def test_tensorflow_plugin_enable():
    plugin = TensorFlowPlugin(TensorFlowPlugin.get_default_metadata())
    
    # Mock dependencies if needed
    with pytest.mock.patch("tensorflow"):
        assert plugin.enable()
        assert plugin.state == PluginState.ENABLED
```

### Integration Tests

```python
def test_tensorflow_plugin_integration():
    from mlflow.plugins import get_plugin_manager
    
    manager = get_plugin_manager()
    manager.initialize()
    
    # Test enabling
    assert manager.enable_plugin("tensorflow")
    assert manager.is_plugin_enabled("tensorflow")
    
    # Test MLflow integration
    import mlflow
    assert hasattr(mlflow, "tensorflow")
    
    # Test disabling
    assert manager.disable_plugin("tensorflow")
    assert not manager.is_plugin_enabled("tensorflow")
```

## Advanced Plugin Features

### Custom Plugin Types

```python
from mlflow.plugins.base import BasePlugin, PluginType

class CustomPluginType(PluginType):
    MONITORING = "monitoring"
    SECURITY = "security"

class MonitoringPlugin(BasePlugin):
    def __init__(self, metadata):
        metadata.plugin_type = CustomPluginType.MONITORING
        super().__init__(metadata)
```

### Plugin Dependencies

```python
class AdvancedPlugin(FrameworkPlugin):
    def check_dependencies(self) -> bool:
        if not super().check_dependencies():
            return False
        
        # Check for other plugins
        manager = get_plugin_manager()
        if not manager.is_plugin_enabled("required_plugin"):
            self._logger.error("Required plugin 'required_plugin' not enabled")
            return False
        
        return True
```

### Plugin Communication

```python
class CollaborativePlugin(BasePlugin):
    def enable(self) -> bool:
        if not super().enable():
            return False
        
        # Register for events from other plugins
        manager = get_plugin_manager()
        other_plugin = manager.get_plugin("other_plugin")
        if other_plugin:
            other_plugin.register_hook("data_processed", self._on_data_processed)
        
        return True
    
    def _on_data_processed(self, data):
        # Handle data from another plugin
        self._process_shared_data(data)
```

## Troubleshooting

### Common Issues

1. **Plugin Not Discovered**
   - Check entry point configuration
   - Verify plugin file placement
   - Check environment variables

2. **Dependencies Not Satisfied**
   - Install required packages
   - Check minimum version requirements
   - Use `--force` flag for testing

3. **Import Errors**
   - Verify module paths
   - Check Python path configuration
   - Ensure MLflow integration exists

### Debug Mode

```bash
# Enable debug logging
export PYTHONPATH=/path/to/genesis-flow
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from mlflow.plugins import get_plugin_manager
manager = get_plugin_manager()
manager.initialize()
"
```

### Plugin Validation

```python
from mlflow.plugins.discovery import PluginDiscovery

discovery = PluginDiscovery()
plugins = discovery.discover_all_plugins()

for name, plugin_class in plugins.items():
    print(f"Validating {name}...")
    if discovery._validate_plugin_class(plugin_class):
        print(f"  ✓ {name} is valid")
    else:
        print(f"  ✗ {name} is invalid")
```

This plugin system provides a powerful, extensible foundation for adding new ML frameworks and capabilities to Genesis-Flow while maintaining backwards compatibility and ease of use.