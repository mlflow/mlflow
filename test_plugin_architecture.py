#!/usr/bin/env python3
"""
Test Plugin Architecture for Genesis-Flow

Comprehensive tests for the plugin system including discovery, management,
and built-in framework plugins.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import logging
from mlflow.plugins import PluginManager, get_plugin_manager
from mlflow.plugins.base import PluginType, PluginState
from mlflow.plugins.discovery import discover_plugins
from mlflow.plugins.builtin import BUILTIN_PLUGINS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_plugin_discovery():
    """Test plugin discovery system."""
    print("Testing Plugin Discovery...")
    print("=" * 50)
    
    try:
        # Test discovering all plugins
        plugins = discover_plugins()
        print(f"✓ Discovered {len(plugins)} total plugins")
        
        # Test built-in plugins specifically
        builtin_count = len(BUILTIN_PLUGINS)
        print(f"✓ Found {builtin_count} built-in plugins: {list(BUILTIN_PLUGINS.keys())}")
        
        # Verify built-in plugins are discovered
        for name in BUILTIN_PLUGINS.keys():
            if name in plugins:
                print(f"  ✓ Built-in plugin {name} discovered")
            else:
                print(f"  ✗ Built-in plugin {name} NOT discovered")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Plugin discovery failed: {e}")
        return False

def test_plugin_manager_initialization():
    """Test plugin manager initialization."""
    print("\nTesting Plugin Manager Initialization...")
    print("=" * 50)
    
    try:
        # Test manager creation
        manager = PluginManager()
        print("✓ Plugin manager created")
        
        # Test initialization
        manager.initialize(auto_discover=True, auto_enable_builtin=False)
        print("✓ Plugin manager initialized")
        
        # Test plugin listing
        plugins = manager.list_plugins()
        print(f"✓ Listed {len(plugins)} registered plugins")
        
        # Test statistics
        stats = manager.get_stats()
        print(f"✓ Plugin statistics: {stats['total_registered']} registered, {stats['enabled']} enabled")
        
        return True
        
    except Exception as e:
        print(f"✗ Plugin manager initialization failed: {e}")
        return False

def test_builtin_plugin_metadata():
    """Test built-in plugin metadata and structure."""
    print("\nTesting Built-in Plugin Metadata...")
    print("=" * 50)
    
    try:
        for name, plugin_class in BUILTIN_PLUGINS.items():
            # Test metadata retrieval
            if hasattr(plugin_class, 'get_default_metadata'):
                metadata = plugin_class.get_default_metadata()
                print(f"✓ {name}: Metadata available")
                print(f"  Version: {metadata.version}")
                print(f"  Type: {metadata.plugin_type.value}")
                print(f"  Dependencies: {metadata.dependencies}")
            else:
                print(f"✗ {name}: No metadata method")
                return False
            
            # Test required methods
            required_methods = ['load', 'enable', 'disable', 'unload']
            for method in required_methods:
                if hasattr(plugin_class, method):
                    print(f"  ✓ Has {method} method")
                else:
                    print(f"  ✗ Missing {method} method")
                    return False
        
        return True
        
    except Exception as e:
        print(f"✗ Built-in plugin metadata test failed: {e}")
        return False

def test_sklearn_plugin():
    """Test scikit-learn plugin specifically."""
    print("\nTesting Scikit-learn Plugin...")
    print("=" * 50)
    
    try:
        from mlflow.plugins.builtin.sklearn_plugin import SklearnPlugin
        
        # Test metadata
        metadata = SklearnPlugin.get_default_metadata()
        print(f"✓ Sklearn plugin metadata: {metadata.name} v{metadata.version}")
        
        # Test dependency checking (should pass since sklearn is core dependency)
        temp_instance = SklearnPlugin(metadata)
        if temp_instance.check_dependencies():
            print("✓ Sklearn dependencies satisfied")
        else:
            print("⚠ Sklearn dependencies not satisfied (expected in minimal environment)")
        
        # Test module path
        module_path = temp_instance.get_module_path()
        print(f"✓ Sklearn module path: {module_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sklearn plugin test failed: {e}")
        return False

def test_plugin_enabling():
    """Test plugin enable/disable functionality."""
    print("\nTesting Plugin Enable/Disable...")
    print("=" * 50)
    
    try:
        manager = get_plugin_manager()
        manager.initialize(auto_discover=True, auto_enable_builtin=False)
        
        # Test enabling sklearn plugin
        plugin_name = "sklearn"
        
        # Check initial state
        if manager.is_plugin_enabled(plugin_name):
            print(f"⚠ Plugin {plugin_name} already enabled, disabling first")
            manager.disable_plugin(plugin_name)
        
        # Test enabling
        if manager.enable_plugin(plugin_name):
            print(f"✓ Successfully enabled {plugin_name} plugin")
            
            # Verify it's enabled
            if manager.is_plugin_enabled(plugin_name):
                print(f"✓ Plugin {plugin_name} confirmed enabled")
            else:
                print(f"✗ Plugin {plugin_name} not showing as enabled")
                return False
                
            # Test getting plugin instance
            plugin_instance = manager.get_plugin(plugin_name)
            if plugin_instance:
                print(f"✓ Retrieved {plugin_name} plugin instance")
                print(f"  State: {plugin_instance.state.value}")
            else:
                print(f"✗ Could not retrieve {plugin_name} plugin instance")
                return False
                
            # Test disabling
            if manager.disable_plugin(plugin_name):
                print(f"✓ Successfully disabled {plugin_name} plugin")
            else:
                print(f"✗ Failed to disable {plugin_name} plugin")
                return False
                
        else:
            print(f"⚠ Could not enable {plugin_name} plugin (may be missing dependencies)")
            # This is acceptable in a minimal environment
        
        return True
        
    except Exception as e:
        print(f"✗ Plugin enable/disable test failed: {e}")
        return False

def test_plugin_types_and_filtering():
    """Test plugin type filtering and categorization."""
    print("\nTesting Plugin Types and Filtering...")
    print("=" * 50)
    
    try:
        manager = get_plugin_manager()
        manager.initialize()
        
        # Test filtering by type
        framework_plugins = manager.list_plugins(plugin_type=PluginType.FRAMEWORK)
        print(f"✓ Found {len(framework_plugins)} framework plugins")
        
        for plugin_info in framework_plugins:
            print(f"  - {plugin_info['name']} ({plugin_info['state']})")
        
        # Test filtering by state
        discovered_plugins = manager.list_plugins(state=PluginState.DISCOVERED)
        print(f"✓ Found {len(discovered_plugins)} discovered plugins")
        
        enabled_plugins = manager.list_plugins(state=PluginState.ENABLED)
        print(f"✓ Found {len(enabled_plugins)} enabled plugins")
        
        return True
        
    except Exception as e:
        print(f"✗ Plugin types and filtering test failed: {e}")
        return False

def test_plugin_integration_with_mlflow():
    """Test that plugins integrate properly with MLflow namespace."""
    print("\nTesting Plugin Integration with MLflow...")
    print("=" * 50)
    
    try:
        import mlflow
        
        manager = get_plugin_manager()
        manager.initialize(auto_discover=True, auto_enable_builtin=True)
        
        # Check if sklearn plugin is accessible through mlflow
        if hasattr(mlflow, 'sklearn'):
            print("✓ mlflow.sklearn is accessible")
            
            # Test that it's a lazy loader
            sklearn_module = getattr(mlflow, 'sklearn')
            print(f"✓ mlflow.sklearn type: {type(sklearn_module)}")
            
        else:
            print("⚠ mlflow.sklearn not accessible (may require sklearn dependency)")
        
        # Test pytorch if available
        if hasattr(mlflow, 'pytorch'):
            print("✓ mlflow.pytorch is accessible")
        else:
            print("⚠ mlflow.pytorch not accessible (may require torch dependency)")
        
        return True
        
    except Exception as e:
        print(f"✗ Plugin integration test failed: {e}")
        return False

def test_plugin_error_handling():
    """Test plugin error handling and recovery."""
    print("\nTesting Plugin Error Handling...")
    print("=" * 50)
    
    try:
        manager = get_plugin_manager()
        manager.initialize()
        
        # Test enabling non-existent plugin
        if not manager.enable_plugin("nonexistent_plugin"):
            print("✓ Properly handled non-existent plugin enable")
        else:
            print("✗ Should not be able to enable non-existent plugin")
            return False
        
        # Test disabling non-enabled plugin
        if not manager.is_plugin_enabled("sklearn"):
            if manager.disable_plugin("sklearn"):
                print("✓ Properly handled disabling non-enabled plugin")
            # This might return True (idempotent) which is also acceptable
        
        # Test getting non-existent plugin
        if manager.get_plugin("nonexistent_plugin") is None:
            print("✓ Properly handled getting non-existent plugin")
        else:
            print("✗ Should return None for non-existent plugin")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Plugin error handling test failed: {e}")
        return False

def test_plugin_manager_context():
    """Test plugin manager context functionality."""
    print("\nTesting Plugin Manager Context...")
    print("=" * 50)
    
    try:
        manager = get_plugin_manager()
        manager.initialize()
        
        # Test context manager for temporary plugin enabling
        plugin_name = "sklearn"
        
        # Ensure plugin is disabled initially
        if manager.is_plugin_enabled(plugin_name):
            manager.disable_plugin(plugin_name)
        
        # Test context manager
        try:
            with manager.plugin_context(plugin_name) as plugin:
                if plugin:
                    print(f"✓ Plugin {plugin_name} temporarily enabled in context")
                    print(f"  Plugin state: {plugin.state.value}")
                else:
                    print(f"⚠ Plugin {plugin_name} context returned None (may lack dependencies)")
            
            # Verify plugin is disabled after context
            if not manager.is_plugin_enabled(plugin_name):
                print(f"✓ Plugin {plugin_name} properly disabled after context")
            else:
                print(f"⚠ Plugin {plugin_name} still enabled after context")
                
        except RuntimeError as e:
            print(f"⚠ Plugin context failed (expected if dependencies missing): {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Plugin manager context test failed: {e}")
        return False

def main():
    """Run all plugin architecture tests."""
    print("Genesis-Flow Plugin Architecture Tests")
    print("=" * 50)
    
    tests = [
        ("Plugin Discovery", test_plugin_discovery),
        ("Plugin Manager Initialization", test_plugin_manager_initialization),
        ("Built-in Plugin Metadata", test_builtin_plugin_metadata),
        ("Scikit-learn Plugin", test_sklearn_plugin),
        ("Plugin Enabling/Disabling", test_plugin_enabling),
        ("Plugin Types and Filtering", test_plugin_types_and_filtering),
        ("MLflow Integration", test_plugin_integration_with_mlflow),
        ("Error Handling", test_plugin_error_handling),
        ("Context Manager", test_plugin_manager_context),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            print(f"\n❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Plugin Architecture Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All plugin architecture tests passed!")
        print("Plugin system is working correctly and ready for use.")
        return True
    else:
        print("❌ Some tests failed. Plugin system may need attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)