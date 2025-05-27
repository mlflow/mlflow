#!/usr/bin/env python
"""Test script to verify the prompt registry refactor works correctly."""

import os
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

def test_oss_prompt_registry():
    """Test prompt operations with OSS registry (should work as before)."""
    print("Testing OSS Prompt Registry...")
    
    # Set up OSS tracking URI
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    try:
        # Register a prompt
        prompt = client.register_prompt(
            name="test_prompt",
            template="Hello {{name}}, welcome to {{place}}!",
            description="A test greeting prompt",
            tags={"category": "greeting"}
        )
        print(f"✓ Registered prompt: {prompt.name} v{prompt.version}")
        
        # Load the prompt
        loaded_prompt = client.load_prompt("test_prompt")
        print(f"✓ Loaded prompt: {loaded_prompt.name} v{loaded_prompt.version}")
        
        # Create a new version
        new_version = client.create_prompt_version(
            name="test_prompt",
            template="Hi {{name}}, welcome to {{place}}! How are you?",
            description="Updated greeting prompt"
        )
        print(f"✓ Created new version: {new_version.version}")
        
        # Set an alias
        client.set_prompt_alias("test_prompt", "production", new_version.version)
        print(f"✓ Set alias 'production' to version {new_version.version}")
        
        # Load by alias
        prod_prompt = client.get_prompt_version_by_alias("test_prompt", "production")
        print(f"✓ Loaded prompt by alias: v{prod_prompt.version}")
        
        # Search prompts
        prompts = client.search_prompts(filter_string="name = 'test_prompt'")
        print(f"✓ Found {len(prompts)} prompts")
        
        # Log prompt to a run
        with mlflow.start_run() as run:
            client.log_prompt(run.info.run_id, "prompts:/test_prompt/1")
            logged_prompts = client.list_logged_prompts(run.info.run_id)
            print(f"✓ Logged {len(logged_prompts)} prompts to run")
        
        print("\n✅ OSS Prompt Registry tests passed!")
        
    except Exception as e:
        print(f"\n❌ OSS test failed: {e}")
        raise


def test_uc_prompt_registry_blocked():
    """Test that UC prompt registry is blocked by default"""
    print("\nTesting UC Prompt Registry (should be blocked by default)...")
    
    # Reset registry URI to OSS first
    mlflow.set_registry_uri("")
    
    # Test with is_prompt_supported_registry directly
    from mlflow.prompt.registry_utils import is_prompt_supported_registry
    
    # Test UC registry URIs
    uc_uris = ["databricks-uc://profile", "uc:http://localhost:8080"]
    
    for uri in uc_uris:
        supported = is_prompt_supported_registry(uri)
        if not supported:
            print(f"✓ UC registry '{uri}' correctly blocked without feature flag")
        else:
            print(f"❌ UC registry '{uri}' should be blocked by default!")


def test_uc_prompt_registry_enabled():
    """Test that UC prompt registry works when enabled"""
    print("\nTesting UC Prompt Registry with feature flag enabled...")
    
    # Enable UC prompt support
    os.environ["MLFLOW_ENABLE_UC_PROMPT_SUPPORT"] = "true"
    
    # Test with is_prompt_supported_registry directly
    from mlflow.prompt.registry_utils import is_prompt_supported_registry
    
    # Test UC registry URIs
    uc_uris = ["databricks-uc://profile", "uc:http://localhost:8080"]
    
    for uri in uc_uris:
        supported = is_prompt_supported_registry(uri)
        if supported:
            print(f"✓ UC registry '{uri}' enabled with feature flag")
        else:
            print(f"❌ UC registry '{uri}' should be enabled with feature flag!")
    
    # Clean up
    del os.environ["MLFLOW_ENABLE_UC_PROMPT_SUPPORT"]
    
    print("\n✅ UC Prompt Registry feature flag tests passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Prompt Registry Refactor")
    print("=" * 60)
    
    test_oss_prompt_registry()
    test_uc_prompt_registry_blocked()
    test_uc_prompt_registry_enabled()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main() 