#!/usr/bin/env python3
"""
Script to populate MLflow tracking server with test secrets and bindings.
Run this during development to test the secrets UI with realistic data.

Usage:
    python dev/test-secrets-data.py [--host HOST] [--port PORT]
"""

import argparse
import random
import requests
import uuid
from typing import List, Dict

# Configuration
PROVIDERS = [
    {
        "id": "openai",
        "name": "OpenAI",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    },
    {
        "id": "anthropic",
        "name": "Anthropic",
        "models": ["claude-sonnet-4-5-20250929", "claude-opus-4-1-20250805", "claude-haiku-4-5-20251001"],
    },
    {
        "id": "vertex_ai",
        "name": "Google Vertex AI",
        "models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
    },
    {
        "id": "bedrock",
        "name": "AWS Bedrock",
        "models": ["anthropic.claude-sonnet-4-5-20250929-v1:0", "meta.llama3-3-70b-instruct-v1:0"],
    },
    {
        "id": "databricks",
        "name": "Databricks",
        "models": ["databricks-claude-sonnet-4-5", "databricks-llama-4-maverick"],
    },
]

RESOURCE_TYPES = ["SCORER_JOB", "GLOBAL"]

ADJECTIVES = ["fast", "smart", "efficient", "reliable", "robust", "scalable", "innovative", "advanced", "optimized", "production"]
NOUNS = ["model", "assistant", "agent", "service", "endpoint", "pipeline", "workflow", "system", "engine", "processor"]
ENVIRONMENTS = ["dev", "staging", "prod", "test", "demo"]


def generate_secret_name(provider: str, model: str) -> str:
    """Generate a realistic secret name."""
    adj = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    env = random.choice(ENVIRONMENTS)
    return f"{adj}-{noun}-{env}"


def generate_api_key(provider: str) -> str:
    """Generate a fake API key."""
    if provider == "openai":
        return f"sk-proj-{uuid.uuid4().hex[:48]}"
    elif provider == "anthropic":
        return f"sk-ant-{uuid.uuid4().hex[:48]}"
    elif provider == "vertex_ai":
        return f"AIza{uuid.uuid4().hex[:35]}"
    elif provider == "bedrock":
        return f"AKIA{uuid.uuid4().hex[:32]}"
    else:
        return f"dbx-{uuid.uuid4().hex[:40]}"


def generate_resource_id(resource_type: str) -> str:
    """Generate a fake resource ID."""
    if resource_type == "SCORER_JOB":
        return f"scorer-job-{uuid.uuid4().hex[:8]}"
    elif resource_type == "EVALUATION_RUN":
        return f"eval-run-{uuid.uuid4().hex[:8]}"
    elif resource_type == "SERVING_ENDPOINT":
        return f"endpoint-{uuid.uuid4().hex[:8]}"
    return f"resource-{uuid.uuid4().hex[:8]}"


def generate_field_name(provider: str) -> str:
    """Generate environment variable name for the secret."""
    return f"{provider.upper()}_API_KEY"


def create_additional_binding(base_url: str, secret_id: str, provider: str) -> bool:
    """Add an additional binding to an existing secret."""
    # Don't bind to GLOBAL resources - only bind to things that USE the secret
    # (e.g., scorer jobs, gateway endpoints that consume the secret)
    non_global_types = [rt for rt in RESOURCE_TYPES if rt != "GLOBAL"]
    resource_type = random.choice(non_global_types)
    resource_id = generate_resource_id(resource_type)
    field_name = generate_field_name(provider)

    payload = {
        "secret_id": secret_id,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "field_name": field_name,
    }

    response = requests.post(
        f"{base_url}/ajax-api/3.0/mlflow/secrets/bind",
        json=payload,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code != 200:
        print(f"    ❌ Failed to add binding: {response.status_code} - {response.text}")
        return False

    result = response.json()
    binding_id = result['binding']['binding_id']
    print(f"    ✅ Additional binding: {resource_type} -> {resource_id} (binding_id={binding_id})")
    return True


def create_secret_with_bindings(base_url: str, provider_data: Dict, num_bindings: int, is_shared: bool = False) -> None:
    """Create a secret and bind it to random resources."""
    provider_id = provider_data["id"]
    model = random.choice(provider_data["models"])
    secret_name = generate_secret_name(provider_id, model)
    api_key = generate_api_key(provider_id)

    # Pick one random binding for the initial create-and-bind
    resource_type = random.choice(RESOURCE_TYPES)
    resource_id = generate_resource_id(resource_type)
    field_name = generate_field_name(provider_id)

    payload = {
        "secret_name": secret_name,
        "secret_value": api_key,
        "provider": provider_id,
        "model": model,
        "is_shared": is_shared,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "field_name": field_name,
    }

    print(f"Creating {'shared' if is_shared else 'private'} secret: {secret_name}")
    print(f"  Provider: {provider_id}, Model: {model}")
    print(f"  Initial binding: {resource_type} -> {resource_id}")

    response = requests.post(
        f"{base_url}/ajax-api/3.0/mlflow/secrets/create-and-bind",
        json=payload,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code != 200:
        print(f"  ❌ Failed: {response.status_code} - {response.text}")
        return

    result = response.json()
    secret_id = result['secret']['secret_id']
    print(f"  ✅ Created: secret_id={secret_id}")

    # Add additional bindings only for shared secrets (private secrets can only have one binding)
    # Also skip adding additional bindings for GLOBAL resource types (looks weird to have multiple GLOBAL bindings)
    additional_bindings = num_bindings - 1
    if additional_bindings > 0 and is_shared and resource_type != "GLOBAL":
        print(f"  Adding {additional_bindings} more binding(s)...")
        for _ in range(additional_bindings):
            create_additional_binding(base_url, secret_id, provider_id)
    elif additional_bindings > 0 and not is_shared:
        print(f"  ℹ️  Skipping additional bindings (private secrets can only have one binding)")
    elif additional_bindings > 0 and resource_type == "GLOBAL":
        print(f"  ℹ️  Skipping additional bindings (GLOBAL resource type already bound)")

    print()


def main():
    parser = argparse.ArgumentParser(description="Populate MLflow with test secrets and bindings")
    parser.add_argument("--host", default="localhost", help="MLflow tracking server host")
    parser.add_argument("--port", type=int, default=5000, help="MLflow tracking server port")
    parser.add_argument("--num-secrets", type=int, default=10, help="Number of secrets to create")
    parser.add_argument("--min-bindings", type=int, default=1, help="Minimum bindings per secret")
    parser.add_argument("--max-bindings", type=int, default=5, help="Maximum bindings per secret")
    parser.add_argument("--shared-ratio", type=float, default=0.3, help="Ratio of shared secrets (0.0-1.0)")

    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print("=" * 80)
    print(f"Creating {args.num_secrets} test secrets at {base_url}")
    print(f"Bindings per secret: {args.min_bindings}-{args.max_bindings}")
    print(f"Shared secret ratio: {args.shared_ratio * 100:.0f}%")
    print("=" * 80)
    print()

    for i in range(args.num_secrets):
        # Randomly select a provider
        provider_data = random.choice(PROVIDERS)

        # Decide if this should be a shared secret
        is_shared = random.random() < args.shared_ratio

        # Random number of bindings
        num_bindings = random.randint(args.min_bindings, args.max_bindings)

        print(f"[{i+1}/{args.num_secrets}] Provider: {provider_data['name']}")
        create_secret_with_bindings(base_url, provider_data, num_bindings, is_shared)

    print("=" * 80)
    print("✅ Test data creation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
