"""
Validate authentication for agent evaluation.

This script tests authentication to required services:
- MLflow tracking server (Databricks or local)
- LLM provider (if configured)

Performs lightweight API calls to verify credentials before expensive operations.

Usage:
    python scripts/validate_auth.py
"""

import os
import sys

from utils import check_databricks_config, validate_env_vars


def check_databricks_auth():
    """Test Databricks authentication."""
    print("Testing Databricks authentication...")

    is_databricks, profile = check_databricks_config()

    if not is_databricks:
        print("  ⊘ Not using Databricks (skipped)")
        print()
        return []

    # Check for auth credentials
    token = os.getenv("DATABRICKS_TOKEN")
    host = os.getenv("DATABRICKS_HOST")

    if not token and not host:
        # Try using databricks SDK (more robust) or fallback to CLI
        try:
            # Try new Databricks SDK first
            try:
                from databricks import sdk

                print("  ↻ Using Databricks SDK...")

                # Try to create workspace client
                try:
                    w = sdk.WorkspaceClient()
                    # Test with a simple API call
                    current_user = w.current_user.me()
                    print(f"  ✓ Authenticated as: {current_user.user_name}")
                    print()
                    return []

                except AttributeError as e:
                    # Handle NoneType error gracefully
                    if "'NoneType'" in str(e):
                        print("  ✗ Databricks configuration incomplete or corrupted")
                        print()
                        return ["Run: databricks auth login --profile DEFAULT"]
                    raise

            except ImportError:
                # Fall back to old databricks-cli
                from databricks_cli.sdk.api_client import ApiClient

                print("  ↻ Using Databricks CLI profile...")

                try:
                    api_client = ApiClient()

                    # Check if api_client is properly initialized
                    if api_client is None or not hasattr(api_client, "host"):
                        print("  ✗ Databricks CLI profile not configured")
                        print()
                        return ["Run: databricks auth login --profile DEFAULT"]

                except (AttributeError, TypeError) as e:
                    print(f"  ✗ Profile configuration error: {str(e)[:80]}")
                    print()
                    return ["Run: databricks auth login --profile DEFAULT"]

            # Test with MLflow client
            from mlflow import MlflowClient

            client = MlflowClient()
            client.search_experiments(max_results=1)
            print("  ✓ Databricks profile authenticated")
            print()
            return []

        except ImportError:
            print("  ✗ Neither databricks-sdk nor databricks-cli installed")
            print()
            return ["Install databricks SDK: pip install databricks-sdk"]
        except Exception as e:
            print(f"  ✗ Authentication failed: {str(e)[:100]}")
            print()
            return ["Run: databricks auth login --profile DEFAULT"]

    # Test with environment variables
    try:
        from mlflow import MlflowClient

        client = MlflowClient()
        client.search_experiments(max_results=1)

        print("  ✓ Databricks token valid")
        print()
        return []

    except Exception as e:
        print(f"  ✗ Token validation failed: {str(e)[:100]}")
        print()
        return [
            "Check DATABRICKS_TOKEN is set correctly",
            "Run: databricks auth login --host <workspace-url>",
        ]


def check_mlflow_tracking():
    """Test MLflow tracking server connectivity."""
    print("Testing MLflow tracking server...")

    # Use utility to validate env vars
    errors = validate_env_vars()

    if errors:
        for error in errors:
            print(f"  ✗ {error}")
        print()
        return [f"Set environment variable: {error}" for error in errors]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")

    try:
        from mlflow import MlflowClient

        client = MlflowClient()

        # Test connectivity by getting experiment
        experiment = client.get_experiment(experiment_id)

        print(f"  ✓ Connected to: {tracking_uri}")
        print(f"  ✓ Experiment: {experiment.name}")
        print()
        return []

    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ Connection failed: {error_msg[:100]}")
        print()

        if "404" in error_msg or "not found" in error_msg.lower():
            return [f"Experiment {experiment_id} not found - check MLFLOW_EXPERIMENT_ID"]
        elif "401" in error_msg or "403" in error_msg or "authentication" in error_msg.lower():
            return ["Authentication failed - check credentials"]
        else:
            return [f"Cannot connect to {tracking_uri} - check tracking URI and network"]


def check_llm_provider():
    """Check LLM provider configuration (optional)."""
    print("Checking LLM provider configuration...")

    # Check for common LLM provider env vars
    providers_found = []

    if os.getenv("OPENAI_API_KEY"):
        providers_found.append("OpenAI")

    if os.getenv("ANTHROPIC_API_KEY"):
        providers_found.append("Anthropic")

    if os.getenv("DATABRICKS_TOKEN") or os.getenv("DATABRICKS_HOST"):
        providers_found.append("Databricks")

    if providers_found:
        print(f"  ✓ Found credentials for: {', '.join(providers_found)}")
        print()
    else:
        print("  ⚠ No LLM provider credentials detected")
        print("    This is OK if your agent uses Databricks profile auth")
        print()

    return []  # Warning only, not blocking


def main():
    """Main validation workflow."""
    print("=" * 60)
    print("Authentication Validation")
    print("=" * 60)
    print()

    all_issues = []

    # Check 1: MLflow tracking
    tracking_issues = check_mlflow_tracking()
    all_issues.extend(tracking_issues)

    # Check 2: Databricks auth (if using Databricks)
    databricks_issues = check_databricks_auth()
    all_issues.extend(databricks_issues)

    # Check 3: LLM provider (optional check)
    llm_issues = check_llm_provider()
    all_issues.extend(llm_issues)

    # Summary
    print("=" * 60)
    print("Validation Report")
    print("=" * 60)
    print()

    if not all_issues:
        print("✓ ALL AUTHENTICATION CHECKS PASSED")
        print()
        print("Your authentication is configured correctly.")
        print()
        print("Next steps:")
        print("  1. Integrate tracing: See references/tracing-integration.md")
        print("  2. Test runtime tracing: Edit and run scripts/validate_agent_tracing.py")
        print()
    else:
        print(f"✗ Found {len(all_issues)} issue(s):")
        print()
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        print()
        print("=" * 60)
        print("Fix the authentication issues above before proceeding.")
        print("=" * 60)
        print()
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
