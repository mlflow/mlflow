#!/usr/bin/env python3
"""
Validate MLflow environment setup for agent evaluation.

This script runs `mlflow doctor` and adds custom checks for:
- Environment variables (MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_ID)
- MLflow version compatibility (>=3.6.0)
- Agent package installation
- Basic connectivity test

Usage:
    python scripts/validate_environment.py
"""

import os
import sys
import subprocess
import importlib.util
from packaging import version


def run_mlflow_doctor():
    """Run mlflow doctor and return output."""
    print("Running MLflow diagnostics...")
    print()

    try:
        result = subprocess.run(
            ["mlflow", "doctor"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Print output (mlflow doctor goes to stderr)
        output = result.stderr + result.stdout
        print(output)

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⚠ mlflow doctor timed out")
        return False
    except FileNotFoundError:
        print("✗ mlflow command not found")
        print("  Install: pip install mlflow")
        return False


def check_environment_variables():
    """Check that required environment variables are set."""
    print("Checking environment variables...")

    issues = []

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")

    if not tracking_uri:
        print("  ✗ MLFLOW_TRACKING_URI not set")
        issues.append("Set MLFLOW_TRACKING_URI: export MLFLOW_TRACKING_URI=<uri>")
    else:
        print(f"  ✓ MLFLOW_TRACKING_URI: {tracking_uri}")

    if not experiment_id:
        print("  ✗ MLFLOW_EXPERIMENT_ID not set")
        issues.append("Set MLFLOW_EXPERIMENT_ID: export MLFLOW_EXPERIMENT_ID=<id>")
    else:
        print(f"  ✓ MLFLOW_EXPERIMENT_ID: {experiment_id}")

    print()
    return issues


def check_mlflow_version():
    """Check MLflow version is compatible."""
    print("Checking MLflow version...")

    try:
        import mlflow
        current_version = mlflow.__version__

        # Remove dev/rc suffixes for comparison
        clean_version = current_version.split("dev")[0].split("rc")[0]

        if version.parse(clean_version) >= version.parse("3.8.0"):
            print(f"  ✓ MLflow {current_version} (>=3.8.0)")
            print()
            return []
        else:
            print(f"  ✗ MLflow {current_version} (need >=3.8.0)")
            print()
            return ["Upgrade MLflow: pip install --upgrade 'mlflow>=3.8.0'"]
    except ImportError:
        print("  ✗ MLflow not installed")
        print()
        return ["Install MLflow: pip install mlflow"]


def check_agent_package():
    """Check if agent package is installed and importable."""
    print("Checking agent package...")

    # Try to find the agent package by checking for common agent entry points
    agent_patterns = [
        "mlflow_agent",
        "agent",
        "src.agent",
        "app.agent"
    ]

    for pattern in agent_patterns:
        spec = importlib.util.find_spec(pattern)
        if spec is not None:
            print(f"  ✓ Agent package found: {pattern}")
            print()
            return []

    print("  ⚠ Agent package not found in common locations")
    print("    This may be OK if your agent is in a custom location")
    print()
    return []  # Warning only, not blocking


def test_connectivity():
    """Test basic connectivity to MLflow tracking server."""
    print("Testing MLflow connectivity...")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")

    if not tracking_uri or not experiment_id:
        print("  ⊘ Skipped (environment variables not set)")
        print()
        return []

    try:
        from mlflow import MlflowClient

        client = MlflowClient()
        experiment = client.get_experiment(experiment_id)

        print(f"  ✓ Connected to experiment: {experiment.name}")
        print()
        return []
    except Exception as e:
        print(f"  ✗ Connection failed: {str(e)[:100]}")
        print()
        return [f"Check connectivity and authentication to {tracking_uri}"]


def main():
    """Main validation workflow."""
    print("=" * 60)
    print("MLflow Environment Validation")
    print("=" * 60)
    print()

    all_issues = []

    # Check 1: Run mlflow doctor
    doctor_ok = run_mlflow_doctor()
    if not doctor_ok:
        all_issues.append("mlflow doctor reported issues")

    # Check 2: Environment variables
    env_issues = check_environment_variables()
    all_issues.extend(env_issues)

    # Check 3: MLflow version
    version_issues = check_mlflow_version()
    all_issues.extend(version_issues)

    # Check 4: Agent package
    agent_issues = check_agent_package()
    all_issues.extend(agent_issues)

    # Check 5: Connectivity (only if env vars set)
    connectivity_issues = test_connectivity()
    all_issues.extend(connectivity_issues)

    # Summary
    print("=" * 60)
    print("Validation Report")
    print("=" * 60)
    print()

    if not all_issues:
        print("✓ ALL CHECKS PASSED")
        print()
        print("Your environment is ready for agent evaluation.")
        print()
        print("Next steps:")
        print("  1. Integrate tracing: See references/tracing-integration.md")
        print("  2. Validate tracing: python scripts/validate_tracing_static.py")
        print("  3. Prepare dataset: python scripts/list_datasets.py")
        print()
    else:
        print(f"✗ Found {len(all_issues)} issue(s):")
        print()
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        print()
        print("=" * 60)
        print("Fix the issues above and re-run this script.")
        print("=" * 60)
        print()
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
