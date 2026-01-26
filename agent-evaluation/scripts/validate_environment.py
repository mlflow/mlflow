"""
Validate MLflow environment setup for agent evaluation.

This script runs `mlflow doctor` and adds custom checks for:
- Environment variables (MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_ID)
- MLflow version compatibility (>=3.8.0)
- Agent package installation
- Basic connectivity test

Usage:
    python scripts/validate_environment.py
"""

import importlib.util
import subprocess
import sys

from utils import test_mlflow_connection, validate_env_vars, validate_mlflow_version


def run_mlflow_doctor():
    """Run mlflow doctor and return output."""
    print("Running MLflow diagnostics...")
    print()

    try:
        result = subprocess.run(["mlflow", "doctor"], capture_output=True, text=True, timeout=10)

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

    errors = validate_env_vars()

    if not errors:
        env_vars = {}
        import os

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")

        if tracking_uri:
            print(f"  ✓ MLFLOW_TRACKING_URI: {tracking_uri}")
        if experiment_id:
            print(f"  ✓ MLFLOW_EXPERIMENT_ID: {experiment_id}")
    else:
        for error in errors:
            print(f"  ✗ {error}")

    print()
    return ["Set environment variables" for _ in errors] if errors else []


def check_mlflow_version():
    """Check MLflow version is compatible."""
    print("Checking MLflow version...")

    is_valid, version_str = validate_mlflow_version("3.8.0")

    if is_valid:
        print(f"  ✓ MLflow {version_str} (>=3.8.0)")
        print()
        return []
    elif version_str == "not installed":
        print(f"  ✗ MLflow not installed")
        print()
        return ["Install MLflow: pip install mlflow"]
    else:
        print(f"  ✗ MLflow {version_str} (need >=3.8.0)")
        print()
        return ["Upgrade MLflow: pip install --upgrade 'mlflow>=3.8.0'"]


def check_agent_package():
    """Remind user to verify agent package is importable."""
    print("Agent package check...")
    print("  ℹ Verify your agent is importable:")
    print("    python -c 'from your_module import your_agent'")
    print("  Replace 'your_module' and 'your_agent' with your actual package/function names")
    print()
    return []  # Informational only, not blocking


def test_connectivity():
    """Test basic connectivity to MLflow tracking server."""
    print("Testing MLflow connectivity...")

    import os

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")

    if not tracking_uri or not experiment_id:
        print("  ⊘ Skipped (environment variables not set)")
        print()
        return []

    success, result = test_mlflow_connection(tracking_uri, experiment_id)

    if success:
        print(f"  ✓ Connected to experiment: {result}")
        print()
        return []
    else:
        print(f"  ✗ Connection failed: {result}")
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
        print("  2. Prepare dataset: python scripts/list_datasets.py")
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
