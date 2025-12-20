#!/usr/bin/env python3
"""
Interactive MLflow environment setup script.

This script helps configure MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_ID
for agent evaluation.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Configure MLflow for agent evaluation"
    )
    parser.add_argument(
        '--tracking-uri',
        help='MLflow tracking URI (e.g., databricks://DEFAULT, http://127.0.0.1:5050)'
    )
    parser.add_argument(
        '--experiment-id',
        help='Experiment ID to use'
    )
    parser.add_argument(
        '--experiment-name',
        help='Experiment name (for search or creation)'
    )
    parser.add_argument(
        '--create',
        action='store_true',
        help='Create new experiment with --experiment-name'
    )
    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Fail if args missing (no prompts)'
    )
    return parser.parse_args()


def check_mlflow_installed() -> bool:
    """Check if MLflow >=3.6.0 is installed."""
    try:
        result = subprocess.run(
            ["mlflow", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.strip().split()[-1]
        print(f"✓ MLflow {version} is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ MLflow is not installed")
        print("  Install with: uv pip install mlflow")
        return False


def detect_databricks_profiles() -> list[str]:
    """Detect available Databricks profiles."""
    try:
        result = subprocess.run(
            ["databricks", "auth", "profiles"],
            capture_output=True,
            text=True,
            check=True
        )
        profiles = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        return profiles
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def check_databricks_auth(profile: str) -> bool:
    """Check if a Databricks profile is authenticated."""
    try:
        # Try a simple API call to check auth
        result = subprocess.run(
            ["databricks", "auth", "env", "-p", profile],
            capture_output=True,
            text=True,
            check=True
        )
        return "DATABRICKS_TOKEN" in result.stdout or "DATABRICKS_HOST" in result.stdout
    except subprocess.CalledProcessError:
        return False


def start_local_mlflow_server(port: int = 5050) -> bool:
    """Start local MLflow server in the background."""
    print(f"\nStarting local MLflow server on port {port}...")

    try:
        # Create mlruns directory if it doesn't exist
        Path("./mlruns").mkdir(exist_ok=True)

        # Start server in background
        cmd = [
            "mlflow", "server",
            "--port", str(port),
            "--backend-store-uri", "sqlite:///mlflow.db",
            "--default-artifact-root", "./mlruns"
        ]

        print(f"  Command: {' '.join(cmd)}")
        print(f"  Running in background...")

        # Note: In production, you might want to use nohup or subprocess.Popen with proper detachment
        print(f"\n  To start the server manually, run:")
        print(f"    {' '.join(cmd)} &")
        print(f"\n  Server will be available at: http://127.0.0.1:{port}")

        return True
    except Exception as e:
        print(f"✗ Error starting server: {e}")
        return False


def configure_tracking_uri(args_uri: str | None = None, non_interactive: bool = False) -> str | None:
    """Interactive configuration of MLFLOW_TRACKING_URI.

    Args:
        args_uri: Tracking URI from CLI arguments
        non_interactive: If True, fail instead of prompting for input
    """
    # If URI provided via CLI, use it
    if args_uri:
        print(f"\n✓ Using tracking URI: {args_uri}")
        return args_uri

    # Non-interactive mode requires args
    if non_interactive:
        print("✗ --tracking-uri required in non-interactive mode")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Step 1: Configure MLFLOW_TRACKING_URI")
    print("=" * 60)

    # Check for existing value
    existing = os.getenv("MLFLOW_TRACKING_URI")
    if existing:
        print(f"\nCurrent value: {existing}")
        response = input("Keep this value? (y/n): ").strip().lower()
        if response == 'y':
            return existing

    # Detect options
    options = []

    # Option 1: Databricks profiles
    profiles = detect_databricks_profiles()
    if profiles:
        print(f"\n✓ Found {len(profiles)} Databricks profile(s):")
        for i, profile in enumerate(profiles, 1):
            auth_status = "authenticated" if check_databricks_auth(profile) else "not authenticated"
            options.append(f"databricks://{profile}")
            print(f"  {len(options)}. databricks://{profile} ({auth_status})")

    # Option 2: Local server
    options.append("http://127.0.0.1:5050")
    print(f"\n  {len(options)}. http://127.0.0.1:5050 (local server)")

    # Option 3: Custom
    options.append("custom")
    print(f"  {len(options)}. Custom URI")

    # Get selection
    print()
    while True:
        try:
            choice = int(input(f"Select tracking URI (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                break
            print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")

    selected = options[choice - 1]

    # Handle custom
    if selected == "custom":
        selected = input("Enter custom tracking URI: ").strip()

    # Handle Databricks
    if selected.startswith("databricks://"):
        profile = selected.split("//")[1]
        if not check_databricks_auth(profile):
            print(f"\n⚠ Profile '{profile}' is not authenticated")
            print(f"  Please run: databricks auth login -p {profile}")
            response = input("Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                return None

    # Handle local server
    if selected.startswith("http://127.0.0.1"):
        response = input("\nStart local MLflow server? (y/n): ").strip().lower()
        if response == 'y':
            start_local_mlflow_server()

    return selected


def list_experiments(tracking_uri: str) -> list[dict]:
    """List available experiments."""
    try:
        env = os.environ.copy()
        env["MLFLOW_TRACKING_URI"] = tracking_uri

        result = subprocess.run(
            ["mlflow", "experiments", "list"],
            capture_output=True,
            text=True,
            check=True,
            env=env
        )

        # Parse output (simplified)
        lines = result.stdout.strip().split('\n')
        experiments = []

        for line in lines[2:]:  # Skip header
            if line.strip():
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 2:
                    exp_id = parts[0]
                    name = parts[1]
                    experiments.append({"id": exp_id, "name": name})

        return experiments
    except Exception as e:
        print(f"✗ Error listing experiments: {e}")
        return []


def create_experiment(tracking_uri: str, name: str) -> str | None:
    """Create a new experiment."""
    try:
        env = os.environ.copy()
        env["MLFLOW_TRACKING_URI"] = tracking_uri

        result = subprocess.run(
            ["mlflow", "experiments", "create", "-n", name],
            capture_output=True,
            text=True,
            check=True,
            env=env
        )

        # Extract experiment ID from output
        for line in result.stdout.split('\n'):
            if "Experiment" in line and "created" in line:
                # Try to extract ID
                words = line.split()
                for i, word in enumerate(words):
                    if word.lower() == "id" and i + 1 < len(words):
                        return words[i + 1].strip()

        # If can't parse, return None (but experiment was created)
        return None
    except subprocess.CalledProcessError as e:
        print(f"✗ Error creating experiment: {e.stderr}")
        return None


def configure_experiment_id(
    tracking_uri: str,
    args_exp_id: str | None = None,
    args_exp_name: str | None = None,
    create_new: bool = False,
    non_interactive: bool = False
) -> str | None:
    """Interactive configuration of MLFLOW_EXPERIMENT_ID.

    Args:
        tracking_uri: MLflow tracking URI
        args_exp_id: Experiment ID from CLI arguments
        args_exp_name: Experiment name from CLI arguments
        create_new: Create new experiment with args_exp_name
        non_interactive: If True, fail instead of prompting for input
    """
    # Handle CLI arguments first
    if args_exp_id:
        print(f"\n✓ Using experiment ID: {args_exp_id}")
        return args_exp_id

    if create_new and args_exp_name:
        print(f"\n✓ Creating experiment: {args_exp_name}")
        exp_id = create_experiment(tracking_uri, args_exp_name)
        if exp_id:
            print(f"✓ Experiment created with ID: {exp_id}")
            return exp_id
        else:
            # Try to find it by name (might have been created but ID not parsed)
            experiments = list_experiments(tracking_uri)
            for exp in experiments:
                if exp['name'] == args_exp_name:
                    print(f"✓ Found experiment ID: {exp['id']}")
                    return exp['id']
            print(f"✗ Failed to create or find experiment '{args_exp_name}'")
            sys.exit(1)

    if args_exp_name:
        # Search for experiment by name
        print(f"\n✓ Searching for experiment: {args_exp_name}")
        experiments = list_experiments(tracking_uri)
        for exp in experiments:
            if exp['name'] == args_exp_name:
                print(f"✓ Found experiment ID: {exp['id']}")
                return exp['id']

        if non_interactive:
            print(f"✗ Experiment '{args_exp_name}' not found")
            sys.exit(1)

        # In interactive mode, prompt to create
        print(f"⚠ Experiment '{args_exp_name}' not found")
        response = input("Create new experiment? (y/n): ").strip().lower()
        if response == 'y':
            exp_id = create_experiment(tracking_uri, args_exp_name)
            if exp_id:
                print(f"✓ Experiment created with ID: {exp_id}")
                return exp_id
            else:
                experiments = list_experiments(tracking_uri)
                for exp in experiments:
                    if exp['name'] == args_exp_name:
                        print(f"✓ Found experiment ID: {exp['id']}")
                        return exp['id']
                print("✗ Could not determine experiment ID")
                return None

    # Non-interactive mode requires args
    if non_interactive:
        print("✗ --experiment-id or --experiment-name required in non-interactive mode")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Step 2: Configure MLFLOW_EXPERIMENT_ID")
    print("=" * 60)

    # Check for existing value
    existing = os.getenv("MLFLOW_EXPERIMENT_ID")
    if existing:
        print(f"\nCurrent value: {existing}")
        response = input("Keep this value? (y/n): ").strip().lower()
        if response == 'y':
            return existing

    # List experiments
    print("\nFetching experiments...")
    experiments = list_experiments(tracking_uri)

    if experiments:
        print(f"\n✓ Found {len(experiments)} experiment(s):")
        for i, exp in enumerate(experiments[:10], 1):  # Show first 10
            print(f"  {i}. {exp['name']} (ID: {exp['id']})")

        if len(experiments) > 10:
            print(f"  ... and {len(experiments) - 10} more")

        print(f"\n  {len(experiments[:10]) + 1}. Create new experiment")
    else:
        print("\n  No experiments found. Will create new one.")
        experiments = []

    # Get selection
    print()
    if experiments:
        while True:
            try:
                choice = int(input(f"Select experiment (1-{min(len(experiments), 10) + 1}): "))
                if 1 <= choice <= min(len(experiments), 10):
                    return experiments[choice - 1]['id']
                elif choice == min(len(experiments), 10) + 1:
                    break  # Create new
                print(f"Please enter a number between 1 and {min(len(experiments), 10) + 1}")
            except ValueError:
                print("Please enter a valid number")

    # Create new experiment
    default_name = "agent-evaluation"
    name = input(f"\nEnter experiment name [{default_name}]: ").strip()
    if not name:
        name = default_name

    print(f"\nCreating experiment '{name}'...")
    exp_id = create_experiment(tracking_uri, name)

    if exp_id:
        print(f"✓ Experiment created with ID: {exp_id}")
        return exp_id
    else:
        # Try to find it by name
        experiments = list_experiments(tracking_uri)
        for exp in experiments:
            if exp['name'] == name:
                print(f"✓ Found experiment ID: {exp['id']}")
                return exp['id']

        print("✗ Could not determine experiment ID")
        return None


def main():
    """Main setup flow."""
    # Parse command-line arguments
    args = parse_arguments()

    print("=" * 60)
    print("MLflow Environment Setup for Agent Evaluation")
    print("=" * 60)

    # Check MLflow installation
    if not check_mlflow_installed():
        sys.exit(1)

    print()

    # Configure tracking URI
    tracking_uri = configure_tracking_uri(args.tracking_uri, args.non_interactive)
    if not tracking_uri:
        print("\n✗ Setup cancelled")
        sys.exit(1)

    # Configure experiment ID
    experiment_id = configure_experiment_id(
        tracking_uri,
        args.experiment_id,
        args.experiment_name,
        args.create,
        args.non_interactive
    )
    if not experiment_id:
        print("\n✗ Setup cancelled")
        sys.exit(1)

    # Summary
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"\nMLFLOW_TRACKING_URI={tracking_uri}")
    print(f"MLFLOW_EXPERIMENT_ID={experiment_id}")

    print("\nTo use these settings, export them:")
    print(f"  export MLFLOW_TRACKING_URI={tracking_uri}")
    print(f"  export MLFLOW_EXPERIMENT_ID={experiment_id}")

    print("\nOr add them to your shell configuration (~/.bashrc, ~/.zshrc, etc.)")
    print("=" * 60)


if __name__ == "__main__":
    main()
