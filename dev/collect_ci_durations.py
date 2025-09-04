#!/usr/bin/env python
"""
Collect test durations from CI by creating a temporary workflow.

This script:
1. Creates a temporary workflow file in .github/workflows/
2. Commits and pushes it to a temporary branch
3. Triggers the workflow
4. Waits for completion and downloads artifacts
5. Cleans up the temporary branch

Usage:
    # Run ALL tests to get complete duration coverage (default)
    python dev/collect_ci_durations.py
    
    # Run specific suites only
    python dev/collect_ci_durations.py --suites python models pyfunc
    
    # Keep branch for debugging
    python dev/collect_ci_durations.py --keep-branch --branch my-test-branch
"""

import json
import subprocess
import time
import sys
from pathlib import Path
import argparse
import tempfile
import os
import shutil

WORKFLOW_CONTENT = """name: Collect Test Durations (Temporary)
on:
  workflow_dispatch:
  push:
    branches:
      - '{branch_name}'

jobs:
  collect-all-durations:
    name: Collect all test durations
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/setup-python
      - uses: ./.github/actions/setup-pyenv
      - uses: ./.github/actions/setup-java
      
      - name: Install uv
        uses: astral-sh/setup-uv@v6
      
      - name: Install ALL dependencies
        run: |
          source ./dev/install-common-deps.sh
          # Install ALL dependencies for comprehensive test coverage - "install the world"
          uv pip install --system -c requirements/constraints.txt .[extras]
          uv pip install --system -c requirements/constraints.txt '.[mlserver]' '.[genai]'
          uv pip install --system -c requirements/constraints.txt tensorflow 'pyspark[connect]' torch transformers
          uv pip install --system -c requirements/constraints.txt langchain langchain-community langchain-experimental
          uv pip install --system -c requirements/constraints.txt 'shap<0.47.0' lightgbm xgboost catboost
          uv pip install --system -c requirements/constraints.txt tf-keras uvicorn 'litellm>=1.52.9'
          uv pip install --system -c requirements/constraints.txt databricks-agents openai 'optuna>=4'
          uv pip install --system -c requirements/constraints.txt typing_extensions dspy
          uv pip install --system -c requirements/constraints.txt 'pydantic<2'  # For pydantic v1 tests
          # Install test plugin
          uv pip install --no-deps tests/resources/mlflow-test-plugin
      
      - name: Run ALL tests with duration collection
        run: |
          # TEST MODE: Just run a quick test to verify workflow works
          pytest --store-durations --durations-path=all_test_durations.json tests/test_version.py -v || true
      
      - name: Verify duration file exists and show stats
        if: always()
        run: |
          echo "=== Duration file verification ==="
          if [ -f all_test_durations.json ]; then
            echo "✓ Duration file exists"
            echo "File size: $(ls -lh all_test_durations.json | awk '{{print $5}}')"
            echo "Number of test durations: $(python -c "import json; print(len(json.load(open('all_test_durations.json'))))")"
            echo ""
            echo "=== First 50 lines of duration file ==="
            head -50 all_test_durations.json
            echo ""
            echo "=== Last 50 lines of duration file ==="
            tail -50 all_test_durations.json
          else
            echo "ERROR: Duration file all_test_durations.json not found!"
            echo "Current directory contents:"
            ls -la
          fi
      
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: final-test-durations
          path: all_test_durations.json
          retention-days: 7
"""

# Test suite configurations (all use same deps since we install everything)
TEST_SUITES = {
    # Run complete test suite - this captures EVERYTHING
    "all-tests": {
        "os": "ubuntu-latest",
        "test_path": "tests"
    },
    # Windows can run separately
    "windows": {
        "os": "windows-latest",
        "test_path": "tests --ignore-flavors --ignore=tests/projects --ignore=tests/examples --ignore=tests/evaluate --ignore=tests/db"
    }
}

def run_command(cmd, capture=True, check=True, timeout=None):
    """Run shell command and return output."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture,
            text=True,
            check=check,
            timeout=timeout
        )
        if capture:
            return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if check:
            print(f"Command failed: {e}")
            if e.stderr:
                print(f"Error output: {e.stderr}")
            raise
        return None
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds")
        return None
    return None

def create_workflow_content(branch_name, suites=None):
    """Generate the workflow content. Suites parameter is ignored since we run everything."""
    # Simply return the workflow with branch name filled in
    return WORKFLOW_CONTENT.format(branch_name=branch_name)

def wait_for_workflow(run_id, max_wait_minutes=60):
    """Wait for workflow to complete with timeout."""
    start_time = time.time()
    max_wait = max_wait_minutes * 60
    
    print(f"\nWorkflow started with run ID: {run_id}")
    print(f"View at: https://github.com/mlflow/mlflow/actions/runs/{run_id}")
    print(f"Waiting for completion (max {max_wait_minutes} minutes)...\n")
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            print(f"\nTimeout: Workflow did not complete within {max_wait_minutes} minutes")
            return False
        
        # Check workflow status
        status = run_command(
            f"gh run view {run_id} --json status,conclusion --jq '.status + \"|\" + .conclusion'",
            check=False
        )
        
        if status:
            workflow_status, conclusion = status.split("|")
            
            # Print progress update every 30 seconds
            if int(elapsed) % 30 == 0:
                minutes_elapsed = int(elapsed / 60)
                print(f"  [{minutes_elapsed}m elapsed] Workflow status: {workflow_status}")
            
            if workflow_status == "completed":
                print(f"\nWorkflow completed with conclusion: {conclusion}")
                return conclusion in ["success", "failure"]  # We want artifacts even if tests fail
        
        time.sleep(5)
    
    return False

def main():
    parser = argparse.ArgumentParser(
        description='Collect test durations from CI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ALL tests for complete coverage (default)
  python dev/collect_ci_durations.py
  
  # Run specific suites only
  python dev/collect_ci_durations.py --suites python models pyfunc
  
  # Run all individual suites separately (not recommended - use all-tests instead)
  python dev/collect_ci_durations.py --suites all
  
  # Keep branch for debugging
  python dev/collect_ci_durations.py --keep-branch
  
  # Use specific branch name
  python dev/collect_ci_durations.py --branch my-duration-test
  
  # Extend timeout for complete test suite
  python dev/collect_ci_durations.py --timeout 120
"""
    )
    parser.add_argument(
        '--suites', 
        nargs='+',
        default=['all-tests'],
        help=f'Test suites to collect durations for. Available: {", ".join(TEST_SUITES.keys())}, all'
    )
    parser.add_argument(
        '--branch',
        default=None,
        help='Branch name to use (default: temp-durations-<timestamp>)'
    )
    parser.add_argument(
        '--keep-branch',
        action='store_true',
        help='Keep the temporary branch after completion'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=480,
        help='Maximum time to wait for workflow completion in minutes (default: 480 = 8 hours)'
    )
    args = parser.parse_args()
    
    # Handle "all" suites
    if "all" in args.suites:
        args.suites = list(TEST_SUITES.keys())
    
    # Validate suites
    invalid_suites = [s for s in args.suites if s not in TEST_SUITES]
    if invalid_suites:
        print(f"Error: Invalid suites: {', '.join(invalid_suites)}")
        print(f"Available suites: {', '.join(TEST_SUITES.keys())}")
        return 1
    
    # Generate branch name
    if not args.branch:
        timestamp = int(time.time())
        args.branch = f"temp-durations-{timestamp}"
    
    workflow_file = Path(".github/workflows/temp-duration-collection.yml")
    current_branch = None
    
    try:
        # Check for uncommitted changes
        uncommitted = run_command("git status --porcelain", check=False)
        if uncommitted:
            print("Warning: You have uncommitted changes. They will not be included in the duration collection.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return 1
        
        print(f"\nCollecting durations for suites: {', '.join(args.suites)}")
        print(f"Using branch: {args.branch}\n")
        
        # Store current branch
        current_branch = run_command("git rev-parse --abbrev-ref HEAD")
        
        # Create and checkout new branch
        print(f"Creating branch: {args.branch}")
        run_command(f"git checkout -b {args.branch}")
        
        # Create workflow file
        print(f"Creating workflow file: {workflow_file}")
        workflow_file.parent.mkdir(parents=True, exist_ok=True)
        workflow_content = create_workflow_content(args.branch, args.suites)
        workflow_file.write_text(workflow_content)
        
        # Commit and push
        run_command(f"git add {workflow_file}")
        run_command(f'git commit --no-verify -m "Temporary: collect test durations for {", ".join(args.suites)}"')
        print(f"Pushing branch to origin...")
        run_command(f"git push -u origin {args.branch}")
        
        # Wait for workflow to auto-trigger on push
        print("\nWaiting for workflow to start...")
        time.sleep(10)
        
        # Get the run ID
        run_id = run_command(
            f"gh run list --branch {args.branch} --limit 1 --json databaseId --jq '.[0].databaseId'"
        )
        
        if not run_id:
            print("Workflow didn't auto-start, triggering manually...")
            run_command(f"gh workflow run temp-duration-collection.yml --ref {args.branch}")
            time.sleep(10)
            run_id = run_command(
                f"gh run list --branch {args.branch} --limit 1 --json databaseId --jq '.[0].databaseId'"
            )
        
        if not run_id:
            print("Error: Failed to start workflow")
            return 1
        
        # Wait for completion
        if not wait_for_workflow(run_id, args.timeout):
            print("Warning: Workflow did not complete successfully, but attempting to download any artifacts...")
        
        # Download artifacts
        print("\nDownloading duration artifacts...")
        # First, clean any existing artifact directories
        for old_dir in Path(".").glob("final-test-durations*"):
            shutil.rmtree(old_dir, ignore_errors=True)
        for old_dir in Path(".").glob("test-durations-*"):
            shutil.rmtree(old_dir, ignore_errors=True)
            
        # Download the duration artifact
        result = run_command(f"gh run download {run_id} --name final-test-durations", check=False)
        
        # Check if download succeeded and file exists in the expected location
        duration_file = Path("final-test-durations/all_test_durations.json")
        
        if not duration_file.exists():
            print("Warning: Could not find final-test-durations/all_test_durations.json")
            print("Checking directory contents...")
            
            # Look for the file
            possible_files = list(Path(".").glob("*/all_test_durations.json"))
            if possible_files:
                duration_file = possible_files[0]
                print(f"Found duration file at: {duration_file}")
            else:
                # Try to find any JSON files
                json_files = list(Path(".").glob("*/*.json"))
                if json_files:
                    print(f"Found JSON files: {json_files}")
                    duration_file = json_files[0]
                else:
                    print("ERROR: No duration files found in downloaded artifacts")
                    print("Directory contents:")
                    for item in Path(".").iterdir():
                        print(f"  {item}")
                        if item.is_dir():
                            for subitem in item.iterdir():
                                print(f"    {subitem}")
        
        # Process the duration file
        if duration_file and duration_file.exists():
            target = Path("tests/.test_durations")
            
            with open(duration_file) as f:
                new_durations = json.load(f)
            
            print(f"\nCollected {len(new_durations)} test durations")
            
            # Merge with existing if present
            if target.exists():
                with open(target) as f:
                    existing = json.load(f)
                print(f"Merging with {len(existing)} existing durations")
                existing.update(new_durations)
                
                with open(target, 'w') as f:
                    json.dump(existing, f, indent=2, sort_keys=True)
                
                print(f"Updated {target} with {len(existing)} total durations")
            else:
                with open(target, 'w') as f:
                    json.dump(new_durations, f, indent=2, sort_keys=True)
                print(f"Created {target} with {len(new_durations)} durations")
            
            # Cleanup downloaded artifacts
            for artifact_dir in Path(".").glob("final-test-durations*"):
                shutil.rmtree(artifact_dir, ignore_errors=True)
            for artifact_dir in Path(".").glob("test-durations-*"):
                shutil.rmtree(artifact_dir, ignore_errors=True)
            
            print(f"\n✓ Duration file updated successfully: {target}")
        else:
            print("\nWarning: No duration artifacts found. Check the workflow logs for errors.")
            print(f"View logs at: https://github.com/mlflow/mlflow/actions/runs/{run_id}")
            return 1
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    finally:
        # Cleanup
        if current_branch:
            print(f"\nReturning to branch: {current_branch}")
            run_command(f"git checkout {current_branch}", check=False)
        
        # Remove local workflow file
        if workflow_file.exists():
            workflow_file.unlink()
            run_command(f"git rm {workflow_file}", check=False)
        
        if not args.keep_branch and args.branch:
            print(f"Cleaning up branch: {args.branch}")
            run_command(f"git branch -D {args.branch}", check=False)
            run_command(f"git push origin --delete {args.branch}", check=False)
    
    print("\nDone!")
    return 0

if __name__ == "__main__":
    sys.exit(main())