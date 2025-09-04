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

env:
  MLFLOW_HOME: ${{{{ github.workspace }}}}
  MLFLOW_CONDA_HOME: /usr/share/miniconda
  SPARK_LOCAL_IP: localhost
  PIP_EXTRA_INDEX_URL: https://download.pytorch.org/whl/cpu
  PIP_CONSTRAINT: ${{{{ github.workspace }}}}/requirements/constraints.txt
  PYTHONUTF8: "1"
  _MLFLOW_TESTING_TELEMETRY: "true"

jobs:
  collect-durations:
    name: Collect test durations
    runs-on: ubuntu-latest
    timeout-minutes: 480
    strategy:
      fail-fast: false
      matrix:
        group: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        include:
          - splits: 10
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/untracked
      - name: Install uv
        uses: astral-sh/setup-uv@v6
      - uses: ./.github/actions/free-disk-space
      - uses: ./.github/actions/setup-python
      - uses: ./.github/actions/setup-pyenv
      - uses: ./.github/actions/setup-java
      - uses: ./.github/actions/cache-pip
      - name: Install dependencies
        run: |
          python -m venv .venv
          source .venv/bin/activate
          source ./dev/install-common-deps.sh --ml
          # transformers doesn't support Keras 3 yet. tf-keras needs to be installed as a workaround.
          uv pip install -c requirements/constraints.txt tf-keras
      - uses: ./.github/actions/show-versions
      - uses: ./.github/actions/pipdeptree
      - name: Import check
        run: |
          source .venv/bin/activate
          # `-I` is used to avoid importing modules from user-specific site-packages
          # that might conflict with the built-in modules (e.g. `types`).
          python -I tests/check_mlflow_lazily_imports_ml_packages.py
      
      - name: Run tests
        run: |
          source .venv/bin/activate
          source dev/setup-ssh.sh
          pytest --splits=${{{{ matrix.splits }}}} --group=${{{{ matrix.group }}}} \
            --store-durations --durations-path=group_${{{{ matrix.group }}}}_durations.json --quiet --requires-ssh \
            --ignore-flavors --ignore=tests/examples --ignore=tests/evaluate \
            --ignore tests/genai tests || true
      
      - name: Verify duration file exists and show stats
        if: always()
        run: |
          echo "=== Duration file verification for group ${{{{ matrix.group }}}} ==="
          duration_file="group_${{{{ matrix.group }}}}_durations.json"
          if [ -f "$duration_file" ]; then
            echo "✓ Duration file exists: $duration_file"
            echo "File size: $(ls -lh "$duration_file" | awk '{{print $5}}')"
            echo "Number of test durations: $(python -c "import json; print(len(json.load(open('$duration_file'))))")"
            echo ""
            echo "=== First 20 lines of duration file ==="
            head -20 "$duration_file"
          else
            echo "ERROR: Duration file $duration_file not found!"
            echo "Current directory contents:"
            ls -la
          fi
      
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-durations-group-${{{{ matrix.group }}}}
          path: group_${{{{ matrix.group }}}}_durations.json
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

def wait_for_workflow(run_id, repo, max_wait_minutes=60):
    """Wait for workflow to complete with timeout."""
    start_time = time.time()
    max_wait = max_wait_minutes * 60
    
    print(f"\nWorkflow started with run ID: {run_id}")
    print(f"View at: https://github.com/{repo}/actions/runs/{run_id}")
    print(f"Waiting for completion (max {max_wait_minutes} minutes)...\n")
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            print(f"\nTimeout: Workflow did not complete within {max_wait_minutes} minutes")
            return False
        
        # Check workflow status - with repo for reliability
        status = run_command(
            f"gh run view {run_id} --repo {repo} --json status,conclusion --jq '.status + \"|\" + .conclusion'",
            check=False
        )
        
        if status:
            try:
                workflow_status, conclusion = status.split("|")
                
                # Print progress update every 30 seconds
                if int(elapsed) % 30 == 0:
                    minutes_elapsed = int(elapsed / 60)
                    print(f"  [{minutes_elapsed}m elapsed] Workflow status: {workflow_status}")
                
                if workflow_status == "completed":
                    print(f"\nWorkflow completed with conclusion: {conclusion}")
                    return conclusion in ["success", "failure"]  # We want artifacts even if tests fail
            except ValueError:
                # Handle case where split fails
                print(f"Warning: Unexpected status format: {status}")
        else:
            # Command failed - print warning but continue waiting
            if int(elapsed) % 30 == 0:
                print(f"  [{int(elapsed/60)}m elapsed] Unable to check workflow status, retrying...")
        
        time.sleep(5)
    
    return False

def get_github_repo():
    """Get GitHub repo from git remote."""
    remote_url = run_command("git remote get-url origin")
    if remote_url:
        # Parse github.com:user/repo.git or https://github.com/user/repo.git
        if "github.com" in remote_url:
            if remote_url.startswith("git@"):
                # SSH format: git@github.com:user/repo.git
                repo_path = remote_url.split(":")[-1]
            else:
                # HTTPS format: https://github.com/user/repo.git
                repo_path = remote_url.split("github.com/")[-1]
            # Remove .git suffix if present
            repo_path = repo_path.replace(".git", "")
            return repo_path
    return "mlflow/mlflow"  # fallback

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
    repo = get_github_repo()
    
    try:
        # Check for uncommitted changes
        uncommitted = run_command("git status --porcelain", check=False)
        if uncommitted:
            print("Warning: You have uncommitted changes. They will not be included in the duration collection.")
            print("Continuing anyway...")
        
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
        
        # Get the run ID - add repo explicitly for safety
        run_id = run_command(
            f"gh run list --repo {repo} --branch {args.branch} --limit 1 --json databaseId --jq '.[0].databaseId'"
        )
        
        if not run_id:
            print("Workflow didn't auto-start, triggering manually...")
            run_command(f"gh workflow run temp-duration-collection.yml --repo {repo} --ref {args.branch}")
            time.sleep(10)
            run_id = run_command(
                f"gh run list --repo {repo} --branch {args.branch} --limit 1 --json databaseId --jq '.[0].databaseId'"
            )
        
        if not run_id:
            print("Error: Failed to start workflow")
            return 1
        
        # Wait for completion
        if not wait_for_workflow(run_id, repo, args.timeout):
            print("Warning: Workflow did not complete successfully, but attempting to download any artifacts...")
        
        # Download all matrix artifacts and merge them
        print("\nDownloading matrix duration artifacts...")
        
        # Create a temp directory for download
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download all matrix artifacts (test-durations-group-1, test-durations-group-2, etc.)
            merged_durations = {}
            groups_found = 0
            
            # Try to download each group artifact (we used groups 1, 2)
            for group in range(1, 11):  # Check up to 10 groups
                artifact_name = f"test-durations-group-{group}"
                download_cmd = f"cd {tmpdir} && gh run download {run_id} --repo {repo} --name {artifact_name}"
                result = run_command(download_cmd, check=False)
                
                # Check if this group's duration file exists
                duration_file = Path(tmpdir) / f"group_{group}_durations.json"
                if duration_file.exists():
                    groups_found += 1
                    print(f"Found artifact: {artifact_name}")
                    
                    # Read and merge this group's durations
                    with open(duration_file) as f:
                        group_durations = json.load(f)
                    
                    if isinstance(group_durations, dict):
                        merged_durations.update(group_durations)
                        print(f"  Merged {len(group_durations)} durations from group {group}")
                    else:
                        print(f"  Warning: Group {group} data is not a dictionary")
            
            if groups_found == 0:
                print(f"ERROR: No matrix artifacts found in {tmpdir}")
                print(f"Directory contents: {list(Path(tmpdir).iterdir())}")
                print(f"View logs at: https://github.com/{repo}/actions/runs/{run_id}")
                return 1
            
            # Validate merged durations
            if not merged_durations:
                print("ERROR: No test durations were collected from any group")
                return 1
                
            # Check that keys look like test names (should contain "::" for pytest format)
            if not any("::" in key for key in merged_durations.keys()):
                print(f"ERROR: Merged data doesn't appear to contain test durations")
                print(f"Sample keys: {list(merged_durations.keys())[:5]}")
                return 1
            
            print(f"\nCollected {len(merged_durations)} test durations from {groups_found} matrix groups")
            
            # Copy to tests/.test_durations (overwrite)
            target = Path("tests/.test_durations")
            with open(target, 'w') as f:
                json.dump(merged_durations, f, indent=2, sort_keys=True)
            
            print(f"✓ Wrote {len(merged_durations)} durations to {target}")
        
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