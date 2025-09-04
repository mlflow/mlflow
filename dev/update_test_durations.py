#!/usr/bin/env python
"""
Script to update test duration files used by pytest-split for CI parallelization.

Usage:
    python dev/update_test_durations.py [--quick] [--platform linux|windows|both]
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.absolute()


def check_uv_installed() -> bool:
    """Check if uv is installed."""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def run_pytest(
    test_paths: list[str], 
    duration_path: str, 
    verbose: bool = True
) -> bool:
    """
    Run pytest with duration collection.
    
    Returns True if duration file was created successfully.
    """
    cmd = [
        "uv", "run", "pytest",
        "--store-durations",
        f"--durations-path={duration_path}",
    ]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend([
        "tests/",
        "--ignore=tests/examples",
        "--ignore=tests/projects", 
        "--ignore-flavors",
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run pytest, don't fail on test failures (we just want durations)
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    # Check if duration file was created
    return Path(duration_path).exists()


def merge_duration_files(existing_path: Path, new_path: Path) -> Dict:
    """Merge two duration files, preferring newer durations."""
    existing_durations = {}
    if existing_path.exists():
        with open(existing_path, 'r') as f:
            existing_durations = json.load(f)
    
    new_durations = {}
    if new_path.exists():
        with open(new_path, 'r') as f:
            new_durations = json.load(f)
    
    # Merge, preferring new durations
    merged = existing_durations.copy()
    merged.update(new_durations)
    
    # Sort by test name for consistency
    return dict(sorted(merged.items()))


def update_durations() -> bool:
    """Update test durations."""
    project_root = get_project_root()
    duration_file = project_root / "tests" / ".test_durations"
    temp_file = Path(f"/tmp/test_durations_{os.getpid()}.json")
    
    print(f"\n{'='*60}")
    print(f"Collecting test durations")
    print(f"Output file: {duration_file}")
    print(f"{'='*60}\n")
    
    try:
        # Run pytest to collect durations
        success = run_pytest(
            test_paths=["tests/"],
            duration_path=str(temp_file)
        )
        
        if not success:
            print(f"Warning: No duration file was created")
            return False
        
        # Load the new durations
        with open(temp_file, 'r') as f:
            new_durations = json.load(f)
        
        print(f"Collected {len(new_durations)} test durations")
        
        # Merge with existing if present
        if duration_file.exists():
            print(f"Merging with existing durations from {duration_file}")
            merged = merge_duration_files(duration_file, temp_file)
            print(f"Total tests with durations after merge: {len(merged)}")
        else:
            merged = new_durations
            print(f"Creating new duration file with {len(merged)} tests")
        
        # Write the updated durations
        with open(duration_file, 'w') as f:
            json.dump(merged, f, indent=2, sort_keys=True)
        
        print(f"✅ Successfully updated {duration_file}")
        return True
        
    finally:
        # Clean up temp file
        if temp_file.exists():
            temp_file.unlink()




def main():
    parser = argparse.ArgumentParser(
        description="Update test duration files for pytest-split CI parallelization"
    )
    
    args = parser.parse_args()
    
    print("MLflow Test Duration Update Script")
    print("===================================\n")
    
    # Check dependencies
    if not check_uv_installed():
        print("Error: uv is required but not installed")
        print("Install with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return 1
    
    # Update durations
    success = update_durations()
    
    # Print next steps
    print("\n" + "="*60)
    print("Duration update complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the changes:")
    print("   git diff tests/.test_durations*")
    print("\n2. Commit the updated durations:")
    print("   git add tests/.test_durations")
    print("   git commit -m 'Update test durations for pytest-split'")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())