#!/usr/bin/env python
"""
Combines test duration files from multiple pytest-split test groups.

This script is used in CI to merge duration data from parallel test jobs
into a single consolidated duration file for better test splitting.
"""
import json
import glob
import sys
import argparse
from pathlib import Path


def combine_duration_files(artifact_dir="duration-artifacts", output_dir="."):
    """
    Combine multiple .test_durations files into consolidated duration files.
    
    Args:
        artifact_dir: Directory containing downloaded artifacts with duration files
        output_dir: Directory to write combined duration files
    
    Returns:
        Number of test durations combined
    """
    combined = {}
    
    # Find all duration files
    duration_files = glob.glob(f"{artifact_dir}/*/.test_durations")
    
    if not duration_files:
        print("No duration files found", file=sys.stderr)
        return 0
    
    print(f"Found {len(duration_files)} duration files to combine")
    
    for filepath in duration_files:
        try:
            with open(filepath) as f:
                data = json.load(f)
                # Merge, keeping the maximum duration for each test
                # (conservative approach - use longest observed time)
                for test, duration in data.items():
                    if test not in combined or duration > combined[test]:
                        combined[test] = duration
                print(f"  - Processed {filepath}: {len(data)} tests")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
    
    # Write combined durations for Ubuntu
    output_path_ubuntu = Path(output_dir) / ".test_durations_ubuntu"
    with open(output_path_ubuntu, 'w') as f:
        json.dump(combined, f, indent=2, sort_keys=True)
    print(f"Wrote combined Ubuntu durations to {output_path_ubuntu}")
    
    # Write combined durations for Windows
    # (same data for now, but could be customized in the future)
    output_path_windows = Path(output_dir) / ".test_durations_windows"
    with open(output_path_windows, 'w') as f:
        json.dump(combined, f, indent=2, sort_keys=True)
    print(f"Wrote combined Windows durations to {output_path_windows}")
    
    print(f"\nSuccessfully combined {len(combined)} test durations from {len(duration_files)} files")
    return len(combined)


def main():
    parser = argparse.ArgumentParser(description="Combine pytest-split duration files")
    parser.add_argument(
        "--artifact-dir",
        default="duration-artifacts",
        help="Directory containing duration artifacts (default: duration-artifacts)"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write combined duration files (default: current directory)"
    )
    
    args = parser.parse_args()
    
    combined_count = combine_duration_files(args.artifact_dir, args.output_dir)
    
    if combined_count == 0:
        print("Warning: No test durations were combined", file=sys.stderr)
        sys.exit(0)  # Don't fail CI if no durations found (e.g., first run)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())