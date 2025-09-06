#!/usr/bin/env python
"""
Combines test duration files from multiple pytest-split test groups.

This script is used in CI to merge duration data from parallel test jobs
into a single consolidated duration file for better test splitting.
"""

import argparse
import glob
import json
import sys
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
    combined_linux = {}
    combined_windows = {}

    # Find all duration files
    duration_files = glob.glob(f"{artifact_dir}/*/.test_durations")

    if not duration_files:
        print("No duration files found", file=sys.stderr)
        return 0

    print(f"Found {len(duration_files)} duration files to combine")

    for filepath in duration_files:
        # Determine platform from artifact directory name
        artifact_name = Path(filepath).parent.name
        is_windows = "windows" in artifact_name.lower()

        try:
            with open(filepath) as f:
                data = json.load(f)
                # Merge into platform-specific dict, keeping maximum duration
                target = combined_windows if is_windows else combined_linux
                for test, duration in data.items():
                    if test not in target or duration > target[test]:
                        target[test] = duration
                platform = "Windows" if is_windows else "Linux"
                print(f"  - Processed {filepath} ({platform}): {len(data)} tests")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)

    # Write combined durations for Linux (used by python and flavors jobs)
    output_path_linux = Path(output_dir) / ".test_durations_linux"
    with open(output_path_linux, "w") as f:
        json.dump(combined_linux, f, indent=2, sort_keys=True)
    print(f"Wrote combined Linux durations to {output_path_linux}: {len(combined_linux)} tests")

    # Write combined durations for Windows
    output_path_windows = Path(output_dir) / ".test_durations_windows"
    with open(output_path_windows, "w") as f:
        json.dump(combined_windows, f, indent=2, sort_keys=True)
    print(
        f"Wrote combined Windows durations to {output_path_windows}: {len(combined_windows)} tests"
    )

    total_combined = len(set(combined_linux.keys()) | set(combined_windows.keys()))
    print(f"\nSuccessfully combined {total_combined} unique tests across platforms")
    return total_combined


def main():
    parser = argparse.ArgumentParser(description="Combine pytest-split duration files")
    parser.add_argument(
        "--artifact-dir",
        default="duration-artifacts",
        help="Directory containing duration artifacts (default: duration-artifacts)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write combined duration files (default: current directory)",
    )

    args = parser.parse_args()

    combined_count = combine_duration_files(args.artifact_dir, args.output_dir)

    if combined_count == 0:
        print("Warning: No test durations were combined", file=sys.stderr)
        sys.exit(0)  # Don't fail CI if no durations found (e.g., first run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
