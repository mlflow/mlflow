#!/usr/bin/env python
"""Merge test duration files from multiple sources."""

import glob
import json
import sys


def merge_duration_files():
    """Merge all duration JSON files in subdirectories."""
    merged = {}
    total_tests = 0

    # Find all JSON files in artifact directories
    for pattern in ["*/*.json", "*.json"]:
        for f in glob.glob(pattern):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    if isinstance(data, dict):
                        merged.update(data)
                        total_tests += len(data)
                        print(f"Loaded {len(data)} durations from {f}")
            except Exception as e:
                print(f"Error loading {f}: {e}", file=sys.stderr)

    # Write merged file
    with open("all_durations.json", "w") as fp:
        json.dump(merged, fp, indent=2, sort_keys=True)

    print(f"Total: {len(merged)} unique test durations from {total_tests} total tests")
    return 0 if merged else 1


if __name__ == "__main__":
    sys.exit(merge_duration_files())
