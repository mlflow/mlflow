#!/usr/bin/env python
"""Analyze test split distribution across groups."""

import json
from collections import defaultdict
from pathlib import Path


def analyze_test_splits(duration_file, num_groups=10):
    """Analyze how tests would be split across groups based on durations."""

    # Load duration data
    with open(duration_file) as f:
        durations = json.load(f)

    # Sort tests by duration (longest first for better distribution)
    sorted_tests = sorted(durations.items(), key=lambda x: x[1], reverse=True)

    # Initialize groups
    groups = defaultdict(list)
    group_times = [0] * num_groups

    # Distribute tests using greedy algorithm (put each test in group with minimum time)
    for test_name, duration in sorted_tests:
        # Find group with minimum total time
        min_group = min(range(num_groups), key=lambda i: group_times[i])
        groups[min_group].append((test_name, duration))
        group_times[min_group] += duration

    # Print analysis
    print(f"\nAnalyzing {duration_file}")
    print(f"Total tests: {len(durations)}")
    print(f"Total duration: {sum(durations.values()):.1f}s")
    print(f"\nDistribution across {num_groups} groups:")
    print("-" * 60)

    for i in range(num_groups):
        group_tests = groups[i]
        group_time = group_times[i]
        mins = group_time / 60
        print(f"Group {i + 1:2d}: {len(group_tests):4d} tests, {group_time:7.1f}s ({mins:5.1f}m)")

        # Show top 3 slowest tests in this group
        if group_tests:
            print("  Top 3 slowest:")
            for test, duration in sorted(group_tests, key=lambda x: x[1], reverse=True)[:3]:
                test_short = test.split("::")[-1] if "::" in test else test[-50:]
                print(f"    - {test_short[:60]:60s} {duration:6.1f}s")

    # Calculate variance metrics
    avg_time = sum(group_times) / num_groups
    max_time = max(group_times)
    min_time = min(group_times)
    variance_ratio = max_time / min_time if min_time > 0 else float("inf")

    print("-" * 60)
    print(f"Average group time: {avg_time:.1f}s ({avg_time / 60:.1f}m)")
    print(f"Min group time: {min_time:.1f}s ({min_time / 60:.1f}m)")
    print(f"Max group time: {max_time:.1f}s ({max_time / 60:.1f}m)")
    print(f"Variance ratio (max/min): {variance_ratio:.2f}x")

    # Check what pytest-split would actually do
    print("\n" + "=" * 60)
    print("What pytest-split SHOULD produce (ideal distribution):")
    print("This is based on greedy bin packing algorithm")
    print("=" * 60)

    return groups, group_times


def check_actual_split(duration_file, num_groups=10):
    """Simulate what pytest-split actually does."""
    # pytest-split uses a slightly different algorithm
    # It tries to balance based on cumulative time

    with open(duration_file) as f:
        durations = json.load(f)

    # Sort by test name (pytest-split maintains order)
    sorted_tests = sorted(durations.items())

    # Calculate target time per group
    total_time = sum(durations.values())
    target_time = total_time / num_groups

    groups = defaultdict(list)
    current_group = 0
    current_time = 0

    for test_name, duration in sorted_tests:
        if current_time >= target_time and current_group < num_groups - 1:
            current_group += 1
            current_time = 0

        groups[current_group].append((test_name, duration))
        current_time += duration

    print("\nWhat pytest-split MIGHT produce (sequential distribution):")
    print("This assumes tests are processed in alphabetical order")
    print("-" * 60)

    group_times = []
    for i in range(num_groups):
        group_tests = groups[i]
        group_time = sum(d for _, d in group_tests)
        group_times.append(group_time)
        mins = group_time / 60
        print(f"Group {i + 1:2d}: {len(group_tests):4d} tests, {group_time:7.1f}s ({mins:5.1f}m)")

    if group_times:
        max_time = max(group_times)
        min_time = min(group_times) if min(group_times) > 0 else 0.1
        variance_ratio = max_time / min_time

        print("-" * 60)
        print(f"Variance ratio with sequential: {variance_ratio:.2f}x")


if __name__ == "__main__":
    # Analyze main python test durations
    duration_files = [
        ".github/workflows/test_durations/python.test_duration",
        ".github/workflows/test_durations/flavors.test_duration",
        ".github/workflows/test_durations/windows.test_duration",
    ]

    for file in duration_files:
        if Path(file).exists():
            analyze_test_splits(file)
            check_actual_split(file)
            print("\n" + "=" * 80 + "\n")
