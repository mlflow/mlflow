#!/usr/bin/env python3
"""
List and compare MLflow evaluation datasets in an experiment.

This script discovers existing datasets before prompting to create new ones,
preventing duplicate work and helping users make informed choices.

Usage:
    python scripts/list_datasets.py

Environment variables required:
    MLFLOW_TRACKING_URI
    MLFLOW_EXPERIMENT_ID
"""

import argparse
import os
import signal
import sys

import numpy as np

from mlflow import MlflowClient
from mlflow.genai.datasets import get_dataset


class TimeoutError(Exception):
    """Custom timeout exception."""


def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise TimeoutError()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="List and compare MLflow evaluation datasets")
    parser.add_argument("--dataset-name", help="Specific dataset to display")
    parser.add_argument(
        "--show-samples", type=int, default=5, help="Number of sample queries to show (default: 5)"
    )
    parser.add_argument(
        "--non-interactive", action="store_true", help="List without interactive selection"
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Timeout in seconds for dataset search (default: 30)"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Include detailed diversity analysis (slower)"
    )
    return parser.parse_args()


def check_environment():
    """Check that required environment variables are set."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")

    if not tracking_uri:
        print("✗ MLFLOW_TRACKING_URI not set")
        print("  Run: export MLFLOW_TRACKING_URI=<uri>")
        return None, None

    if not experiment_id:
        print("✗ MLFLOW_EXPERIMENT_ID not set")
        print("  Run: export MLFLOW_EXPERIMENT_ID=<id>")
        return None, None

    return tracking_uri, experiment_id


def calculate_diversity_metrics(queries):
    """Calculate diversity metrics for a list of queries."""
    if not queries:
        return 0.0, 0.0

    # Query length statistics
    lengths = [len(q) for q in queries]
    avg_length = np.mean(lengths)
    std_length = np.std(lengths)

    # Unique word count (simple diversity measure)
    all_words = set()
    for query in queries:
        words = query.lower().split()
        all_words.update(words)

    unique_word_ratio = len(all_words) / len(queries) if queries else 0

    return avg_length, std_length, unique_word_ratio


def classify_diversity(std_length, unique_word_ratio, query_count):
    """Classify diversity as HIGH, MEDIUM, or LOW."""
    # Heuristics based on variability and vocabulary
    if query_count < 5:
        return "LOW (too few queries)"

    if std_length > 15 and unique_word_ratio > 3:
        return "HIGH (varied lengths and topics)"
    elif std_length > 8 or unique_word_ratio > 2:
        return "MEDIUM"
    else:
        return "LOW (repetitive patterns)"


def main():
    """Main workflow."""
    # Parse command-line arguments
    args = parse_arguments()

    print("=" * 60)
    print("MLflow Dataset Discovery")
    print("=" * 60)
    print()

    # Check environment
    tracking_uri, experiment_id = check_environment()
    if not tracking_uri or not experiment_id:
        sys.exit(1)

    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment ID: {experiment_id}")
    print()

    # Search for datasets with timeout protection
    print(f"Searching for datasets in experiment... (timeout: {args.timeout}s)")
    client = MlflowClient()

    datasets = []
    try:
        # Set up timeout (Unix only)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(args.timeout)

            try:
                datasets = client.search_datasets(experiment_ids=[experiment_id])
                signal.alarm(0)  # Cancel timeout
            except TimeoutError:
                print(f"\n⚠ Search timed out after {args.timeout}s")
                print("  Try increasing timeout with: --timeout 60")
                print()
                sys.exit(1)
        else:
            # No timeout support on Windows
            print("  (No timeout support on Windows)")
            datasets = client.search_datasets(experiment_ids=[experiment_id])

    except Exception as e:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancel timeout
        print(f"✗ Error searching datasets: {e}")
        sys.exit(1)

    if not datasets:
        print("\nNo datasets found in this experiment.")
        print("\nYou'll need to create a new dataset.")
        print("Run: python scripts/create_dataset_template.py")
        return

    print(f"✓ Found {len(datasets)} dataset(s)")
    print()

    # Analyze each dataset
    dataset_info = []

    if not args.detailed:
        print("\n(Using fast mode - skipping diversity analysis)")
        print("(Use --detailed flag for full analysis)")
        print()

    for i, dataset in enumerate(datasets, 1):
        if args.detailed:
            print(f"Analyzing dataset {i}/{len(datasets)}: {dataset.name}")
        else:
            print(f"Loading dataset {i}/{len(datasets)}: {dataset.name}")

        try:
            # Load dataset
            ds = get_dataset(dataset.name)
            df = ds.to_df()

            # Extract queries
            queries = []
            for _, row in df.iterrows():
                inputs = row["inputs"]
                query = inputs.get("query", inputs.get("question", str(inputs)))
                queries.append(query)

            # Calculate metrics (only in detailed mode)
            if args.detailed:
                avg_len, std_len, unique_ratio = calculate_diversity_metrics(queries)
                diversity = classify_diversity(std_len, unique_ratio, len(queries))
            else:
                avg_len = std_len = unique_ratio = 0
                diversity = "N/A (use --detailed)"

            dataset_info.append(
                {
                    "name": dataset.name,
                    "id": dataset.dataset_id,
                    "count": len(queries),
                    "avg_length": avg_len,
                    "std_length": std_len,
                    "unique_ratio": unique_ratio,
                    "diversity": diversity,
                    "queries": queries[: args.show_samples],  # Use CLI arg for sample count
                }
            )

        except Exception as e:
            print(f"  ⚠ Could not analyze: {e}")
            dataset_info.append(
                {"name": dataset.name, "id": dataset.dataset_id, "count": "?", "error": str(e)}
            )

    print()

    # If specific dataset requested via CLI
    if args.dataset_name:
        matching = [d for d in dataset_info if d["name"] == args.dataset_name]
        if not matching:
            print(f"✗ Dataset '{args.dataset_name}' not found")
            sys.exit(1)

        info = matching[0]
        print(f"\nDataset: {info['name']}")

        if "error" in info:
            print(f"⚠ Could not analyze: {info['error']}")
            sys.exit(1)

        print(f"Queries: {info['count']}")
        print(f"Avg length: {info['avg_length']:.1f} chars (σ={info['std_length']:.1f})")
        print(f"Diversity: {info['diversity']}")
        print(f"\nSample queries (first {args.show_samples}):")
        for j, query in enumerate(info["queries"], 1):
            preview = query[:70] + "..." if len(query) > 70 else query
            print(f"  {j}. {preview}")
        return

    # Display results
    print("=" * 60)
    print("Available Datasets")
    print("=" * 60)
    print()

    # Sort by query count (descending)
    dataset_info.sort(
        key=lambda x: x.get("count", 0) if isinstance(x.get("count"), int) else 0, reverse=True
    )

    for i, info in enumerate(dataset_info, 1):
        print(f"{i}. {info['name']}")

        if "error" in info:
            print(f"   ⚠ Could not analyze: {info['error']}")
        else:
            print(f"   Queries: {info['count']}")
            print(f"   Avg length: {info['avg_length']:.1f} chars (σ={info['std_length']:.1f})")
            print(f"   Diversity: {info['diversity']}")

            # Sample queries
            print("   Sample:")
            for j, query in enumerate(info["queries"], 1):
                preview = query[:70] + "..." if len(query) > 70 else query
                print(f"     {j}. {preview}")

            # Recommendation
            if info["count"] > 20 and "HIGH" in info["diversity"]:
                print("   → RECOMMENDED (most comprehensive)")

        print()

    # Non-interactive mode: just list and exit
    if args.non_interactive:
        return

    # Interactive selection
    print("[C] Create new dataset")
    print()

    while True:
        try:
            choice = input(f"Select dataset (1-{len(dataset_info)}) or C to create new: ").strip()

            if choice.upper() == "C":
                print("\nTo create a new dataset, run:")
                print("  python scripts/create_dataset_template.py")
                break

            idx = int(choice) - 1
            if 0 <= idx < len(dataset_info):
                selected = dataset_info[idx]
                print(f"\n✓ Selected: {selected['name']}")
                print(f"  Queries: {selected.get('count', '?')}")
                print("\nUse this dataset name in your evaluation script:")
                print(f'  DATASET_NAME = "{selected["name"]}"')
                break
            else:
                print(f"Please enter a number between 1 and {len(dataset_info)} or C")

        except ValueError:
            print("Invalid input. Enter a number or C")
        except KeyboardInterrupt:
            print("\n\nCancelled")
            sys.exit(0)


if __name__ == "__main__":
    main()
