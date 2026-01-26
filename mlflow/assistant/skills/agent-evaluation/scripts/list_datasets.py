"""
List and compare MLflow evaluation datasets in an experiment.

This script discovers existing datasets before prompting to create new ones,
preventing duplicate work and helping users make informed choices.

Features:
- Diversity metrics (query length variability, unique vocabulary)
- Timeout protection for large experiments
- Multiple output formats (table, JSON, names-only)
- Sample query preview

Usage:
    python scripts/list_datasets.py                      # Table format (default)
    python scripts/list_datasets.py --format json        # JSON output
    python scripts/list_datasets.py --format names-only  # Names only (for piping)
    python scripts/list_datasets.py --detailed          # Include diversity analysis

Environment variables required:
    MLFLOW_TRACKING_URI
    MLFLOW_EXPERIMENT_ID
"""

import argparse
import json
import os
import signal
import sys

import numpy as np

from mlflow import MlflowClient
from mlflow.genai.datasets import get_dataset
from utils import validate_env_vars


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
        "--format",
        choices=["table", "json", "names-only"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds for dataset search (default: 30)",
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Include detailed diversity analysis (slower)"
    )
    return parser.parse_args()


def calculate_diversity_metrics(queries):
    """Calculate diversity metrics for a list of queries."""
    if not queries:
        return 0.0, 0.0, 0.0

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

    if std_length > 30 and unique_word_ratio > 5:
        return "HIGH"
    elif std_length > 15 and unique_word_ratio > 3:
        return "MEDIUM"
    else:
        return "LOW"


def get_datasets_with_timeout(client, experiment_ids, timeout_seconds):
    """Get datasets with timeout protection."""
    # Set alarm for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        datasets = client.search_datasets(experiment_ids=experiment_ids)
        signal.alarm(0)  # Cancel alarm
        return datasets
    except TimeoutError:
        signal.alarm(0)
        print(f"⚠ Dataset search timed out after {timeout_seconds}s")
        print("  Try: --timeout <seconds> to increase timeout")
        return []
    except Exception as e:
        signal.alarm(0)
        print(f"✗ Error searching datasets: {str(e)[:100]}")
        return []


def print_table_format(dataset_info, args):
    """Print datasets in table format."""
    if not dataset_info:
        print("\n✗ No datasets found in this experiment")
        print("\nTo create a new dataset:")
        print("  python scripts/create_dataset_template.py --test-cases-file test_cases.txt")
        return

    print(f"\n✓ Found {len(dataset_info)} dataset(s):")
    print("=" * 80)

    for i, info in enumerate(dataset_info, 1):
        print(f"\n{i}. {info['name']}")
        print(f"   Queries: {info.get('count', '?')}")

        if args.detailed:
            if "avg_length" in info:
                print(f"   Avg length: {info['avg_length']:.1f} chars")
                print(f"   Std length: {info['std_length']:.1f} chars")
                print(f"   Unique words/query: {info['unique_word_ratio']:.1f}")
                print(f"   Diversity: {info.get('diversity', 'N/A')}")

            if "samples" in info:
                print(f"\n   Sample queries:")
                for j, sample in enumerate(info["samples"], 1):
                    preview = sample[:60] + "..." if len(sample) > 60 else sample
                    print(f"     {j}. {preview}")

    print("\n" + "=" * 80)
    print("\nTo use a dataset in evaluation:")
    print('  python scripts/run_evaluation_template.py --dataset-name "dataset_name"')


def print_json_format(dataset_info):
    """Print datasets in JSON format."""
    print(json.dumps(dataset_info, indent=2))


def print_names_only(dataset_info):
    """Print dataset names only (one per line)."""
    for info in dataset_info:
        print(info["name"])


def main():
    """Main workflow."""
    args = parse_arguments()

    print("=" * 80)
    print("MLflow Evaluation Datasets")
    print("=" * 80)

    # Check environment using utility
    errors = validate_env_vars()
    if errors:
        print("\n✗ Environment validation failed:")
        for error in errors:
            print(f"  - {error}")
        print("\nRun scripts/setup_mlflow.py to configure environment")
        sys.exit(1)

    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
    print(f"\nExperiment ID: {experiment_id}")

    # Get datasets
    print("\nSearching for datasets...")
    client = MlflowClient()

    try:
        if args.dataset_name:
            # Search for specific dataset
            print(f"  Looking for: {args.dataset_name}")
            datasets = get_datasets_with_timeout(client, [experiment_id], args.timeout)
            datasets = [d for d in datasets if d.name == args.dataset_name]

            if not datasets:
                print(f"\n✗ Dataset '{args.dataset_name}' not found")
                sys.exit(1)
        else:
            # Get all datasets
            datasets = get_datasets_with_timeout(client, [experiment_id], args.timeout)

    except Exception as e:
        print(f"\n✗ Error: {str(e)[:200]}")
        sys.exit(1)

    # Process datasets
    dataset_info = []

    for dataset in datasets:
        info = {"name": dataset.name}

        # Try to load dataset for detailed info
        if args.detailed or args.show_samples > 0:
            try:
                ds = get_dataset(dataset.name)
                df = ds.to_df()

                info["count"] = len(df)

                # Extract queries (flexible extraction from various input formats)
                queries = []
                for _, row in df.iterrows():
                    inputs = row.get("inputs", {})
                    if isinstance(inputs, dict):
                        # Try common keys first, then use first non-empty value
                        query = (
                            inputs.get("query")
                            or inputs.get("question")
                            or inputs.get("input")
                            or inputs.get("prompt")
                            or next((v for v in inputs.values() if v), str(inputs))
                        )
                        queries.append(str(query))
                    else:
                        # If inputs is not a dict, use it directly
                        queries.append(str(inputs))

                # Calculate diversity metrics
                if queries and args.detailed:
                    avg_len, std_len, unique_ratio = calculate_diversity_metrics(queries)
                    info["avg_length"] = avg_len
                    info["std_length"] = std_len
                    info["unique_word_ratio"] = unique_ratio
                    info["diversity"] = classify_diversity(std_len, unique_ratio, len(queries))

                # Sample queries
                if queries and args.show_samples > 0:
                    info["samples"] = queries[: args.show_samples]

            except Exception as e:
                info["count"] = "?"
                info["error"] = str(e)[:50]

        dataset_info.append(info)

    # Output in requested format
    if args.format == "json":
        print_json_format(dataset_info)
    elif args.format == "names-only":
        print_names_only(dataset_info)
    else:  # table
        print_table_format(dataset_info, args)


if __name__ == "__main__":
    main()
