"""
Quick test script for mlflow.genai.discover_issues().

Usage:
    # Against a local MLflow server (start one with `mlflow server`)
    uv run python scripts/test_discover_issues.py

    # Against Databricks
    MLFLOW_TRACKING_URI=databricks \
    DATABRICKS_HOST=https://your-workspace.databricks.com \
    DATABRICKS_TOKEN=your-token \
    uv run python scripts/test_discover_issues.py --experiment-id 123456

    # Custom model
    uv run python scripts/test_discover_issues.py --model "openai:/gpt-4.1"
"""

import argparse
import json
import time
from pathlib import Path

import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--tracking-uri", default="http://localhost:5000")
parser.add_argument("--experiment-id", default=None)
parser.add_argument("--experiment-name", default=None)
parser.add_argument("--model", default=None, help="LLM model URI (e.g. openai:/gpt-4.1)")
parser.add_argument("--sample-size", type=int, default=20)
parser.add_argument("--max-issues", type=int, default=5)
parser.add_argument("--output-dir", default="scripts/discover_issues_output")
args = parser.parse_args()

mlflow.set_tracking_uri(args.tracking_uri)

if args.experiment_name:
    mlflow.set_experiment(args.experiment_name)
elif args.experiment_id:
    mlflow.set_experiment(experiment_id=args.experiment_id)
else:
    print("No experiment specified. Listing available experiments:")
    from mlflow import MlflowClient

    client = MlflowClient()
    for exp in client.search_experiments(max_results=10):
        print(f"  [{exp.experiment_id}] {exp.name}")
    print("\nRe-run with --experiment-id or --experiment-name")
    raise SystemExit(1)

# Show what we're working with
exp_id = mlflow.tracking.fluent._get_experiment_id()
traces = mlflow.search_traces(locations=[exp_id], return_type="list", include_spans=False)
print(f"Experiment {exp_id}: {len(traces)} traces")

# Run it
print(f"\nRunning discover_issues(sample_size={args.sample_size}, max_issues={args.max_issues})...")
start = time.time()

kwargs = {"sample_size": args.sample_size, "max_issues": args.max_issues}
if args.model:
    kwargs["model"] = args.model

result = mlflow.genai.discover_issues(**kwargs)
elapsed = time.time() - start

# Print results
print(f"\nDone in {elapsed:.1f}s — {len(result.issues)} issues found\n")

for i, issue in enumerate(result.issues, 1):
    print(f"  {i}. {issue.name} ({issue.frequency:.0%} of traces)")
    print(f"     {issue.description}")
    print(f"     Root cause: {issue.root_cause}")
    print()

print(result.summary)

# Save artifacts
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

(output_dir / "summary.md").write_text(result.summary)
(output_dir / "issues.json").write_text(
    json.dumps(
        [
            {
                "name": i.name,
                "description": i.description,
                "root_cause": i.root_cause,
                "frequency": i.frequency,
                "example_trace_ids": i.example_trace_ids,
            }
            for i in result.issues
        ],
        indent=2,
    )
)
print(f"Artifacts saved to {output_dir}/")
