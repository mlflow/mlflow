"""
Test mlflow.genai.discover_issues() against an experiment.

Usage:
    uv run python scripts/test_discover_issues.py --experiment-id <ID>
    uv run python scripts/test_discover_issues.py --experiment-id <ID> --tracking-uri http://localhost:5000
"""

import argparse
import time

import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--experiment-id", required=True)
parser.add_argument("--tracking-uri", default="http://localhost:5000")
parser.add_argument("--sample-size", type=int, default=100)
args = parser.parse_args()

mlflow.set_tracking_uri(args.tracking_uri)
mlflow.set_experiment(experiment_id=args.experiment_id)

start = time.time()
result = mlflow.genai.discover_issues(sample_size=args.sample_size)
elapsed = time.time() - start

print(f"\nDone in {elapsed:.1f}s — {len(result.issues)} issues found\n")
print(result.summary)

for issue in result.issues:
    print(f"\n--- {issue.name} ({issue.frequency:.0%}, confidence: {issue.confidence}/100) ---")
    print(f"  {issue.description}")
    print(f"  Root cause: {issue.root_cause}")
    print(f"  Examples: {issue.example_trace_ids[:3]}")
