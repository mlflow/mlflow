"""
Run discover_issues() against a local MLflow server with a local model.

Requires:
    - Local MLflow server running (e.g. `mlflow server --port 5000`)
    - OPENAI_API_KEY set for the OpenAI model

Usage:
    uv run python scripts/discover_issues_local_server.py
    uv run python scripts/discover_issues_local_server.py --experiment-id 2 --tracking-uri http://localhost:5000
"""

import argparse
import time

import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--experiment-id", default="1")
parser.add_argument("--tracking-uri", default="http://localhost:5000")
parser.add_argument("--triage-sample-size", type=int, default=10)
parser.add_argument("--validation-sample-size", type=int, default=50)
args = parser.parse_args()

JUDGE_MODEL = "openai:/gpt-5-mini"
ANALYSIS_MODEL = "openai:/gpt-5"

mlflow.set_tracking_uri(args.tracking_uri)
mlflow.set_experiment(experiment_id=args.experiment_id)

start = time.time()
result = mlflow.genai.discover_issues(
    triage_sample_size=args.triage_sample_size,
    validation_sample_size=args.validation_sample_size,
    judge_model=JUDGE_MODEL,
    analysis_model=ANALYSIS_MODEL,
)
elapsed = time.time() - start

print(f"\nDone in {elapsed:.1f}s — {len(result.issues)} issues found\n")
print(result.summary)

for issue in result.issues:
    print(f"\n--- {issue.name} ({issue.frequency:.0%}, confidence: {issue.confidence}/100) ---")
    print(f"  {issue.description}")
    print(f"  Root cause: {issue.root_cause}")
    print(f"  Examples: {issue.example_trace_ids[:3]}")
