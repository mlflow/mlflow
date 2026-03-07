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
import os
import time

# Ensure API key is available (OPENAI_TOKEN -> OPENAI_API_KEY)
if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_TOKEN"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_TOKEN"]

import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--experiment-id", default="1")
parser.add_argument("--tracking-uri", default="http://localhost:5000")
parser.add_argument("--triage-sample-size", type=int, default=10)
args = parser.parse_args()

MODEL = "openai:/gpt-5-mini"

mlflow.set_tracking_uri(args.tracking_uri)
mlflow.set_experiment(experiment_id=args.experiment_id)

start = time.time()
result = mlflow.genai.discover_issues(
    triage_sample_size=args.triage_sample_size,
    model=MODEL,
)
elapsed = time.time() - start

print(f"\nDone in {elapsed:.1f}s — {len(result.issues)} issues found\n")
print(result.summary)

for issue in result.issues:
    print(f"\n--- {issue.name} ({issue.frequency:.0%}, confidence: {issue.confidence}/100) ---")
    print(f"  {issue.description}")
    print(f"  Root cause: {issue.root_cause}")
    print(f"  Examples: {issue.example_trace_ids[:3]}")
