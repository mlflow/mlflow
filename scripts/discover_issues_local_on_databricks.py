"""
Run discover_issues() locally against a Databricks experiment with Databricks model serving.

Requires environment variables:
    DATABRICKS_HOST  — workspace URL (e.g. https://my-workspace.databricks.com)
    DATABRICKS_TOKEN — personal access token

Usage:
    uv run python scripts/discover_issues_local_on_databricks.py
"""

import os
import time

os.environ.setdefault("DATABRICKS_HOST", os.environ.get("E2_URL", ""))
os.environ.setdefault("DATABRICKS_TOKEN", os.environ.get("E2_TOKEN", ""))

import mlflow

EXPERIMENT_ID = "1116807276482355"
JUDGE_MODEL = "databricks:/databricks-gpt-5-nano"
ANALYSIS_MODEL = "databricks:/databricks-claude-sonnet-4-5"

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_id=EXPERIMENT_ID)

start = time.time()
result = mlflow.genai.discover_issues(
    triage_sample_size=10,
    validation_sample_size=50,
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
