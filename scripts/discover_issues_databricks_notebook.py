# Databricks notebook source
# MAGIC %md
# MAGIC # Discover Issues — Databricks Notebook
# MAGIC
# MAGIC Runs `discover_issues()` on a Databricks experiment using a Databricks model serving endpoint.
# MAGIC
# MAGIC **Requirements:**
# MAGIC - Run in a Databricks workspace notebook
# MAGIC - Model serving endpoint accessible (e.g. `databricks:/databricks-claude-sonnet-4-5`)

# COMMAND ----------

import time

import mlflow

EXPERIMENT_ID = "3239668363401461"  # <-- replace with your experiment ID
MODEL = "databricks:/databricks-claude-sonnet-4-5"

mlflow.set_experiment(experiment_id=EXPERIMENT_ID)

# COMMAND ----------

start = time.time()
result = mlflow.genai.discover_issues(
    triage_sample_size=100,
    judge_model=MODEL,
    analysis_model=MODEL,
)
elapsed = time.time() - start

# COMMAND ----------

print(f"Done in {elapsed:.1f}s — {len(result.issues)} issues found\n")
print(result.summary)

for issue in result.issues:
    print(f"\n--- {issue.name} ({issue.frequency:.0%}, confidence: {issue.confidence}/100) ---")
    print(f"  {issue.description}")
    print(f"  Root cause: {issue.root_cause}")
    print(f"  Examples: {issue.example_trace_ids[:3]}")

# COMMAND ----------

# Inspect generated scorers
for issue in result.issues:
    print(f"=== {issue.name} ===")
    print(f"Instructions: {issue.scorer.instructions}\n")
