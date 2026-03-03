"""
Manual test: Create evaluation datasets and log them to MLflow runs.
Run this script, then open http://localhost:5000 to see datasets in the UI.

Usage:
    uv run python manual_test_dataset.py
"""

import pandas as pd

import mlflow
from mlflow.data.pandas_dataset import from_pandas
from mlflow.genai.datasets import create_dataset

# Point at local MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Create a test experiment
experiment_name = "dataset-link-test"
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Sample dataframe
df = pd.DataFrame(
    {
        "question": [
            "What is MLflow?",
            "What is a dataset?",
            "How do I log metrics?",
        ],
        "answer": [
            "An open-source ML platform",
            "A collection of data",
            "Use mlflow.log_metric()",
        ],
    }
)

# --- Create evaluation datasets on the Datasets tab ---
print("Creating evaluation datasets...")

eval_dataset_1 = create_dataset(
    name="qa-training-data",
    experiment_id=experiment_id,
    tags={"version": "1.0", "type": "training"},
)
print(f"  Created: {eval_dataset_1.name} (ID: {eval_dataset_1.dataset_id})")

eval_dataset_2 = create_dataset(
    name="qa-evaluation-data",
    experiment_id=experiment_id,
    tags={"version": "1.0", "type": "evaluation"},
)
print(f"  Created: {eval_dataset_2.name} (ID: {eval_dataset_2.dataset_id})")

# --- Run 1: Log run input dataset ---
with mlflow.start_run(run_name="training-run"):
    dataset = from_pandas(df, name="qa-training-data")
    mlflow.log_input(dataset, context="training")
    mlflow.log_metric("accuracy", 0.95)
    print(f"Run 1 (training): {mlflow.active_run().info.run_id}")

# --- Run 2: Log another run input dataset ---
with mlflow.start_run(run_name="evaluation-run"):
    dataset = from_pandas(df, name="qa-evaluation-data")
    mlflow.log_input(dataset, context="evaluation")
    mlflow.log_metric("f1_score", 0.88)
    print(f"Run 2 (evaluation): {mlflow.active_run().info.run_id}")

print(f"\nDone! Open http://localhost:5000 and navigate to experiment '{experiment_name}'")
print("Check the Datasets tab for evaluation datasets, and run details for dataset links.")
