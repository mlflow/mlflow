#!/usr/bin/env python
"""
Test script for creating a distillation dataset from traces.

Prerequisites:
1. MLflow server running at http://127.0.0.1:5000
2. Traces linked to a prompt (via load_prompt() in traced code)

Usage:
    python prompt_optimization_backend_test_script/test_distillation_dataset_creation.py
"""

import mlflow
from mlflow.genai.optimize.distillation import create_distillation_dataset_from_prompt

# Configuration - update these for your setup
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "optimization_backend"
SOURCE_PROMPT_URI = "prompts:/aime_solver/9"
MAX_TRACES = 50

# Setup
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

experiment_id = experiment.experiment_id

print(f"Experiment ID: {experiment_id}")
print(f"Source Prompt: {SOURCE_PROMPT_URI}")

# Create distillation dataset
result = create_distillation_dataset_from_prompt(
    experiment_id=experiment_id,
    source_prompt_uri=SOURCE_PROMPT_URI,
    max_traces=MAX_TRACES,
)

print(f"Dataset ID: {result.dataset_id}")
print(f"Dataset Name: {result.dataset_name}")
print(f"Traces: {result.num_traces}")
print(f"Samples: {result.num_samples}")
