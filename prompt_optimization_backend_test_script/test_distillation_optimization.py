#!/usr/bin/env python
"""
Test script for distillation optimization (skips dataset creation).

Prerequisites:
1. MLflow server running at http://127.0.0.1:5000
2. A distillation dataset already created (with expected_response)
3. A student prompt to optimize

Usage:
    python prompt_optimization_backend_test_script/test_distillation_optimization.py
"""

import time

import mlflow
from mlflow.genai.datasets import get_dataset
from mlflow.genai.optimize import optimize_prompts
from mlflow.genai.optimize.job import _build_predict_fn, _create_optimizer, _load_scorers
from mlflow.genai.prompts import load_prompt

# Configuration - update these for your setup
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "optimization_backend"

# Paste your dataset ID here (created from test_distillation_dataset_creation.py)
DISTILLATION_DATASET_ID = "d-3c71957e5f804f06be65ce387c0f3ab6"

# Student prompt to optimize
STUDENT_PROMPT_URI = "prompts:/aime_solver/10"

# Optimizer settings
OPTIMIZER_TYPE = "gepa"
OPTIMIZER_CONFIG = {
    "reflection_model": "openai:/gpt-5",  # Use gpt-5 for reflection (not student's model)
    "max_metric_calls": 200,
    # GEPA-specific kwargs to improve optimization
    "gepa_kwargs": {
        "reflection_minibatch_size": 5,  # Use smaller minibatch for more diverse feedback
        "skip_perfect_score": False,  # Don't skip even if scores are high
    },
}

# Setup
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
experiment_id = experiment.experiment_id

print(f"Experiment ID: {experiment_id}")
print(f"Dataset ID: {DISTILLATION_DATASET_ID}")
print(f"Student Prompt: {STUDENT_PROMPT_URI}")

# Load dataset
print("\nLoading distillation dataset...")
dataset = get_dataset(dataset_id=DISTILLATION_DATASET_ID)
df = dataset.to_df()
print(f"Dataset has {len(df)} records")

# Load student prompt
print("\nLoading student prompt...")
student_prompt = load_prompt(STUDENT_PROMPT_URI)
print(f"Prompt template type: {type(student_prompt.template)}")

# Build predict function
print("\nBuilding predict function...")
predict_fn = _build_predict_fn(STUDENT_PROMPT_URI)

# Create optimizer and scorers
print("\nCreating optimizer and scorers...")
optimizer = _create_optimizer(OPTIMIZER_TYPE, OPTIMIZER_CONFIG)
scorers = _load_scorers(["SemanticMatch"], experiment_id)

# Run optimization
print("\nStarting optimization...")
start_time = time.time()

result = optimize_prompts(
    predict_fn=predict_fn,
    train_data=dataset,
    prompt_uris=[STUDENT_PROMPT_URI],
    optimizer=optimizer,
    scorers=scorers,
    enable_tracking=True,
)

elapsed = time.time() - start_time
print(f"\nOptimization completed in {elapsed:.1f}s")
print(f"Optimizer: {result.optimizer_name}")
print(f"Initial score: {result.initial_eval_score}")
print(f"Final score: {result.final_eval_score}")

if result.optimized_prompts:
    print(f"Optimized prompt URI: {result.optimized_prompts[0].uri}")
