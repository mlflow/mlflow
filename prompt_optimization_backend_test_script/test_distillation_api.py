#!/usr/bin/env python
"""
Test script for the distillation API endpoint.

This tests the backend API for prompt distillation using the existing
CreatePromptOptimizationJob endpoint with optimizer_type=DISTILLATION (3).

Prerequisites:
1. MLflow server running at http://127.0.0.1:5000
2. Teacher prompt registered (e.g., prompts:/aime_solver/9 with gpt-4o)
3. Student prompt registered (e.g., prompts:/aime_solver/10 with gpt-4o-mini)
4. Dataset created with input questions

Usage:
    python prompt_optimization_backend_test_script/test_distillation_api.py
"""

import json
import time

import requests

# Configuration - update these for your setup
MLFLOW_SERVER = "http://127.0.0.1:5000"
EXPERIMENT_ID = "0"  # Update with your experiment ID

# Prompt URIs - teacher uses stronger model, student uses weaker model
TEACHER_PROMPT_URI = "prompts:/aime_solver/9"  # Should have gpt-4o in model_config
STUDENT_PROMPT_URI = "prompts:/aime_solver/10"  # Should have gpt-4o-mini in model_config

# Dataset with input questions (no expected_response needed - teacher will generate them)
DATASET_ID = "d-xxx"  # Update with your dataset ID


def create_distillation_job():
    """Create a distillation job via the API."""
    # GEPA optimizer config (distillation uses GEPA internally)
    optimizer_config = {
        "reflection_model": "openai:/gpt-5",  # Model for GEPA reflection
        "max_metric_calls": 100,
    }

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "source_prompt_uri": STUDENT_PROMPT_URI,  # Student prompt to optimize
        "config": {
            "optimizer_type": 3,  # OPTIMIZER_TYPE_DISTILLATION
            "dataset_id": DATASET_ID,
            "teacher_prompt_uri": TEACHER_PROMPT_URI,  # Required for distillation
            "optimizer_config_json": json.dumps(optimizer_config),
        },
    }

    response = requests.post(
        f"{MLFLOW_SERVER}/ajax-api/2.0/mlflow/prompt-optimization/jobs/create",
        json=payload,
    )
    response.raise_for_status()
    return response.json()


def get_job_status(job_id: str):
    """Check job status."""
    response = requests.get(
        f"{MLFLOW_SERVER}/ajax-api/2.0/mlflow/prompt-optimization/jobs/get",
        params={"job_id": job_id},
    )
    response.raise_for_status()
    return response.json()


def main():
    print("Creating distillation job...")
    print(f"  Teacher prompt: {TEACHER_PROMPT_URI}")
    print(f"  Student prompt: {STUDENT_PROMPT_URI}")
    print(f"  Dataset: {DATASET_ID}")
    print()

    try:
        result = create_distillation_job()
    except requests.exceptions.HTTPError as e:
        print(f"Error creating job: {e}")
        print(f"Response: {e.response.text}")
        return

    job = result.get("job", {})
    job_id = job.get("job_id")
    run_id = job.get("run_id")

    print("Job created successfully!")
    print(f"  Job ID: {job_id}")
    print(f"  Run ID: {run_id}")
    print(f"  View in MLflow UI: {MLFLOW_SERVER}/#/experiments/{EXPERIMENT_ID}/runs/{run_id}")
    print()

    # Poll for completion
    print("Polling for job completion...")
    while True:
        try:
            status_result = get_job_status(job_id)
        except requests.exceptions.HTTPError as e:
            print(f"Error getting status: {e}")
            break

        job_state = status_result.get("job", {}).get("state", {})
        job_status = job_state.get("status")

        # Map status codes to names
        status_names = {
            0: "UNSPECIFIED",
            1: "PENDING",
            2: "RUNNING",
            3: "SUCCEEDED",
            4: "FAILED",
            5: "CANCELED",
        }
        status_name = status_names.get(job_status, f"UNKNOWN({job_status})")
        print(f"  Status: {status_name}")

        if job_status in [3, 4, 5]:  # SUCCEEDED, FAILED, CANCELED
            print()
            if job_status == 3:  # SUCCEEDED
                job_info = status_result.get("job", {})
                optimized_uri = job_info.get("optimized_prompt_uri")
                initial_scores = job_info.get("initial_eval_scores", {})
                final_scores = job_info.get("final_eval_scores", {})
                print("Distillation completed successfully!")
                print(f"  Optimized prompt: {optimized_uri}")
                print(f"  Initial scores: {initial_scores}")
                print(f"  Final scores: {final_scores}")
            elif job_status == 4:  # FAILED
                error_msg = job_state.get("error_message", "Unknown error")
                print(f"Job failed: {error_msg}")
            else:
                print("Job was canceled")
            break

        time.sleep(10)


if __name__ == "__main__":
    main()
