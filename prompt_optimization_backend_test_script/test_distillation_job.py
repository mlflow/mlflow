#!/usr/bin/env python
"""
Test script to invoke prompt distillation job creation.

This script demonstrates how to call the createOptimizationJob API with
optimizer_type=DISTILLATION for knowledge distillation (ICL-KD).

Distillation optimizes a student prompt to match teacher model responses:
1. Searches for existing traces linked to the teacher prompt
2. Extracts inputs and teacher responses from those traces
3. Creates a distillation dataset on the fly (inputs + teacher responses as expected_response)
4. Student prompt is optimized using GEPA with SemanticMatch scorer

Prerequisites:
1. Start the MLflow server with job execution enabled:
   ```
   export MLFLOW_BACKEND_STORE_URI="sqlite:///mlflow.db"
   export MLFLOW_SERVER_ENABLE_JOB_EXECUTION="true"
   mlflow server --host 127.0.0.1 --port 5000
   ```

2. Register two prompts with different model configs:
   - Teacher prompt: stronger model (e.g., gpt-4o) - must have traces already
   - Student prompt: weaker model to be optimized (e.g., gpt-4o-mini)

3. Generate traces for the teacher prompt by running inference on your dataset.
   The distillation job will find these traces and extract the teacher's responses.

Usage:
    python prompt_optimization_backend_test_script/test_distillation_job.py
"""

import json

import requests

# Configuration
MLFLOW_SERVER_URL = "http://127.0.0.1:5000"
API_VERSION = 3  # API version from proto (prompt optimization APIs are v3)


def get_optimization_job(job_id: str) -> dict:
    """
    Get the status of an optimization job.

    Args:
        job_id: The job ID to query.

    Returns:
        The job details.
    """
    url = f"{MLFLOW_SERVER_URL}/ajax-api/{API_VERSION}.0/mlflow/prompt-optimization/jobs/{job_id}"

    print(f"Getting job status for: {job_id}")

    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None

    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    return result


def search_optimization_jobs(experiment_id: str = None) -> dict:
    """
    Search for optimization jobs (includes both standard and distillation jobs).

    Args:
        experiment_id: Optional experiment ID to filter by.

    Returns:
        List of matching jobs.
    """
    url = f"{MLFLOW_SERVER_URL}/ajax-api/{API_VERSION}.0/mlflow/prompt-optimization/jobs/search"

    payload = {}
    if experiment_id:
        payload["experiment_id"] = experiment_id

    print("Searching optimization jobs...")

    response = requests.post(url, json=payload)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None

    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    return result


def delete_optimization_job(job_id: str) -> dict:
    """
    Delete an optimization job.

    Args:
        job_id: The job ID to delete.

    Returns:
        Empty response on success.
    """
    url = f"{MLFLOW_SERVER_URL}/ajax-api/{API_VERSION}.0/mlflow/prompt-optimization/jobs/{job_id}"

    print(f"Deleting job: {job_id}")

    response = requests.delete(url)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None

    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    return result


def create_distillation_job(
    experiment_id: str,
    student_prompt_uri: str,
    teacher_prompt_uri: str,
    optimizer_config: dict | None = None,
    tags: list[dict] | None = None,
) -> dict:
    """
    Create a distillation job.

    The job will:
    1. Search for traces linked to the teacher prompt
    2. Extract inputs and responses from those traces
    3. Create a distillation dataset on the fly
    4. Optimize the student prompt using GEPA with SemanticMatch scorer

    Args:
        experiment_id: The experiment ID to track the job.
        student_prompt_uri: URI of the student prompt to optimize.
        teacher_prompt_uri: URI of the teacher prompt (must have existing traces).
        optimizer_config: GEPA optimizer config (reflection_model, max_metric_calls, etc.)
        tags: Optional tags for the job.

    Returns:
        The created job details.
    """
    url = f"{MLFLOW_SERVER_URL}/ajax-api/{API_VERSION}.0/mlflow/prompt-optimization/jobs"

    # Build config with OPTIMIZER_TYPE_DISTILLATION (3)
    # Note: dataset_id is NOT required - distillation creates its dataset from traces
    config = {
        "optimizer_type": 3,  # OPTIMIZER_TYPE_DISTILLATION
        "teacher_prompt_uri": teacher_prompt_uri,
    }

    # Add optimizer_config_json if provided (GEPA config for internal optimization)
    if optimizer_config:
        config["optimizer_config_json"] = json.dumps(optimizer_config)

    payload = {
        "experiment_id": experiment_id,
        "source_prompt_uri": student_prompt_uri,  # Student prompt to optimize
        "config": config,
        "tags": tags or [],
    }

    print("Creating distillation job...")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    response = requests.post(url, json=payload)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None

    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    return result


def cancel_optimization_job(job_id: str) -> dict:
    """
    Cancel an optimization job.

    Args:
        job_id: The job ID to cancel.

    Returns:
        The job details after cancellation.
    """
    base_url = f"{MLFLOW_SERVER_URL}/ajax-api/{API_VERSION}.0/mlflow/prompt-optimization"
    url = f"{base_url}/jobs/{job_id}/cancel"

    print(f"Cancelling job: {job_id}")

    response = requests.post(url)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None

    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    return result


def main():
    import mlflow

    # Configuration - update these for your setup
    # Teacher prompt must have existing traces (run inference on it first)
    teacher_prompt_uri = "prompts:/aime_solver/9"  # gpt-5.2 (with traces)
    student_prompt_uri = "prompts:/aime_solver/10"  # gpt-4.1-mini (to optimize)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("optimization_backend")

    # GEPA optimizer config for internal optimization
    # (distillation uses GEPA with SemanticMatch scorer internally)
    optimizer_config = {
        "reflection_model": "openai:/gpt-5",
        "max_metric_calls": 160,
    }

    experiment = mlflow.get_experiment_by_name("optimization_backend")
    experiment_id = experiment.experiment_id

    result = create_distillation_job(
        experiment_id=experiment_id,
        student_prompt_uri=student_prompt_uri,
        teacher_prompt_uri=teacher_prompt_uri,
        optimizer_config=optimizer_config,
        tags=[{"key": "test", "value": "distillation"}],
    )

    if result and "job" in result:
        job_id = result["job"].get("job_id")
        print(f"\nCreated distillation job with ID: {job_id}")

        import pdb

        pdb.set_trace()

        get_optimization_job(job_id)
        search_optimization_jobs(experiment_id)
        cancel_optimization_job(job_id)
        delete_optimization_job(job_id)
    else:
        print("Failed to create distillation job")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
