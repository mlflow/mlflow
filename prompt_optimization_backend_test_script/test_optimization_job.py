#!/usr/bin/env python
"""
Test script to invoke prompt optimization job creation.

This script demonstrates how to call the createOptimizationJob API.

Prerequisites:
1. Start the MLflow server with job execution enabled:
   ```
   export MLFLOW_BACKEND_STORE_URI="sqlite:///mlflow.db"
   export MLFLOW_SERVER_ENABLE_JOB_EXECUTION="true"
   mlflow server --host 127.0.0.1 --port 5000
   ```

2. Set tracking URI and create a dataset:
   ```
   export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
   python script_tmp/create_aime_dataset.py --max-samples 10
   ```

3. Make sure you have a prompt registered in MLflow with model_config set.

Usage:
    python script_tmp/test_optimization_job.py --dataset-id <dataset_id>
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
    Search for optimization jobs.

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


def create_optimization_job(
    experiment_id: str,
    prompt_uri: str,
    optimizer_type: str = "gepa",
    scorers: list[str] | None = None,
    dataset_id: str | None = None,
    optimizer_config: dict | None = None,
    tags: list[dict] | None = None,
) -> dict:
    url = f"{MLFLOW_SERVER_URL}/ajax-api/{API_VERSION}.0/mlflow/prompt-optimization/jobs"

    # Convert string optimizer_type to proto enum value
    optimizer_type_to_enum = {
        "gepa": 1,  # OPTIMIZER_TYPE_GEPA
        "metaprompt": 2,  # OPTIMIZER_TYPE_METAPROMPT
    }
    optimizer_type_enum = optimizer_type_to_enum.get(optimizer_type.lower(), 0)

    # Build config - source_prompt_uri is now a top-level field, not in config
    config = {
        "optimizer_type": optimizer_type_enum,
        "scorers": scorers or [],  # Empty list if None
    }

    # Add dataset_id if provided (optional for zero-shot metaprompting)
    if dataset_id:
        config["dataset_id"] = dataset_id

    # Add optimizer_config_json if provided
    if optimizer_config:
        config["optimizer_config_json"] = json.dumps(optimizer_config)

    payload = {
        "experiment_id": experiment_id,
        "source_prompt_uri": prompt_uri,  # Top-level field, not in config
        "config": config,
        "tags": tags or [],
    }

    print("Creating optimization job...")
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
    url = f"{MLFLOW_SERVER_URL}/ajax-api/{API_VERSION}.0/mlflow/prompt-optimization/jobs/{job_id}/cancel"

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

    dataset_id = "d-ecaeda79a412460bb6f0560af6bb7321"
    prompt_uri = "prompts:/aime_solver/1"

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("optimization_backend")

    # Scorers - use built-in scorer names (e.g., "Correctness", "Safety")
    # or registered scorer names from the experiment
    scorers = ["Correctness"]  # Built-in scorer for checking expected_response

    # Optimizer-specific config
    optimizer_config = {
        "reflection_model": "openai:/gpt-5-mini",
        "max_metric_calls": 60,
    }

    experiment = mlflow.get_experiment_by_name("optimization_backend")
    experiment_id = experiment.experiment_id

    result = create_optimization_job(
        experiment_id=experiment_id,
        prompt_uri=prompt_uri,
        dataset_id=dataset_id,
        scorers=scorers,
        optimizer_type="gepa",
        optimizer_config=optimizer_config,
        tags=[{"key": "test", "value": "true"}],
    )

    if result and "job" in result:
        job_id = result["job"].get("job_id")
        print(f"\nCreated job with ID: {job_id}")

        import pdb

        pdb.set_trace()

        get_optimization_job(job_id)
        search_optimization_jobs(experiment_id)
        cancel_optimization_job(job_id)
        delete_optimization_job(job_id)
    else:
        print("Failed to create optimization job")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
