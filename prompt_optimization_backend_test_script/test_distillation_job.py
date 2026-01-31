#!/usr/bin/env python
"""
Test script to invoke prompt distillation job creation.

This script demonstrates how to call the distillation API for ICL-KD
(In-Context Learning based Knowledge Distillation).

**Workflow:**
1. Run your agent/prompt over your dataset (traces are auto-captured by MLflow)
2. Call the distillation API with the source prompt URI (where traces are linked)
3. The API automatically finds traces, extracts responses, and optimizes the student prompt

Prerequisites:
1. Start the MLflow server with job execution enabled:
   ```
   export MLFLOW_BACKEND_STORE_URI="sqlite:///mlflow.db"
   export MLFLOW_SERVER_ENABLE_JOB_EXECUTION="true"
   mlflow server --host 127.0.0.1 --port 5000
   ```

2. Run your agent with MLflow tracing enabled, using load_prompt():
   ```python
   import mlflow
   from mlflow.genai.prompts import load_prompt

   mlflow.set_tracking_uri("http://127.0.0.1:5000")
   mlflow.set_experiment("my_experiment")

   @mlflow.trace
   def my_agent(question):
       prompt = load_prompt("prompts:/my_prompt/1")  # Links trace to prompt
       # ... call LLM with prompt ...
       return response

   # Run over your dataset - traces are captured automatically
   for item in dataset:
       my_agent(item["question"])
   ```

3. Create a student prompt with the model config you want to distill to
   (e.g., gpt-4o-mini instead of gpt-4o)

Usage:
    python prompt_optimization_backend_test_script/test_distillation_job.py
"""

import json
import time

import requests

# Configuration
MLFLOW_SERVER_URL = "http://127.0.0.1:5000"
API_VERSION = 3


def create_distillation_job(
    experiment_id: str,
    student_prompt_uri: str,
    source_prompt_uri: str,
    optimizer_type: str = "gepa",
    optimizer_config: dict | None = None,
    max_traces: int | None = None,
) -> dict:
    """
    Create a distillation job via the API.

    Args:
        experiment_id: MLflow experiment ID.
        student_prompt_uri: URI of the student prompt to optimize (e.g., "prompts:/solver/2").
        source_prompt_uri: URI of the prompt whose traces to use (e.g., "prompts:/solver/1").
        optimizer_type: Optimizer type ("gepa" or "metaprompt").
        optimizer_config: Optimizer-specific configuration.
        max_traces: Maximum number of traces to use. None for all.

    Returns:
        Response dict containing the job information.
    """
    url = f"{MLFLOW_SERVER_URL}/ajax-api/{API_VERSION}.0/mlflow/prompt-optimization/distill"

    payload = {
        "experiment_id": experiment_id,
        "student_prompt_uri": student_prompt_uri,
        "source_prompt_uri": source_prompt_uri,
        "optimizer_type": f"OPTIMIZER_TYPE_{optimizer_type.upper()}",
    }

    if optimizer_config:
        payload["optimizer_config_json"] = json.dumps(optimizer_config)

    if max_traces is not None:
        payload["max_traces"] = max_traces

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


def get_optimization_job(job_id: str) -> dict:
    """Get the status of a job."""
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


def poll_job_status(job_id: str, poll_interval: int = 10, timeout: int = 600) -> dict:
    """
    Poll job status until completion or timeout.

    Args:
        job_id: The job ID to monitor.
        poll_interval: Seconds between polls.
        timeout: Maximum seconds to wait.

    Returns:
        Final job state.
    """
    start_time = time.time()
    print(f"\nPolling job status every {poll_interval}s (timeout: {timeout}s)...")

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"Timeout after {elapsed:.0f}s")
            return None

        result = get_optimization_job(job_id)
        if not result:
            print("Failed to get job status")
            return None

        job = result.get("job", {})
        status = job.get("state", {}).get("status", "UNKNOWN")
        progress = job.get("progress", {})

        print(f"[{elapsed:.0f}s] Status: {status}")
        if progress:
            print(f"  Progress: {json.dumps(progress, indent=4)}")

        if status in ["COMPLETED", "FAILED", "CANCELED"]:
            print(f"\nJob finished with status: {status}")
            return result

        time.sleep(poll_interval)


def main():
    import mlflow

    # Configuration - update these values for your setup
    # =====================================================
    # Source prompt - the prompt whose traces will be used as training data
    # This is typically your "teacher" prompt (e.g., using gpt-4o)
    source_prompt_uri = "prompts:/aime_solver/1"

    # Student prompt - the prompt you want to optimize
    # This should use a smaller/cheaper model (e.g., gpt-4o-mini)
    student_prompt_uri = "prompts:/aime_solver/2"

    # Maximum number of traces to use (None for all available)
    max_traces = 100
    # =====================================================

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("distillation_backend")

    # Optimizer configuration
    optimizer_config = {
        "reflection_model": "openai:/gpt-4o-mini",
        "max_metric_calls": 50,
    }

    experiment = mlflow.get_experiment_by_name("distillation_backend")
    if not experiment:
        mlflow.create_experiment("distillation_backend")
        experiment = mlflow.get_experiment_by_name("distillation_backend")

    experiment_id = experiment.experiment_id

    print("=" * 60)
    print("Starting Distillation Test")
    print("=" * 60)
    print(f"Experiment ID: {experiment_id}")
    print(f"Source Prompt (traces): {source_prompt_uri}")
    print(f"Student Prompt (to optimize): {student_prompt_uri}")
    print(f"Max Traces: {max_traces}")
    print("=" * 60)

    result = create_distillation_job(
        experiment_id=experiment_id,
        student_prompt_uri=student_prompt_uri,
        source_prompt_uri=source_prompt_uri,
        optimizer_type="gepa",
        optimizer_config=optimizer_config,
        max_traces=max_traces,
    )

    if result and "job" in result:
        job_id = result["job"].get("job_id")
        run_id = result["job"].get("run_id")
        print(f"\nCreated distillation job with ID: {job_id}")
        print(f"MLflow Run ID: {run_id}")
        print(f"View run at: {MLFLOW_SERVER_URL}/#/experiments/{experiment_id}/runs/{run_id}")

        # Poll for completion
        final_result = poll_job_status(job_id, poll_interval=15, timeout=1800)

        if final_result:
            job = final_result.get("job", {})
            scores = job.get("scores", {})
            optimized_prompt_uri = job.get("optimized_prompt_uri")

            print("\n" + "=" * 60)
            print("Distillation Results")
            print("=" * 60)
            print(f"Traces Used: {job.get('num_traces', 'N/A')}")
            print(f"Samples Extracted: {job.get('num_samples', 'N/A')}")
            print(f"Initial Score: {scores.get('initial_eval_score')}")
            print(f"Final Score: {scores.get('final_eval_score')}")
            print(f"Optimized Prompt: {optimized_prompt_uri}")
    else:
        print("Failed to create distillation job")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
