"""
E2E tests for Gateway Scorer Architecture.

Prerequisites:
1. Start MLflow server with job execution + passphrase enabled:
   MLFLOW_SERVER_ENABLE_JOB_EXECUTION=true MLFLOW_CRYPTO_KEY_PASSPHRASE="your-secret-passphrase-here" uv run mlflow server \
     --backend-store-uri sqlite:////Users/daniel.seong/news-assistant/mlflow.db --port 5555

Run tests:
   python tests/server/test_gateway_scorer_e2e.py
"""

import json
import time

import requests

import mlflow
from mlflow.genai.judges.instructions_judge import InstructionsJudge

SERVER_URL = "http://localhost:5555"
EXPERIMENT_ID = "1"
GATEWAY_ENDPOINT = "my-endpoint"

# Define all test scorers here
TEST_SCORERS = [
    {
        "name": "answer_sufficiency",
        "description": "Basic scorer with {{inputs}} and {{outputs}}",
        "scorer": InstructionsJudge(
            name="answer_sufficiency",
            model=f"gateway:/{GATEWAY_ENDPOINT}",
            instructions="""You are evaluating whether an AI assistant output sufficiently answers the user input.

Input: {{ inputs }}
Output: {{ outputs }}

Return only Yes or No.""",
        ),
    },
    {
        "name": "trace_analyzer",
        "description": "Agentic scorer with {{trace}} variable (triggers tool calls)",
        "scorer": InstructionsJudge(
            name="trace_analyzer",
            model=f"gateway:/{GATEWAY_ENDPOINT}",
            instructions="""You are analyzing a trace to evaluate its structure.

Trace: {{ trace }}

Analyze the trace and determine if it has a valid structure with at least one span.
Return Yes if the trace has valid spans, No otherwise.""",
        ),
    },
]


def serialize_scorer(scorer) -> str:
    return json.dumps(scorer.model_dump())


def poll_job_until_complete(job_id: str, timeout: int = 120) -> dict:
    start_time = time.time()
    while time.time() - start_time < timeout:
        response = requests.get(f"{SERVER_URL}/ajax-api/3.0/jobs/{job_id}")
        result = response.json()
        if result.get("status") in ("SUCCEEDED", "FAILED"):
            return result
        time.sleep(1)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def test_ui_flow(scorer_config: dict, trace_id: str) -> bool:
    scorer = scorer_config["scorer"]
    serialized = serialize_scorer(scorer)

    response = requests.post(
        f"{SERVER_URL}/api/3.0/mlflow/scorer/invoke-async",
        json={
            "experiment_id": EXPERIMENT_ID,
            "trace_ids": [trace_id],
            "serialized_scorer": serialized,
            "log_assessments": False,
        },
    )

    if response.status_code != 200:
        print(f"    Submit failed: {response.status_code} - {response.text}")
        return False

    result = response.json()
    if "jobs" not in result:
        print(f"    No jobs in response: {result}")
        return False

    job_id = result["jobs"][0]["job_id"]
    job_result = poll_job_until_complete(job_id)

    if job_result.get("status") == "SUCCEEDED":
        assessments = job_result.get("result", {}).get("assessments", [])
        if assessments:
            source_id = assessments[0].get("source", {}).get("source_id", "")
            value = assessments[0].get("value")
            rationale = assessments[0].get("rationale", "")[:100]
            print(f"    Value: {value}")
            print(f"    Rationale: {rationale}...")
            assert source_id == f"gateway:/{GATEWAY_ENDPOINT}", f"Wrong source_id: {source_id}"
            return True
        else:
            print("    Job succeeded but no assessments")
            return False
    else:
        error = job_result.get("error", "Unknown error")
        print(f"    Job failed: {error}")
        return False


def test_sdk_flow(scorer_config: dict, traces) -> bool:
    scorer = scorer_config["scorer"]

    try:
        results = mlflow.genai.evaluate(data=traces, scorers=[scorer])
        success = bool(results.metrics or results.tables)
        if success:
            print(f"    Metrics: {results.metrics}")
        return success
    except Exception as e:
        print(f"    Error: {e}")
        return False


def run_all_tests():
    print("Gateway Scorer E2E Tests")
    print("=" * 60)
    print(f"Server: {SERVER_URL}")
    print(f"Gateway endpoint: {GATEWAY_ENDPOINT}")
    print(f"Scorers to test: {len(TEST_SCORERS)}")
    print("=" * 60)

    mlflow.set_tracking_uri(SERVER_URL)

    # Get test traces
    traces = mlflow.search_traces(experiment_ids=[EXPERIMENT_ID], max_results=1)
    if traces.empty:
        print("SKIP: No traces found in experiment")
        return

    trace_id = traces.iloc[0]["trace_id"]
    print(f"Using trace: {trace_id}\n")

    results = []

    for i, scorer_config in enumerate(TEST_SCORERS, 1):
        name = scorer_config["name"]
        description = scorer_config["description"]

        print(f"[{i}/{len(TEST_SCORERS)}] Testing: {name}")
        print(f"    {description}")

        # Test UI flow
        print("  UI Flow:")
        ui_success = test_ui_flow(scorer_config, trace_id)
        print(f"    Result: {'PASS' if ui_success else 'FAIL'}")

        # Test SDK flow
        print("  SDK Flow:")
        sdk_success = test_sdk_flow(scorer_config, traces)
        print(f"    Result: {'PASS' if sdk_success else 'FAIL'}")

        results.append(
            {
                "name": name,
                "ui": ui_success,
                "sdk": sdk_success,
            }
        )
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Scorer':<25} {'UI':<10} {'SDK':<10}")
    print("-" * 45)

    all_passed = True
    for r in results:
        ui_status = "PASS" if r["ui"] else "FAIL"
        sdk_status = "PASS" if r["sdk"] else "FAIL"
        print(f"{r['name']:<25} {ui_status:<10} {sdk_status:<10}")
        if not r["ui"] or not r["sdk"]:
            all_passed = False

    print("-" * 45)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")


if __name__ == "__main__":
    run_all_tests()
