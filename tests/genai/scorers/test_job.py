"""
E2E integration tests for async scorer invocation via the MLflow server.

These tests spin up a real MLflow server with job execution enabled and test
the full flow of invoking scorers on traces asynchronously.

The MLflow AI Gateway is mocked to avoid real LLM calls during testing.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Literal

import pytest
import requests

import mlflow
from mlflow.genai.judges import make_judge
from mlflow.tracing.assessment import log_expectation

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="MLflow job execution is not supported on Windows"
)


class MockGatewayHandler(BaseHTTPRequestHandler):
    """Mock handler for MLflow gateway chat completions endpoint."""

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))

        messages = body.get("messages", [])
        prompt_text = str(messages)

        # Check for agentic scorers ({{trace}} template) - indicated by tool calls
        tools = body.get("tools", [])
        if tools:
            # {{trace}} scorers use tools to fetch trace data - just return a valid response
            response = self._make_response("3", "Counted 3 spans")
        # Check for conversation data (session-level scorers)
        elif "conversation" in prompt_text.lower():
            # Validate that {{conversation}} was parsed - should contain conversation turns
            # Session 1: "Hello, how are you?" / "What's your name?"
            # Session 2: "What's the weather?" / "Thanks!"
            prompt_lower = prompt_text.lower()
            # Check for either session's content
            has_session_1 = "hello" in prompt_lower and (
                "name" in prompt_lower or "assistant" in prompt_lower
            )
            has_session_2 = "weather" in prompt_lower and "thanks" in prompt_lower
            if has_session_1:
                response = self._make_response("Good", "Session 1: Good conversation")
            elif has_session_2:
                response = self._make_response("Average", "Session 2: Average conversation")
            else:
                self._send_error(
                    "Conversation content not found. Expected session 1 (hello/name) "
                    f"or session 2 (weather/thanks). Got: {prompt_text[:500]}"
                )
                return
        # Check for safety evaluation (builtin scorer)
        elif "safe" in prompt_text.lower() or "harmful" in prompt_text.lower():
            # Builtin Safety scorer uses its own prompt template with trace data
            # Just validate that some trace content is present
            if "what is" not in prompt_text.lower() and "answer" not in prompt_text.lower():
                self._send_error("Trace data not found in safety prompt")
                return
            response = self._make_response("yes", "Content is safe")
        else:
            # Default: single-turn scorers with {{inputs}}/{{outputs}}/{{expectations}}
            if "what is" not in prompt_text.lower() or "the answer is" not in prompt_text.lower():
                self._send_error(f"Trace inputs/outputs not found in prompt: {prompt_text[:500]}")
                return
            # Validate that expectations were parsed (expected_answer values are "0", "2", or "4")
            if "expected_answer" not in prompt_text.lower():
                self._send_error(f"Expectations not found in prompt: {prompt_text[:500]}")
                return
            response = self._make_response("Yes", "Mock response")

        response_body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def _make_response(self, result: str, rationale: str) -> dict[str, Any]:
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"result": result, "rationale": rationale}),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def _send_error(self, message: str):
        error_response = {"error": {"message": message, "type": "invalid_request_error"}}
        response_body = json.dumps(error_response).encode()
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format, *args):
        pass  # Suppress logging


class Client:
    """HTTP client for interacting with MLflow server endpoints."""

    def __init__(self, server_url: str):
        self.server_url = server_url

    def invoke_scorer(
        self,
        experiment_id: str,
        serialized_scorer: str,
        trace_ids: list[str],
        log_assessments: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "experiment_id": experiment_id,
            "serialized_scorer": serialized_scorer,
            "trace_ids": trace_ids,
            "log_assessments": log_assessments,
        }
        response = requests.post(
            f"{self.server_url}/ajax-api/3.0/mlflow/scorer/invoke",
            json=payload,
        )
        if not response.ok:
            raise AssertionError(
                f"invoke_scorer failed with status {response.status_code}: {response.text}"
            )
        return response.json()

    def get_job(self, job_id: str) -> dict[str, Any]:
        response = requests.get(f"{self.server_url}/ajax-api/3.0/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def wait_job(self, job_id: str, timeout: float = 30) -> dict[str, Any]:
        beg_time = time.time()
        while time.time() - beg_time <= timeout:
            job_json = self.get_job(job_id)
            if job_json["status"] in ["SUCCEEDED", "FAILED", "TIMEOUT"]:
                return job_json
            time.sleep(0.5)
        raise TimeoutError("The job did not complete within the timeout.")

    def wait_job_succeeded(self, job_id: str) -> dict[str, Any]:
        result = self.wait_job(job_id)
        assert result["status"] == "SUCCEEDED", f"Job failed: {result}"
        return result


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory, mock_gateway_server: str) -> Client:
    """Start an MLflow server with job execution enabled for scorer invocation."""
    from tests.helper_functions import get_safe_port

    tmp_path = tmp_path_factory.mktemp("scorer_job_server")
    backend_store_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    port = get_safe_port()
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "-h",
            "127.0.0.1",
            "-p",
            str(port),
            "--backend-store-uri",
            backend_store_uri,
        ],
        env={
            **os.environ,
            "MLFLOW_SERVER_ENABLE_JOB_EXECUTION": "true",
            # Register the scorer invoke job function
            "_MLFLOW_SUPPORTED_JOB_FUNCTION_LIST": ("mlflow.genai.scorers.job.invoke_scorer_job"),
            "_MLFLOW_ALLOWED_JOB_NAME_LIST": "invoke_scorer",
            # Allow loading custom scorers (normally restricted to Databricks runtime)
            "DATABRICKS_RUNTIME_VERSION": "15.0",
            # Point gateway calls to our mock server (test override)
            "_MLFLOW_GATEWAY_BASE_URL_TEST_OVERRIDE": mock_gateway_server,
            # Set batch size to 2 for testing job batching behavior
            "MLFLOW_SERVER_SCORER_INVOKE_BATCH_SIZE": "2",
        },
        start_new_session=True,
    ) as server_proc:
        try:
            # Wait for job runner to start
            time.sleep(10)
            # Wait for server to be healthy
            deadline = time.time() + 15
            while time.time() < deadline:
                time.sleep(1)
                try:
                    resp = requests.get(f"http://127.0.0.1:{port}/health")
                except requests.ConnectionError:
                    continue
                if resp.status_code == 200:
                    break
            else:
                raise TimeoutError("Server did not report healthy within 15 seconds")
            yield Client(f"http://127.0.0.1:{port}")
        finally:
            os.killpg(server_proc.pid, signal.SIGKILL)


@pytest.fixture
def experiment_with_traces(client: Client):
    """Create an experiment with traces for testing, including expectations."""
    mlflow.set_tracking_uri(client.server_url)

    experiment_name = f"test_scorer_job_{time.time()}"
    experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    trace_ids = []
    for i in range(3):
        with mlflow.start_span(name=f"test_span_{i}") as span:
            span.set_inputs({"question": f"What is {i} + {i}?"})
            span.set_outputs(f"The answer is {i + i}")
            trace_ids.append(span.trace_id)

        # Add expectation (ground truth) to each trace
        log_expectation(
            trace_id=trace_ids[-1],
            name="expected_answer",
            value=str(i + i),
        )

    return experiment_id, trace_ids


@pytest.fixture(scope="module")
def mock_gateway_server():
    """Start a mock server that handles gateway chat completion requests."""
    from tests.helper_functions import get_safe_port

    port = get_safe_port()
    server = HTTPServer(("127.0.0.1", port), MockGatewayHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_invoke_scorer_basic(client: Client, experiment_with_traces):
    experiment_id, trace_ids = experiment_with_traces

    judge = make_judge(
        name="answer_quality",
        instructions="Input: {{ inputs }}\nOutput: {{ outputs }}\nExpected: {{ expectations }}",
        model="gateway:/mock-endpoint",
        feedback_value_type=Literal["Yes", "No"],
    )

    response = client.invoke_scorer(
        experiment_id=experiment_id,
        serialized_scorer=json.dumps(judge.model_dump()),
        trace_ids=trace_ids,
    )

    # 3 traces with batch size 2 -> 2 jobs (sizes [2, 1])
    jobs = response["jobs"]
    assert len(jobs) == 2
    assert sorted(len(j["trace_ids"]) for j in jobs) == [1, 2]

    # Verify all trace IDs are accounted for
    all_job_trace_ids = {tid for j in jobs for tid in j["trace_ids"]}
    assert all_job_trace_ids == set(trace_ids)

    # Wait for all jobs and verify results
    for job_info in jobs:
        result = client.wait_job_succeeded(job_info["job_id"])["result"]
        assert result["success"] is True
        assert result["failures"] == []

        for trace_id in job_info["trace_ids"]:
            assessment = result["assessments"][trace_id][0]
            assert assessment["assessment_name"] == "answer_quality"
            assert assessment["feedback"]["value"] == "Yes"


def test_invoke_scorer_missing_trace(client: Client, experiment_with_traces):
    experiment_id, _ = experiment_with_traces
    fake_trace_id = "tr-does-not-exist-00000000000000"

    judge = make_judge(
        name="answer_quality",
        instructions="Input: {{ inputs }}\nOutput: {{ outputs }}",
        model="gateway:/mock-endpoint",
        feedback_value_type=Literal["Yes", "No"],
    )

    response = client.invoke_scorer(
        experiment_id=experiment_id,
        serialized_scorer=json.dumps(judge.model_dump()),
        trace_ids=[fake_trace_id],
    )

    # Job succeeds but result indicates trace not found
    result = client.wait_job_succeeded(response["jobs"][0]["job_id"])["result"]
    assert result["success"] is False
    assert result["failures"][0]["trace_id"] == fake_trace_id
    assert result["failures"][0]["error_code"] == "TRACE_NOT_FOUND"


@pytest.fixture
def experiment_with_agentic_trace(client: Client):
    """Create an experiment with a multi-span trace for agentic scorer testing."""
    mlflow.set_tracking_uri(client.server_url)

    experiment_name = f"test_agentic_scorer_{time.time()}"
    experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    # Create a trace with multiple spans (simulating an agentic workflow)
    with mlflow.start_span(name="agent_main") as parent_span:
        parent_span.set_inputs({"query": "What is the weather?"})

        with mlflow.start_span(name="tool_call_1") as tool_span1:
            tool_span1.set_inputs({"tool": "get_weather"})
            tool_span1.set_outputs({"temperature": 72})

        with mlflow.start_span(name="tool_call_2") as tool_span2:
            tool_span2.set_inputs({"tool": "format_response"})
            tool_span2.set_outputs({"message": "It's 72 degrees"})

        parent_span.set_outputs("The weather is 72 degrees")
        trace_id = parent_span.trace_id

    return experiment_id, trace_id


@pytest.fixture
def experiment_with_conversation_traces(client: Client):
    """Create an experiment with conversation traces from two different sessions."""
    mlflow.set_tracking_uri(client.server_url)

    experiment_name = f"test_conversation_scorer_{time.time()}"
    experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    session_1_id = f"session_1_{time.time()}"
    session_2_id = f"session_2_{time.time()}"

    session_1_trace_ids = []
    session_2_trace_ids = []

    # Session 1: Two turns
    for i, (user_msg, assistant_msg) in enumerate(
        [
            ("Hello, how are you?", "I'm doing well, thank you!"),
            ("What's your name?", "I'm an AI assistant."),
        ]
    ):
        with mlflow.start_span(name=f"session1_turn_{i}") as span:
            mlflow.update_current_trace(metadata={"mlflow.trace.session": session_1_id})
            span.set_inputs({"messages": [{"role": "user", "content": user_msg}]})
            span.set_outputs({"choices": [{"message": {"content": assistant_msg}}]})
            session_1_trace_ids.append(span.trace_id)

    # Session 2: Two turns (different conversation)
    for i, (user_msg, assistant_msg) in enumerate(
        [
            ("What's the weather?", "It's sunny today."),
            ("Thanks!", "You're welcome!"),
        ]
    ):
        with mlflow.start_span(name=f"session2_turn_{i}") as span:
            mlflow.update_current_trace(metadata={"mlflow.trace.session": session_2_id})
            span.set_inputs({"messages": [{"role": "user", "content": user_msg}]})
            span.set_outputs({"choices": [{"message": {"content": assistant_msg}}]})
            session_2_trace_ids.append(span.trace_id)

    return {
        "experiment_id": experiment_id,
        "session_1_id": session_1_id,
        "session_1_trace_ids": session_1_trace_ids,
        "session_2_id": session_2_id,
        "session_2_trace_ids": session_2_trace_ids,
    }


def test_invoke_agentic_scorer(client: Client, experiment_with_agentic_trace):
    experiment_id, trace_id = experiment_with_agentic_trace

    # Scorer using {{trace}} template variable (triggers tool-based flow)
    judge = make_judge(
        name="span_counter",
        instructions="Count spans in: {{ trace }}",
        model="gateway:/mock-endpoint",
        feedback_value_type=Literal["1", "2", "3", "4", "5"],
    )

    response = client.invoke_scorer(
        experiment_id=experiment_id,
        serialized_scorer=json.dumps(judge.model_dump()),
        trace_ids=[trace_id],
    )

    result = client.wait_job_succeeded(response["jobs"][0]["job_id"])["result"]
    assert result["success"] is True
    assert result["assessments"][trace_id][0]["assessment_name"] == "span_counter"
    assert result["assessments"][trace_id][0]["feedback"]["value"] == "3"


def test_invoke_conversation_scorer(client: Client, experiment_with_conversation_traces):
    fixture = experiment_with_conversation_traces
    session_1_trace_ids = fixture["session_1_trace_ids"]
    session_2_trace_ids = fixture["session_2_trace_ids"]

    # Scorer using {{conversation}} template variable (session-level)
    judge = make_judge(
        name="conversation_quality",
        instructions="Evaluate: {{ conversation }}",
        model="gateway:/mock-endpoint",
        feedback_value_type=Literal["Good", "Average", "Poor"],
    )

    response = client.invoke_scorer(
        experiment_id=fixture["experiment_id"],
        serialized_scorer=json.dumps(judge.model_dump()),
        trace_ids=session_1_trace_ids + session_2_trace_ids,
    )

    # 2 sessions -> 2 jobs, each grouped by session
    jobs = response["jobs"]
    assert len(jobs) == 2
    job_trace_sets = [set(j["trace_ids"]) for j in jobs]
    assert set(session_1_trace_ids) in job_trace_sets
    assert set(session_2_trace_ids) in job_trace_sets

    # Verify each session got expected response
    results_by_session = {}
    for job_info in jobs:
        result = client.wait_job_succeeded(job_info["job_id"])["result"]
        assert result["success"] is True
        # Session-level scorers log to first trace only
        assert len(result["assessments"]) == 1

        value = list(result["assessments"].values())[0][0]["feedback"]["value"]
        if set(job_info["trace_ids"]) == set(session_1_trace_ids):
            results_by_session["session_1"] = value
        else:
            results_by_session["session_2"] = value

    assert results_by_session == {"session_1": "Good", "session_2": "Average"}


def test_invoke_builtin_safety_scorer(client: Client, experiment_with_traces):
    experiment_id, trace_ids = experiment_with_traces
    trace_id = trace_ids[0]

    # Builtin Safety scorer (uses builtin_scorer_class)
    serialized_scorer = json.dumps(
        {
            "name": "safety",
            "aggregations": [],
            "description": None,
            "mlflow_version": "3.6.0rc0",
            "serialization_version": 1,
            "builtin_scorer_class": "Safety",
            "builtin_scorer_pydantic_data": {"name": "safety", "model": "gateway:/mock-endpoint"},
            "call_source": None,
            "call_signature": None,
            "original_func_name": None,
            "instructions_judge_pydantic_data": None,
        }
    )

    response = client.invoke_scorer(
        experiment_id=experiment_id,
        serialized_scorer=serialized_scorer,
        trace_ids=[trace_id],
    )

    result = client.wait_job_succeeded(response["jobs"][0]["job_id"])["result"]
    assert result["success"] is True
    assert result["assessments"][trace_id][0]["assessment_name"] == "safety"
    assert result["assessments"][trace_id][0]["feedback"]["value"].lower() in ("yes", "no")


def test_invoke_scorer_with_log_assessments(client: Client, experiment_with_traces):
    experiment_id, trace_ids = experiment_with_traces
    trace_id = trace_ids[0]

    judge = make_judge(
        name="answer_quality",
        instructions="Input: {{ inputs }}\nOutput: {{ outputs }}\nExpected: {{ expectations }}",
        model="gateway:/mock-endpoint",
        feedback_value_type=Literal["Yes", "No"],
    )

    response = client.invoke_scorer(
        experiment_id=experiment_id,
        serialized_scorer=json.dumps(judge.model_dump()),
        trace_ids=[trace_id],
        log_assessments=True,
    )

    job_result = client.wait_job(response["jobs"][0]["job_id"])
    assert job_result["status"] == "SUCCEEDED"

    # Get assessment ID from job result
    assessment_id = job_result["result"]["assessments"][trace_id][0]["assessment_id"]

    # Verify assessment was persisted to trace
    trace = mlflow.get_trace(trace_id)
    persisted = next(a for a in trace.info.assessments if a.assessment_id == assessment_id)
    assert persisted.name == "answer_quality"
    assert persisted.value == "Yes"


def test_invoke_scorer_partial_success(client: Client, experiment_with_traces):
    experiment_id, trace_ids = experiment_with_traces
    valid_trace_id = trace_ids[0]
    fake_trace_id = "tr-does-not-exist-00000000000000"

    judge = make_judge(
        name="answer_quality",
        instructions="Input: {{ inputs }}\nOutput: {{ outputs }}\nExpected: {{ expectations }}",
        model="gateway:/mock-endpoint",
        feedback_value_type=Literal["Yes", "No"],
    )

    response = client.invoke_scorer(
        experiment_id=experiment_id,
        serialized_scorer=json.dumps(judge.model_dump()),
        trace_ids=[valid_trace_id, fake_trace_id],
    )

    # Job succeeds but result has partial failure
    result = client.wait_job_succeeded(response["jobs"][0]["job_id"])["result"]
    assert result["success"] is False

    # Valid trace got assessment
    assert result["assessments"][valid_trace_id][0]["assessment_name"] == "answer_quality"

    # Invalid trace recorded as failure
    assert result["failures"][0]["trace_id"] == fake_trace_id
    assert result["failures"][0]["error_code"] == "TRACE_NOT_FOUND"
