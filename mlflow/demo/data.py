"""Demo data definitions for traces.

This module contains the trace data used by the demo generators.
Each trace has v1 (baseline) and v2 (improved) responses to demonstrate
agent improvement over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """Tool call with input/output for agent traces."""

    name: str
    input: dict[str, Any]
    output: dict[str, Any]


@dataclass
class DemoTrace:
    """Demo trace with query, two response versions, and expected ground truth.

    - v1_response: Initial/baseline agent output (less accurate, more verbose)
    - v2_response: Improved agent output (better quality, closer to expected)
    - expected_response: Ground truth for evaluation
    """

    query: str
    v1_response: str
    v2_response: str
    expected_response: str
    trace_type: str
    tools: list[ToolCall] = field(default_factory=list)
    session_id: str | None = None
    session_user: str | None = None
    turn_index: int | None = None


# =============================================================================
# RAG Traces (2 traces)
# =============================================================================

RAG_TRACES: list[DemoTrace] = [
    DemoTrace(
        query="What is MLflow Tracing?",
        v1_response=(
            "MLflow Tracing is a feature that helps you understand what's happening "
            "in your LLM applications. It captures information about your app's execution."
        ),
        v2_response=(
            "MLflow Tracing provides observability for LLM applications by capturing "
            "the execution flow as hierarchical spans, including inputs, outputs, and latency."
        ),
        expected_response=(
            "MLflow Tracing provides observability for LLM applications, capturing "
            "prompts, model calls, and tool invocations as hierarchical spans."
        ),
        trace_type="rag",
    ),
    DemoTrace(
        query="How do I evaluate LLM outputs?",
        v1_response=(
            "MLflow has an evaluate() function that runs your data through scorers "
            "and gives you metrics. The results get logged automatically."
        ),
        v2_response=(
            "Use mlflow.evaluate() with your model and scorers. Results include "
            "per-row scores and aggregate metrics, logged automatically to MLflow."
        ),
        expected_response=(
            "Use mlflow.evaluate() with your model and scorers. Results include "
            "per-row scores and aggregate metrics, all logged to MLflow."
        ),
        trace_type="rag",
    ),
]

# =============================================================================
# Agent Traces (2 traces)
# =============================================================================

AGENT_TRACES: list[DemoTrace] = [
    DemoTrace(
        query="What's the weather in San Francisco and should I bring an umbrella?",
        v1_response=(
            "The weather in San Francisco is currently 62°F with partly cloudy skies. "
            "There's a low chance of rain today."
        ),
        v2_response=(
            "It's 62°F and partly cloudy in San Francisco with a 15% chance of rain. "
            "You probably don't need an umbrella today!"
        ),
        expected_response=(
            "It's 62°F and partly cloudy in San Francisco with only a 15% chance of "
            "rain. You probably don't need an umbrella today!"
        ),
        trace_type="agent",
        tools=[
            ToolCall(
                name="get_weather",
                input={"city": "San Francisco"},
                output={"temp": 62, "condition": "partly cloudy", "rain_chance": 15},
            ),
        ],
    ),
    DemoTrace(
        query="Calculate the compound interest on $10,000 at 5% for 10 years",
        v1_response=(
            "Based on my calculation, $10,000 invested at 5% annual interest "
            "compounded yearly for 10 years would grow to approximately $16,289."
        ),
        v2_response=(
            "With compound interest, $10,000 at 5% annual interest for 10 years "
            "grows to $16,288.95 (calculated as 10000 × 1.05^10)."
        ),
        expected_response=(
            "With compound interest, $10,000 at 5% annual interest for 10 years "
            "grows to $16,288.95."
        ),
        trace_type="agent",
        tools=[
            ToolCall(
                name="calculator",
                input={"expression": "10000 * (1 + 0.05)^10"},
                output={"result": 16288.95},
            ),
        ],
    ),
]

# =============================================================================
# Session Traces (3 sessions x 2 turns = 6 traces)
# =============================================================================

SESSION_TRACES: list[DemoTrace] = [
    # Session 1: MLflow Setup (2 turns)
    DemoTrace(
        query="I'm new to MLflow. How do I get started?",
        v1_response=(
            "To get started with MLflow, first install it with pip. Then you can "
            "start using the tracking API to log your experiments."
        ),
        v2_response=(
            "Welcome! Install MLflow with `pip install mlflow`, then run `mlflow server` "
            "to launch the tracking server. Start logging with mlflow.start_run()."
        ),
        expected_response=(
            "Install MLflow with `pip install mlflow`, then run `mlflow server` to launch "
            "the tracking server. Start logging experiments with mlflow.start_run()."
        ),
        trace_type="session",
        session_id="session-mlflow-setup",
        session_user="alice@example.com",
        turn_index=1,
    ),
    DemoTrace(
        query="Great, I installed it. How do I log my first experiment?",
        v1_response=(
            "You can use mlflow.log_param() and mlflow.log_metric() to log "
            "parameters and metrics. Make sure to start a run first."
        ),
        v2_response=(
            "Use `with mlflow.start_run():` as a context manager, then call "
            "mlflow.log_param('learning_rate', 0.01) and mlflow.log_metric('accuracy', 0.95)."
        ),
        expected_response=(
            "Use `with mlflow.start_run():` then call mlflow.log_param() for "
            "hyperparameters and mlflow.log_metric() for results."
        ),
        trace_type="session",
        session_id="session-mlflow-setup",
        session_user="alice@example.com",
        turn_index=2,
    ),
    # Session 2: Model Deployment (2 turns)
    DemoTrace(
        query="How do I deploy a model with MLflow?",
        v1_response=(
            "First you need to log your model, then register it in the model registry. "
            "After that you can serve it using MLflow's serving capabilities."
        ),
        v2_response=(
            "Log your model with mlflow.sklearn.log_model() (or other flavors), register it "
            "with mlflow.register_model(), then serve with `mlflow models serve -m models:/name/1`."
        ),
        expected_response=(
            "Log your model with mlflow.<flavor>.log_model(), register with "
            "mlflow.register_model(), then serve with `mlflow models serve`."
        ),
        trace_type="session",
        session_id="session-deployment",
        session_user="bob@example.com",
        turn_index=1,
    ),
    DemoTrace(
        query="Can I deploy to Kubernetes instead?",
        v1_response=(
            "Yes, MLflow can deploy to Kubernetes. You'll need to build a Docker image "
            "and then deploy it to your cluster."
        ),
        v2_response=(
            "Yes! Use `mlflow models build-docker -m models:/name/1` to create a Docker image, "
            "then deploy to Kubernetes with your standard k8s deployment manifests."
        ),
        expected_response=(
            "Use `mlflow models build-docker` to create a container image, then deploy "
            "to Kubernetes with standard deployment manifests."
        ),
        trace_type="session",
        session_id="session-deployment",
        session_user="bob@example.com",
        turn_index=2,
    ),
    # Session 3: Evaluation (2 turns)
    DemoTrace(
        query="How do I evaluate my RAG application?",
        v1_response=(
            "MLflow has evaluation tools that can help you assess your RAG application. "
            "You can use built-in metrics or create custom ones."
        ),
        v2_response=(
            "Use mlflow.evaluate() with RAG-specific scorers like relevance() and "
            "faithfulness(). Pass your dataset and model, and results are logged automatically."
        ),
        expected_response=(
            "Use mlflow.evaluate() with relevance() and faithfulness() scorers. "
            "Results including per-row scores are logged to MLflow."
        ),
        trace_type="session",
        session_id="session-evaluation",
        session_user="carol@example.com",
        turn_index=1,
    ),
    DemoTrace(
        query="What if I need custom evaluation metrics?",
        v1_response=(
            "You can create custom scorers in MLflow. Define a function that takes "
            "inputs and outputs and returns a score."
        ),
        v2_response=(
            "Create custom scorers with @mlflow.scorer decorator or use make_genai_metric() "
            "for LLM-as-judge evaluations. Return a Score object with value and rationale."
        ),
        expected_response=(
            "Use @mlflow.scorer decorator for custom metrics or make_genai_metric() "
            "for LLM-as-judge evaluations."
        ),
        trace_type="session",
        session_id="session-evaluation",
        session_user="carol@example.com",
        turn_index=2,
    ),
]

# =============================================================================
# Combined Trace Data
# =============================================================================

ALL_DEMO_TRACES: list[DemoTrace] = RAG_TRACES + AGENT_TRACES + SESSION_TRACES


def get_expected_answers() -> dict[str, str]:
    """Build a dict mapping queries to expected responses for evaluation."""
    return {trace.query.lower(): trace.expected_response for trace in ALL_DEMO_TRACES}
