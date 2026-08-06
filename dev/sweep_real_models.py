"""Run `evaluate_sweep` against real Databricks serving endpoints.

Unlike the other sweep fixtures in this directory, this one calls real models and a real LLM judge,
so it costs tokens and takes a few minutes. It is the end-to-end check: generation variance,
judge variance, trace latency, and token-usage-derived cost are all genuine.

Requires a Databricks workspace with foundation-model endpoints:

    databricks auth login --profile e2-dogfood
    DATABRICKS_CONFIG_PROFILE=e2-dogfood \
      MLFLOW_TRACKING_URI=http://localhost:5000 \
      uv run --no-sync python dev/sweep_real_models.py
"""

import os
import sys

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from mlflow.genai.scorers import Correctness, scorer
from mlflow.tracing.constant import SpanAttributeKey

EXPERIMENT_NAME = "evaluate_sweep_real_models"

# A strong model judges, so judge quality is not the thing under test. Pinned explicitly rather
# than relying on the "databricks" managed judge, which needs the agents package installed.
JUDGE_MODEL = "databricks:/databricks-claude-sonnet-4"

# Built-in judges send a structured-output tool schema that some serving endpoints reject
# ("tools.0.custom.strict: Extra inputs are not permitted"), which fails every scorer call and
# leaves the sweep with no scores. Default to a code-based scorer so the sweep always produces
# quality numbers from real generations; pass --judge to use the LLM judge instead.
USE_LLM_JUDGE = "--judge" in sys.argv

# Deliberately spread across the quality/cost/latency range, so the comparison has something to
# show rather than three near-identical points.
# Each config maps to a serving endpoint plus the model name LiteLLM prices it under. The
# endpoint alias ("databricks-...") is not in any pricing catalogue, so cost needs the real name.
ENDPOINTS = {
    "llama-3.1-8b": ("databricks-meta-llama-3-1-8b-instruct", "databricks/databricks-meta-llama-3-1-8b-instruct"),
    "gemini-2.5-flash": ("databricks-gemini-2-5-flash", "gemini/gemini-2.5-flash"),
    "claude-sonnet-4": ("databricks-claude-sonnet-4", "anthropic/claude-sonnet-4-20250514"),
}

DATA = [
    {
        "inputs": {"question": "What does MLflow Tracking record about a machine learning run?"},
        "expectations": {
            "expected_response": "Parameters, metrics, tags, and artifacts for each run, so runs can be compared.",
            "keywords": ["metric", "param"],
        },
    },
    {
        "inputs": {"question": "In one sentence, what is an MLflow trace?"},
        "expectations": {
            "expected_response": "A record of a GenAI request's execution, made of spans capturing inputs, outputs, and timing.",
            "keywords": ["span"],
        },
    },
    {
        "inputs": {"question": "What is the MLflow Model Registry for?"},
        "expectations": {
            "expected_response": "Versioning registered models and moving those versions through lifecycle stages.",
            "keywords": ["version"],
        },
    },
]


@scorer
def keyword_coverage(outputs, expectations) -> float:
    """Fraction of the expected keywords the answer mentions.

    A deterministic stand-in for a judge: it still measures the real model's real output, and
    differing model verbosity makes the scores genuinely differ between configs.
    """
    keywords = expectations.get("keywords", [])
    if not keywords:
        return 0.0
    answer = str(outputs).lower()
    return sum(1 for keyword in keywords if keyword in answer) / len(keywords)


@scorer
def conciseness(outputs) -> float:
    """1.0 for a one-sentence answer, tapering off as the model over-explains.

    A second, deliberately uncorrelated scorer: models that score well on coverage by saying more
    tend to score worse here, so the per-scorer charts show genuinely different rankings.
    """
    words = len(str(outputs).split())
    if words == 0:
        return 0.0
    return max(0.0, min(1.0, 30 / words))


@scorer
def mentions_mlflow(outputs) -> bool:
    """Whether the answer names MLflow at all — a pass/fail scorer, so its interval uses Wilson."""
    return "mlflow" in str(outputs).lower()


def make_predict_fn(endpoint: str, model_name: str):
    client = WorkspaceClient().serving_endpoints

    def predict_fn(question: str) -> str:
        response = client.query(
            name=endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content="Answer in one short sentence."),
                ChatMessage(role=ChatMessageRole.USER, content=question),
            ],
            max_tokens=120,
        )

        # The Databricks SDK's `query` is not autologged, so nothing records the token usage it
        # returns and MLflow has no basis to price the trace. Put the usage on the span ourselves,
        # which is what an autologged client would do, so cost is computed from real token counts.
        span = mlflow.get_current_active_span()
        usage = response.usage
        if span is not None and usage is not None:
            span.set_attribute(SpanAttributeKey.MODEL, model_name)
            span.set_attribute(
                SpanAttributeKey.CHAT_USAGE,
                {
                    "input_tokens": usage.prompt_tokens or 0,
                    "output_tokens": usage.completion_tokens or 0,
                    "total_tokens": usage.total_tokens or 0,
                },
            )
        return response.choices[0].message.content

    return predict_fn


def main() -> None:
    if not os.environ.get("DATABRICKS_CONFIG_PROFILE") and not os.environ.get("DATABRICKS_HOST"):
        raise SystemExit("Set DATABRICKS_CONFIG_PROFILE (or DATABRICKS_HOST/TOKEN) first.")

    mlflow.set_experiment(EXPERIMENT_NAME)
    # Autologging captures each endpoint call as a span carrying the model name and token usage,
    # which is what lets MLflow price the trace.
    mlflow.openai.autolog()

    result = mlflow.genai.evaluate_sweep(
        data=DATA,
        scorers=(
            [Correctness(model=JUDGE_MODEL), keyword_coverage, conciseness, mentions_mlflow]
            if USE_LLM_JUDGE
            else [keyword_coverage, conciseness, mentions_mlflow]
        ),
        predict_fns={
            name: make_predict_fn(endpoint, model_name)
            for name, (endpoint, model_name) in ENDPOINTS.items()
        },
        n_repeats=3,
    )

    print(f"\nparent_run_id: {result.parent_run_id}")
    for name, config in result.configs.items():
        intervals = {s: (round(i.mean, 3), i.method) for s, i in config.scorer_intervals.items()}
        print(f"\n{name}")
        print(f"  scorers:  {intervals}")
        print(f"  latency:  {config.latency}")
        print(f"  cost:     {config.cost}")
        if config.failed_repeats:
            print(f"  failed repeats: {config.failed_repeats}")


if __name__ == "__main__":
    main()
