"""
Smoke tests validating the TruLens documentation examples work end-to-end.

Requires:
    pip install trulens trulens-providers-litellm
    export OPENAI_API_KEY=...

Run:
    MLFLOW_TRACKING_URI=http://localhost:5000 \
        python -m pytest tests/genai/scorers/trulens/test_trulens_docs.py -v
"""

import pytest

import mlflow
from mlflow.genai.scorers.trulens import (
    AnswerRelevance,
    Coherence,
    ContextRelevance,
    ExecutionEfficiency,
    Groundedness,
    LogicalConsistency,
    PlanAdherence,
    PlanQuality,
    ToolCalling,
    ToolSelection,
    get_scorer,
)

MODEL = "openai:/gpt-5-mini"


@pytest.fixture(autouse=True)
def tracking_uri():
    mlflow.set_tracking_uri("http://localhost:5000")


# ── Direct RAG scorer calls ──────────────────────────────────────────


def test_groundedness():
    scorer = Groundedness(model=MODEL)
    feedback = scorer(
        inputs="What is MLflow?",
        outputs="MLflow is an open-source platform for managing machine learning workflows.",
        expectations={
            "context": "MLflow is an ML platform for experiment tracking and model deployment."
        },
    )
    assert feedback.value in ("yes", "no")
    assert feedback.error is None
    assert "score" in feedback.metadata


def test_context_relevance():
    scorer = ContextRelevance(model=MODEL)
    feedback = scorer(
        inputs="What is MLflow?",
        expectations={
            "context": "MLflow is an ML platform for experiment tracking and model deployment."
        },
    )
    assert feedback.value in ("yes", "no")
    assert feedback.error is None


def test_answer_relevance():
    scorer = AnswerRelevance(model=MODEL)
    feedback = scorer(
        inputs="What is MLflow?",
        outputs="MLflow is an open-source platform for managing machine learning workflows.",
    )
    assert feedback.value in ("yes", "no")
    assert feedback.error is None


def test_coherence():
    scorer = Coherence(model=MODEL)
    feedback = scorer(
        outputs="MLflow is an open-source platform. It provides experiment tracking, "
        "model versioning, and deployment capabilities.",
    )
    assert feedback.value in ("yes", "no")
    assert feedback.error is None


def test_groundedness_custom_threshold():
    scorer = Groundedness(model=MODEL, threshold=0.8)
    feedback = scorer(
        inputs="What is MLflow?",
        outputs="MLflow is an open-source platform.",
        expectations={"context": "MLflow is an ML platform."},
    )
    assert feedback.value in ("yes", "no")
    assert feedback.metadata["threshold"] == 0.8


# ── get_scorer ────────────────────────────────────────────────────────


def test_get_scorer_groundedness():
    scorer = get_scorer("Groundedness", model=MODEL)
    feedback = scorer(
        inputs="What is MLflow?",
        outputs="MLflow is a platform for ML workflows.",
        expectations={"context": "MLflow is an ML platform."},
    )
    assert feedback.value in ("yes", "no")
    assert feedback.error is None


def test_get_scorer_answer_relevance():
    scorer = get_scorer("AnswerRelevance", model=MODEL)
    feedback = scorer(
        inputs="What is MLflow?",
        outputs="MLflow is an open-source platform.",
    )
    assert feedback.value in ("yes", "no")
    assert feedback.error is None


# ── mlflow.genai.evaluate (RAG) ──────────────────────────────────────


def test_evaluate_rag_scorers():
    eval_dataset = [
        {
            "inputs": {"query": "What is MLflow?"},
            "outputs": "MLflow is an open-source platform for managing ML workflows.",
            "expectations": {
                "context": "MLflow is an ML platform for experiment tracking and deployment."
            },
        },
        {
            "inputs": {"query": "How do I track experiments?"},
            "outputs": "You can use mlflow.start_run() to begin tracking experiments.",
            "expectations": {
                "context": "MLflow provides APIs like mlflow.start_run() for experiment tracking."
            },
        },
    ]

    results = mlflow.genai.evaluate(
        data=eval_dataset,
        scorers=[
            Groundedness(model=MODEL),
            AnswerRelevance(model=MODEL),
        ],
    )
    assert results is not None
    assert results.metrics is not None


# ── Agent trace scorers ───────────────────────────────────────────────


@pytest.fixture
def sample_traces():
    """Create a simple traced function and collect traces for agent scorer testing."""

    @mlflow.trace
    def simple_agent(query: str) -> str:
        return f"Answer to: {query}"

    experiment_name = "test_trulens_agent_scorers"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        simple_agent("What is MLflow?")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    return mlflow.search_traces(experiment_ids=[experiment.experiment_id])


def test_logical_consistency(sample_traces):
    scorer = LogicalConsistency(model=MODEL)
    results = mlflow.genai.evaluate(
        data=sample_traces,
        scorers=[scorer],
    )
    assert results is not None


def test_execution_efficiency(sample_traces):
    scorer = ExecutionEfficiency(model=MODEL)
    results = mlflow.genai.evaluate(
        data=sample_traces,
        scorers=[scorer],
    )
    assert results is not None


def test_plan_adherence(sample_traces):
    scorer = PlanAdherence(model=MODEL)
    results = mlflow.genai.evaluate(
        data=sample_traces,
        scorers=[scorer],
    )
    assert results is not None


def test_plan_quality(sample_traces):
    scorer = PlanQuality(model=MODEL)
    results = mlflow.genai.evaluate(
        data=sample_traces,
        scorers=[scorer],
    )
    assert results is not None


def test_tool_selection(sample_traces):
    scorer = ToolSelection(model=MODEL)
    results = mlflow.genai.evaluate(
        data=sample_traces,
        scorers=[scorer],
    )
    assert results is not None


def test_tool_calling(sample_traces):
    scorer = ToolCalling(model=MODEL)
    results = mlflow.genai.evaluate(
        data=sample_traces,
        scorers=[scorer],
    )
    assert results is not None


def test_evaluate_multiple_agent_scorers(sample_traces):
    results = mlflow.genai.evaluate(
        data=sample_traces,
        scorers=[
            LogicalConsistency(model=MODEL),
            ToolSelection(model=MODEL),
        ],
    )
    assert results is not None
    assert results.metrics is not None
