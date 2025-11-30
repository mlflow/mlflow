import importlib.metadata

import dspy
import pandas as pd
import pytest
from dspy.utils.dummies import DummyLM
from packaging.version import Version

import mlflow
import mlflow.utils
import mlflow.utils.autologging_utils
from mlflow.tracing.constant import TraceMetadataKey

from tests.openai.test_openai_evaluate import purge_traces
from tests.tracing.helper import get_traces, reset_autolog_state  # noqa: F401

if Version(importlib.metadata.version("dspy")) < Version("2.5.17"):
    pytest.skip("Evaluation test requires dspy>=2.5.17", allow_module_level=True)

_EVAL_DATA = pd.DataFrame(
    {
        "inputs": [
            "What is MLflow?",
            "What is Spark?",
        ],
        "ground_truth": [
            "MLflow is an open-source platform to manage the ML lifecycle.",
            "Spark is a unified analytics engine for big data processing.",
        ],
    }
)


def get_fake_model():
    dspy.settings.configure(
        lm=DummyLM(
            {
                "What is MLflow?": {
                    "answer": "MLflow is an open-source platform to manage the ML lifecycle.",
                    "reasoning": "No reasoning provided.",
                },
                "What is Spark?": {
                    "answer": "Spark is a unified analytics engine for big data processing.",
                    "reasoning": "No reasoning provided.",
                },
            }
        )
    )

    class CoT(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate_answer = dspy.ChainOfThought("question -> answer")

        def forward(self, question):
            prediction = self.generate_answer(question=question)
            return dspy.Prediction(answer=prediction.answer)

    return CoT()


@pytest.mark.parametrize(
    "config",
    [
        None,
        {"log_traces": False},
        {"log_traces": True},
    ],
)
@pytest.mark.usefixtures("reset_autolog_state")
def test_dspy_evaluate(config):
    if config:
        mlflow.dspy.autolog(**config)

    is_trace_disabled = config and not config.get("log_traces", True)
    is_trace_enabled = config and config.get("log_traces", True)

    cot = get_fake_model()

    def model(inputs):
        return [cot(question).answer for question in inputs["inputs"]]

    with mlflow.start_run() as run:
        eval_result = mlflow.evaluate(
            model,
            data=_EVAL_DATA,
            targets="ground_truth",
            extra_metrics=[mlflow.metrics.exact_match()],
        )
    assert eval_result.metrics["exact_match/v1"] == 1.0

    # Traces should not be logged when disabled explicitly
    if is_trace_disabled:
        assert len(get_traces()) == 0
    else:
        assert len(get_traces()) == 2
        assert run.info.run_id == get_traces()[0].info.request_metadata[TraceMetadataKey.SOURCE_RUN]

    purge_traces()

    # Test original autolog configs is restored
    cot(question="What is MLflow?")
    assert len(get_traces()) == (1 if is_trace_enabled else 0)


@pytest.mark.skip(
    reason="DSPy pyfunc wrapper does not support batch inputs, which is required for eval. "
    "Unskip when we add support for batch inputs."
)
@pytest.mark.usefixtures("reset_autolog_state")
def test_dspy_pyfunc_evaluate():
    with mlflow.start_run() as run:
        model_info = mlflow.dspy.log_model(get_fake_model(), name="model")
        eval_result = mlflow.evaluate(
            model_info.model_uri,
            data=_EVAL_DATA,
            targets="ground_truth",
            extra_metrics=[mlflow.metrics.exact_match()],
        )
    assert eval_result.metrics["exact_match/v1"] == 1.0

    # Traces should be automatically enabled during evaluation
    assert len(get_traces()) == 2
    assert run.info.run_id == get_traces()[0].info.request_metadata[TraceMetadataKey.SOURCE_RUN]


@pytest.mark.parametrize("globally_disabled", [True, False])
@pytest.mark.usefixtures("reset_autolog_state")
def test_dspy_evaluate_should_not_log_traces_when_disabled(globally_disabled):
    if globally_disabled:
        mlflow.autolog(disable=True)
    else:
        mlflow.dspy.autolog(disable=True)

    cot = get_fake_model()

    def model(inputs):
        return [cot(question).answer for question in inputs["inputs"]]

    with mlflow.start_run():
        eval_result = mlflow.evaluate(
            model,
            data=_EVAL_DATA,
            targets="ground_truth",
            extra_metrics=[mlflow.metrics.exact_match()],
        )
    assert eval_result.metrics["exact_match/v1"] == 1.0
    assert len(get_traces()) == 0
