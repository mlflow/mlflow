from unittest import mock

import openai
import pandas as pd
import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation import evaluate
from mlflow.models.evaluation.evaluators.default import DefaultEvaluator
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.fluent import TRACE_BUFFER

from tests.tracing.helper import get_traces


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


@pytest.fixture
def client(monkeypatch, mock_openai):
    monkeypatch.setenvs(
        {
            "OPENAI_API_KEY": "test",
            "OPENAI_API_BASE": mock_openai,
        }
    )
    return openai.OpenAI(api_key="test", base_url=mock_openai)


@pytest.mark.parametrize(
    "original_autolog_config", [
        None,
        {"log_traces": False},
        {"log_models": True},
        {"log_traces": False, "log_models": False},
    ]
)
def test_openai_evaluate(client, original_autolog_config):
    if original_autolog_config:
        mlflow.openai.autolog(**original_autolog_config)

    def model(inputs):
        return [
            client.chat.completions.create(
                messages=[{"role": "user", "content": question}],
                model="gpt-4o-mini",
                temperature=0.0,
            ).choices[0].message.content
            for question in inputs["inputs"]
        ]

    with mock.patch("mlflow.openai.log_model") as log_model_mock:
        with mlflow.start_run() as run:
            evaluate(
                model,
                data=_EVAL_DATA,
                targets="ground_truth",
                extra_metrics=[mlflow.metrics.exact_match()],
            )
        log_model_mock.assert_not_called()

    assert len(get_traces()) == 2
    assert run.info.run_id == get_traces()[0].info.request_metadata[TraceMetadataKey.SOURCE_RUN]

    # Test original autolog configs is restored
    with mock.patch("mlflow.openai.log_model") as log_model_mock:
        client.chat.completions.create(messages=[{"role": "user", "content": "hi"}], model="gpt-4o-mini")

        if original_autolog_config and original_autolog_config.get("log_models", False):
            log_model_mock.assert_called_once()

        if original_autolog_config and original_autolog_config.get("log_traces", True):
            assert len(get_traces()) == 3
        else:
            assert len(get_traces()) == 2


def test_openai_pyfunc_evaluate(client):
    with mlflow.start_run() as run:
        model_info = mlflow.openai.log_model(
            model="gpt-4o-mini",
            task="chat.completions",
            artifact_path="model",
            messages=[{"role": "system", "content": "You are an MLflow expert."}],
        )

        evaluate(
            model_info.model_uri,
            data=_EVAL_DATA,
            targets="ground_truth",
            extra_metrics=[mlflow.metrics.exact_match()],
        )
    assert len(get_traces()) == 2
    assert run.info.run_id == get_traces()[0].info.request_metadata[TraceMetadataKey.SOURCE_RUN]


def test_openai_evaluate_should_not_log_traces_when_disabled(client):
    mlflow.openai.autolog(disable=True)  # Our chain contains OpenAI call as well

    def model(inputs):
        return [
            client.chat.completions.create(
                messages=[{"role": "user", "content": question}],
                model="gpt-4o-mini",
                temperature=0.0,
            ).choices[0].message.content
            for question in inputs["inputs"]
        ]

    with mlflow.start_run() as run:
        evaluate(
            model,
            data=_EVAL_DATA,
            targets="ground_truth",
            extra_metrics=[mlflow.metrics.exact_match()],
        )

    assert len(get_traces()) == 0
