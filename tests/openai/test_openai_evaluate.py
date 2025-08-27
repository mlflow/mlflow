from unittest import mock

import openai
import pandas as pd
import pytest

import mlflow
from mlflow.models.evaluation import evaluate
from mlflow.tracing.constant import TraceMetadataKey

from tests.tracing.helper import get_traces, purge_traces, reset_autolog_state  # noqa: F401

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
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_BASE", mock_openai)
    return openai.OpenAI(api_key="test", base_url=mock_openai)


@pytest.mark.parametrize(
    "config",
    [
        None,
        {"log_traces": False},
        {"log_traces": True},
    ],
)
@pytest.mark.usefixtures("reset_autolog_state")
def test_openai_evaluate(client, config):
    if config:
        mlflow.openai.autolog(**config)

    is_trace_disabled = config and not config.get("log_traces", True)
    is_trace_enabled = config and config.get("log_traces", True)

    def model(inputs):
        return [
            client.chat.completions.create(
                messages=[{"role": "user", "content": question}],
                model="gpt-4o-mini",
                temperature=0.0,
            )
            .choices[0]
            .message.content
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

    # Traces should not be logged when disabled explicitly
    if is_trace_disabled:
        assert len(get_traces()) == 0
    else:
        assert len(get_traces()) == 2
        assert run.info.run_id == get_traces()[0].info.request_metadata[TraceMetadataKey.SOURCE_RUN]

    purge_traces()

    # Test original autolog configs is restored
    client.chat.completions.create(
        messages=[{"role": "user", "content": "hi"}], model="gpt-4o-mini"
    )

    assert len(get_traces()) == (1 if is_trace_enabled else 0)


@pytest.mark.usefixtures("reset_autolog_state")
def test_openai_pyfunc_evaluate(client):
    with mlflow.start_run() as run:
        model_info = mlflow.openai.log_model(
            "gpt-4o-mini",
            "chat.completions",
            name="model",
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


@pytest.mark.parametrize("globally_disabled", [True, False])
@pytest.mark.usefixtures("reset_autolog_state")
def test_openai_evaluate_should_not_log_traces_when_disabled(client, globally_disabled):
    if globally_disabled:
        mlflow.autolog(disable=True)
    else:
        mlflow.openai.autolog(disable=True)

    def model(inputs):
        return [
            client.chat.completions.create(
                messages=[{"role": "user", "content": question}],
                model="gpt-4o-mini",
                temperature=0.0,
            )
            .choices[0]
            .message.content
            for question in inputs["inputs"]
        ]

    with mlflow.start_run():
        evaluate(
            model,
            data=_EVAL_DATA,
            targets="ground_truth",
            extra_metrics=[mlflow.metrics.exact_match()],
        )

    assert len(get_traces()) == 0
