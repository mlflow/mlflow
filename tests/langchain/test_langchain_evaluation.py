import inspect
from unittest import mock

import pandas as pd
import pytest
from langchain.prompts import PromptTemplate
from langchain_community.llms import FakeListLLM

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation import evaluate
from mlflow.models.evaluation.evaluators.default import DefaultEvaluator
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.fluent import TRACE_BUFFER

from tests.tracing.helper import get_traces


_EVAL_DATA = pd.DataFrame(
    {
        "question": [
            "What is MLflow?",
            "What is Spark?",
        ],
        "ground_truth": ["What is MLflow?", "Not what is Spark?"],
    }
)

def create_fake_chain():
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Answer user's question: {question}",
    )
    llm = FakeListLLM(responses=["response"])
    return prompt | llm


@pytest.fixture(autouse=True)
def reset_autolog_state():
    # Autologging state is global, so we need to reset it between tests
    mlflow.langchain.autolog(disable=True)
    # Reset the fact that it is 'disabled'
    del mlflow.utils.autologging_utils.AUTOLOGGING_INTEGRATIONS["langchain"]



@pytest.mark.parametrize(
    "original_autolog_config", [
        None,
        {"log_traces": False},
        {"log_models": True},
        {"log_traces": False, "log_models": False},
    ]
)
def test_langchain_evaluate(original_autolog_config):
    if original_autolog_config:
        mlflow.langchain.autolog(**original_autolog_config)

    chain = create_fake_chain()

    with mock.patch("mlflow.langchain.log_model") as log_model_mock:

        def model(inputs):
            return [chain.invoke({"question": input}) for input in inputs["question"]]

        with mlflow.start_run() as run:
            evaluate(
                model,
                data=_EVAL_DATA,
                targets="ground_truth",
                extra_metrics=[mlflow.metrics.exact_match()],
            )
        log_model_mock.assert_not_called()

    assert len(get_traces()) == 2
    for trace in get_traces():
        assert len(trace.data.spans) == 3
    assert run.info.run_id == get_traces()[0].info.request_metadata[TraceMetadataKey.SOURCE_RUN]

    # Test original langchain autolog configs is restored
    with mock.patch("mlflow.langchain.log_model") as log_model_mock:
        chain.invoke({"question": "text"})

        if original_autolog_config and original_autolog_config.get("log_models", False):
            log_model_mock.assert_called_once()

        if original_autolog_config and original_autolog_config.get("log_traces", True):
            assert len(get_traces()) == 3
        else:
            assert len(get_traces()) == 2


def test_langchain_evaluate_fails_with_an_exception():
    # Check langchain autolog parameters are restored after evaluation
    mlflow.langchain.autolog(log_models=True)

    chain = create_fake_chain()

    with (
        mock.patch("mlflow.langchain.log_model") as log_model_mock,
        mock.patch.object(
            DefaultEvaluator, "evaluate", side_effect=MlflowException("evaluate mock error")
        ),
    ):
        def model(inputs):
            return [chain.invoke({"question": input}) for input in inputs["question"]]

        with mlflow.start_run():
            with pytest.raises(MlflowException, match="evaluate mock error"):
                evaluate(
                    model,
                    data=_EVAL_DATA,
                    targets="ground_truth",
                    extra_metrics=[mlflow.metrics.exact_match()],
                )
            log_model_mock.assert_not_called()

    assert len(get_traces()) == 0

    # Test original langchain autolog configs is restored
    with mock.patch("mlflow.langchain.log_model") as log_model_mock:
        chain.invoke({"question": "text"})

        log_model_mock.assert_called_once()
        assert len(get_traces()) == 1
        assert len(get_traces()[0].data.spans) == 3


def test_langchain_pyfunc_evaluate():
    chain = create_fake_chain()

    with mlflow.start_run() as run:
        model_info = mlflow.langchain.log_model(chain, "model")
        evaluate(
            model_info.model_uri,
            data=_EVAL_DATA,
            targets="ground_truth",
            extra_metrics=[mlflow.metrics.exact_match()],
        )
    assert len(get_traces()) == 2
    assert len(get_traces()[0].data.spans) == 3
    assert run.info.run_id == get_traces()[0].info.request_metadata[TraceMetadataKey.SOURCE_RUN]


def test_langchain_evaluate_should_not_log_traces_when_disabled():
    mlflow.langchain.autolog(disable=True)

    chain = create_fake_chain()

    def model(inputs):
        return [chain.invoke({"question": input}) for input in inputs["question"]]

    with mlflow.start_run() as run:
        evaluate(
            model,
            data=_EVAL_DATA,
            targets="ground_truth",
            extra_metrics=[mlflow.metrics.exact_match()],
        )

    assert len(get_traces()) == 0
