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
from mlflow.utils.autologging_utils import (
    MLFLOW_EVALUATE_RESTRICT_LANGCHAIN_AUTOLOG_TO_TRACES_CONFIG,
)

from tests.tracing.helper import get_traces

INFERENCE_FILE_NAME = "inference_inputs_outputs.json"


def test_langchain_evaluate_autologs_traces():
    # Check langchain autolog parameters are restored after evaluation
    mlflow.langchain.autolog(log_models=True)

    prompt = PromptTemplate(
        input_variables=["input"],
        template="Test Prompt {input}",
    )
    llm = FakeListLLM(responses=["response"])
    chain = prompt | llm

    with mock.patch("mlflow.langchain.log_model") as log_model_mock:

        def model(inputs):
            return [chain.invoke({"input": input}) for input in inputs["inputs"]]

        eval_data = pd.DataFrame(
            {
                "inputs": [
                    "What is MLflow?",
                    "What is Spark?",
                ],
                "ground_truth": ["What is MLflow?", "Not what is Spark?"],
            }
        )

        with mlflow.start_run() as run:
            evaluate(
                model,
                eval_data,
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
        with mlflow.start_run() as run:
            chain.invoke({"input": "text"})

        log_model_mock.assert_called_once()
        assert len(get_traces()) == 3
        assert len(get_traces()[0].data.spans) == 3


def test_langchain_pyfunc_autologs_traces():
    prompt = PromptTemplate(
        input_variables=["inputs"],
        template="Test Prompt {inputs}",
    )
    llm = FakeListLLM(responses=["response"])
    chain = prompt | llm

    eval_data = pd.DataFrame(
        {
            "inputs": ["What is MLflow?"],
            "ground_truth": ["What is MLflow?"],
        }
    )

    with mlflow.start_run() as run:
        model_info = mlflow.langchain.log_model(chain, "model")
        evaluate(
            model_info.model_uri,
            eval_data,
            targets="ground_truth",
            extra_metrics=[mlflow.metrics.exact_match()],
        )
    assert len(get_traces()) == 1
    assert len(get_traces()[0].data.spans) == 3
    assert run.info.run_id == get_traces()[0].info.request_metadata[TraceMetadataKey.SOURCE_RUN]


def test_langchain_evaluate_fails_with_an_exception():
    # Check langchain autolog parameters are restored after evaluation
    mlflow.langchain.autolog(log_models=True)

    prompt = PromptTemplate(
        input_variables=["input"],
        template="Test Prompt {input}",
    )
    llm = FakeListLLM(responses=["response"])
    chain = prompt | llm

    with (
        mock.patch("mlflow.langchain.log_model") as log_model_mock,
        mock.patch.object(
            DefaultEvaluator, "evaluate", side_effect=MlflowException("evaluate mock error")
        ),
    ):

        def model(inputs):
            return [chain.invoke({"input": input}) for input in inputs["inputs"]]

        eval_data = pd.DataFrame(
            {
                "inputs": [
                    "What is MLflow?",
                    "What is Spark?",
                ],
                "ground_truth": ["What is MLflow?", "Not what is Spark?"],
            }
        )
        with mlflow.start_run():
            with pytest.raises(MlflowException, match="evaluate mock error"):
                evaluate(
                    model,
                    eval_data,
                    targets="ground_truth",
                    extra_metrics=[mlflow.metrics.exact_match()],
                )
            log_model_mock.assert_not_called()

    assert len(get_traces()) == 0

    TRACE_BUFFER.clear()

    # Test original langchain autolog configs is restored
    with mock.patch("mlflow.langchain.log_model") as log_model_mock:
        with mlflow.start_run():
            chain.invoke({"input": "text"})

        log_model_mock.assert_called_once()
        assert len(get_traces()) == 1
        assert len(get_traces()[0].data.spans) == 3


def test_langchain_autolog_parameters_matches_default_parameters():
    # The custom config is to restrict langchain autologging to only log traces.
    # The parameters in this configuration should match the signature of
    # mlflow.langchain.autolog exactly. The values of the parameters should be set
    # in a way that disables logging anything but traces.
    params = inspect.signature(mlflow.langchain.autolog).parameters
    for name in params:
        assert name in MLFLOW_EVALUATE_RESTRICT_LANGCHAIN_AUTOLOG_TO_TRACES_CONFIG
    for name in MLFLOW_EVALUATE_RESTRICT_LANGCHAIN_AUTOLOG_TO_TRACES_CONFIG:
        assert name in params


def test_evaluate_works_with_no_langchain_installed():
    with mock.patch.dict("sys.modules", {"langchain": None}):
        # Import within the test context
        with pytest.raises(ImportError, match="import of langchain halted"):
            import langchain  # noqa: F401
        eval_data = pd.DataFrame(
            {
                "inputs": [1],
                "ground_truth": [1],
            }
        )

        @mlflow.trace
        def model(inputs):
            return inputs

        evaluate(
            model, eval_data, targets="ground_truth", extra_metrics=[mlflow.metrics.exact_match()]
        )
        assert len(get_traces()) == 1
