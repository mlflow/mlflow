import pandas as pd
import pytest

import mlflow
from mlflow.metrics import latency
from mlflow.tracing.constant import TraceMetadataKey

import mlflow.utils
import mlflow.utils.autologging_utils
from tests.tracing.helper import get_traces


_EVAL_DATA = pd.DataFrame(
    {
        "inputs": [
            "What is MLflow?",
            "What is Spark?",
        ],
        "ground_truth": ["What is MLflow?", "Not what is Spark?"],
    }
)


@pytest.fixture(autouse=True)
def reset_autolog_state():
    # Autologging state is global, so we need to reset it between tests
    mlflow.llama_index.autolog(disable=True)
    # Reset the fact that it is 'disabled'
    del mlflow.utils.autologging_utils.AUTOLOGGING_INTEGRATIONS["llama_index"]


@pytest.mark.parametrize(
    "original_autolog_config", [
        None,
        {"log_traces": False},
        {"log_traces": True},
    ]
)
def test_llama_index_evaluate(single_index, original_autolog_config):
    if original_autolog_config:
        mlflow.llama_index.autolog(**original_autolog_config)

    engine = single_index.as_query_engine()

    def model(inputs):
        return [engine.query(question) for question in inputs["inputs"]]

    with mlflow.start_run() as run:
        eval_result = mlflow.evaluate(
            model,
            data=_EVAL_DATA,
            targets="ground_truth",
            extra_metrics=[latency()],
        )
    assert eval_result.metrics["latency/mean"] > 0

    # Traces should be automatically enabled during evaluation
    assert len(get_traces()) == 2
    assert run.info.run_id == get_traces()[0].info.request_metadata[TraceMetadataKey.SOURCE_RUN]

    # Test original autolog configs is restored
    engine.query("text")
    if original_autolog_config and original_autolog_config.get("log_traces", True):
        assert len(get_traces()) == 3
    else:
        assert len(get_traces()) == 2


@pytest.mark.parametrize("engine_type", ["query", "chat"])
def test_llama_index_pyfunc_evaluate(engine_type, single_index):
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            single_index,
            "llama_index",
            engine_type=engine_type,
        )

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)

    with mlflow.start_run() as run:
        eval_result = mlflow.evaluate(
            pyfunc_model,
            data=_EVAL_DATA,
            targets="ground_truth",
            extra_metrics=[latency()],
        )
    assert eval_result.metrics["latency/mean"] > 0

    # Traces should be automatically enabled during evaluation
    assert len(get_traces()) == 2
    assert run.info.run_id == get_traces()[0].info.request_metadata[TraceMetadataKey.SOURCE_RUN]



def test_llama_index_evaluate_should_not_log_traces_when_disabled(single_index):
    mlflow.llama_index.autolog(disable=True)

    def model(inputs):
        engine = single_index.as_query_engine()
        return [engine.query(question) for question in inputs["inputs"]]

    with mlflow.start_run() as run:
        eval_result = mlflow.evaluate(
            model,
            data=_EVAL_DATA,
            targets="ground_truth",
            extra_metrics=[latency()],
        )
    assert eval_result.metrics["latency/mean"] > 0
    assert len(get_traces()) == 0
