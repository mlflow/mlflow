from pathlib import Path

import pytest

from promptflow import load_flow
from promptflow._sdk.entities._flow import Flow

import mlflow
from mlflow import MlflowException
from mlflow.openai.utils import _mock_request, _mock_chat_completion_response, TEST_CONTENT


def get_promptflow_example_model():

    flow_path = Path(__file__).parent / "flow_with_additional_includes"
    return load_flow(flow_path)


def test_promptflow_log_and_load_model():
    model = get_promptflow_example_model()
    with mlflow.start_run():
        logged_model = mlflow.promptflow.log_model(model, "promptflow_model")

    loaded_model = mlflow.promptflow.load_model(logged_model.model_uri)

    assert "promptflow" in logged_model.flavors
    assert str(logged_model.signature.inputs) == "['text': string]"
    assert str(logged_model.signature.outputs) == "['output': string]"

    assert isinstance(loaded_model, Flow)


def test_pyfunc_load_promptflow_model():
    model = get_promptflow_example_model()
    with mlflow.start_run():
        logged_model = mlflow.promptflow.log_model(model, "promptflow_model")

    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)

    assert "promptflow" in logged_model.flavors
    assert type(loaded_model) == mlflow.pyfunc.PyFuncModel


def test_promptflow_model_predict():
    with _mock_request(return_value=_mock_chat_completion_response()):
        model = get_promptflow_example_model()
        with mlflow.start_run():
            logged_model = mlflow.promptflow.log_model(model, "promptflow_model")
        loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
        result = loaded_model.predict([{"text": "Python Hello World!"}])
        assert result == [{"output": TEST_CONTENT}]


def test_unsupported_class():
    mock_model = object()
    with pytest.raises(
        MlflowException,
        match="only supports instances loaded by ~promptflow.load_flow"
    ):
        with mlflow.start_run():
            mlflow.promptflow.log_model(mock_model, "mock_model_path")
