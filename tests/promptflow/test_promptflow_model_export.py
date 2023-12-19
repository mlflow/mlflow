from pathlib import Path

import pytest

from promptflow import load_flow
from promptflow._sdk.entities._flow import Flow

import mlflow
from mlflow import MlflowException


def get_promptflow_example_model():

    flow_path = Path(__file__).parent / "flow_with_additional_includes"
    return load_flow(flow_path)


def test_promptflow_log_and_load_model():
    model = get_promptflow_example_model()
    with mlflow.start_run():
        logged_model = mlflow.promptflow.log_model(
            model, "promptflow_model", input_example={"text": "Python Hello World!"}
        )

    loaded_model = mlflow.promptflow.load_model(logged_model.model_uri)

    assert "promptflow" in logged_model.flavors
    assert logged_model.signature is not None
    assert str(logged_model.signature.inputs) == "['text': string]"
    assert str(logged_model.signature.outputs) == "['output': string]"
    assert isinstance(loaded_model, Flow)


def test_log_model_with_config():
    model = get_promptflow_example_model()
    model_config = {"connection.provider": "local"}
    with mlflow.start_run():
        logged_model = mlflow.promptflow.log_model(model, "promptflow_model", model_config=model_config)

    assert mlflow.pyfunc.FLAVOR_NAME in logged_model.flavors
    assert mlflow.pyfunc.MODEL_CONFIG in logged_model.flavors[mlflow.pyfunc.FLAVOR_NAME]
    logged_model_config = logged_model.flavors[mlflow.pyfunc.FLAVOR_NAME][mlflow.pyfunc.MODEL_CONFIG]
    assert logged_model_config == model_config


def test_pyfunc_load_promptflow_model():
    model = get_promptflow_example_model()
    with mlflow.start_run():
        logged_model = mlflow.promptflow.log_model(model, "promptflow_model")

    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)

    assert "promptflow" in logged_model.flavors
    assert type(loaded_model) == mlflow.pyfunc.PyFuncModel


def test_promptflow_model_predict():
    model = get_promptflow_example_model()
    with mlflow.start_run():
        logged_model = mlflow.promptflow.log_model(model, "promptflow_model")
    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    input_value = "Python Hello World!"
    result = loaded_model.predict({"text": input_value})
    expected_result = f"Write a simple {input_value} program that displays the greeting message when executed.\n"
    assert result == {"output": expected_result}


def test_unsupported_class():
    mock_model = object()
    with pytest.raises(
        MlflowException,
        match="only supports instances loaded by ~promptflow.load_flow"
    ):
        with mlflow.start_run():
            mlflow.promptflow.log_model(mock_model, "mock_model_path")
