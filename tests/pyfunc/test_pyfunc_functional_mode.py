"""A module for testing pyfunc functional and class-based models."""
import os
import logging
from typing import List

import json
import pytest
import pandas as pd
import mlflow
import mlflow.pyfunc
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow.pyfunc.model import _FunctionPythonModel
from mlflow.deployments import PredictionsResponse

_logger = logging.getLogger(__name__)

# This test suite is included as a code dependency when testing PyFunc models in new
# processes and docker containers. In these environments, the `tests` module is not available.
# Therefore, we attempt to import from `tests` and gracefully emit a warning if it's unavailable.
try:
    from tests.helper_functions import pyfunc_serve_and_score_model
except ImportError:
    _logger.warning(
        "Failed to import test helper functions. Tests depending on these functions may fail!"
    )

@pytest.fixture
def model_path(tmp_path):
    """Return a temporary path to which we can log models."""
    return os.path.join(tmp_path, "model")

@pytest.fixture
def input_example():
    """Return an example input for testing models."""
    return ["a", "b", "c"]

@pytest.fixture
def output_example():
    """Return an example output for testing models."""
    return ["A", "B", "C"]

class TestPythonModelWithCallMethod(mlflow.pyfunc.PythonModel):
    """A class-based pyfunc model."""

    def predict(self, context, model_input, params=None):
        return self.__call__(model_input)

    def __call__(self, model_input):
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.values.flatten().tolist()
        return [x.upper() for x in model_input]

class TestFunctionalModelClassWithCallMethod:
    """A functional pyfunc model based on a class."""

    def __call__(self, model_input):
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.values.flatten().tolist()
        return [x.upper() for x in model_input]

def functional_model_fn(model_input: List[str]) -> List[str]:
    """A functional pyfunc model based on a function."""
    return [i.upper() for i in model_input]


def test_class_python_model_with_call_method(model_path, input_example, output_example):
    """
    Test that class-based pyfunc models with a __call__ method
    and inherited from PythonModel are PyFuncModel instances.
    """

    model = TestPythonModelWithCallMethod()
    assert (
        model.predict(None, input_example) == output_example
    ), f"Expected {output_example}, got {model.predict(None, input_example)}"
    mlflow.pyfunc.save_model(
        path=model_path, python_model=TestPythonModelWithCallMethod(), input_example=input_example
    )

    loaded_model = mlflow.pyfunc.load_model(model_path)

    assert (
        loaded_model.predict(input_example) == output_example
    ), f"Expected {output_example}, got {loaded_model.predict(input_example)}"
    assert isinstance(
        loaded_model, mlflow.pyfunc.PyFuncModel
    ), f"Expected mlflow.pyfunc.PyFuncModel, got {type(loaded_model)}"

    unwarpped_model = loaded_model.unwrap_python_model()
    assert isinstance(
        unwarpped_model, TestPythonModelWithCallMethod
    ), f"Expected TestClassModel, got {type(unwarpped_model)}"


def test_functional_model_class_with_call_method(model_path, input_example, output_example):
    """
    Test that class-based pyfunc models with a __call__ method
    and not inherited from PythonModel are _FunctionPythonModel instances.
    """

    model = TestFunctionalModelClassWithCallMethod()
    assert (
        model(input_example) == output_example
    ), f"Expected {output_example}, got {model(input_example)}"
    mlflow.pyfunc.save_model(model_path, python_model=model, input_example=input_example)

    loaded_model = mlflow.pyfunc.load_model(model_path)
    assert (
        loaded_model.predict(input_example) == output_example
    ), f"Expected {output_example}, got {loaded_model.predict(input_example)}"
    assert isinstance(
        loaded_model, mlflow.pyfunc.PyFuncModel
    ), f"Expected mlflow.pyfunc.PyFuncModel, got {type(loaded_model)}"

    unwarpped_model = loaded_model.unwrap_python_model()
    assert isinstance(
        unwarpped_model, _FunctionPythonModel
    ), f"Expected _FunctionPythonModel, got {type(unwarpped_model)}"


def test_functional_model_func(model_path, input_example, output_example):
    """
    Test that functional pyfunc models based on a function
    are _FunctionPythonModel instances.
    """

    assert functional_model_fn(input_example) == output_example
    mlflow.pyfunc.save_model(
        model_path, python_model=functional_model_fn, input_example=input_example
    )

    loaded_model = mlflow.pyfunc.load_model(model_path)
    assert (
        loaded_model.predict(input_example) == output_example
    ), f"Expected {output_example}, got {loaded_model.predict(input_example)}"
    assert isinstance(
        loaded_model, mlflow.pyfunc.PyFuncModel
    ), f"Expected mlflow.pyfunc.PyFuncModel, got {type(loaded_model)}"

    unwarpped_model = loaded_model.unwrap_python_model()
    assert isinstance(
        unwarpped_model, _FunctionPythonModel
    ), f"Expected _FunctionPythonModel, got {type(unwarpped_model)}"

@pytest.mark.skipcacheclean
def test_pyfunc_serve_class_python_model_with_call_method(input_example, output_example):
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=TestPythonModelWithCallMethod(),
            input_example=input_example
        )

    # Test inputs format
    inference_payload = json.dumps({"inputs": input_example})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert values == output_example, f"Expected {output_example}, got {values}"

@pytest.mark.skipcacheclean
def test_pyfunc_serve_functional_model_class_with_call_method(input_example, output_example):
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=TestFunctionalModelClassWithCallMethod(),
            input_example=input_example
        )

    # Test inputs format
    inference_payload = json.dumps({"inputs": input_example})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert values == output_example, f"Expected {output_example}, got {values}"

@pytest.mark.skipcacheclean
def test_pyfunc_serve_functional_model_func(input_example, output_example):
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=functional_model_fn,
            input_example=input_example
        )

    # Test inputs format
    inference_payload = json.dumps({"inputs": input_example})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert values == output_example, f"Expected {output_example}, got {values}"
