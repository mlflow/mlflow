import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.python_api import (
    _CONTENT_TYPE_CSV,
    _CONTENT_TYPE_JSON,
    _serialize_input_data,
)
from mlflow.utils.env_manager import CONDA, VIRTUALENV


@pytest.mark.parametrize(
    ("input_data", "expected_data", "content_type"),
    [
        (
            "x,y\n1,3\n2,4",
            pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
            _CONTENT_TYPE_CSV,
        ),
        (
            {"inputs": {"a": [1]}},
            {"a": np.array([1])},
            _CONTENT_TYPE_JSON,
        ),
    ],
)
def test_predict(input_data, expected_data, content_type):
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            if type(model_input) == pd.DataFrame:
                assert model_input.equals(expected_data)
            else:
                assert model_input == expected_data
            return {}

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=TestModel(),
        )
        run_id = run.info.run_id

    mlflow.models.predict(
        model_uri=f"runs:/{run_id}/model",
        input_data=input_data,
        content_type=content_type,
        install_mlflow=True,
    )


@pytest.mark.parametrize(
    "env_manager",
    [VIRTUALENV, CONDA],
)
def test_predict_with_pip_requirements_override(env_manager):
    if env_manager == CONDA and sys.platform == "win32":
        pytest.skip("Skipping conda tests on Windows")

    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            # XGBoost should be installed by pip_requirements_override
            import xgboost

            assert xgboost.__version__ == "2.0.3"

            # Scikit-learn version should be overridden to 1.3.0 by pip_requirements_override
            import sklearn

            assert sklearn.__version__ == "1.3.0"

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=TestModel(),
            extra_pip_requirements=["scikit-learn==1.3.2"],
        )
        run_id = run.info.run_id

    mlflow.models.predict(
        model_uri=f"runs:/{run_id}/model",
        input_data={"inputs": [1, 2, 3]},
        content_type=_CONTENT_TYPE_JSON,
        pip_requirements_override=["xgboost==2.0.3", "scikit-learn==1.3.0"],
        env_manager=env_manager,
        install_mlflow=True,
    )


@pytest.fixture
def mock_backend():
    mock_backend = mock.MagicMock()
    with mock.patch("mlflow.models.python_api.get_flavor_backend", return_value=mock_backend):
        yield mock_backend


def test_predict_with_both_input_data_and_path_raise(mock_backend):
    with pytest.raises(MlflowException, match=r"Both input_data and input_path are provided"):
        mlflow.models.predict(
            model_uri="runs:/test/Model",
            input_data={"inputs": [1, 2, 3]},
            input_path="input.csv",
            content_type=_CONTENT_TYPE_CSV,
        )


def test_predict_invalid_content_type(mock_backend):
    with pytest.raises(MlflowException, match=r"Content type must be one of"):
        mlflow.models.predict(
            model_uri="runs:/test/Model",
            input_data={"inputs": [1, 2, 3]},
            content_type="any",
        )


def test_predict_with_input_none(mock_backend):
    mlflow.models.predict(
        model_uri="runs:/test/Model",
        content_type=_CONTENT_TYPE_CSV,
    )

    mock_backend.predict.assert_called_once_with(
        model_uri="runs:/test/Model",
        input_path=None,
        output_path=None,
        content_type=_CONTENT_TYPE_CSV,
        pip_requirements_override=None,
    )


@pytest.mark.parametrize(
    ("input_data", "content_type", "expected"),
    [
        # String (no change)
        ('{"inputs": [1, 2, 3]}', _CONTENT_TYPE_JSON, '{"inputs": [1, 2, 3]}'),
        ("x,y,z\n1,2,3\n4,5,6", _CONTENT_TYPE_CSV, "x,y,z\n1,2,3\n4,5,6"),
        # List
        ([1, 2, 3], _CONTENT_TYPE_CSV, "0\n1\n2\n3\n"),  # a header '0' is added by pandas
        ([[1, 2, 3], [4, 5, 6]], _CONTENT_TYPE_CSV, "0,1,2\n1,2,3\n4,5,6\n"),
        # Dict (pandas)
        (
            {
                "x": [
                    1,
                    2,
                ],
                "y": [3, 4],
            },
            _CONTENT_TYPE_CSV,
            "x,y\n1,3\n2,4\n",
        ),
        # Dict (json)
        ({"inputs": [1, 2, 3]}, _CONTENT_TYPE_JSON, '{"inputs": [1, 2, 3]}'),
        # Pandas DataFrame
        (pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}), _CONTENT_TYPE_CSV, "x,y\n1,4\n2,5\n3,6\n"),
    ],
)
def test_serialize_input_data(input_data, content_type, expected):
    assert _serialize_input_data(input_data, content_type) == expected


@pytest.mark.parametrize(
    ("input_data", "content_type"),
    [
        # Invalid input datatype for the content type
        (1, _CONTENT_TYPE_CSV),
        ({1, 2, 3}, _CONTENT_TYPE_CSV),
        (1, _CONTENT_TYPE_JSON),
        (True, _CONTENT_TYPE_JSON),
        ([1, 2, 3], _CONTENT_TYPE_JSON),
        # Invalid string
        ("{inputs: [1, 2, 3]}", _CONTENT_TYPE_JSON),
        ("x,y\n1,2\n3,4,5\n", _CONTENT_TYPE_CSV),
        # Invalid list
        ([[1, 2], [3, 4], 5], _CONTENT_TYPE_CSV),
        # Invalid dict (unserealizable)
        ({"x": 1, "y": {1, 2, 3}}, _CONTENT_TYPE_JSON),
    ],
)
def test_serialize_input_data_invalid_format(input_data, content_type):
    with pytest.raises(MlflowException):  # noqa: PT011
        _serialize_input_data(input_data, content_type)
