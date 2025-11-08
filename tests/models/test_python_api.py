import datetime
import json
import os
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.python_api import (
    _CONTENT_TYPE_CSV,
    _CONTENT_TYPE_JSON,
    _serialize_input_data,
)
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.utils.env_manager import CONDA, LOCAL, UV, VIRTUALENV

from tests.tracing.helper import get_traces


@pytest.mark.parametrize(
    ("input_data", "expected_data", "content_type"),
    [
        (
            "x,y\n1,3\n2,4",
            pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
            _CONTENT_TYPE_CSV,
        ),
        (
            {"a": [1]},
            {"a": np.array([1])},
            _CONTENT_TYPE_JSON,
        ),
        (
            1,
            np.array(1),
            _CONTENT_TYPE_JSON,
        ),
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            _CONTENT_TYPE_JSON,
        ),
        (
            scipy.sparse.csc_matrix([[1, 2], [3, 4]]),
            np.array([[1, 2], [3, 4]]),
            _CONTENT_TYPE_JSON,
        ),
        (
            # uLLM input, no change
            {"input": "some_data"},
            {"input": "some_data"},
            _CONTENT_TYPE_JSON,
        ),
    ],
)
@pytest.mark.parametrize(
    "env_manager",
    [VIRTUALENV, UV],
)
def test_predict(input_data, expected_data, content_type, env_manager):
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            if isinstance(model_input, pd.DataFrame):
                assert model_input.equals(expected_data)
            elif isinstance(model_input, np.ndarray):
                assert np.array_equal(model_input, expected_data)
            else:
                assert model_input == expected_data
            return {}

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=TestModel(),
            extra_pip_requirements=["pytest"],
        )

    mlflow.models.predict(
        model_uri=model_info.model_uri,
        input_data=input_data,
        content_type=content_type,
        env_manager=env_manager,
    )


@pytest.mark.parametrize(
    "env_manager",
    [VIRTUALENV, CONDA, UV],
)
def test_predict_with_pip_requirements_override(env_manager):
    if env_manager == CONDA:
        if sys.platform == "win32":
            pytest.skip("Skipping conda tests on Windows")

    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            # XGBoost should be installed by pip_requirements_override
            import xgboost

            assert xgboost.__version__ == "1.7.3"

            # Scikit-learn version should be overridden to 1.3.0 by pip_requirements_override
            import sklearn

            assert sklearn.__version__ == "1.3.0"

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=TestModel(),
            extra_pip_requirements=["scikit-learn==1.3.2", "pytest"],
        )

    requirements_override = ["xgboost==1.7.3", "scikit-learn==1.3.0"]
    if env_manager == CONDA:
        # Install charset-normalizer with conda-forge to work around pip-vs-conda issue during
        # CI tests. At the beginning of the CI test, it installs MLflow dependencies via pip,
        # which includes charset-normalizer. Then when it runs this test case, the conda env
        # is created but charset-normalizer is installed via the default channel, which is one
        # major version behind the version installed via pip (as of 2024 Jan). As a result,
        # Python env confuses pip and conda versions and cause errors like "ImportError: cannot
        # import name 'COMMON_SAFE_ASCII_CHARACTERS' from 'charset_normalizer.constant'".
        # To work around this, we install the latest cversion from the conda-forge.
        # TODO: Implement better isolation approach for pip and conda environments during testing.
        requirements_override.append("conda-forge::charset-normalizer")

    mlflow.models.predict(
        model_uri=model_info.model_uri,
        input_data={"inputs": [1, 2, 3]},
        content_type=_CONTENT_TYPE_JSON,
        pip_requirements_override=requirements_override,
        env_manager=env_manager,
    )


@pytest.mark.parametrize("env_manager", [VIRTUALENV, CONDA, UV])
def test_predict_with_model_alias(env_manager):
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            assert os.environ["TEST"] == "test"
            return model_input

    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            name="model",
            python_model=TestModel(),
            registered_model_name="model_name",
        )
    client = mlflow.MlflowClient()
    client.set_registered_model_alias("model_name", "test_alias", 1)

    mlflow.models.predict(
        model_uri="models:/model_name@test_alias",
        input_data="abc",
        env_manager=env_manager,
        extra_envs={"TEST": "test"},
    )


@pytest.mark.parametrize("env_manager", [VIRTUALENV, CONDA, UV])
def test_predict_with_extra_envs(env_manager):
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            assert os.environ["TEST"] == "test"
            return model_input

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=TestModel(),
        )

    mlflow.models.predict(
        model_uri=model_info.model_uri,
        input_data="abc",
        content_type=_CONTENT_TYPE_JSON,
        env_manager=env_manager,
        extra_envs={"TEST": "test"},
    )


def test_predict_with_extra_envs_errors():
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            assert os.environ["TEST"] == "test"
            return model_input

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=TestModel(),
        )

    with pytest.raises(
        MlflowException,
        match=r"Extra environment variables are only "
        r"supported when env_manager is set to 'virtualenv', 'conda' or 'uv'",
    ):
        mlflow.models.predict(
            model_uri=model_info.model_uri,
            input_data="abc",
            content_type=_CONTENT_TYPE_JSON,
            env_manager=LOCAL,
            extra_envs={"TEST": "test"},
        )

    with pytest.raises(
        MlflowException, match=r"An exception occurred while running model prediction"
    ):
        mlflow.models.predict(
            model_uri=model_info.model_uri,
            input_data="abc",
            content_type=_CONTENT_TYPE_JSON,
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
        extra_envs=None,
    )


@pytest.mark.parametrize(
    ("input_data", "content_type", "expected"),
    [
        # String (convert to serving input)
        ("[1, 2, 3]", _CONTENT_TYPE_JSON, '{"inputs": "[1, 2, 3]"}'),
        # uLLM String (no change)
        ({"input": "data"}, _CONTENT_TYPE_JSON, '{"input": "data"}'),
        ("x,y,z\n1,2,3\n4,5,6", _CONTENT_TYPE_CSV, "x,y,z\n1,2,3\n4,5,6"),
        # Bool
        (True, _CONTENT_TYPE_JSON, '{"inputs": true}'),
        # Int
        (1, _CONTENT_TYPE_JSON, '{"inputs": 1}'),
        # Float
        (1.0, _CONTENT_TYPE_JSON, '{"inputs": 1.0}'),
        # Datetime
        (
            datetime.datetime(2021, 1, 1, 0, 0, 0),
            _CONTENT_TYPE_JSON,
            '{"inputs": "2021-01-01T00:00:00"}',
        ),
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
        ({"a": [1, 2, 3]}, _CONTENT_TYPE_JSON, '{"inputs": {"a": [1, 2, 3]}}'),
        # Pandas DataFrame (csv)
        (pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}), _CONTENT_TYPE_CSV, "x,y\n1,4\n2,5\n3,6\n"),
        # Pandas DataFrame (json)
        (
            pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
            _CONTENT_TYPE_JSON,
            '{"dataframe_split": {"columns": ["x", "y"], "data": [[1, 4], [2, 5], [3, 6]]}}',
        ),
        # Numpy Array
        (np.array([1, 2, 3]), _CONTENT_TYPE_JSON, '{"inputs": [1, 2, 3]}'),
        # CSC Matrix
        (
            scipy.sparse.csc_matrix([[1, 2], [3, 4]]),
            _CONTENT_TYPE_JSON,
            '{"inputs": [[1, 2], [3, 4]]}',
        ),
        # CSR Matrix
        (
            scipy.sparse.csr_matrix([[1, 2], [3, 4]]),
            _CONTENT_TYPE_JSON,
            '{"inputs": [[1, 2], [3, 4]]}',
        ),
    ],
)
def test_serialize_input_data(input_data, content_type, expected):
    if content_type == _CONTENT_TYPE_JSON:
        assert json.loads(_serialize_input_data(input_data, content_type)) == json.loads(expected)
    else:
        assert _serialize_input_data(input_data, content_type) == expected


@pytest.mark.parametrize(
    ("input_data", "content_type"),
    [
        # Invalid input datatype for the content type
        (1, _CONTENT_TYPE_CSV),
        ({1, 2, 3}, _CONTENT_TYPE_CSV),
        # Invalid string
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


def test_predict_use_current_experiment():
    class TestModel(mlflow.pyfunc.PythonModel):
        @mlflow.trace
        def predict(self, context, model_input: list[str]):
            return model_input

    exp_id = mlflow.set_experiment("test_experiment").experiment_id
    client = mlflow.MlflowClient()
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=TestModel(),
        )

    assert len(client.search_traces(experiment_ids=[exp_id])) == 0
    mlflow.models.predict(
        model_uri=model_info.model_uri,
        input_data=["a", "b", "c"],
        env_manager=VIRTUALENV,
    )
    traces = client.search_traces(experiment_ids=[exp_id])
    assert len(traces) == 1
    assert json.loads(traces[0].data.request)["model_input"] == ["a", "b", "c"]


def test_predict_traces_link_to_active_model():
    model = mlflow.set_active_model(name="test_model")

    class TestModel(mlflow.pyfunc.PythonModel):
        @mlflow.trace
        def predict(self, context, model_input: list[str]):
            return model_input

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=TestModel(),
        )

    traces = get_traces()
    assert len(traces) == 0

    mlflow.models.predict(
        model_uri=model_info.model_uri,
        input_data=["a", "b", "c"],
        env_manager=VIRTUALENV,
    )
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.request_metadata[TraceMetadataKey.MODEL_ID] == model.model_id
