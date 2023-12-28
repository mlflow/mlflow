import json
from tempfile import NamedTemporaryFile
from unittest import mock

import pandas as pd
import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.python_api import (
    _CONTENT_TYPE_CSV,
    _CONTENT_TYPE_JSON,
    _serialize_input_data,
)
from mlflow.utils.file_utils import TempDir


@pytest.fixture
def mock_backend():
    mock_backend = mock.MagicMock()
    with mock.patch("mlflow.models.python_api.get_flavor_backend", return_value=mock_backend):
        yield mock_backend


def test_predict_with_input_path(mock_backend):
    with NamedTemporaryFile() as input_file:
        mlflow.models.predict(
            model_uri="runs:/test/Model",
            input_data_or_path=input_file.name,
            content_type=_CONTENT_TYPE_CSV,
        )

    mock_backend.predict.assert_called_once_with(
        model_uri="runs:/test/Model",
        input_path=input_file.name,
        output_path=None,
        content_type=_CONTENT_TYPE_CSV,
        pip_requirements_override=None,
    )


def test_predict_with_input_csv(mock_backend):
    input_data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    with TempDir() as tmp:
        # Mock TempDir in the predict function to return the same temp dir we created
        mock_tempdir = mock.MagicMock()
        mock_tempdir.__enter__.return_value.path.return_value = tmp.path()
        with mock.patch("mlflow.models.python_api.TempDir", return_value=mock_tempdir):
            mlflow.models.predict(
                model_uri="runs:/test/Model",
                input_data_or_path=input_data,
                content_type=_CONTENT_TYPE_CSV,
            )

            mock_backend.predict.assert_called_once_with(
                model_uri="runs:/test/Model",
                input_path=tmp.path() + "/input.csv",
                output_path=None,
                content_type=_CONTENT_TYPE_CSV,
                pip_requirements_override=None,
            )

            with open(tmp.path() + "/input.csv") as f:
                df = pd.read_csv(f)
                assert df.equals(input_data)


def test_predict_with_input_json(mock_backend):
    input_data = {"inputs": [1, 2, 3]}

    with TempDir() as tmp:
        # Mock TempDir in the predict function to return the same temp dir we created
        mock_tempdir = mock.MagicMock()
        mock_tempdir.__enter__.return_value.path.return_value = tmp.path()
        with mock.patch("mlflow.models.python_api.TempDir", return_value=mock_tempdir):
            mlflow.models.predict(
                model_uri="runs:/test/Model",
                input_data_or_path=input_data,
                content_type=_CONTENT_TYPE_JSON,
            )

            mock_backend.predict.assert_called_once_with(
                model_uri="runs:/test/Model",
                input_path=tmp.path() + "/input.json",
                output_path=None,
                content_type=_CONTENT_TYPE_JSON,
                pip_requirements_override=None,
            )

            with open(tmp.path() + "/input.json") as f:
                assert json.load(f) == input_data


def test_predict_with_input_none(mock_backend):
    mlflow.models.predict(
        model_uri="runs:/test/Model",
        input_data_or_path=None,
        content_type=_CONTENT_TYPE_CSV,
    )

    mock_backend.predict.assert_called_once_with(
        model_uri="runs:/test/Model",
        input_path=None,
        output_path=None,
        content_type=_CONTENT_TYPE_CSV,
        pip_requirements_override=None,
    )


def test_predict_invalid_content_type(mock_backend):
    with pytest.raises(MlflowException, match=r"Content type must be one of"):
        mlflow.models.predict(
            model_uri="runs:/test/Model",
            input_data_or_path={"inputs": [1, 2, 3]},
            content_type="any",
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
