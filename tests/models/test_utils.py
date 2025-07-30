import os
import random
from typing import Any, NamedTuple
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import sklearn.neighbors as knn
from sklearn import datasets

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.environment_variables import MLFLOW_DISABLE_SCHEMA_DETAILS
from mlflow.exceptions import MlflowException
from mlflow.models import add_libraries_to_model
from mlflow.models.utils import (
    _config_context,
    _convert_llm_input_data,
    _enforce_array,
    _enforce_datatype,
    _enforce_object,
    _enforce_property,
    _flatten_nested_params,
    _validate_and_get_model_code_path,
    _validate_model_code_from_notebook,
    get_model_version_from_model_uri,
)
from mlflow.pyfunc import _enforce_schema, _validate_prediction_input
from mlflow.types import DataType, Schema
from mlflow.types.schema import Array, ColSpec, Object, Property


class ModelWithData(NamedTuple):
    model: Any
    inference_data: Any


@pytest.fixture(scope="module")
def sklearn_knn_model():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    knn_model = knn.KNeighborsClassifier()
    knn_model.fit(X, y)
    return ModelWithData(model=knn_model, inference_data=X)


def random_int(lo=1, hi=1000000000):
    return random.randint(lo, hi)


def test_adding_libraries_to_model_default(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    artifact_path = "model"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{model_name}/2"

    # Log a model
    with mlflow.start_run():
        run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        mlflow.sklearn.log_model(
            sklearn_knn_model.model,
            name=artifact_path,
            registered_model_name=model_name,
        )

    wheeled_model_info = add_libraries_to_model(model_uri)
    assert wheeled_model_info.run_id == run_id

    # Verify new model version created
    wheeled_model_version = get_model_version_from_model_uri(wheeled_model_uri)
    assert wheeled_model_version.run_id == run_id
    assert wheeled_model_version.name == model_name


def test_adding_libraries_to_model_new_run(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    artifact_path = "model"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{model_name}/2"

    # Log a model
    with mlflow.start_run():
        original_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        mlflow.sklearn.log_model(
            sklearn_knn_model.model,
            name=artifact_path,
            registered_model_name=model_name,
        )

    with mlflow.start_run():
        wheeled_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        wheeled_model_info = add_libraries_to_model(model_uri)
    assert original_run_id != wheeled_run_id
    assert wheeled_model_info.run_id == wheeled_run_id

    # Verify new model version created
    wheeled_model_version = get_model_version_from_model_uri(wheeled_model_uri)
    assert wheeled_model_version.run_id == wheeled_run_id
    assert wheeled_model_version.name == model_name


def test_adding_libraries_to_model_run_id_passed(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    artifact_path = "model"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{model_name}/2"

    # Log a model
    with mlflow.start_run():
        original_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        mlflow.sklearn.log_model(
            sklearn_knn_model.model,
            name=artifact_path,
            registered_model_name=model_name,
        )

    with mlflow.start_run():
        wheeled_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id

    wheeled_model_info = add_libraries_to_model(model_uri, run_id=wheeled_run_id)
    assert original_run_id != wheeled_run_id
    assert wheeled_model_info.run_id == wheeled_run_id

    # Verify new model version created
    wheeled_model_version = get_model_version_from_model_uri(wheeled_model_uri)
    assert wheeled_model_version.run_id == wheeled_run_id
    assert wheeled_model_version.name == model_name


def test_adding_libraries_to_model_new_model_name(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    wheeled_model_name = f"wheels-test-{random_int()}"
    artifact_path = "model"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{wheeled_model_name}/1"

    # Log a model
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sklearn_knn_model.model,
            name=artifact_path,
            registered_model_name=model_name,
        )

    with mlflow.start_run():
        new_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        wheeled_model_info = add_libraries_to_model(
            model_uri, registered_model_name=wheeled_model_name
        )
    assert wheeled_model_info.run_id == new_run_id

    # Verify new model version created
    wheeled_model_version = get_model_version_from_model_uri(wheeled_model_uri)
    assert wheeled_model_version.run_id == new_run_id
    assert wheeled_model_version.name == wheeled_model_name
    assert wheeled_model_name != model_name


def test_adding_libraries_to_model_when_version_source_None(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    artifact_path = "model"
    model_uri = f"models:/{model_name}/1"

    # Log a model
    with mlflow.start_run():
        original_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        mlflow.sklearn.log_model(
            sklearn_knn_model.model,
            name=artifact_path,
            registered_model_name=model_name,
        )

    model_version_without_source = ModelVersion(name=model_name, version=1, creation_timestamp=124)
    assert model_version_without_source.run_id is None
    with mock.patch.object(
        MlflowClient, "get_model_version", return_value=model_version_without_source
    ) as mlflow_client_mock:
        wheeled_model_info = add_libraries_to_model(model_uri)
        assert wheeled_model_info.run_id is not None
        assert wheeled_model_info.run_id != original_run_id
        mlflow_client_mock.assert_called_once_with(model_name, "1")


@pytest.mark.parametrize(
    ("data", "data_type"),
    [
        ("string", DataType.string),
        (np.int32(1), DataType.integer),
        (np.int32(1), DataType.long),
        (np.int32(1), DataType.double),
        (True, DataType.boolean),
        (1.0, DataType.double),
        (np.float32(0.1), DataType.float),
        (np.float32(0.1), DataType.double),
        (np.int64(100), DataType.long),
        (np.datetime64("2023-10-13 00:00:00"), DataType.datetime),
    ],
)
def test_enforce_datatype(data, data_type):
    assert _enforce_datatype(data, data_type) == data


def test_enforce_datatype_with_errors():
    with pytest.raises(MlflowException, match=r"Expected dtype to be DataType, got str"):
        _enforce_datatype("string", "string")

    with pytest.raises(
        MlflowException, match=r"Failed to enforce schema of data `123` with dtype `string`"
    ):
        _enforce_datatype(123, DataType.string)


def test_enforce_object():
    data = {
        "a": "some_sentence",
        "b": b"some_bytes",
        "c": ["sentence1", "sentence2"],
        "d": {"str": "value", "arr": [0.1, 0.2]},
    }
    obj = Object(
        [
            Property("a", DataType.string),
            Property("b", DataType.binary, required=False),
            Property("c", Array(DataType.string)),
            Property(
                "d",
                Object(
                    [
                        Property("str", DataType.string),
                        Property("arr", Array(DataType.double), required=False),
                    ]
                ),
            ),
        ]
    )
    assert _enforce_object(data, obj) == data

    data = {"a": "some_sentence", "c": ["sentence1", "sentence2"], "d": {"str": "some_value"}}
    assert _enforce_object(data, obj) == data


def test_enforce_object_with_errors():
    with pytest.raises(MlflowException, match=r"Expected data to be dictionary, got list"):
        _enforce_object(["some_sentence"], Object([Property("a", DataType.string)]))

    with pytest.raises(MlflowException, match=r"Expected obj to be Object, got Property"):
        _enforce_object({"a": "some_sentence"}, Property("a", DataType.string))

    obj = Object([Property("a", DataType.string), Property("b", DataType.string, required=False)])
    with pytest.raises(MlflowException, match=r"Missing required properties: {'a'}"):
        _enforce_object({}, obj)

    with pytest.raises(
        MlflowException, match=r"Invalid properties not defined in the schema found: {'c'}"
    ):
        _enforce_object({"a": "some_sentence", "c": "some_sentence"}, obj)

    with pytest.raises(
        MlflowException,
        match=r"Failed to enforce schema for key `a`. Expected type string, received type int",
    ):
        _enforce_object({"a": 1}, obj)


def test_enforce_property():
    data = "some_sentence"
    prop = Property("a", DataType.string)
    assert _enforce_property(data, prop) == data

    data = ["some_sentence1", "some_sentence2"]
    prop = Property("a", Array(DataType.string))
    assert _enforce_property(data, prop) == data

    prop = Property("a", Array(DataType.binary))
    assert _enforce_property(data, prop) == [b"some_sentence1", b"some_sentence2"]

    data = np.array([np.int32(1), np.int32(2)])
    prop = Property("a", Array(DataType.integer))
    assert (_enforce_property(data, prop) == data).all()

    data = {
        "a": "some_sentence",
        "b": b"some_bytes",
        "c": ["sentence1", "sentence2"],
        "d": {"str": "value", "arr": [0.1, 0.2]},
    }
    prop = Property(
        "any_name",
        Object(
            [
                Property("a", DataType.string),
                Property("b", DataType.binary, required=False),
                Property("c", Array(DataType.string), required=False),
                Property(
                    "d",
                    Object(
                        [
                            Property("str", DataType.string),
                            Property("arr", Array(DataType.double), required=False),
                        ]
                    ),
                ),
            ]
        ),
    )
    assert _enforce_property(data, prop) == data
    data = {"a": "some_sentence", "d": {"str": "some_value"}}
    assert _enforce_property(data, prop) == data


def test_enforce_property_with_errors():
    with pytest.raises(
        MlflowException, match=r"Failed to enforce schema of data `123` with dtype `string`"
    ):
        _enforce_property(123, Property("a", DataType.string))

    with pytest.raises(MlflowException, match=r"Missing required properties: {'a'}"):
        _enforce_property(
            {"b": ["some_sentence1", "some_sentence2"]},
            Property(
                "any_name",
                Object([Property("a", DataType.string), Property("b", Array(DataType.string))]),
            ),
        )

    with pytest.raises(
        MlflowException,
        match=r"Failed to enforce schema for key `a`. Expected type string, received type list",
    ):
        _enforce_property(
            {"a": ["some_sentence1", "some_sentence2"]},
            Property("any_name", Object([Property("a", DataType.string)])),
        )


@pytest.mark.parametrize(
    ("data", "schema"),
    [
        # 1. Flat list
        (["some_sentence1", "some_sentence2"], Array(DataType.string)),
        # 2. Nested list
        (
            [
                [["a", "b"], ["c", "d"]],
                [["e", "f", "g"], ["h"]],
                [[]],
            ],
            Array(Array(Array(DataType.string))),
        ),
        # 3. Array of Object
        (
            [
                {"a": "some_sentence1", "b": "some_sentence2"},
                {"a": "some_sentence3", "c": ["some_sentence4", "some_sentence5"]},
            ],
            Array(
                Object(
                    [
                        Property("a", DataType.string),
                        Property("b", DataType.string, required=False),
                        Property("c", Array(DataType.string), required=False),
                    ]
                )
            ),
        ),
        # 4. Empty list
        ([], Array(DataType.string)),
    ],
)
def test_enforce_array_on_list(data, schema):
    assert _enforce_array(data, schema) == data


@pytest.mark.parametrize(
    ("data", "schema"),
    [
        # 1. 1D array
        (np.array(["some_sentence1", "some_sentence2"]), Array(DataType.string)),
        # 2. 2D array
        (
            np.array(
                [
                    ["a", "b"],
                    ["c", "d"],
                ]
            ),
            Array(Array(DataType.string)),
        ),
        # 3. Empty array
        (np.array([[], []]), Array(Array(DataType.string))),
    ],
)
def test_enforce_array_on_numpy_array(data, schema):
    assert (_enforce_array(data, schema) == data).all()


def test_enforce_array_with_errors():
    with pytest.raises(MlflowException, match=r"Expected data to be list or numpy array, got str"):
        _enforce_array("abc", Array(DataType.string))

    with pytest.raises(MlflowException, match=r"Incompatible input types"):
        _enforce_array([123, 456, 789], Array(DataType.string))

    # Nested array with mixed type elements
    with pytest.raises(MlflowException, match=r"Incompatible input types"):
        _enforce_array([["a", "b"], [1, 2]], Array(Array(DataType.string)))

    # Nested array with different nest level
    with pytest.raises(MlflowException, match=r"Expected data to be list or numpy array, got str"):
        _enforce_array([["a", "b"], "c"], Array(Array(DataType.string)))

    # Missing priperties in Object
    with pytest.raises(MlflowException, match=r"Missing required properties: {'b'}"):
        _enforce_array(
            [
                {"a": "some_sentence1", "b": "some_sentence2"},
                {"a": "some_sentence3", "c": ["some_sentence4", "some_sentence5"]},
            ],
            Array(Object([Property("a", DataType.string), Property("b", DataType.string)])),
        )

    # Extra properties
    with pytest.raises(
        MlflowException, match=r"Invalid properties not defined in the schema found: {'c'}"
    ):
        _enforce_array(
            [
                {"a": "some_sentence1", "b": "some_sentence2"},
                {"a": "some_sentence3", "c": ["some_sentence4", "some_sentence5"]},
            ],
            Array(
                Object(
                    [Property("a", DataType.string), Property("b", DataType.string, required=False)]
                )
            ),
        )


def test_model_code_validation():
    # Invalid code with dbutils
    invalid_code = "dbutils.library.restartPython()\nsome_python_variable = 5"

    with mock.patch("mlflow.models.utils._logger.warning") as mock_warning:
        _validate_model_code_from_notebook(invalid_code)
        mock_warning.assert_called_once_with(
            "The model file uses 'dbutils' commands which are not supported. To ensure your "
            "code functions correctly, make sure that it does not rely on these dbutils "
            "commands for correctness."
        )

    # Code with commented magic commands displays warning
    warning_code = "# dbutils.library.restartPython()\n# MAGIC %run ../wheel_installer"

    with mock.patch("mlflow.models.utils._logger.warning") as mock_warning:
        _validate_model_code_from_notebook(warning_code)
        mock_warning.assert_called_once_with(
            "The model file uses magic commands which have been commented out. To ensure your code "
            "functions correctly, make sure that it does not rely on these magic commands for "
            "correctness."
        )

    # Code with commented pip magic commands does not warn
    warning_code = "# MAGIC %pip install mlflow"
    with mock.patch("mlflow.models.utils._logger.warning") as mock_warning:
        _validate_model_code_from_notebook(warning_code)
        mock_warning.assert_not_called()

    # Test valid code
    valid_code = "some_valid_python_code = 'valid'"

    validated_code = _validate_model_code_from_notebook(valid_code).decode("utf-8")
    assert validated_code == valid_code

    # Test uncommented magic commands
    code_with_magic_command = (
        "valid_python_code = 'valid'\n%pip install sqlparse\nvalid_python_code = 'valid'\n# Comment"
    )
    expected_validated_code = (
        "valid_python_code = 'valid'\n# MAGIC %pip install sqlparse\nvalid_python_code = "
        "'valid'\n# Comment"
    )

    validated_code_with_magic_command = _validate_model_code_from_notebook(
        code_with_magic_command
    ).decode("utf-8")
    assert validated_code_with_magic_command == expected_validated_code


def test_config_context():
    with _config_context("tests/langchain/config.yml"):
        assert mlflow.models.model_config.__mlflow_model_config__ == "tests/langchain/config.yml"

    assert mlflow.models.model_config.__mlflow_model_config__ is None


def test_flatten_nested_params():
    nested_params = {
        "a": 1,
        "b": {"c": 2, "d": {"e": 3}},
        "f": {"g": {"h": 4}},
    }
    expected_flattened_params = {
        "a": 1,
        "b.c": 2,
        "b.d.e": 3,
        "f.g.h": 4,
    }
    assert _flatten_nested_params(nested_params, sep=".") == expected_flattened_params
    assert _flatten_nested_params(nested_params, sep="/") == {
        "a": 1,
        "b/c": 2,
        "b/d/e": 3,
        "f/g/h": 4,
    }
    assert _flatten_nested_params({}) == {}

    params = {"a": 1, "b": 2, "c": 3}
    assert _flatten_nested_params(params) == params

    params = {
        "a": 1,
        "b": {"c": 2, "d": {"e": 3, "f": [1, 2, 3]}, "g": "hello"},
        "h": {"i": None},
    }
    expected_flattened_params = {
        "a": 1,
        "b/c": 2,
        "b/d/e": 3,
        "b/d/f": [1, 2, 3],
        "b/g": "hello",
        "h/i": None,
    }
    assert _flatten_nested_params(params) == expected_flattened_params

    nested_params = {1: {2: {3: 4}}, "a": {"b": {"c": 5}}}
    expected_flattened_params_mixed = {
        "1/2/3": 4,
        "a/b/c": 5,
    }
    assert _flatten_nested_params(nested_params) == expected_flattened_params_mixed

    rag_params = {
        "workspace_url": "https://e2-dogfood.staging.cloud.databricks.com",
        "vector_search_endpoint_name": "dbdemos_vs_endpoint",
        "vector_search_index": "monitoring.rag.databricks_docs_index",
        "embedding_model_endpoint_name": "databricks-bge-large-en",
        "embedding_model_query_instructions": "Represent this sentence for searching",
        "llm_model": "databricks-dbrx-instruct",
        "llm_prompt_template": "You are a trustful assistant for Databricks users.",
        "retriever_config": {"k": 5, "use_mmr": "false"},
        "llm_parameters": {"temperature": 0.01, "max_tokens": 200},
        "llm_prompt_template_variables": ["chat_history", "context", "question"],
        "secret_scope": "dbdemos",
        "secret_key": "rag_sunish",
    }

    expected_rag_flattened_params = {
        "workspace_url": "https://e2-dogfood.staging.cloud.databricks.com",
        "vector_search_endpoint_name": "dbdemos_vs_endpoint",
        "vector_search_index": "monitoring.rag.databricks_docs_index",
        "embedding_model_endpoint_name": "databricks-bge-large-en",
        "embedding_model_query_instructions": "Represent this sentence for searching",
        "llm_model": "databricks-dbrx-instruct",
        "llm_prompt_template": "You are a trustful assistant for Databricks users.",
        "retriever_config/k": 5,
        "retriever_config/use_mmr": "false",
        "llm_parameters/temperature": 0.01,
        "llm_parameters/max_tokens": 200,
        "llm_prompt_template_variables": ["chat_history", "context", "question"],
        "secret_scope": "dbdemos",
        "secret_key": "rag_sunish",
    }

    assert _flatten_nested_params(rag_params) == expected_rag_flattened_params


@pytest.mark.parametrize(
    ("data", "target", "target_type"),
    [
        (pd.DataFrame([{"a": [1, 2, 3]}]), [{"a": [1, 2, 3]}], list),
        (pd.DataFrame([{"a": np.array([1, 2, 3])}]), [{"a": [1, 2, 3]}], list),
        (pd.DataFrame([{0: np.array(["abc"])[0]}]), ["abc"], list),
        (np.array([1, 2, 3]), [1, 2, 3], list),
        (np.array([123])[0], 123, int),
        (np.array(["abc"])[0], "abc", str),
    ],
)
def test_convert_llm_input_data(data, target, target_type):
    result = _convert_llm_input_data(data)
    assert result == target
    assert type(result) == target_type


@pytest.mark.parametrize(
    ("model_path", "error_message"),
    [
        (
            "model.py",
            f"The provided model path '{os.getcwd()}/model.py' does not exist. "
            "Ensure the file path is valid and try again.",
        ),
        (
            "model",
            f"The provided model path '{os.getcwd()}/model' does not exist. "
            "Ensure the file path is valid and try again. "
            f"Perhaps you meant '{os.getcwd()}/model.py'?",
        ),
    ],
)
def test_validate_and_get_model_code_path_not_found(model_path, error_message, tmp_path):
    with pytest.raises(MlflowException, match=error_message):
        _validate_and_get_model_code_path(model_path, tmp_path)


def test_validate_and_get_model_code_path_success(tmp_path):
    # if the model file exists, return the path as is
    model_path = os.path.abspath(__file__)
    actual = _validate_and_get_model_code_path(model_path, tmp_path)

    assert actual == model_path


def test_suppress_schema_error(monkeypatch):
    schema = Schema(
        [
            ColSpec("double", "id"),
            ColSpec("string", "name"),
        ]
    )
    monkeypatch.setenv(MLFLOW_DISABLE_SCHEMA_DETAILS.name, "true")
    data = pd.DataFrame({"id": [1, 2]}, dtype="float64")

    with pytest.raises(
        MlflowException,
        match=r"Failed to enforce model input schema. Please check your input data.",
    ):
        _validate_prediction_input(data, None, schema, None)


def test_enforce_schema_with_missing_and_extra_columns(monkeypatch):
    schema = Schema(
        [
            ColSpec("long", "id"),
            ColSpec("string", "name"),
        ]
    )
    monkeypatch.setenv(MLFLOW_DISABLE_SCHEMA_DETAILS.name, "true")
    input_data = pd.DataFrame({"id": [1, 2], "extra_col": ["mlflow", "oss"]})
    with pytest.raises(
        MlflowException, match=r"Input schema validation failed.*extra inputs provided"
    ):
        _enforce_schema(input_data, schema)
