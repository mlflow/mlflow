import json
import math
import os
import random
import signal
from collections import namedtuple
from io import StringIO

import keras
import numpy as np
import pandas as pd
import pytest
import sklearn.neighbors as knn
from packaging.version import Version
from sklearn import datasets

import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.sklearn
from mlflow.models import ModelSignature, infer_signature
from mlflow.protos.databricks_pb2 import BAD_REQUEST, ErrorCode
from mlflow.pyfunc import PythonModel
from mlflow.pyfunc.scoring_server import get_cmd
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema
from mlflow.types.schema import Array, Object, Property
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import NumpyEncoder
from mlflow.version import VERSION

from tests.helper_functions import (
    expect_status_code,
    pyfunc_serve_and_score_model,
    random_int,
    random_str,
)

if Version(keras.__version__) >= Version("2.6.0"):
    from tensorflow.keras.layers import Concatenate, Dense, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import SGD
else:
    from keras.layers import Concatenate, Dense, Input
    from keras.models import Model
    from keras.optimizers import SGD


ModelWithData = namedtuple("ModelWithData", ["model", "inference_data"])


def build_and_save_sklearn_model(model_path):
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    X, y = load_iris(return_X_y=True)
    model = LogisticRegression().fit(X, y)

    mlflow.sklearn.save_model(model, path=model_path)


class MyChatLLM(PythonModel):
    def predict(self, context, model_input, params=None):
        # If (and only-if) we define model signature, input is converted
        # to pandas DataFrame in _enforce_schema applied in Pyfunc.predict.
        # TODO: Confirm if this is ok, for me it sounds confusing.
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.to_dict(orient="records")[0]

        messages = model_input["messages"]
        ret = " ".join([m["content"] for m in messages])

        return {
            "id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            "object": "chat.completion",
            "created": 1698916461,
            "model": "llama-2-70b-chat-hf",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": ret,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 47, "completion_tokens": 49, "total_tokens": 96},
            # Echo model input and params for testing purposes
            "model_input": model_input,
            "params": params,
        }


class MyCompletionsLLM(PythonModel):
    # Example model that takes "prompt" as model input
    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.to_dict(orient="records")[0]

        ret = model_input["prompt"]

        return {
            "choices": [
                {
                    "index": 0,
                    "text": ret,
                    "finish_reason": "stop",
                }
            ],
            # Echo model input and params for testing purposes
            "model_input": model_input,
            "params": params,
        }


class MyEmbeddingsLLM(PythonModel):
    # Example model that takes "input" as model input
    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.to_dict(orient="records")[0]

        return {
            "data": [
                {
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3],
                }
            ],
            # Echo model input and params for testing purposes
            "model_input": model_input,
            "params": params,
        }


@pytest.fixture
def pandas_df_with_all_types():
    pdf = pd.DataFrame(
        {
            "boolean": [True, False, True],
            "integer": np.array([1, 2, 3], np.int32),
            "long": np.array([1, 2, 3], np.int64),
            "float": np.array([math.pi, 2 * math.pi, 3 * math.pi], np.float32),
            "double": [math.pi, 2 * math.pi, 3 * math.pi],
            "binary": [bytearray([1, 2, 3]), bytearray([4, 5, 6]), bytearray([7, 8, 9])],
            "datetime": [
                np.datetime64("2021-01-01 00:00:00"),
                np.datetime64("2021-02-02 00:00:00"),
                np.datetime64("2021-03-03 12:00:00"),
            ],
        }
    )
    pdf["string"] = pd.Series(["a", "b", "c"], dtype=DataType.string.to_pandas())
    return pdf


@pytest.fixture
def pandas_df_with_csv_types():
    pdf = pd.DataFrame(
        {
            "boolean": [True, False, True],
            "integer": np.array([1, 2, 3], np.int32),
            "long": np.array([1, 2, 3], np.int64),
            "float": np.array([math.pi, 2 * math.pi, 3 * math.pi], np.float32),
            "double": [math.pi, 2 * math.pi, 3 * math.pi],
        }
    )
    pdf["string"] = pd.Series(["a", "b", "c"], dtype=DataType.string.to_pandas())
    return pdf


@pytest.fixture(scope="module")
def sklearn_model():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    knn_model = knn.KNeighborsClassifier()
    knn_model.fit(X, y)
    return ModelWithData(model=knn_model, inference_data=X)


@pytest.fixture(scope="module")
def keras_model():
    iris = datasets.load_iris()
    data = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
    )
    y = data["target"]
    X = data.drop("target", axis=1).values
    input_a = Input(shape=(2,), name="a")
    input_b = Input(shape=(2,), name="b")
    output = Dense(1)(Dense(3, input_dim=4)(Concatenate()([input_a, input_b])))
    model = Model(inputs=[input_a, input_b], outputs=output)
    model.compile(loss="mean_squared_error", optimizer=SGD())
    model.fit([X[:, :2], X[:, -2:]], y)
    return ModelWithData(model=model, inference_data=X)


@pytest.fixture
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


def test_scoring_server_responds_to_malformed_json_input_with_error_code_and_message(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    malformed_json_content = "this is,,,, not valid json"
    response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=malformed_json_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    response_json = json.loads(response.content)
    assert response_json.get("error_code") == ErrorCode.Name(BAD_REQUEST)
    message = response_json.get("message")
    expected_message = "Invalid input. Ensure that input is a valid JSON formatted string."
    assert expected_message in message


def test_scoring_server_responds_to_invalid_json_format_with_error_code_and_message(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)
    for not_a_dict_content in [1, "1", [1]]:
        incorrect_json_content = json.dumps(not_a_dict_content)
        response = pyfunc_serve_and_score_model(
            model_uri=os.path.abspath(model_path),
            data=incorrect_json_content,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        )
        response_json = json.loads(response.content)
        assert response_json.get("error_code") == ErrorCode.Name(BAD_REQUEST)
        assert "message" in response_json
        message = response_json.get("message")
        assert "The input must be a JSON dictionary with exactly one of the input fields" in message

    for incorrect_format in [
        {"not": "a serialized dataframe"},
        {"dataframe_records": [], "dataframe_split": {"data": []}},
    ]:
        incorrect_json_content = json.dumps(incorrect_format)
        response = pyfunc_serve_and_score_model(
            model_uri=os.path.abspath(model_path),
            data=incorrect_json_content,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        )
        response_json = json.loads(response.content)
        assert response_json.get("error_code") == ErrorCode.Name(BAD_REQUEST)
        message = response_json.get("message")
        assert "The input must be a JSON dictionary with exactly one of the input fields" in message


def test_scoring_server_responds_to_invalid_pandas_input_format_with_stacktrace_and_error_code(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    pdf = pd.DataFrame(sklearn_model.inference_data)
    wrong_records_content = json.dumps({"dataframe_records": pdf.to_dict(orient="split")})
    wrong_split_content = json.dumps({"dataframe_split": pdf.to_dict(orient="records")})

    response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=wrong_split_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    response_json = json.loads(response.content)
    assert response_json.get("error_code") == ErrorCode.Name(BAD_REQUEST)
    message = response_json.get("message")
    assert "Dataframe split format must be a dictionary. Got list" in message

    response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=wrong_records_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    response_json = json.loads(response.content)
    assert response_json.get("error_code") == ErrorCode.Name(BAD_REQUEST)
    message = response_json.get("message")
    assert "Dataframe records format must be a list of records. Got dictionary." in message


def test_scoring_server_responds_to_invalid_dataframe_with_stacktrace_and_error_code(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    invalid_dataframe_content = json.dumps(
        {"dataframe_split": {"index": [1, 2], "data": [[1], [2], [3]]}}
    )

    response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=invalid_dataframe_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    response_json = json.loads(response.content)
    assert response_json.get("error_code") == ErrorCode.Name(BAD_REQUEST)
    message = response_json.get("message")
    assert "Provided dataframe_split field is not a valid dataframe representation" in message


def test_scoring_server_responds_to_incompatible_inference_dataframe_with_stacktrace_and_error_code(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)
    incompatible_df = pd.DataFrame(np.array(range(10)))

    response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=incompatible_df,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    response_json = json.loads(response.content)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(BAD_REQUEST)
    assert "message" in response_json
    assert "stack_trace" in response_json


def test_scoring_server_responds_to_invalid_csv_input_with_stacktrace_and_error_code(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    # Any empty string is not valid pandas CSV
    incorrect_csv_content = ""
    response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=incorrect_csv_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_CSV,
    )
    response_json = json.loads(response.content)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(BAD_REQUEST)
    assert "message" in response_json
    assert "stack_trace" in response_json


def test_scoring_server_successfully_evaluates_correct_dataframes_with_pandas_records_orientation(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    pandas_record_content = json.dumps(
        {"dataframe_records": pd.DataFrame(sklearn_model.inference_data).to_dict(orient="records")}
    )

    response_records_content_type = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=pandas_record_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    expect_status_code(response_records_content_type, 200)

    # Testing the charset parameter
    response_records_content_type = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=pandas_record_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON + "; charset=UTF-8",
    )
    expect_status_code(response_records_content_type, 200)


def test_scoring_server_successfully_evaluates_correct_dataframes_with_pandas_split_orientation(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    pandas_split_content = json.dumps(
        {"dataframe_split": pd.DataFrame(sklearn_model.inference_data).to_dict(orient="split")}
    )

    # Testing the charset parameter
    response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=pandas_split_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON + "; charset=UTF-8",
    )

    expect_status_code(response, 200)

    response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=pandas_split_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    expect_status_code(response, 200)


def test_scoring_server_responds_to_invalid_content_type_request_with_unsupported_content_type_code(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    pandas_split_content = pd.DataFrame(sklearn_model.inference_data).to_json(orient="split")
    response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=pandas_split_content,
        content_type="not_a_supported_content_type",
    )
    expect_status_code(response, 415)


def test_scoring_server_responds_to_invalid_content_type_request_with_unrecognized_content_param(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)
    pandas_split_content = pd.DataFrame(sklearn_model.inference_data).to_json(orient="split")
    response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=pandas_split_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON + "; something=something",
    )
    expect_status_code(response, 415)


def test_scoring_server_successfully_evaluates_correct_tf_serving_sklearn(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    inp_dict = {"instances": sklearn_model.inference_data.tolist()}
    response_records_content_type = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=json.dumps(inp_dict),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    expect_status_code(response_records_content_type, 200)


def test_scoring_server_successfully_evaluates_correct_tf_serving_keras_instances(
    keras_model, model_path
):
    mlflow.tensorflow.save_model(keras_model.model, path=model_path)

    inp_dict = {
        "instances": [
            {"a": a.tolist(), "b": b.tolist()}
            for (a, b) in zip(keras_model.inference_data[:, :2], keras_model.inference_data[:, -2:])
        ]
    }
    response_records_content_type = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=json.dumps(inp_dict),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    expect_status_code(response_records_content_type, 200)


def test_scoring_server_successfully_evaluates_correct_tf_serving_keras_inputs(
    keras_model, model_path
):
    mlflow.tensorflow.save_model(keras_model.model, path=model_path)

    inp_dict = {
        "inputs": {
            "a": keras_model.inference_data[:, :2].tolist(),
            "b": keras_model.inference_data[:, -2:].tolist(),
        }
    }
    response_records_content_type = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=json.dumps(inp_dict),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    expect_status_code(response_records_content_type, 200)


def test_parse_json_input_records_oriented():
    size = 2
    data = {
        "col_m": [random_int(0, 1000) for _ in range(size)],
        "col_z": [random_str() for _ in range(size)],
        "col_a": [random_int() for _ in range(size)],
    }
    p1 = pd.DataFrame.from_dict(data)
    records_content = json.dumps({"dataframe_records": p1.to_dict(orient="records")})
    records_content, _ = pyfunc_scoring_server._split_data_and_params(records_content)
    p2 = pyfunc_scoring_server.infer_and_parse_data(records_content)
    # "records" orient may shuffle column ordering. Hence comparing each column Series
    for col in data:
        assert all(p1[col] == p2[col])


def test_parse_json_input_split_oriented():
    size = 200
    data = {
        "col_m": [random_int(0, 1000) for _ in range(size)],
        "col_z": [random_str() for _ in range(size)],
        "col_a": [random_int() for _ in range(size)],
    }
    p1 = pd.DataFrame.from_dict(data)
    split_content = json.dumps({"dataframe_split": p1.to_dict(orient="split")})
    split_content, _ = pyfunc_scoring_server._split_data_and_params(split_content)
    p2 = pyfunc_scoring_server.infer_and_parse_data(split_content)
    assert all(p1 == p2)


def test_records_oriented_json_to_df():
    # test that datatype for "zip" column is not converted to "int64"
    jstr = """
      {
        "dataframe_records": [
          {"zip":"95120","cost":10.45,"score":8},
          {"zip":"95128","cost":23.0,"score":0},
          {"zip":"95128","cost":12.1,"score":10}
        ]
      }
    """
    jstr, _ = pyfunc_scoring_server._split_data_and_params(jstr)
    df = pyfunc_scoring_server.infer_and_parse_data(jstr)
    assert set(df.columns) == {"zip", "cost", "score"}
    assert {str(dt) for dt in df.dtypes} == {"object", "float64", "int64"}


def _shuffle_pdf(pdf):
    cols = list(pdf.columns)
    random.shuffle(cols)
    return pdf[cols]


def test_split_oriented_json_to_df():
    # test that datatype for "zip" column is not converted to "int64"
    jstr = """
      {
        "dataframe_split": {
          "columns":["zip","cost","count"],
          "index":[0,1,2],
          "data":[["95120",10.45,-8],["95128",23.0,-1],["95128",12.1,1000]]
        }
      }
    """
    jstr, _ = pyfunc_scoring_server._split_data_and_params(jstr)
    df = pyfunc_scoring_server.infer_and_parse_data(jstr)

    assert set(df.columns) == {"zip", "cost", "count"}
    assert {str(dt) for dt in df.dtypes} == {"object", "float64", "int64"}


def test_parse_with_schema_csv(pandas_df_with_csv_types):
    schema = Schema([ColSpec(c, c) for c in pandas_df_with_csv_types.columns])
    df = _shuffle_pdf(pandas_df_with_csv_types)
    csv_str = df.to_csv(index=False)
    df = pyfunc_scoring_server.parse_csv_input(StringIO(csv_str), schema=schema)
    assert schema == infer_signature(df[schema.input_names()]).inputs


def test_parse_with_schema(pandas_df_with_all_types):
    schema = Schema([ColSpec(c, c) for c in pandas_df_with_all_types.columns])
    df = _shuffle_pdf(pandas_df_with_all_types)
    json_str = json.dumps({"dataframe_split": df.to_dict(orient="split")}, cls=NumpyEncoder)
    json_str, _ = pyfunc_scoring_server._split_data_and_params(json_str)
    df = pyfunc_scoring_server.infer_and_parse_data(json_str, schema=schema)
    json_str = json.dumps({"dataframe_records": df.to_dict(orient="records")}, cls=NumpyEncoder)
    json_str, _ = pyfunc_scoring_server._split_data_and_params(json_str)
    df = pyfunc_scoring_server.infer_and_parse_data(json_str, schema=schema)
    assert schema == infer_signature(df[schema.input_names()]).inputs

    # The current behavior with pandas json parse with type hints is weird. In some cases, the
    # types are forced ignoring overflow and loss of precision:

    bad_df = """
    {
      "dataframe_split": {
        "columns":["bad_integer", "bad_float", "bad_string", "bad_boolean"],
        "data":[
          [9007199254740991.0, 1.1,                1, 1.5],
          [9007199254740992.0, 9007199254740992.0, 2, 0],
          [9007199254740994.0, 3.3,                3, "some arbitrary string"]
        ]
      }
    }
    """
    schema = Schema(
        [
            ColSpec("integer", "bad_integer"),
            ColSpec("float", "bad_float"),
            ColSpec("string", "bad_string"),
            ColSpec("boolean", "bad_boolean"),
        ]
    )
    bad_df, _ = pyfunc_scoring_server._split_data_and_params(bad_df)
    df = pyfunc_scoring_server.infer_and_parse_data(bad_df, schema=schema)
    # Unfortunately, the current behavior of pandas parse is to force numbers to int32 even if
    # they don't fit:
    assert df["bad_integer"].dtype == np.int32
    assert all(df["bad_integer"] == [-2147483648, -2147483648, -2147483648])

    # The same goes for floats:
    assert df["bad_float"].dtype == np.float32
    assert all(df["bad_float"] == np.array([1.1, 9007199254740992, 3.3], dtype=np.float32))
    # However bad string is recognized as int64:
    assert all(df["bad_string"] == np.array([1, 2, 3], dtype=object))

    # Boolean is forced - zero and empty string is false, everything else is true:
    assert df["bad_boolean"].dtype == bool
    assert all(df["bad_boolean"] == [True, False, True])


def test_serving_model_with_schema(pandas_df_with_all_types):
    class TestModel(PythonModel):
        def predict(self, context, model_input, params=None):
            return [[k, str(v)] for k, v in model_input.dtypes.items()]

    schema = Schema([ColSpec(c, c) for c in pandas_df_with_all_types.columns])
    df = _shuffle_pdf(pandas_df_with_all_types)
    with TempDir(chdr=True):
        with mlflow.start_run():
            model_info = mlflow.pyfunc.log_model(
                "model", python_model=TestModel(), signature=ModelSignature(schema)
            )
        response = pyfunc_serve_and_score_model(
            model_uri=model_info.model_uri,
            data=json.dumps({"dataframe_split": df.to_dict(orient="split")}, cls=NumpyEncoder),
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
            extra_args=["--env-manager", "local"],
        )
        response_json = json.loads(response.content)["predictions"]

        # objects are not converted to pandas Strings at the moment
        expected_types = {**pandas_df_with_all_types.dtypes, "string": np.dtype(object)}
        assert response_json == [[k, str(v)] for k, v in expected_types.items()]
        response = pyfunc_serve_and_score_model(
            model_uri=model_info.model_uri,
            data=json.dumps(
                {"dataframe_records": pandas_df_with_all_types.to_dict(orient="records")},
                cls=NumpyEncoder,
            ),
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
            extra_args=["--env-manager", "local"],
        )
        response_json = json.loads(response.content)["predictions"]
        assert response_json == [[k, str(v)] for k, v in expected_types.items()]

        # Test 'inputs' format
        response = pyfunc_serve_and_score_model(
            model_uri=model_info.model_uri,
            data=json.dumps({"inputs": df.to_dict(orient="list")}, cls=NumpyEncoder),
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
            extra_args=["--env-manager", "local"],
        )
        response_json = json.loads(response.content)["predictions"]
        assert response_json == [[k, str(v)] for k, v in expected_types.items()]


def test_serving_model_with_param_schema(sklearn_model, model_path):
    dataframe = {
        "dataframe_split": pd.DataFrame(sklearn_model.inference_data).to_dict(orient="split")
    }
    signature = infer_signature(sklearn_model.inference_data)
    param_schema = ParamSchema(
        [ParamSpec("param1", DataType.datetime, np.datetime64("2023-07-01"))]
    )
    signature.params = param_schema
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path, signature=signature)

    # Success if passing no parameters
    response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=json.dumps(dataframe),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON + "; charset=UTF-8",
        extra_args=["--env-manager", "local"],
    )
    expect_status_code(response, 200)

    # Raise error if invalid value is passed
    payload = dataframe.copy()
    payload.update({"params": {"param1": "invalid_value1"}})
    response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=json.dumps(payload),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON + "; charset=UTF-8",
        extra_args=["--env-manager", "local"],
    )
    expect_status_code(response, 400)
    assert (
        "Failed to convert value invalid_value1 from type str to "
        "DataType.datetime for param 'param1'"
        in json.loads(response.content.decode("utf-8"))["message"]
    )

    # Ignore parameters specified in payload if it is not defined in ParamSchema
    payload = dataframe.copy()
    payload.update({"params": {"invalid_param": "value"}})
    response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=json.dumps(payload),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON + "; charset=UTF-8",
        extra_args=["--env-manager", "local"],
    )
    expect_status_code(response, 200)


def test_get_jsonnable_obj():
    from mlflow.pyfunc.scoring_server import _get_jsonable_obj

    py_ary = [["a", "b", "c"], ["e", "f", "g"]]
    np_ary = _get_jsonable_obj(np.array(py_ary))
    assert json.dumps(py_ary, cls=NumpyEncoder) == json.dumps(np_ary, cls=NumpyEncoder)
    np_ary = _get_jsonable_obj(np.array(py_ary, dtype=type(str)))
    assert json.dumps(py_ary, cls=NumpyEncoder) == json.dumps(np_ary, cls=NumpyEncoder)


def test_parse_json_input_including_path():
    class TestModel(PythonModel):
        def predict(self, context, model_input, params=None):
            return 1

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=TestModel())

    pandas_split_content = pd.DataFrame(
        {
            "url": ["http://foo.com", "https://bar.com"],
            "bad_protocol": ["aaa://bbb", "address:/path"],
        }
    )

    response_records_content_type = pyfunc_serve_and_score_model(
        model_uri=f"runs:/{run.info.run_id}/model",
        data=pandas_split_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    expect_status_code(response_records_content_type, 200)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            {"port": 5000, "host": "0.0.0.0", "nworkers": 4, "timeout": 60},
            "--timeout=60 -b 0.0.0.0:5000 -w 4",
        ),
        ({"host": "0.0.0.0", "nworkers": 4, "timeout": 60}, "--timeout=60 -b 0.0.0.0 -w 4"),
        ({"port": 5000, "nworkers": 4, "timeout": 60}, "--timeout=60 -w 4"),
        ({"nworkers": 4, "timeout": 60}, "--timeout=60 -w 4"),
        ({"timeout": 60}, "--timeout=60"),
    ],
)
def test_get_cmd(args: dict, expected: str):
    cmd, _ = get_cmd(model_uri="foo", **args)

    assert cmd == (
        f"gunicorn {expected} ${{GUNICORN_CMD_ARGS}} -- mlflow.pyfunc.scoring_server.wsgi:app"
    )


def test_scoring_server_client(sklearn_model, model_path):
    from mlflow.models.flavor_backend_registry import get_flavor_backend
    from mlflow.pyfunc.scoring_server.client import ScoringServerClient
    from mlflow.utils import find_free_port

    mlflow.sklearn.save_model(
        sk_model=sklearn_model.model, path=model_path, metadata={"metadata_key": "value"}
    )
    expected_result = sklearn_model.model.predict(sklearn_model.inference_data)

    port = find_free_port()
    timeout = 60
    server_proc = None
    try:
        server_proc = get_flavor_backend(
            model_path, env_manager=_EnvManager.CONDA, workers=1, install_mlflow=False
        ).serve(
            model_uri=model_path,
            port=port,
            host="127.0.0.1",
            timeout=timeout,
            enable_mlserver=False,
            synchronous=False,
        )

        client = ScoringServerClient(host="127.0.0.1", port=port)
        client.wait_server_ready()

        data = pd.DataFrame(sklearn_model.inference_data)
        result = client.invoke(data).get_predictions().to_numpy()[:, 0]
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

        version = client.get_version()
        assert version == VERSION
    finally:
        if server_proc is not None:
            os.kill(server_proc.pid, signal.SIGTERM)


_LLM_CHAT_INPUT_SCHEMA = Schema(
    [
        ColSpec(
            Array(
                Object(
                    [
                        Property("role", DataType.string),
                        Property("content", DataType.string),
                    ]
                ),
            ),
            name="messages",
        )
    ]
)


@pytest.mark.parametrize(
    ("signature", "expected_model_input", "expected_params"),
    [
        # Test case: no signature, everything should go to data
        (
            None,
            {
                "messages": [{"role": "user", "content": "hello!"}],
                "max_tokens": 20,
                "temperature": 0.5,
            },
            {},
        ),
        # Test case: signature with params, split params and data
        (
            ModelSignature(
                inputs=_LLM_CHAT_INPUT_SCHEMA,
                params=ParamSchema(
                    [
                        ParamSpec("temperature", DataType.double, default=0.5),
                        ParamSpec("max_tokens", DataType.integer, default=20),
                        ParamSpec("top_p", DataType.double, default=0.9),
                    ]
                ),
            ),
            {
                "messages": [{"role": "user", "content": "hello!"}],
            },
            {
                "temperature": 0.5,
                "max_tokens": 20,
                "top_p": 0.9,  # filled with the default value
            },
        ),
        # Test case: if some params are not defeind in either input and params schema,
        # they will be dropped
        (
            ModelSignature(
                inputs=_LLM_CHAT_INPUT_SCHEMA,
                params=ParamSchema(
                    [
                        ParamSpec("temperature", DataType.double, default=0.5),
                    ]
                ),
            ),
            {
                "messages": [{"role": "user", "content": "hello!"}],
            },
            {
                # only params defined in the schema are passed
                "temperature": 0.5,
            },
        ),
        # Test case: params can be defined in the input schema
        (
            ModelSignature(
                inputs=Schema(
                    [
                        *_LLM_CHAT_INPUT_SCHEMA.inputs,
                        ColSpec(DataType.long, "max_tokens", required=False),
                        ColSpec(DataType.double, "temperature", required=False),
                    ]
                ),
            ),
            {
                "messages": [{"role": "user", "content": "hello!"}],
                "temperature": 0.5,
                "max_tokens": 20,
            },
            {},
        ),
    ],
)
def test_scoring_server_allows_payloads_with_llm_chat_keys_for_pyfunc(
    model_path, signature, expected_model_input, expected_params
):
    mlflow.pyfunc.save_model(model_path, python_model=MyChatLLM(), signature=signature)

    payload = json.dumps(
        {
            "messages": [{"role": "user", "content": "hello!"}],
            "temperature": 0.5,
            "max_tokens": 20,
        }
    )
    response = pyfunc_serve_and_score_model(
        model_uri=model_path,
        data=payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    expect_status_code(response, 200)
    assert json.loads(response.content)["choices"][0]["message"]["content"] == "hello!"
    assert json.loads(response.content)["model_input"] == expected_model_input
    assert json.loads(response.content)["params"] == expected_params


_LLM_COMPLETIONS_INPUT_SCHEMA = Schema(
    [
        ColSpec(
            DataType.string,
            name="prompt",
        )
    ]
)


@pytest.mark.parametrize(
    ("signature", "expected_model_input", "expected_params"),
    [
        # Test case: no signature, everything should go to data
        (
            None,
            {
                "prompt": "hello!",
                "max_tokens": 20,
                "temperature": 0.5,
            },
            {},
        ),
        # Test case: signature with params, split params and data
        (
            ModelSignature(
                inputs=_LLM_COMPLETIONS_INPUT_SCHEMA,
                params=ParamSchema(
                    [
                        ParamSpec("temperature", DataType.double, default=0.5),
                        ParamSpec("max_tokens", DataType.integer, default=20),
                        ParamSpec("top_p", DataType.double, default=0.9),
                    ]
                ),
            ),
            {
                "prompt": "hello!",
            },
            {
                "temperature": 0.5,
                "max_tokens": 20,
                "top_p": 0.9,  # filled with the default value
            },
        ),
    ],
)
def test_scoring_server_allows_payloads_with_llm_completions_keys_for_pyfunc(
    model_path, signature, expected_model_input, expected_params
):
    mlflow.pyfunc.save_model(model_path, python_model=MyCompletionsLLM(), signature=signature)

    payload = json.dumps(
        {
            "prompt": "hello!",
            "temperature": 0.5,
            "max_tokens": 20,
        }
    )
    response = pyfunc_serve_and_score_model(
        model_uri=model_path,
        data=payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    expect_status_code(response, 200)
    assert json.loads(response.content)["choices"][0]["text"] == "hello!"
    assert json.loads(response.content)["model_input"] == expected_model_input
    assert json.loads(response.content)["params"] == expected_params


_LLM_EMBEDDINGS_INPUT_SCHEMA = Schema(
    [
        ColSpec(
            DataType.string,
            name="input",
        )
    ]
)


@pytest.mark.parametrize(
    ("signature", "expected_model_input", "expected_params"),
    [
        # Test case: no signature, everything should go to data
        (
            None,
            {
                "input": "hello!",
                "random": "test",
            },
            {},
        ),
        # Test case: signature with no params accepted, ignores params
        (
            ModelSignature(
                inputs=_LLM_EMBEDDINGS_INPUT_SCHEMA,
            ),
            {
                "input": "hello!",
            },
            {},
        ),
    ],
)
def test_scoring_server_allows_payloads_with_llm_embeddings_keys_for_pyfunc(
    model_path, signature, expected_model_input, expected_params
):
    mlflow.pyfunc.save_model(model_path, python_model=MyEmbeddingsLLM(), signature=signature)

    payload = json.dumps(
        {
            "input": "hello!",
            "random": "test",
        }
    )
    response = pyfunc_serve_and_score_model(
        model_uri=model_path,
        data=payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    expect_status_code(response, 200)
    assert json.loads(response.content)["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert json.loads(response.content)["model_input"] == expected_model_input
    assert json.loads(response.content)["params"] == expected_params


def test_scoring_server_allows_payloads_with_messages_for_pyfunc_wrapped(model_path):
    sklearn_path = model_path + "-sklearn"
    build_and_save_sklearn_model(sklearn_path)

    # wrapped pyfuncs count as pyfuncs (sklearn is not present in model.metadata.flavors)
    class WrappedSklearn(PythonModel):
        def load_context(self, context):
            self.model = mlflow.pyfunc.load_model(context.artifacts["model_path"])

        # note: model_input is the value of "messages", not a dict
        def predict(self, context, model_input):
            weird_but_valid_parse = [json.loads(model_input["messages"][0]["content"])]
            return self.model.predict(weird_but_valid_parse)

    mlflow.pyfunc.save_model(
        model_path, python_model=WrappedSklearn(), artifacts={"model_path": sklearn_path}
    )

    payload = json.dumps(
        {
            "messages": [{"role": "user", "content": "[2,2,2,2]"}],
            "max_tokens": 20,
        }
    )
    response = pyfunc_serve_and_score_model(
        model_uri=model_path,
        data=payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    expect_status_code(response, 200)


@pytest.mark.parametrize(
    ("dict_input", "param_schema", "expected"),
    [
        (
            # no param signature, everything should go
            # to data no params should get dropped
            {"messages": ["test"], "max_tokens": 20, "random": "test"},
            None,
            ({"messages": ["test"], "max_tokens": 20, "random": "test"}, {}),
        ),
        (
            # params defined in the param schema should go to params
            # rest should go to data
            {"messages": ["test"], "max_tokens": 20, "random": "test"},
            ParamSchema(
                [
                    ParamSpec("max_tokens", DataType.integer, default=20),
                ]
            ),
            ({"messages": ["test"], "random": "test"}, {"max_tokens": 20}),
        ),
    ],
)
def test_split_data_and_params_for_llm_input(dict_input, param_schema, expected):
    data, params = pyfunc_scoring_server._split_data_and_params_for_llm_input(
        dict_input, param_schema
    )
    expected_data, expected_params = expected
    assert data == expected_data
    assert params == expected_params
