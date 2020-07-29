import json
import math
import numpy as np
import os
import pandas as pd
from collections import namedtuple, OrderedDict

import pytest
import random
import sklearn.datasets as datasets
import sklearn.neighbors as knn

import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.sklearn
from mlflow.models import ModelSignature, infer_signature
from mlflow.protos.databricks_pb2 import ErrorCode, MALFORMED_REQUEST, BAD_REQUEST
from mlflow.pyfunc import PythonModel
from mlflow.types import Schema, ColSpec, DataType
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import NumpyEncoder

from tests.helper_functions import pyfunc_serve_and_score_model, random_int, random_str

ModelWithData = namedtuple("ModelWithData", ["model", "inference_data"])


@pytest.fixture
def pandas_df_with_all_types():
    pdf = pd.DataFrame(
        {
            "boolean": [True, False, True],
            "integer": np.array([1, 2, 3], np.int32),
            "long": np.array([1, 2, 3], np.int64),
            "float": np.array([math.pi, 2 * math.pi, 3 * math.pi], np.float32),
            "double": [math.pi, 2 * math.pi, 3 * math.pi],
            "binary": [bytearray([1, 2, 3]), bytearray([4, 5, 6]), bytearray([7, 8, 9]),],
        }
    )
    pdf["string"] = pd.Series(["a", "b", "c"], dtype=DataType.string.to_pandas())
    return pdf


@pytest.fixture(scope="session")
def sklearn_model():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    knn_model = knn.KNeighborsClassifier()
    knn_model.fit(X, y)
    return ModelWithData(model=knn_model, inference_data=X)


@pytest.fixture
def model_path(tmpdir):
    return str(os.path.join(tmpdir.strpath, "model"))


@pytest.mark.large
def test_scoring_server_responds_to_invalid_json_input_with_stacktrace_and_error_code(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    incorrect_json_content = json.dumps({"not": "a serialized dataframe"})
    response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=incorrect_json_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    )
    response_json = json.loads(response.content)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "stack_trace" in response_json


@pytest.mark.large
def test_scoring_server_responds_to_malformed_json_input_with_stacktrace_and_error_code(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    malformed_json_content = "this is,,,, not valid json"
    response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=malformed_json_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    )
    response_json = json.loads(response.content)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "stack_trace" in response_json


@pytest.mark.large
def test_scoring_server_responds_to_invalid_pandas_input_format_with_stacktrace_and_error_code(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    # The pyfunc scoring server expects a serialized Pandas Dataframe in `split` or `records`
    # format; passing a serialized Dataframe in `table` format should yield a readable error
    pandas_table_content = pd.DataFrame(sklearn_model.inference_data).to_json(orient="table")
    response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=pandas_table_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    )
    response_json = json.loads(response.content)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "stack_trace" in response_json


@pytest.mark.large
def test_scoring_server_responds_to_incompatible_inference_dataframe_with_stacktrace_and_error_code(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)
    incompatible_df = pd.DataFrame(np.array(range(10)))

    response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=incompatible_df,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    )
    response_json = json.loads(response.content)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(BAD_REQUEST)
    assert "message" in response_json
    assert "stack_trace" in response_json


@pytest.mark.large
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
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "stack_trace" in response_json


@pytest.mark.large
def test_scoring_server_successfully_evaluates_correct_dataframes_with_pandas_records_orientation(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    pandas_record_content = pd.DataFrame(sklearn_model.inference_data).to_json(orient="records")
    response_records_content_type = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=pandas_record_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_RECORDS_ORIENTED,
    )
    assert response_records_content_type.status_code == 200


@pytest.mark.large
def test_scoring_server_successfully_evaluates_correct_dataframes_with_pandas_split_orientation(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    pandas_split_content = pd.DataFrame(sklearn_model.inference_data).to_json(orient="split")
    response_default_content_type = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=pandas_split_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    assert response_default_content_type.status_code == 200

    response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=pandas_split_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    )
    assert response.status_code == 200


@pytest.mark.large
def test_scoring_server_successfully_evaluates_correct_split_to_numpy(sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    pandas_split_content = pd.DataFrame(sklearn_model.inference_data).to_json(orient="split")
    response_records_content_type = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=pandas_split_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_NUMPY,
    )
    assert response_records_content_type.status_code == 200


@pytest.mark.large
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
    assert response.status_code == 415


@pytest.mark.large
def test_parse_json_input_records_oriented():
    size = 20
    data = {
        "col_m": [random_int(0, 1000) for _ in range(size)],
        "col_z": [random_str(4) for _ in range(size)],
        "col_a": [random_int() for _ in range(size)],
    }
    p1 = pd.DataFrame.from_dict(data)
    p2 = pyfunc_scoring_server.parse_json_input(p1.to_json(orient="records"), orient="records")
    # "records" orient may shuffle column ordering. Hence comparing each column Series
    for col in data.keys():
        assert all(p1[col] == p2[col])


@pytest.mark.large
def test_parse_json_input_split_oriented():
    size = 200
    data = {
        "col_m": [random_int(0, 1000) for _ in range(size)],
        "col_z": [random_str(4) for _ in range(size)],
        "col_a": [random_int() for _ in range(size)],
    }
    p1 = pd.DataFrame.from_dict(data)
    p2 = pyfunc_scoring_server.parse_json_input(p1.to_json(orient="split"), orient="split")
    assert all(p1 == p2)


@pytest.mark.large
def test_parse_json_input_split_oriented_to_numpy_array():
    size = 200
    data = OrderedDict(
        [
            ("col_m", [random_int(0, 1000) for _ in range(size)]),
            ("col_z", [random_str(4) for _ in range(size)]),
            ("col_a", [random_int() for _ in range(size)]),
        ]
    )
    p0 = pd.DataFrame.from_dict(data)
    np_array = np.array(
        [[a, b, c] for a, b, c in zip(data["col_m"], data["col_z"], data["col_a"])], dtype=object,
    )
    p1 = pd.DataFrame(np_array).infer_objects()
    p2 = pyfunc_scoring_server.parse_split_oriented_json_input_to_numpy(p0.to_json(orient="split"))
    np.testing.assert_array_equal(p1, p2)


@pytest.mark.large
def test_records_oriented_json_to_df():
    # test that datatype for "zip" column is not converted to "int64"
    jstr = (
        "["
        '{"zip":"95120","cost":10.45,"score":8},'
        '{"zip":"95128","cost":23.0,"score":0},'
        '{"zip":"95128","cost":12.1,"score":10}'
        "]"
    )
    df = pyfunc_scoring_server.parse_json_input(jstr, orient="records")

    assert set(df.columns) == {"zip", "cost", "score"}
    assert set(str(dt) for dt in df.dtypes) == {"object", "float64", "int64"}


def _shuffle_pdf(pdf):
    cols = list(pdf.columns)
    random.shuffle(cols)
    return pdf[cols]


@pytest.mark.large
def test_split_oriented_json_to_df():
    # test that datatype for "zip" column is not converted to "int64"
    jstr = (
        '{"columns":["zip","cost","count"],"index":[0,1,2],'
        '"data":[["95120",10.45,-8],["95128",23.0,-1],["95128",12.1,1000]]}'
    )
    df = pyfunc_scoring_server.parse_json_input(jstr, orient="split")

    assert set(df.columns) == {"zip", "cost", "count"}
    assert set(str(dt) for dt in df.dtypes) == {"object", "float64", "int64"}


def test_parse_with_schema(pandas_df_with_all_types):
    schema = Schema([ColSpec(c, c) for c in pandas_df_with_all_types.columns])
    df = _shuffle_pdf(pandas_df_with_all_types)
    json_str = json.dumps(df.to_dict(orient="split"), cls=NumpyEncoder)
    df = pyfunc_scoring_server.parse_json_input(json_str, orient="split", schema=schema)
    json_str = json.dumps(df.to_dict(orient="records"), cls=NumpyEncoder)
    df = pyfunc_scoring_server.parse_json_input(json_str, orient="records", schema=schema)
    assert schema == infer_signature(df[schema.column_names()]).inputs

    # The current behavior with pandas json parse with type hints is weird. In some cases, the
    # types are forced ignoting overflow and loss of precision:

    bad_df = """{
      "columns":["bad_integer", "bad_float", "bad_string", "bad_boolean"],
      "data":[
        [9007199254740991.0, 1.1,                1, 1.5],
        [9007199254740992.0, 9007199254740992.0, 2, 0],
        [9007199254740994.0, 3.3,                3, "some arbitrary string"]
      ]
    }"""
    schema = Schema(
        [
            ColSpec("integer", "bad_integer"),
            ColSpec("float", "bad_float"),
            ColSpec("float", "good_float"),
            ColSpec("string", "bad_string"),
            ColSpec("boolean", "bad_boolean"),
        ]
    )
    df = pyfunc_scoring_server.parse_json_input(bad_df, orient="split", schema=schema)
    # Unfortunately, the current behavior of pandas parse is to force numbers to int32 even if
    # they don't fit:
    assert df["bad_integer"].dtype == np.int32
    assert all(df["bad_integer"] == [-2147483648, -2147483648, -2147483648])

    # The same goes for floats:
    assert df["bad_float"].dtype == np.float32
    assert all(df["bad_float"] == np.array([1.1, 9007199254740992, 3.3], dtype=np.float32))
    # However bad string is recognized as int64:
    assert all(df["bad_string"] == np.array([1, 2, 3], dtype=np.object))

    # Boolean is forced - zero and empty string is false, everything else is true:
    assert df["bad_boolean"].dtype == np.bool
    assert all(df["bad_boolean"] == [True, False, True])


@pytest.mark.large
def test_serving_model_with_schema(pandas_df_with_all_types):
    class TestModel(PythonModel):
        def predict(self, context, model_input):
            return [[k, str(v)] for k, v in model_input.dtypes.items()]

    schema = Schema([ColSpec(c, c) for c in pandas_df_with_all_types.columns])
    df = _shuffle_pdf(pandas_df_with_all_types)
    with TempDir(chdr=True):
        with mlflow.start_run() as run:
            mlflow.pyfunc.log_model(
                "model", python_model=TestModel(), signature=ModelSignature(schema)
            )
        response = pyfunc_serve_and_score_model(
            model_uri="runs:/{}/model".format(run.info.run_id),
            data=json.dumps(df.to_dict(orient="split"), cls=NumpyEncoder),
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
            extra_args=["--no-conda"],
        )
        response_json = json.loads(response.content)
        assert response_json == [[k, str(v)] for k, v in pandas_df_with_all_types.dtypes.items()]
        response = pyfunc_serve_and_score_model(
            model_uri="runs:/{}/model".format(run.info.run_id),
            data=json.dumps(pandas_df_with_all_types.to_dict(orient="records"), cls=NumpyEncoder),
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_RECORDS_ORIENTED,
            extra_args=["--no-conda"],
        )
        response_json = json.loads(response.content)
        assert response_json == [[k, str(v)] for k, v in pandas_df_with_all_types.dtypes.items()]


@pytest.mark.large
def test_split_oriented_json_to_numpy_array():
    # test that datatype for "zip" column is not converted to "int64"
    jstr = (
        '{"columns":["zip","cost","count"],"index":[0,1,2],'
        '"data":[["95120",10.45,-8],["95128",23.0,-1],["95128",12.1,1000]]}'
    )
    df = pyfunc_scoring_server.parse_split_oriented_json_input_to_numpy(jstr)

    assert set(df.columns) == {"zip", "cost", "count"}
    assert set(str(dt) for dt in df.dtypes) == {"object", "float64", "int64"}


def test_get_jsonnable_obj():
    from mlflow.pyfunc.scoring_server import _get_jsonable_obj

    py_ary = [["a", "b", "c"], ["e", "f", "g"]]
    np_ary = _get_jsonable_obj(np.array(py_ary))
    assert json.dumps(py_ary, cls=NumpyEncoder) == json.dumps(np_ary, cls=NumpyEncoder)
    np_ary = _get_jsonable_obj(np.array(py_ary, dtype=type(str)))
    assert json.dumps(py_ary, cls=NumpyEncoder) == json.dumps(np_ary, cls=NumpyEncoder)
