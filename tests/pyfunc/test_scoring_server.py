import os
import json
import pandas as pd
import numpy as np
from collections import namedtuple, OrderedDict

import pytest
import sklearn.datasets as datasets
import sklearn.neighbors as knn

import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.sklearn
from mlflow.protos.databricks_pb2 import ErrorCode, MALFORMED_REQUEST, BAD_REQUEST

from tests.helper_functions import pyfunc_serve_and_score_model, random_int, random_str


ModelWithData = namedtuple("ModelWithData", ["model", "inference_data"])


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
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    incorrect_json_content = json.dumps({"not": "a serialized dataframe"})
    response = pyfunc_serve_and_score_model(
            model_uri=os.path.abspath(model_path),
            data=incorrect_json_content,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    response_json = json.loads(response.content)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "stack_trace" in response_json


@pytest.mark.large
def test_scoring_server_responds_to_malformed_json_input_with_stacktrace_and_error_code(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    malformed_json_content = "this is,,,, not valid json"
    response = pyfunc_serve_and_score_model(
            model_uri=os.path.abspath(model_path),
            data=malformed_json_content,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    response_json = json.loads(response.content)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "stack_trace" in response_json


@pytest.mark.large
def test_scoring_server_responds_to_invalid_pandas_input_format_with_stacktrace_and_error_code(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    # The pyfunc scoring server expects a serialized Pandas Dataframe in `split` or `records`
    # format; passing a serialized Dataframe in `table` format should yield a readable error
    pandas_table_content = pd.DataFrame(sklearn_model.inference_data).to_json(orient="table")
    response = pyfunc_serve_and_score_model(
            model_uri=os.path.abspath(model_path),
            data=pandas_table_content,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    response_json = json.loads(response.content)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "stack_trace" in response_json


@pytest.mark.large
def test_scoring_server_responds_to_incompatible_inference_dataframe_with_stacktrace_and_error_code(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)
    incompatible_df = pd.DataFrame(np.array(range(10)))

    response = pyfunc_serve_and_score_model(
            model_uri=os.path.abspath(model_path),
            data=incompatible_df,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    response_json = json.loads(response.content)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(BAD_REQUEST)
    assert "message" in response_json
    assert "stack_trace" in response_json


@pytest.mark.large
def test_scoring_server_responds_to_invalid_csv_input_with_stacktrace_and_error_code(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    # Any empty string is not valid pandas CSV
    incorrect_csv_content = ""
    response = pyfunc_serve_and_score_model(
            model_uri=os.path.abspath(model_path),
            data=incorrect_csv_content,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_CSV)
    response_json = json.loads(response.content)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "stack_trace" in response_json


@pytest.mark.large
def test_scoring_server_successfully_evaluates_correct_dataframes_with_pandas_records_orientation(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    pandas_record_content = pd.DataFrame(sklearn_model.inference_data).to_json(orient="records")
    response_records_content_type = pyfunc_serve_and_score_model(
            model_uri=os.path.abspath(model_path),
            data=pandas_record_content,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_RECORDS_ORIENTED)
    assert response_records_content_type.status_code == 200


@pytest.mark.large
def test_scoring_server_successfully_evaluates_correct_dataframes_with_pandas_split_orientation(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    pandas_split_content = pd.DataFrame(sklearn_model.inference_data).to_json(orient="split")
    response_default_content_type = pyfunc_serve_and_score_model(
            model_uri=os.path.abspath(model_path),
            data=pandas_split_content,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON)
    assert response_default_content_type.status_code == 200

    response = pyfunc_serve_and_score_model(
            model_uri=os.path.abspath(model_path),
            data=pandas_split_content,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    assert response.status_code == 200


@pytest.mark.large
def test_scoring_server_successfully_evaluates_correct_split_to_numpy(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    pandas_split_content = pd.DataFrame(sklearn_model.inference_data).to_json(orient="split")
    response_records_content_type = pyfunc_serve_and_score_model(
            model_uri=os.path.abspath(model_path),
            data=pandas_split_content,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_NUMPY)
    assert response_records_content_type.status_code == 200


@pytest.mark.large
def test_scoring_server_responds_to_invalid_content_type_request_with_unsupported_content_type_code(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    pandas_split_content = pd.DataFrame(sklearn_model.inference_data).to_json(orient="split")
    response = pyfunc_serve_and_score_model(
            model_uri=os.path.abspath(model_path),
            data=pandas_split_content,
            content_type="not_a_supported_content_type")
    assert response.status_code == 415


@pytest.mark.large
def test_parse_json_input_records_oriented():
    size = 20
    data = {"col_m": [random_int(0, 1000) for _ in range(size)],
            "col_z": [random_str(4) for _ in range(size)],
            "col_a": [random_int() for _ in range(size)]}
    p1 = pd.DataFrame.from_dict(data)
    p2 = pyfunc_scoring_server.parse_json_input(p1.to_json(orient="records"), orient="records")
    # "records" orient may shuffle column ordering. Hence comparing each column Series
    for col in data.keys():
        assert all(p1[col] == p2[col])


@pytest.mark.large
def test_parse_json_input_split_oriented():
    size = 200
    data = {"col_m": [random_int(0, 1000) for _ in range(size)],
            "col_z": [random_str(4) for _ in range(size)],
            "col_a": [random_int() for _ in range(size)]}
    p1 = pd.DataFrame.from_dict(data)
    p2 = pyfunc_scoring_server.parse_json_input(p1.to_json(orient="split"), orient="split")
    assert all(p1 == p2)


@pytest.mark.large
def test_parse_json_input_split_oriented_to_numpy_array():
    size = 200
    data = OrderedDict([("col_m", [random_int(0, 1000) for _ in range(size)]),
                        ("col_z", [random_str(4) for _ in range(size)]),
                        ("col_a", [random_int() for _ in range(size)])])
    p0 = pd.DataFrame.from_dict(data)
    np_array = np.array([[a, b, c] for a, b, c in
                         zip(data['col_m'], data['col_z'], data['col_a'])],
                        dtype=object)
    p1 = pd.DataFrame(np_array).infer_objects()
    p2 = pyfunc_scoring_server.parse_split_oriented_json_input_to_numpy(
        p0.to_json(orient="split"))
    np.testing.assert_array_equal(p1, p2)


@pytest.mark.large
def test_records_oriented_json_to_df():
    # test that datatype for "zip" column is not converted to "int64"
    jstr = '[' \
           '{"zip":"95120","cost":10.45,"score":8},' \
           '{"zip":"95128","cost":23.0,"score":0},' \
           '{"zip":"95128","cost":12.1,"score":10}' \
           ']'
    df = pyfunc_scoring_server.parse_json_input(jstr, orient="records")

    assert set(df.columns) == {'zip', 'cost', 'score'}
    assert set(str(dt) for dt in df.dtypes) == {'object', 'float64', 'int64'}


@pytest.mark.large
def test_split_oriented_json_to_df():
    # test that datatype for "zip" column is not converted to "int64"
    jstr = '{"columns":["zip","cost","count"],"index":[0,1,2],' \
           '"data":[["95120",10.45,-8],["95128",23.0,-1],["95128",12.1,1000]]}'
    df = pyfunc_scoring_server.parse_json_input(jstr, orient="split")

    assert set(df.columns) == {'zip', 'cost', 'count'}
    assert set(str(dt) for dt in df.dtypes) == {'object', 'float64', 'int64'}


@pytest.mark.large
def test_split_oriented_json_to_numpy_array():
    # test that datatype for "zip" column is not converted to "int64"
    jstr = '{"columns":["zip","cost","count"],"index":[0,1,2],' \
           '"data":[["95120",10.45,-8],["95128",23.0,-1],["95128",12.1,1000]]}'
    df = pyfunc_scoring_server.parse_split_oriented_json_input_to_numpy(jstr)

    assert set(df.columns) == {'zip', 'cost', 'count'}
    assert set(str(dt) for dt in df.dtypes) == {'object', 'float64', 'int64'}


def test_get_jsonnable_obj():
    from mlflow.pyfunc.scoring_server import _get_jsonable_obj
    from mlflow.pyfunc.scoring_server import NumpyEncoder
    py_ary = [["a", "b", "c"], ["e", "f", "g"]]
    np_ary = _get_jsonable_obj(np.array(py_ary))
    assert json.dumps(py_ary, cls=NumpyEncoder) == json.dumps(np_ary, cls=NumpyEncoder)
    np_ary = _get_jsonable_obj(np.array(py_ary, dtype=type(str)))
    assert json.dumps(py_ary, cls=NumpyEncoder) == json.dumps(np_ary, cls=NumpyEncoder)
