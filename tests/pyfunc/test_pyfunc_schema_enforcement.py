import base64
import datetime
import decimal
import json
import os
import re
from unittest import mock

import cloudpickle
import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model
from packaging.version import Version

import mlflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow.exceptions import MlflowException
from mlflow.models import (
    Model,
    ModelSignature,
    convert_input_example_to_serving_input,
    infer_signature,
)
from mlflow.models.utils import (
    _enforce_params_schema,
    _enforce_schema,
)
from mlflow.pyfunc import PyFuncModel
from mlflow.pyfunc.scoring_server import is_unified_llm_input
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema, TensorSpec
from mlflow.types.schema import Array, Map, Object, Property
from mlflow.utils.proto_json_utils import dump_input_data

from tests.helper_functions import pyfunc_scoring_endpoint, pyfunc_serve_and_score_model
from tests.tracing.helper import get_traces


class TestModel:
    @staticmethod
    def predict(pdf, params=None):
        return pdf


@pytest.fixture(scope="module")
def sample_params_basic():
    return {
        "str_param": "str_a",
        "int_param": np.int32(1),
        "bool_param": True,
        "double_param": 1.0,
        "float_param": np.float32(0.1),
        "long_param": 100,
        "datetime_param": np.datetime64("2023-06-26 00:00:00"),
        "str_list": ["a", "b", "c"],
        "bool_list": [True, False],
        "double_array": np.array([1.0, 2.0]),
    }


@pytest.fixture(scope="module")
def param_schema_basic():
    return ParamSchema(
        [
            ParamSpec("str_param", DataType.string, "str_a", None),
            ParamSpec("int_param", DataType.integer, np.int32(1), None),
            ParamSpec("bool_param", DataType.boolean, True, None),
            ParamSpec("double_param", DataType.double, 1.0, None),
            ParamSpec("float_param", DataType.float, np.float32(0.1), None),
            ParamSpec("long_param", DataType.long, 100, None),
            ParamSpec(
                "datetime_param", DataType.datetime, np.datetime64("2023-06-26 00:00:00"), None
            ),
            ParamSpec("str_list", DataType.string, ["a", "b", "c"], (-1,)),
            ParamSpec("bool_list", DataType.boolean, [True, False], (-1,)),
            ParamSpec("double_array", DataType.double, [1.0, 2.0], (-1,)),
        ]
    )


class PythonModelWithBasicParams(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        assert isinstance(params, dict)
        assert DataType.is_string(params["str_param"])
        assert DataType.is_integer(params["int_param"])
        assert DataType.is_boolean(params["bool_param"])
        assert DataType.is_double(params["double_param"])
        assert DataType.is_float(params["float_param"])
        assert DataType.is_long(params["long_param"])
        assert DataType.is_datetime(params["datetime_param"])
        assert isinstance(params["str_list"], list)
        assert all(DataType.is_string(x) for x in params["str_list"])
        assert isinstance(params["bool_list"], list)
        assert all(DataType.is_boolean(x) for x in params["bool_list"])
        assert isinstance(params["double_array"], list)
        assert all(DataType.is_double(x) for x in params["double_array"])
        return params


@pytest.fixture(scope="module")
def sample_params_with_arrays():
    return {
        "int_array": np.array([np.int32(1), np.int32(2)]),
        "double_array": np.array([1.0, 2.0]),
        "float_array": np.array([np.float32(1.0), np.float32(2.0)]),
        "long_array": np.array([1, 2]),
        "datetime_array": np.array(
            [np.datetime64("2023-06-26 00:00:00"), np.datetime64("2023-06-26 00:00:00")]
        ),
    }


class PythonModelWithArrayParams(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        assert isinstance(params, dict)
        assert all(DataType.is_integer(x) for x in params["int_array"])
        assert all(DataType.is_double(x) for x in params["double_array"])
        assert all(DataType.is_float(x) for x in params["float_array"])
        assert all(DataType.is_long(x) for x in params["long_array"])
        assert all(DataType.is_datetime(x) for x in params["datetime_array"])
        return params


def test_schema_enforcement_single_column_2d_array():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)
    signature = infer_signature(X, y)
    assert signature.inputs.inputs[0].shape == (-1, 1)
    assert signature.outputs.inputs[0].shape == (-1,)

    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(model, "model", signature=signature)

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    pdf = pd.DataFrame(X)
    np.testing.assert_almost_equal(loaded_model.predict(pdf), model.predict(pdf))


def test_column_schema_enforcement():
    m = Model()
    input_schema = Schema(
        [
            ColSpec("integer", "a"),
            ColSpec("long", "b"),
            ColSpec("float", "c"),
            ColSpec("double", "d"),
            ColSpec("boolean", "e"),
            ColSpec("string", "g"),
            ColSpec("binary", "f"),
            ColSpec("datetime", "h"),
        ]
    )
    m.signature = ModelSignature(inputs=input_schema)
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())
    pdf = pd.DataFrame(
        data=[[1, 2, 3, 4, True, "x", bytes([1]), "2021-01-01 00:00:00.1234567"]],
        columns=["b", "d", "a", "c", "e", "g", "f", "h"],
        dtype=object,
    )
    pdf["a"] = pdf["a"].astype(np.int32)
    pdf["b"] = pdf["b"].astype(np.int64)
    pdf["c"] = pdf["c"].astype(np.float32)
    pdf["d"] = pdf["d"].astype(np.float64)
    pdf["h"] = pdf["h"].astype(np.dtype("datetime64[ns]"))
    # test that missing column raises
    match_missing_inputs = "Model is missing inputs"
    with pytest.raises(MlflowException, match=match_missing_inputs):
        res = pyfunc_model.predict(pdf[["b", "d", "a", "e", "g", "f", "h"]])

    # test that extra column is ignored
    pdf["x"] = 1

    # test that columns are reordered, extra column is ignored
    res = pyfunc_model.predict(pdf)
    assert all((res == pdf[input_schema.input_names()]).all())

    expected_types = dict(zip(input_schema.input_names(), input_schema.pandas_types()))
    # MLflow datetime type in input_schema does not encode precision, so add it for assertions
    expected_types["h"] = np.dtype("datetime64[ns]")
    # object cannot be converted to pandas Strings at the moment
    expected_types["f"] = object
    expected_types["g"] = object
    actual_types = res.dtypes.to_dict()
    assert expected_types == actual_types

    # Test conversions
    # 1. long -> integer raises
    pdf["a"] = pdf["a"].astype(np.int64)
    match_incompatible_inputs = "Incompatible input types"
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    pdf["a"] = pdf["a"].astype(np.int32)
    # 2. integer -> long works
    pdf["b"] = pdf["b"].astype(np.int32)
    res = pyfunc_model.predict(pdf)
    assert all((res == pdf[input_schema.input_names()]).all())
    assert res.dtypes.to_dict() == expected_types
    pdf["b"] = pdf["b"].astype(np.int64)

    # 3. unsigned int -> long works
    pdf["b"] = pdf["b"].astype(np.uint32)
    res = pyfunc_model.predict(pdf)
    assert all((res == pdf[input_schema.input_names()]).all())
    assert res.dtypes.to_dict() == expected_types
    pdf["b"] = pdf["b"].astype(np.int64)

    # 4. unsigned int -> int raises
    pdf["a"] = pdf["a"].astype(np.uint32)
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    pdf["a"] = pdf["a"].astype(np.int32)

    # 5. double -> float raises
    pdf["c"] = pdf["c"].astype(np.float64)
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    pdf["c"] = pdf["c"].astype(np.float32)

    # 6. float -> double works, double -> float does not
    pdf["d"] = pdf["d"].astype(np.float32)
    res = pyfunc_model.predict(pdf)
    assert res.dtypes.to_dict() == expected_types
    pdf["d"] = pdf["d"].astype(np.float64)
    pdf["c"] = pdf["c"].astype(np.float64)
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    pdf["c"] = pdf["c"].astype(np.float32)

    # 7. int -> float raises
    pdf["c"] = pdf["c"].astype(np.int32)
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    pdf["c"] = pdf["c"].astype(np.float32)

    # 8. int -> double works
    pdf["d"] = pdf["d"].astype(np.int32)
    pyfunc_model.predict(pdf)
    assert all((res == pdf[input_schema.input_names()]).all())
    assert res.dtypes.to_dict() == expected_types

    # 9. long -> double raises
    pdf["d"] = pdf["d"].astype(np.int64)
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    pdf["d"] = pdf["d"].astype(np.float64)

    # 10. any float -> any int raises
    pdf["a"] = pdf["a"].astype(np.float32)
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    # 10. any float -> any int raises
    pdf["a"] = pdf["a"].astype(np.float64)
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    pdf["a"] = pdf["a"].astype(np.int32)
    pdf["b"] = pdf["b"].astype(np.float64)
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    pdf["b"] = pdf["b"].astype(np.int64)

    pdf["b"] = pdf["b"].astype(np.float64)
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    pdf["b"] = pdf["b"].astype(np.int64)

    # 11. objects work
    pdf["b"] = pdf["b"].astype(object)
    pdf["d"] = pdf["d"].astype(object)
    pdf["e"] = pdf["e"].astype(object)
    pdf["f"] = pdf["f"].astype(object)
    pdf["g"] = pdf["g"].astype(object)
    res = pyfunc_model.predict(pdf)
    assert res.dtypes.to_dict() == expected_types

    # 12. datetime64[D] (date only) -> datetime64[x] works
    pdf["h"] = pdf["h"].values.astype("datetime64[D]")
    res = pyfunc_model.predict(pdf)
    assert res.dtypes.to_dict() == expected_types
    pdf["h"] = pdf["h"].astype("datetime64[s]")

    # 13. np.ndarrays can be converted to dataframe but have no columns
    with pytest.raises(MlflowException, match=match_missing_inputs):
        pyfunc_model.predict(pdf.values)

    # 14. dictionaries of str -> list/nparray work,
    # including extraneous multi-dimensional arrays and lists
    arr = np.array([1, 2, 3])
    d = {
        "a": arr.astype("int32"),
        "b": arr.astype("int64"),
        "c": arr.astype("float32"),
        "d": arr.astype("float64"),
        "e": [True, False, True],
        "g": ["a", "b", "c"],
        "f": [bytes(0), bytes(1), bytes(1)],
        "h": np.array(["2020-01-01", "2020-02-02", "2020-03-03"], dtype=np.datetime64),
        # Extraneous multi-dimensional numpy array should be silenty dropped
        "i": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        # Extraneous multi-dimensional list should be silently dropped
        "j": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    }
    res = pyfunc_model.predict(d)
    assert res.dtypes.to_dict() == expected_types

    # 15. dictionaries of str -> list[list] fail
    d = {
        "a": [arr.astype("int32")],
        "b": [arr.astype("int64")],
        "c": [arr.astype("float32")],
        "d": [arr.astype("float64")],
        "e": [[True, False, True]],
        "g": np.array([["a", "b", "c"]]),
        "f": [[bytes(0), bytes(1), bytes(1)]],
        "h": [np.array(["2020-01-01", "2020-02-02", "2020-03-03"], dtype=np.datetime64)],
    }
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(d)

    # 16. conversion to dataframe fails
    d = {
        "a": [1],
        "b": [1, 2],
        "c": [1, 2, 3],
    }
    with pytest.raises(
        MlflowException,
        match="This model contains a column-based signature, which suggests a DataFrame input.",
    ):
        pyfunc_model.predict(d)

    # 17. conversion from Decimal to float is allowed since numpy currently has no support for the
    #  data type.
    pdf["d"] = [decimal.Decimal(1.0)]
    res = pyfunc_model.predict(pdf)
    assert res.dtypes.to_dict() == expected_types


def _compare_exact_tensor_dict_input(d1, d2):
    """Return whether two dicts of np arrays are exactly equal"""
    if d1.keys() != d2.keys():
        return False
    return all(np.array_equal(d1[key], d2[key]) for key in d1)


def test_tensor_multi_named_schema_enforcement():
    m = Model()
    input_schema = Schema(
        [
            TensorSpec(np.dtype(np.uint64), (-1, 5), "a"),
            TensorSpec(np.dtype(np.short), (-1, 2), "b"),
            TensorSpec(np.dtype(np.float32), (2, -1, 2), "c"),
        ]
    )
    m.signature = ModelSignature(inputs=input_schema)
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())
    inp = {
        "a": np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=np.uint64),
        "b": np.array([[0, 0], [1, 1], [2, 2]], dtype=np.short),
        "c": np.array([[[0, 0], [1, 1]], [[2, 2], [3, 3]]], dtype=np.float32),
    }

    # test that missing column raises
    inp1 = inp.copy()
    with pytest.raises(MlflowException, match="Model is missing inputs"):
        pyfunc_model.predict(inp1.pop("b"))

    # test that extra column is ignored
    inp2 = inp.copy()
    inp2["x"] = 1

    # test that extra column is removed
    res = pyfunc_model.predict(inp2)
    assert res == {k: v for k, v in inp.items() if k in {"a", "b", "c"}}
    expected_types = dict(zip(input_schema.input_names(), input_schema.input_types()))
    actual_types = {k: v.dtype for k, v in res.items()}
    assert expected_types == actual_types

    # test that variable axes are supported
    inp3 = {
        "a": np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2]], dtype=np.uint64),
        "b": np.array([[0, 0], [1, 1]], dtype=np.short),
        "c": np.array([[[0, 0]], [[2, 2]]], dtype=np.float32),
    }
    res = pyfunc_model.predict(inp3)
    assert _compare_exact_tensor_dict_input(res, inp3)
    expected_types = dict(zip(input_schema.input_names(), input_schema.input_types()))
    actual_types = {k: v.dtype for k, v in res.items()}
    assert expected_types == actual_types

    # test that type casting is not supported
    inp4 = inp.copy()
    inp4["a"] = inp4["a"].astype(np.int32)
    with pytest.raises(
        MlflowException, match="dtype of input int32 does not match expected dtype uint64"
    ):
        pyfunc_model.predict(inp4)

    # test wrong shape
    inp5 = {
        "a": np.array([[0, 0, 0, 0]], dtype=np.uint),
        "b": np.array([[0, 0], [1, 1]], dtype=np.short),
        "c": np.array([[[0, 0]]], dtype=np.float32),
    }
    with pytest.raises(
        MlflowException,
        match=re.escape("Shape of input (1, 4) does not match expected shape (-1, 5)"),
    ):
        pyfunc_model.predict(inp5)

    # test non-dictionary input
    inp6 = [
        np.array([[0, 0, 0, 0, 0]], dtype=np.uint64),
        np.array([[0, 0], [1, 1]], dtype=np.short),
        np.array([[[0, 0]]], dtype=np.float32),
    ]
    with pytest.raises(
        MlflowException, match=re.escape("Model is missing inputs ['a', 'b', 'c'].")
    ):
        pyfunc_model.predict(inp6)

    # test empty ndarray does not work
    inp7 = inp.copy()
    inp7["a"] = np.array([])
    with pytest.raises(
        MlflowException, match=re.escape("Shape of input (0,) does not match expected shape")
    ):
        pyfunc_model.predict(inp7)

    # test dictionary of str -> list does not work
    inp8 = {k: list(v) for k, v in inp.items()}
    match = (
        r"This model contains a tensor-based model signature with input names.+"
        r"suggests a dictionary input mapping input name to a numpy array, but a dict"
        r" with value type <class 'list'> was found"
    )
    with pytest.raises(MlflowException, match=match):
        pyfunc_model.predict(inp8)

    # test dataframe input fails at shape enforcement
    pdf = pd.DataFrame(data=[[1, 2, 3]], columns=["a", "b", "c"])
    pdf["a"] = pdf["a"].astype(np.uint64)
    pdf["b"] = pdf["b"].astype(np.short)
    pdf["c"] = pdf["c"].astype(np.float32)
    with pytest.raises(
        MlflowException,
        match=re.escape(
            "The input pandas dataframe column 'a' contains scalar values, which requires the "
            "shape to be (-1,) or (-1, 1), but got tensor spec shape of (-1, 5)"
        ),
    ):
        pyfunc_model.predict(pdf)


def test_schema_enforcement_single_named_tensor_schema():
    m = Model()
    input_schema = Schema([TensorSpec(np.dtype(np.uint64), (-1, 2, 3), "a")])
    m.signature = ModelSignature(inputs=input_schema)
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())
    input_array = np.array(range(12), dtype=np.uint64).reshape((2, 2, 3))
    inp = {
        "a": input_array,
    }

    # sanity test that dictionary with correct input works
    res = pyfunc_model.predict(inp)
    assert res == inp
    expected_types = dict(zip(input_schema.input_names(), input_schema.input_types()))
    actual_types = {k: v.dtype for k, v in res.items()}
    assert expected_types == actual_types

    # test single np.ndarray input works and is converted to dictionary
    res = pyfunc_model.predict(inp["a"])
    assert res == inp
    expected_types = dict(zip(input_schema.input_names(), input_schema.input_types()))
    actual_types = {k: v.dtype for k, v in res.items()}
    assert expected_types == actual_types

    # test list does not work
    with pytest.raises(MlflowException, match="Model is missing inputs"):
        pyfunc_model.predict(input_array.tolist())


def test_schema_enforcement_single_unnamed_tensor_schema():
    m = Model()
    input_schema = Schema([TensorSpec(np.dtype(np.uint64), (-1, 3))])
    m.signature = ModelSignature(inputs=input_schema)
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())

    input_array = np.array(range(6), dtype=np.uint64).reshape((2, 3))

    # test single np.ndarray input works and is converted to dictionary
    res = pyfunc_model.predict(input_array)
    np.testing.assert_array_equal(res, input_array)
    expected_types = input_schema.input_types()[0]
    assert expected_types == res.dtype

    input_df = pd.DataFrame(input_array, columns=["c1", "c2", "c3"])
    res = pyfunc_model.predict(input_df)
    np.testing.assert_array_equal(res, input_array)
    assert expected_types == res.dtype

    input_df = input_df.drop("c3", axis=1)
    with pytest.raises(
        expected_exception=MlflowException,
        match=re.escape(
            "This model contains a model signature with an unnamed input. Since the "
            "input data is a pandas DataFrame containing multiple columns, "
            "the input shape must be of the structure "
            "(-1, number_of_dataframe_columns). "
            "Instead, the input DataFrame passed had 2 columns and "
            "an input shape of (-1, 3) with all values within the "
            "DataFrame of scalar type. Please adjust the passed in DataFrame to "
            "match the expected structure",
        ),
    ):
        pyfunc_model.predict(input_df)


def test_schema_enforcement_named_tensor_schema_1d():
    m = Model()
    input_schema = Schema(
        [TensorSpec(np.dtype(np.uint64), (-1,), "a"), TensorSpec(np.dtype(np.float32), (-1,), "b")]
    )
    m.signature = ModelSignature(inputs=input_schema)
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())
    pdf = pd.DataFrame(data=[[0, 0], [1, 1]], columns=["a", "b"])
    pdf["a"] = pdf["a"].astype(np.uint64)
    pdf["b"] = pdf["a"].astype(np.float32)
    d_inp = {
        "a": np.array(pdf["a"], dtype=np.uint64),
        "b": np.array(pdf["b"], dtype=np.float32),
    }

    # test dataframe input works for 1d tensor specs and input is converted to dict
    res = pyfunc_model.predict(pdf)
    assert _compare_exact_tensor_dict_input(res, d_inp)
    expected_types = dict(zip(input_schema.input_names(), input_schema.input_types()))
    actual_types = {k: v.dtype for k, v in res.items()}
    assert expected_types == actual_types

    wrong_m = Model()
    wrong_m.signature = ModelSignature(
        inputs=Schema(
            [
                TensorSpec(np.dtype(np.uint64), (-1, 2), "a"),
                TensorSpec(np.dtype(np.float32), (-1,), "b"),
            ]
        )
    )
    wrong_pyfunc_model = PyFuncModel(model_meta=wrong_m, model_impl=TestModel())
    with pytest.raises(
        expected_exception=MlflowException,
        match=re.escape(
            "The input pandas dataframe column 'a' contains scalar "
            "values, which requires the shape to be (-1,) or (-1, 1), but got tensor spec "
            "shape of (-1, 2)."
        ),
    ):
        wrong_pyfunc_model.predict(pdf)

    wrong_m.signature.inputs = Schema(
        [
            TensorSpec(np.dtype(np.uint64), (2, -1), "a"),
            TensorSpec(np.dtype(np.float32), (-1,), "b"),
        ]
    )
    with pytest.raises(
        expected_exception=MlflowException,
        match=re.escape(
            "For pandas dataframe input, the first dimension of shape must be a variable "
            "dimension and other dimensions must be fixed, but in model signature the shape "
            "of input a is (2, -1)."
        ),
    ):
        wrong_pyfunc_model.predict(pdf)

    # test that dictionary works too
    res = pyfunc_model.predict(d_inp)
    assert res == d_inp
    expected_types = dict(zip(input_schema.input_names(), input_schema.input_types()))
    actual_types = {k: v.dtype for k, v in res.items()}
    assert expected_types == actual_types


def test_schema_enforcement_named_tensor_schema_multidimensional():
    m = Model()
    input_schema = Schema(
        [
            TensorSpec(np.dtype(np.uint64), (-1, 2, 3), "a"),
            TensorSpec(np.dtype(np.float32), (-1, 3, 4), "b"),
        ]
    )
    m.signature = ModelSignature(inputs=input_schema)
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())
    data_a = np.array(range(12), dtype=np.uint64)
    data_b = np.array(range(24), dtype=np.float32) + 10.0
    pdf = pd.DataFrame(
        {"a": data_a.reshape(-1, 2 * 3).tolist(), "b": data_b.reshape(-1, 3 * 4).tolist()}
    )
    d_inp = {
        "a": data_a.reshape((-1, 2, 3)),
        "b": data_b.reshape((-1, 3, 4)),
    }

    # test dataframe input works for 1d tensor specs and input is converted to dict
    res = pyfunc_model.predict(pdf)
    assert _compare_exact_tensor_dict_input(res, d_inp)

    # test dataframe input works for 1d tensor specs and input is converted to dict
    pdf_contains_numpy_array = pd.DataFrame(
        {"a": list(data_a.reshape(-1, 2 * 3)), "b": list(data_b.reshape(-1, 3 * 4))}
    )
    res = pyfunc_model.predict(pdf_contains_numpy_array)
    assert _compare_exact_tensor_dict_input(res, d_inp)

    expected_types = dict(zip(input_schema.input_names(), input_schema.input_types()))
    actual_types = {k: v.dtype for k, v in res.items()}
    assert expected_types == actual_types

    with pytest.raises(
        expected_exception=MlflowException,
        match=re.escape(
            "The value in the Input DataFrame column 'a' could not be converted to the expected "
            "shape of: '(-1, 2, 3)'. Ensure that each of the input list elements are of uniform "
            "length and that the data can be coerced to the tensor type 'uint64'"
        ),
    ):
        pyfunc_model.predict(
            pdf.assign(a=np.array(range(16), dtype=np.uint64).reshape(-1, 8).tolist())
        )

    # test that dictionary works too
    res = pyfunc_model.predict(d_inp)
    assert res == d_inp
    expected_types = dict(zip(input_schema.input_names(), input_schema.input_types()))
    actual_types = {k: v.dtype for k, v in res.items()}
    assert expected_types == actual_types


def test_missing_value_hint_is_displayed_when_it_should():
    m = Model()
    input_schema = Schema([ColSpec("integer", "a")])
    m.signature = ModelSignature(inputs=input_schema)
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())
    pdf = pd.DataFrame(data=[[1], [None]], columns=["a"])
    match = "Incompatible input types"
    with pytest.raises(MlflowException, match=match) as ex:
        pyfunc_model.predict(pdf)
    hint = "Hint: the type mismatch is likely caused by missing values."
    assert hint in str(ex.value.message)
    pdf = pd.DataFrame(data=[[1.5], [None]], columns=["a"])
    with pytest.raises(MlflowException, match=match) as ex:
        pyfunc_model.predict(pdf)
    assert hint not in str(ex.value.message)
    pdf = pd.DataFrame(data=[[1], [2]], columns=["a"], dtype=np.float64)
    with pytest.raises(MlflowException, match=match) as ex:
        pyfunc_model.predict(pdf)
    assert hint not in str(ex.value.message)


def test_column_schema_enforcement_no_col_names():
    m = Model()
    input_schema = Schema([ColSpec("double"), ColSpec("double"), ColSpec("double")])
    m.signature = ModelSignature(inputs=input_schema)
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())
    test_data = [[1.0, 2.0, 3.0]]

    # Can call with just a list
    pd.testing.assert_frame_equal(pyfunc_model.predict(test_data), pd.DataFrame(test_data))

    # Or can call with a DataFrame without column names
    pd.testing.assert_frame_equal(
        pyfunc_model.predict(pd.DataFrame(test_data)), pd.DataFrame(test_data)
    )

    # # Or can call with a np.ndarray
    pd.testing.assert_frame_equal(
        pyfunc_model.predict(pd.DataFrame(test_data).values), pd.DataFrame(test_data)
    )

    # Or with column names!
    pdf = pd.DataFrame(data=test_data, columns=["a", "b", "c"])
    pd.testing.assert_frame_equal(pyfunc_model.predict(pdf), pdf)

    # Must provide the right number of arguments
    with pytest.raises(MlflowException, match="the provided value only has 2 inputs."):
        pyfunc_model.predict([[1.0, 2.0]])

    # Must provide the right types
    with pytest.raises(MlflowException, match="Can not safely convert int64 to float64"):
        pyfunc_model.predict([[1, 2, 3]])

    # Can only provide data type that can be converted to dataframe...
    with pytest.raises(MlflowException, match="Expected input to be DataFrame. Found: set"):
        pyfunc_model.predict({1, 2, 3})

    # 9. dictionaries of str -> list/nparray work
    d = {"a": [1.0], "b": [2.0], "c": [3.0]}
    pd.testing.assert_frame_equal(pyfunc_model.predict(d), pd.DataFrame(d))


def test_tensor_schema_enforcement_no_col_names():
    m = Model()
    input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 3))])
    m.signature = ModelSignature(inputs=input_schema)
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())
    test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    # Can call with numpy array of correct shape
    np.testing.assert_array_equal(pyfunc_model.predict(test_data), test_data)

    # Or can call with a dataframe
    np.testing.assert_array_equal(pyfunc_model.predict(pd.DataFrame(test_data)), test_data)

    # Can not call with a list
    with pytest.raises(
        MlflowException,
        match="This model contains a tensor-based model signature with no input names",
    ):
        pyfunc_model.predict([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Can not call with a dict
    with pytest.raises(
        MlflowException,
        match="This model contains a tensor-based model signature with no input names",
    ):
        pyfunc_model.predict({"blah": test_data})

    # Can not call with a np.ndarray of a wrong shape
    with pytest.raises(
        MlflowException,
        match=re.escape("Shape of input (2, 2) does not match expected shape (-1, 3)"),
    ):
        pyfunc_model.predict(np.array([[1.0, 2.0], [4.0, 5.0]]))

    # Can not call with a np.ndarray of a wrong type
    with pytest.raises(
        MlflowException, match="dtype of input uint32 does not match expected dtype float32"
    ):
        pyfunc_model.predict(test_data.astype(np.uint32))

    # Can call with a np.ndarray with more elements along variable axis
    test_data2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    np.testing.assert_array_equal(pyfunc_model.predict(test_data2), test_data2)

    # Can not call with an empty ndarray
    with pytest.raises(
        MlflowException, match=re.escape("Shape of input () does not match expected shape (-1, 3)")
    ):
        pyfunc_model.predict(np.ndarray([]))


@pytest.mark.parametrize("orient", ["records"])
def test_schema_enforcement_for_inputs_style_orientation_of_dataframe(orient):
    # Test Dict[str, List[Any]]
    test_signature = {
        "inputs": '[{"name": "a", "type": "long"}, {"name": "b", "type": "string"}]',
        "outputs": '[{"name": "response", "type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"a": [4, 5, 6], "b": ["a", "b", "c"]}
    pd_data = pd.DataFrame(data)
    check = _enforce_schema(data, signature.inputs)
    pd.testing.assert_frame_equal(check, pd_data)
    pd_check = _enforce_schema(pd_data.to_dict(orient=orient), signature.inputs)
    pd.testing.assert_frame_equal(pd_check, pd_data)

    # Test Dict[str, str]
    test_signature = {
        "inputs": '[{"name": "a", "type": "string"}]',
        "outputs": '[{"name": "response", "type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"a": "Hi there!"}
    pd_data = pd.DataFrame([data])
    check = _enforce_schema(data, signature.inputs)
    pd.testing.assert_frame_equal(check, pd_data)
    pd_check = _enforce_schema(pd_data.to_dict(orient=orient), signature.inputs)
    pd.testing.assert_frame_equal(pd_check, pd_data)

    # Test List[Dict[str, Union[str, List[str]]]]
    test_signature = {
        "inputs": '[{"name": "query", "type": "string"}, {"name": "inputs", "type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = [{"query": ["test_query1", "test_query2"], "inputs": "test input"}]
    pd_data = pd.DataFrame(data)
    check = _enforce_schema(data, signature.inputs)
    pd.testing.assert_frame_equal(check, pd_data)
    pd_check = _enforce_schema(pd_data.to_dict(orient=orient), signature.inputs)
    pd.testing.assert_frame_equal(pd_check, pd_data)

    # Test List[str]
    test_signature = {
        "inputs": '[{"type": "string"}]',
        "outputs": '[{"name": "response", "type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = ["a", "b", "c"]
    pd_data = pd.DataFrame(data)
    check = _enforce_schema(data, signature.inputs)
    pd.testing.assert_frame_equal(check, pd_data)
    pd_check = _enforce_schema(pd_data.to_dict(orient=orient), signature.inputs)
    pd.testing.assert_frame_equal(pd_check, pd_data)

    # Test Dict[str, np.ndarray]
    test_signature = {
        "inputs": '[{"name": "a", "type": "long"}, {"name": "b", "type": "string"}]',
        "outputs": '[{"name": "response", "type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"a": np.array([1, 2, 3]), "b": np.array(["a", "b", "c"])}
    pd_data = pd.DataFrame(data)
    check = _enforce_schema(data, signature.inputs)
    pd.testing.assert_frame_equal(check, pd_data)
    pd_check = _enforce_schema(pd_data.to_dict(orient=orient), signature.inputs)
    pd.testing.assert_frame_equal(pd_check, pd_data)

    # Test Dict[str, <scalar>] (support added in MLflow 2.3.0)
    test_signature = {
        "inputs": '[{"name": "a", "type": "long"}, {"name": "b", "type": "string"}]',
        "outputs": '[{"name": "response", "type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"a": 12, "b": "a"}
    pd_data = pd.DataFrame([data])
    check = _enforce_schema(data, signature.inputs)
    pd.testing.assert_frame_equal(check, pd_data)
    pd_check = _enforce_schema(pd_data.to_dict(orient=orient), signature.inputs)
    pd.testing.assert_frame_equal(pd_check, pd_data)

    # Test Dict[str, np.ndarray] where array.size == 1
    test_signature = {
        "inputs": '[{"name": "a", "type": "long"}, {"name": "b", "type": "string"}]',
        "outputs": '[{"name": "response", "type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"a": np.array([12]), "b": np.array(["a"])}
    pd_data = pd.DataFrame(data)
    check = _enforce_schema(data, signature.inputs)
    pd.testing.assert_frame_equal(check, pd_data)
    pd_check = _enforce_schema(pd_data.to_dict(orient=orient), signature.inputs)
    pd.testing.assert_frame_equal(pd_check, pd_data)

    # Test Dict[str, np.ndarray] where primitives are supplied
    test_signature = {
        "inputs": '[{"name": "a", "type": "string"}, {"name": "b", "type": "string"}]',
        "outputs": '[{"name": "response", "type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    # simulates the structure that model serving will convert the data to when using
    # a Dict[str, str] with a scalar singular value string
    data = {"a": np.array("a"), "b": np.array("b")}
    pd_data = pd.DataFrame([data])
    check = _enforce_schema(data, signature.inputs)
    pd.testing.assert_frame_equal(check, pd_data)
    pd_check = _enforce_schema(pd_data.to_dict(orient=orient), signature.inputs)
    pd.testing.assert_frame_equal(pd_check, pd_data)

    # Assert that the Dict[str, np.ndarray] casing with primitive does not work on anything
    # but a single string.
    test_signature = {
        "inputs": '[{"name": "a", "type": "long"}, {"name": "b", "type": "long"}]',
        "outputs": '[{"name": "response", "type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"a": np.array(1), "b": np.array(2)}
    pd_data = pd.DataFrame([data])
    # Schema enforcement explicitly only provides support for strings that meet primitives in
    # np.arrays criteria. All other data types should fail.
    with pytest.raises(MlflowException, match="This model contains a column-based"):
        _enforce_schema(data, signature.inputs)
    with pytest.raises(MlflowException, match="Incompatible input types for column a. Can not"):
        _enforce_schema(pd_data.to_dict(orient=orient), signature.inputs)

    # Test bytes
    test_signature = {
        "inputs": '[{"name": "audio", "type": "binary"}]',
        "outputs": '[{"name": "response", "type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"audio": b"Hi I am a bytes string"}
    pd_data = pd.DataFrame([data])
    check = _enforce_schema(data, signature.inputs)
    pd.testing.assert_frame_equal(check, pd_data)
    pd_check = _enforce_schema(pd_data.to_dict(orient=orient), signature.inputs)
    pd.testing.assert_frame_equal(pd_check, pd_data)

    # Test base64 encoded
    test_signature = {
        "inputs": '[{"name": "audio", "type": "binary"}]',
        "outputs": '[{"name": "response", "type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"audio": base64.b64encode(b"Hi I am a bytes string").decode("ascii")}
    pd_data = pd.DataFrame([data])
    check = _enforce_schema(data, signature.inputs)
    pd.testing.assert_frame_equal(check, pd_data)
    pd_check = _enforce_schema(pd_data.to_dict(orient=orient), signature.inputs)
    pd.testing.assert_frame_equal(pd_check, pd_data)


def test_schema_enforcement_for_optional_columns():
    input_schema = Schema(
        [
            ColSpec("double", "a"),
            ColSpec("double", "b"),
            ColSpec("string", "c", required=False),
            ColSpec("long", "d", required=False),
        ]
    )
    signature = ModelSignature(inputs=input_schema)
    test_data_with_all_cols = {"a": [1.0], "b": [1.0], "c": ["something"], "d": [2]}
    test_data_with_only_required_cols = {"a": [1.0], "b": [1.0]}
    test_data_with_one_optional_col = {"a": [1.0], "b": [1.0], "d": [2]}

    for data in [
        test_data_with_all_cols,
        test_data_with_only_required_cols,
        test_data_with_one_optional_col,
    ]:
        pd_data = pd.DataFrame(data)
        check = _enforce_schema(pd_data, signature.inputs)
        pd.testing.assert_frame_equal(check, pd_data)

    # Ensure wrong data type for optional column throws
    test_bad_data = {"a": [1.0], "b": [1.0], "d": ["not the right type"]}
    pd_data = pd.DataFrame(test_bad_data)
    with pytest.raises(MlflowException, match="Incompatible input types for column d."):
        _enforce_schema(pd_data, signature.inputs)

    # Ensure it still validates for required columns
    test_missing_required = {"b": [2.0], "c": ["something"]}
    pd_data = pd.DataFrame(test_missing_required)
    with pytest.raises(MlflowException, match="Model is missing inputs"):
        _enforce_schema(pd_data, signature.inputs)


def test_schema_enforcement_for_list_inputs_back_compatibility_check():
    # Test Dict[str, scalar or List[str]]
    test_signature = {
        "inputs": '[{"name": "prompt", "type": "string"}, {"name": "stop", "type": "string"}]',
        "outputs": '[{"type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"prompt": "this is the prompt", "stop": ["a", "b"]}
    pd_data = pd.DataFrame([data])
    check = _enforce_schema(data, signature.inputs)
    pd.testing.assert_frame_equal(check, pd_data)

    # Test Dict[str, List[str]]
    test_signature = {
        "inputs": '[{"name": "a", "type": "string"}, {"name": "b", "type": "string"}]',
        "outputs": '[{"name": "response", "type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"a": ["Hi there!"], "b": ["Hello there", "Bye!"]}
    pd_data = pd.DataFrame([data])
    check = _enforce_schema(data, signature.inputs)
    pd.testing.assert_frame_equal(check, pd_data)

    # Test Dict[str, List[binary]] with bytes
    test_signature = {
        "inputs": '[{"name": "audio", "type": "binary"}]',
        "outputs": '[{"name": "response", "type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"audio": [b"Hi I am a bytes string"]}
    pd_data = pd.DataFrame([data])
    pd_check = _enforce_schema(pd_data, signature.inputs)
    pd.testing.assert_frame_equal(pd_check, pd_data)

    # Test Dict[str, List[binary]] with base64 encoded
    test_signature = {
        "inputs": '[{"name": "audio", "type": "binary"}]',
        "outputs": '[{"name": "response", "type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"audio": [base64.b64encode(b"Hi I am a bytes string").decode("ascii")]}
    pd_data = pd.DataFrame([data])
    pd_check = _enforce_schema(pd_data, signature.inputs)
    pd.testing.assert_frame_equal(pd_check, pd_data)

    # Test Dict[str, List[Any]]
    test_signature = {
        "inputs": '[{"name": "a", "type": "long"}, {"name": "b", "type": "string"}]',
        "outputs": '[{"name": "response", "type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"a": [4, 5, 6], "b": ["a", "b", "c"]}
    pd_data = pd.DataFrame(data)
    pd_check = _enforce_schema(data, signature.inputs)
    pd.testing.assert_frame_equal(pd_check, pd_data)

    # Test Dict[str, np.ndarray]
    test_signature = {
        "inputs": '[{"name": "a", "type": "long"}, {"name": "b", "type": "string"}]',
        "outputs": '[{"name": "response", "type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"a": np.array([1, 2, 3]), "b": np.array(["a", "b", "c"])}
    pd_data = pd.DataFrame(data)
    pd_check = _enforce_schema(pd_data.to_dict(orient="list"), signature.inputs)
    pd.testing.assert_frame_equal(pd_check, pd_data)

    # Test Dict[str, np.ndarray] where array.size == 1
    test_signature = {
        "inputs": '[{"name": "a", "type": "long"}, {"name": "b", "type": "string"}]',
        "outputs": '[{"name": "response", "type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"a": np.array([12]), "b": np.array(["a"])}
    pd_data = pd.DataFrame(data)
    pd_check = _enforce_schema(pd_data.to_dict(orient="list"), signature.inputs)
    pd.testing.assert_frame_equal(pd_check, pd_data)

    # Test Dict[str, np.ndarray] where primitives are supplied
    test_signature = {
        "inputs": '[{"name": "a", "type": "string"}, {"name": "b", "type": "string"}]',
        "outputs": '[{"name": "response", "type": "string"}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    # simulates the structure that model serving will convert the data to when using
    # a Dict[str, str] with a scalar singular value string
    data = {"a": np.array("a"), "b": np.array("b")}
    pd_data = pd.DataFrame([data])
    pd_check = _enforce_schema(pd_data.to_dict(orient="list"), signature.inputs)
    pd.testing.assert_frame_equal(pd_check, pd_data)


def test_schema_enforcement_for_list_inputs():
    # Test Dict[str, scalar or List[str]]
    test_signature = {
        "inputs": '[{"type": "string", "name": "prompt", "required": true}, '
        '{"type": "array", "items": {"type": "string"}, '
        '"name": "stop", "required": true}]',
        "outputs": '[{"type": "string", "required": true}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"prompt": "this is the prompt", "stop": ["a", "b"]}
    output = "this is the output"
    assert signature == infer_signature(data, output)
    pd_data = pd.DataFrame([data])
    check = _enforce_schema(data, signature.inputs)
    pd.testing.assert_frame_equal(check, pd_data)

    # Test Dict[str, List[str]]
    test_signature = {
        "inputs": '[{"type": "array", "items": {"type": "string"}, '
        '"name": "a", "required": true}, '
        '{"type": "array", "items": {"type": "string"}, '
        '"name": "b", "required": true}]',
        "outputs": '[{"type": "string", "required": true}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"a": ["Hi there!"], "b": ["Hello there", "Bye!"]}
    assert signature == infer_signature(data, output)
    pd_data = pd.DataFrame([data])
    check = _enforce_schema(data, signature.inputs)
    pd.testing.assert_frame_equal(check, pd_data)

    # Test Dict[str, List[binary]] with bytes
    test_signature = {
        "inputs": '[{"type": "array", "items": {"type": "binary"}, '
        '"name": "audio", "required": true}]',
        "outputs": '[{"type": "string", "required": true}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"audio": [b"Hi I am a bytes string"]}
    assert signature == infer_signature(data, output)
    pd_data = pd.DataFrame([data])
    check = _enforce_schema(data, signature.inputs)
    pd.testing.assert_frame_equal(check, pd_data)

    # Test Dict[str, List[binary]] with base64 encoded
    test_signature = {
        "inputs": '[{"type": "array", "items": {"type": "binary"}, '
        '"name": "audio", "required": true}]',
        "outputs": '[{"type": "string", "required": true}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"audio": [base64.b64encode(b"Hi I am a bytes string")]}
    assert signature == infer_signature(data, output)
    pd_data = pd.DataFrame([data])
    check = _enforce_schema(data, signature.inputs)
    pd.testing.assert_frame_equal(check, pd_data)

    # Test Dict[str, List[Any]]
    test_signature = {
        "inputs": '[{"type": "array", "items": {"type": "long"}, '
        '"name": "a", "required": true}, '
        '{"type": "array", "items": {"type": "string"}, '
        '"name": "b", "required": true}]',
        "outputs": '[{"type": "string", "required": true}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"a": [4, 5, 6], "b": ["a", "b", "c"]}
    assert signature == infer_signature(data, output)
    pd_data = pd.DataFrame([data])
    check = _enforce_schema(data, signature.inputs)
    pd.testing.assert_frame_equal(check, pd_data)

    # Test Dict[str, np.ndarray]
    test_signature = {
        "inputs": '[{"name": "a", "type": "tensor", "tensor-spec": '
        '{"dtype": "int64", "shape": [-1]}}, '
        '{"name": "b", "type": "tensor", "tensor-spec": '
        '{"dtype": "str", "shape": [-1]}}]',
        "outputs": '[{"type": "string", "required": true}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"a": np.array([1, 2, 3]), "b": np.array(["a", "b", "c"])}
    pd_check = _enforce_schema(data, signature.inputs)
    assert pd_check == data

    # Test Dict[str, np.ndarray] where array.size == 1
    test_signature = {
        "inputs": '[{"name": "a", "type": "tensor", "tensor-spec": '
        '{"dtype": "int64", "shape": [-1]}}, '
        '{"name": "b", "type": "tensor", "tensor-spec": '
        '{"dtype": "str", "shape": [-1]}}]',
        "outputs": '[{"type": "string", "required": true}]',
    }
    signature = ModelSignature.from_dict(test_signature)
    data = {"a": np.array([12]), "b": np.array(["a"])}
    pd_check = _enforce_schema(data, signature.inputs)
    assert pd_check == data


def test_enforce_params_schema_with_success():
    # Correct parameters & schema
    test_parameters = {
        "str_param": "str_a",
        "int_param": np.int32(1),
        "bool_param": True,
        "double_param": 1.0,
        "float_param": np.float32(0.1),
        "long_param": 100,
        "datetime_param": np.datetime64("2023-06-26 00:00:00"),
        "str_list": ["a", "b", "c"],
        "bool_list": [True, False],
    }
    test_schema = ParamSchema(
        [
            ParamSpec("str_param", DataType.string, "str_a", None),
            ParamSpec("int_param", DataType.integer, np.int32(1), None),
            ParamSpec("bool_param", DataType.boolean, True, None),
            ParamSpec("double_param", DataType.double, 1.0, None),
            ParamSpec("float_param", DataType.float, np.float32(0.1), None),
            ParamSpec("long_param", DataType.long, 100, None),
            ParamSpec(
                "datetime_param", DataType.datetime, np.datetime64("2023-06-26 00:00:00"), None
            ),
            ParamSpec("str_list", DataType.string, ["a", "b", "c"], (-1,)),
            ParamSpec("bool_list", DataType.boolean, [True, False], (-1,)),
        ]
    )
    assert _enforce_params_schema(test_parameters, test_schema) == test_parameters

    # Correct parameters & schema with array
    params = {
        "double_array": np.array([1.0, 2.0]),
        "float_array": np.array([np.float32(1.0), np.float32(2.0)]),
        "long_array": np.array([1, 2]),
        "datetime_array": np.array(
            [np.datetime64("2023-06-26 00:00:00"), np.datetime64("2023-06-26 00:00:00")]
        ),
    }
    schema = ParamSchema(
        [
            ParamSpec("double_array", DataType.double, np.array([1.0, 2.0]), (-1,)),
            ParamSpec(
                "float_array", DataType.float, np.array([np.float32(1.0), np.float32(2.0)]), (-1,)
            ),
            ParamSpec("long_array", DataType.long, np.array([1, 2]), (-1,)),
            ParamSpec(
                "datetime_array",
                DataType.datetime,
                np.array(
                    [np.datetime64("2023-06-26 00:00:00"), np.datetime64("2023-06-26 00:00:00")]
                ),
                (-1,),
            ),
        ]
    )
    for param, value in params.items():
        assert (_enforce_params_schema(params, schema)[param] == value).all()

    # Converting parameters value type to corresponding schema type
    # 1. int -> long, float, double
    assert _enforce_params_schema({"double_param": np.int32(1)}, test_schema)["double_param"] == 1.0
    assert _enforce_params_schema({"float_param": np.int32(1)}, test_schema)["float_param"] == 1.0
    assert _enforce_params_schema({"long_param": np.int32(1)}, test_schema)["long_param"] == 1
    # With array
    for param in ["double_array", "float_array", "long_array"]:
        assert (
            _enforce_params_schema({param: [np.int32(1), np.int32(2)]}, schema)[param]
            == params[param]
        ).all()
        assert (
            _enforce_params_schema({param: np.array([np.int32(1), np.int32(2)])}, schema)[param]
            == params[param]
        ).all()

    # 2. long -> float, double
    assert _enforce_params_schema({"double_param": 1}, test_schema)["double_param"] == 1.0
    assert _enforce_params_schema({"float_param": 1}, test_schema)["float_param"] == 1.0
    # With array
    for param in ["double_array", "float_array"]:
        assert (_enforce_params_schema({param: [1, 2]}, schema)[param] == params[param]).all()
        assert (
            _enforce_params_schema({param: np.array([1, 2])}, schema)[param] == params[param]
        ).all()

    # 3. float -> double
    assert (
        _enforce_params_schema({"double_param": np.float32(1)}, test_schema)["double_param"] == 1.0
    )
    assert np.isclose(
        _enforce_params_schema({"double_param": np.float32(0.1)}, test_schema)["double_param"],
        0.1,
        atol=1e-6,
    )
    # With array
    assert (
        _enforce_params_schema({"double_array": [np.float32(1), np.float32(2)]}, schema)[
            "double_array"
        ]
        == params["double_array"]
    ).all()
    assert (
        _enforce_params_schema({"double_array": np.array([np.float32(1), np.float32(2)])}, schema)[
            "double_array"
        ]
        == params["double_array"]
    ).all()

    # 4. any -> datetime (try conversion)
    assert _enforce_params_schema({"datetime_param": "2023-07-01 00:00:00"}, test_schema)[
        "datetime_param"
    ] == np.datetime64("2023-07-01 00:00:00")

    # With array
    assert (
        _enforce_params_schema(
            {"datetime_array": ["2023-06-26 00:00:00", "2023-06-26 00:00:00"]}, schema
        )["datetime_array"]
        == params["datetime_array"]
    ).all()
    assert (
        _enforce_params_schema(
            {"datetime_array": np.array(["2023-06-26 00:00:00", "2023-06-26 00:00:00"])}, schema
        )["datetime_array"]
        == params["datetime_array"]
    ).all()

    # Add default values if the parameter is not provided
    test_parameters = {"a": "str_a"}
    test_schema = ParamSchema(
        [ParamSpec("a", DataType.string, ""), ParamSpec("b", DataType.long, 1)]
    )
    updated_parameters = {"b": 1}
    updated_parameters.update(test_parameters)
    assert _enforce_params_schema(test_parameters, test_schema) == updated_parameters

    # Ignore values not specified in ParamSchema and log warning
    test_parameters = {"a": "str_a", "invalid_param": "value"}
    test_schema = ParamSchema([ParamSpec("a", DataType.string, "")])
    with mock.patch("mlflow.models.utils._logger.warning") as mock_warning:
        assert _enforce_params_schema(test_parameters, test_schema) == {"a": "str_a"}
        mock_warning.assert_called_once_with(
            "Unrecognized params ['invalid_param'] are ignored for inference. "
            "Supported params are: {'a'}. "
            "To enable them, please add corresponding schema in ModelSignature."
        )

    # Converting parameters keys to string if it is not
    test_parameters = {1: 1.0}
    test_schema = ParamSchema([ParamSpec("1", DataType.double, 1.0)])
    assert _enforce_params_schema(test_parameters, test_schema) == {"1": 1.0}


def test__enforce_params_schema_add_default_values():
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, ctx, model_input, params):
            return list(params.values())

    params = {"str_param": "string", "int_array": [1, 2, 3]}
    signature = infer_signature(["input"], params=params)

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=MyModel(), artifact_path="my_model", signature=signature
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    # Not passing params -- predict with default values
    loaded_predict = loaded_model.predict(["input"])
    assert loaded_predict == ["string", [1, 2, 3]]

    # Passing some params -- add default values
    loaded_predict = loaded_model.predict(["input"], params={"str_param": "new_string"})
    assert loaded_predict == ["new_string", [1, 2, 3]]

    # Passing all params -- override
    loaded_predict = loaded_model.predict(
        ["input"], params={"str_param": "new_string", "int_array": [4, 5, 6]}
    )
    assert loaded_predict == ["new_string", [4, 5, 6]]

    # Raise warning for unrecognized params
    with mock.patch("mlflow.models.utils._logger.warning") as mock_warning:
        loaded_predict = loaded_model.predict(["input"], params={"new_param": "new_string"})
    mock_warning.assert_called_once()
    assert (
        "Unrecognized params ['new_param'] are ignored for inference"
        in mock_warning.call_args[0][0]
    )
    assert loaded_predict == ["string", [1, 2, 3]]


def test_enforce_params_schema_errors():
    # Raise error when failing to convert value to DataType.datetime
    test_schema = ParamSchema(
        [ParamSpec("datetime_param", DataType.datetime, np.datetime64("2023-06-06"))]
    )
    with pytest.raises(
        MlflowException, match=r"Failed to convert value 1.0 from type float to DataType.datetime"
    ):
        _enforce_params_schema({"datetime_param": 1.0}, test_schema)
    # With array
    test_schema = ParamSchema(
        [
            ParamSpec(
                "datetime_array",
                DataType.datetime,
                np.array([np.datetime64("2023-06-06"), np.datetime64("2023-06-06")]),
                (-1,),
            )
        ]
    )
    with pytest.raises(
        MlflowException, match=r"Failed to convert value 1.0 from type float to DataType.datetime"
    ):
        _enforce_params_schema({"datetime_array": [1.0, 2.0]}, test_schema)

    # Raise error when failing to convert value to DataType.float
    test_schema = ParamSchema([ParamSpec("float_param", DataType.float, np.float32(1))])
    with pytest.raises(MlflowException, match=r"Incompatible types for param 'float_param'"):
        _enforce_params_schema({"float_param": "a"}, test_schema)
    # With array
    test_schema = ParamSchema(
        [ParamSpec("float_array", DataType.float, np.array([np.float32(1), np.float32(2)]), (-1,))]
    )
    with pytest.raises(MlflowException, match=r"Incompatible types for param 'float_array'"):
        _enforce_params_schema(
            {"float_array": [np.float32(1), np.float32(2), np.float64(3)]}, test_schema
        )

    # Raise error for any other conversions
    error_msg = r"Incompatible types for param 'int_param'"
    test_schema = ParamSchema([ParamSpec("int_param", DataType.long, np.int32(1))])
    with pytest.raises(MlflowException, match=error_msg):
        _enforce_params_schema({"int_param": np.float32(1)}, test_schema)
    with pytest.raises(MlflowException, match=error_msg):
        _enforce_params_schema({"int_param": "1"}, test_schema)
    with pytest.raises(MlflowException, match=error_msg):
        _enforce_params_schema({"int_param": np.datetime64("2023-06-06")}, test_schema)

    error_msg = r"Incompatible types for param 'str_param'"
    test_schema = ParamSchema([ParamSpec("str_param", DataType.string, "1")])
    with pytest.raises(MlflowException, match=error_msg):
        _enforce_params_schema({"str_param": np.float32(1)}, test_schema)
    with pytest.raises(MlflowException, match=error_msg):
        _enforce_params_schema({"str_param": b"string"}, test_schema)
    with pytest.raises(MlflowException, match=error_msg):
        _enforce_params_schema({"str_param": np.datetime64("2023-06-06")}, test_schema)

    # Raise error if parameters is not dictionary
    with pytest.raises(MlflowException, match=r"Parameters must be a dictionary. Got type 'int'."):
        _enforce_params_schema(100, test_schema)

    # Raise error if invalid parameters are passed
    test_parameters = {"a": True, "b": (1, 2), "c": b"test"}
    test_schema = ParamSchema(
        [
            ParamSpec("a", DataType.boolean, False),
            ParamSpec("b", DataType.string, [], (-1,)),
            ParamSpec("c", DataType.string, ""),
        ]
    )
    with pytest.raises(
        MlflowException,
        match=re.escape(
            "Value must be a 1D array with shape (-1,) for param 'b': string "
            "(default: []) (shape: (-1,)), received tuple"
        ),
    ):
        _enforce_params_schema(test_parameters, test_schema)
    # Raise error for non-1D array
    with pytest.raises(MlflowException, match=r"received list with ndim 2"):
        _enforce_params_schema(
            {"a": [[1, 2], [3, 4]]}, ParamSchema([ParamSpec("a", DataType.long, [], (-1,))])
        )


def test_enforce_params_schema_warns_with_model_without_params():
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, ctx, model_input, params=None):
            return list(params.values()) if isinstance(params, dict) else None

    params = {"str_param": "string", "int_array": [1, 2, 3], "123": 123}
    signature = infer_signature(["input"])

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=MyModel(), artifact_path="model1", signature=signature
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    with mock.patch("mlflow.models.utils._logger.warning") as mock_warning:
        loaded_model.predict(["input"], params=params)
    mock_warning.assert_called_with(
        "`params` can only be specified at inference time if the model signature defines a params "
        "schema. This model does not define a params schema. Ignoring provided params: "
        "['str_param', 'int_array', '123']"
    )


def test_enforce_params_schema_errors_with_model_with_params():
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, ctx, model_input, params=None):
            return list(params.values()) if isinstance(params, dict) else None

    params = {"str_param": "string", "int_array": [1, 2, 3], "123": 123}
    signature = infer_signature(["input"], params=params)

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=MyModel(), artifact_path="model2", signature=signature
        )

    loaded_model_with_params = mlflow.pyfunc.load_model(model_info.model_uri)
    with pytest.raises(MlflowException, match=r"Parameters must be a dictionary. Got type 'list'"):
        loaded_model_with_params.predict(["input"], params=[1, 2, 3])

    with mock.patch("mlflow.models.utils._logger.warning") as mock_warning:
        loaded_model_with_params.predict(["input"], params={123: 456})
    mock_warning.assert_called_with(
        "Keys in parameters should be of type `str`, but received non-string keys."
        "Converting all keys to string..."
    )


def test_param_spec_with_success():
    # Normal cases
    assert ParamSpec("a", DataType.long, 1).default == 1
    assert ParamSpec("a", DataType.string, "1").default == "1"
    assert ParamSpec("a", DataType.boolean, True).default is True
    assert ParamSpec("a", DataType.double, 1.0).default == 1.0
    assert ParamSpec("a", DataType.float, np.float32(1)).default == 1
    assert ParamSpec("a", DataType.datetime, np.datetime64("2023-06-06")).default == datetime.date(
        2023, 6, 6
    )
    assert ParamSpec(
        "a", DataType.datetime, np.datetime64("2023-06-06 00:00:00")
    ).default == datetime.datetime(2023, 6, 6, 0, 0, 0)
    assert ParamSpec("a", DataType.integer, np.int32(1)).default == 1

    # Convert default value type if it is not consistent with provided type
    # 1. int -> long, float, double
    assert ParamSpec("a", DataType.long, np.int32(1)).default == 1
    assert ParamSpec("a", DataType.float, np.int32(1)).default == 1.0
    assert ParamSpec("a", DataType.double, np.int32(1)).default == 1.0
    # 2. long -> float, double
    assert ParamSpec("a", DataType.float, 1).default == 1.0
    assert ParamSpec("a", DataType.double, 1).default == 1.0
    # 3. float -> double
    assert ParamSpec("a", DataType.double, np.float32(1)).default == 1.0
    # 4. any -> datetime (try conversion)
    assert ParamSpec("a", DataType.datetime, "2023-07-01 00:00:00").default == np.datetime64(
        "2023-07-01 00:00:00"
    )


def test_param_spec_errors():
    # Raise error if default value can not be converted to specified type
    with pytest.raises(MlflowException, match=r"Incompatible types for param 'a'"):
        ParamSpec("a", DataType.integer, "1.0")
    with pytest.raises(MlflowException, match=r"Incompatible types for param 'a'"):
        ParamSpec("a", DataType.integer, [1.0, 2.0], (-1,))
    with pytest.raises(MlflowException, match=r"Incompatible types for param 'a'"):
        ParamSpec("a", DataType.string, True)
    with pytest.raises(MlflowException, match=r"Incompatible types for param 'a'"):
        ParamSpec("a", DataType.string, [1.0, 2.0], (-1,))
    with pytest.raises(MlflowException, match=r"Binary type is not supported for parameters"):
        ParamSpec("a", DataType.binary, 1.0)
    with pytest.raises(MlflowException, match=r"Failed to convert value"):
        ParamSpec("a", DataType.datetime, 1.0)
    with pytest.raises(MlflowException, match=r"Failed to convert value"):
        ParamSpec("a", DataType.datetime, [1.0, 2.0], (-1,))
    with pytest.raises(MlflowException, match=r"Invalid value for param 'a'"):
        ParamSpec("a", DataType.datetime, np.datetime64("20230606"))

    # Raise error if shape is not specified for list value
    with pytest.raises(
        MlflowException,
        match=re.escape(
            "Value should be a scalar for param 'a': long (default: [1, 2, 3]) with shape None"
        ),
    ):
        ParamSpec("a", DataType.long, [1, 2, 3], shape=None)
    with pytest.raises(
        MlflowException,
        match=re.escape(
            "Value should be a scalar for param 'a': integer (default: [1 2 3]) with shape None"
        ),
    ):
        ParamSpec("a", DataType.integer, np.array([1, 2, 3]), shape=None)

    # Raise error if shape is specified for scalar value
    with pytest.raises(
        MlflowException,
        match=re.escape(
            "Value must be a 1D array with shape (-1,) for param 'a': boolean (default: True) "
            "(shape: (-1,)), received bool"
        ),
    ):
        ParamSpec("a", DataType.boolean, True, shape=(-1,))

    # Raise error if shape specified is not allowed
    with pytest.raises(
        MlflowException, match=r"Shape must be None for scalar value or \(-1,\) for 1D array value"
    ):
        ParamSpec("a", DataType.boolean, [True, False], (2,))

    # Raise error if default value is not scalar or 1D array
    with pytest.raises(
        MlflowException,
        match=re.escape(
            "Value must be a 1D array with shape (-1,) for param 'a': boolean (default: {'a': 1}) "
            "(shape: (-1,)), received dict"
        ),
    ):
        ParamSpec("a", DataType.boolean, {"a": 1}, (-1,))


def test_enforce_schema_in_python_model_predict(sample_params_basic, param_schema_basic):
    test_params = sample_params_basic
    test_schema = param_schema_basic
    signature = infer_signature(["input1"], params=test_params)
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=PythonModelWithBasicParams(),
            artifact_path="test_model",
            signature=signature,
        )
    assert signature.params == test_schema

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    loaded_predict = loaded_model.predict(["a", "b"], params=test_params)
    for param, value in test_params.items():
        if param == "double_array":
            assert (loaded_predict[param] == value).all()
        else:
            assert loaded_predict[param] == value

    # Automatically convert type if it's not consistent with schema
    # 1. int -> long, float, double
    params_int = {
        "double_param": np.int32(1),
        "float_param": np.int32(1),
        "long_param": np.int32(1),
    }
    expected_params_int = {
        "double_param": 1.0,
        "float_param": np.float32(1),
        "long_param": 1,
    }
    loaded_predict = loaded_model.predict(["a", "b"], params=params_int)
    for param in params_int:
        assert loaded_predict[param] == expected_params_int[param]

    # 2. long -> float, double
    params_long = {
        "double_param": 1,
        "float_param": 1,
    }
    expected_params_long = {
        "double_param": 1.0,
        "float_param": np.float32(1),
    }
    loaded_predict = loaded_model.predict(["a", "b"], params=params_long)
    for param in params_long:
        assert loaded_predict[param] == expected_params_long[param]

    # 3. float -> double
    assert (
        loaded_model.predict(
            ["a", "b"],
            params={
                "double_param": np.float32(1),
            },
        )["double_param"]
        == 1.0
    )

    # 4. any -> datetime (try conversion)
    assert loaded_model.predict(
        ["a", "b"],
        params={
            "datetime_param": "2023-06-26 00:00:00",
        },
    )["datetime_param"] == np.datetime64("2023-06-26 00:00:00")


def test_schema_enforcement_all_feature_types_pandas():
    data = {
        "long": [1, 2, 3],
        "bool": [True, False, False],
        "string": ["a", "b", "c"],
        "datetime": [pd.Timestamp("2020-07-14 00:00:00")] * 3,
        "bool_nullable": [True, None, False],
        "string_nullable": ["a", "b", None],
        "double_nullable": [1.0, 2.0, None],
    }
    df = pd.DataFrame.from_dict(data)
    schema = Schema(
        [
            ColSpec(DataType.long, "long"),
            ColSpec(DataType.boolean, "bool"),
            ColSpec(DataType.string, "string"),
            ColSpec(DataType.datetime, "datetime"),
            ColSpec(DataType.boolean, "bool_nullable", required=False),
            ColSpec(DataType.string, "string_nullable", required=False),
            ColSpec(DataType.double, "double_nullable", required=False),
        ]
    )
    pd.testing.assert_frame_equal(_enforce_schema(df, schema), df, check_dtype=False)


def test_enforce_schema_in_python_model_serving(sample_params_basic):
    signature = infer_signature(["input1"], params=sample_params_basic)
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=PythonModelWithBasicParams(),
            artifact_path="test_model",
            signature=signature,
        )

    # params in payload should be json serializable
    test_params = {
        "str_param": "str_a",
        "int_param": 1,
        "bool_param": True,
        "double_param": 1.0,
        "float_param": 0.1,
        "long_param": 100,
        "datetime_param": datetime.datetime(2023, 6, 6, 0, 0, 0),
        "str_list": ["a", "b", "c"],
        "bool_list": [True, False],
        "double_array": np.array([1.0, 2.0]),
    }
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=dump_input_data(["a", "b"], params=test_params),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert response.status_code == 200
    prediction = json.loads(response.content.decode("utf-8"))["predictions"]
    for param, value in test_params.items():
        if param == "double_array":
            assert (prediction[param] == value).all()
        elif param == "datetime_param":
            assert prediction[param] == value.isoformat()
        else:
            assert prediction[param] == value

    # Test invalid params for model serving
    with pytest.raises(TypeError, match=r"Object of type int32 is not JSON serializable"):
        dump_input_data(["a", "b"], params={"int_param": np.int32(1)})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=dump_input_data(["a", "b"], params={"double_param": "invalid"}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert response.status_code == 400
    assert (
        "Incompatible types for param 'double_param'"
        in json.loads(response.content.decode("utf-8"))["message"]
    )

    # Can not pass bytes to request
    with pytest.raises(TypeError, match=r"Object of type bytes is not JSON serializable"):
        pyfunc_serve_and_score_model(
            model_info.model_uri,
            data=dump_input_data(["a", "b"], params={"str_param": b"bytes"}),
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
            extra_args=["--env-manager", "local"],
        )


def test_python_model_serving_compatible(tmp_path):
    """
    # Code for logging the model in mlflow 2.4.0
    import mlflow
    from mlflow.models import infer_signature

    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, ctx, model_input):
            return model_input

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
                    python_model = MyModel(),
                    artifact_path = "test_model",
                    signature = infer_signature(["input"]),
                    registered_model_name="model")
    """
    tmp_path.joinpath("MLmodel").write_text(
        """
artifact_path: test_model
flavors:
  python_function:
    cloudpickle_version: 2.2.1
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.pyfunc.model
    python_model: python_model.pkl
    python_version: 3.8.16
mlflow_version: 2.4.0
model_uuid: 3cbde93be0114644a6ec900c64cab39d
run_id: 3f87fdff03524c19908c3a47fb99f9cd
signature:
  inputs: '[{"type": "string"}]'
  outputs: null
utc_time_created: '2023-07-13 01:29:55.467561'
        """
    )
    tmp_path.joinpath("python_env.yaml").write_text(
        """
python: 3.8.16
build_dependencies:
    - pip==23.1.2
    - setuptools==56.0.0
    - wheel==0.40.0
dependencies:
    - -r requirements.txt
        """
    )
    tmp_path.joinpath("requirements.txt").write_text(
        """
mlflow==2.4.0
cloudpickle==2.2.1
        """
    )

    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, ctx, model_input):
            return model_input

    python_model = MyModel()

    with open(tmp_path / "python_model.pkl", "wb") as out:
        cloudpickle.dump(python_model, out)

    assert Version(mlflow.__version__) > Version("2.4.0")
    model_uri = str(tmp_path)
    pyfunc_loaded = mlflow.pyfunc.load_model(model_uri)

    assert pyfunc_loaded.metadata.signature == ModelSignature(Schema([ColSpec("string")]))

    # predict is compatible
    local_predict = pyfunc_loaded.predict(["input"])
    assert local_predict.values[0].tolist() == ["input"]

    # model serving is compatible
    response = pyfunc_serve_and_score_model(
        model_uri,
        data=dump_input_data(["a", "b"]),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert response.status_code == 200
    prediction = json.loads(response.content.decode("utf-8"))["predictions"]
    assert prediction == [{"0": "a"}, {"0": "b"}]


def test_function_python_model_serving_compatible(tmp_path):
    """
    # Code for logging the model in mlflow 2.4.0
    import mlflow
    from mlflow.models import infer_signature

    def my_model(model_input):
        return model_input

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
                    python_model = my_model,
                    artifact_path = "test_model",
                    signature = infer_signature(["input"]),
                    registered_model_name="model",
                    input_example=["input"])
    """
    tmp_path.joinpath("MLmodel").write_text(
        """
artifact_path: test_model
flavors:
  python_function:
    cloudpickle_version: 2.2.1
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.pyfunc.model
    python_model: python_model.pkl
    python_version: 3.8.16
mlflow_version: 2.4.0
model_uuid: f19b9a51a34a453282e53ca41d384964
run_id: 9fd7b6e125a547fdbb4505f15e8259ed
saved_input_example_info:
  artifact_path: input_example.json
  pandas_orient: split
  type: dataframe
signature:
  inputs: '[{"type": "string"}]'
  outputs: null
utc_time_created: '2023-07-14 10:18:44.353510'
        """
    )
    tmp_path.joinpath("python_env.yaml").write_text(
        """
python: 3.8.16
build_dependencies:
    - pip==23.1.2
    - setuptools==56.0.0
    - wheel==0.40.0
dependencies:
    - -r requirements.txt
        """
    )
    tmp_path.joinpath("requirements.txt").write_text(
        """
mlflow==2.4.0
cloudpickle==2.2.1
pandas==2.0.3
        """
    )
    tmp_path.joinpath("input_example.json").write_text(
        """
{"data": [["input"]]}
        """
    )

    def my_model(model_input):
        return model_input

    from mlflow.pyfunc.model import _FunctionPythonModel

    python_model = _FunctionPythonModel(my_model, signature=infer_signature(["input"]))

    with open(tmp_path / "python_model.pkl", "wb") as out:
        cloudpickle.dump(python_model, out)

    assert Version(mlflow.__version__) > Version("2.4.0")
    model_uri = str(tmp_path)
    pyfunc_loaded = mlflow.pyfunc.load_model(model_uri)

    assert pyfunc_loaded.metadata.signature == ModelSignature(Schema([ColSpec("string")]))

    # predict is compatible
    local_predict = pyfunc_loaded.predict(["input"])
    assert local_predict.values[0].tolist() == ["input"]

    # model serving is compatible
    response = pyfunc_serve_and_score_model(
        model_uri,
        data=dump_input_data(["a", "b"]),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert response.status_code == 200
    prediction = json.loads(response.content.decode("utf-8"))["predictions"]
    assert prediction == [{"0": "a"}, {"0": "b"}]


def test_enforce_schema_with_arrays_in_python_model_predict(sample_params_with_arrays):
    params = sample_params_with_arrays
    signature = infer_signature(["input1"], params=params)
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=PythonModelWithArrayParams(),
            artifact_path="test_model",
            signature=signature,
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    loaded_predict = loaded_model.predict(["a", "b"], params=params)
    for param, value in params.items():
        assert (loaded_predict[param] == value).all()

    # Automatically convert type if it's not consistent with schema
    # 1. int -> long, float, double
    for param in ["double_array", "float_array", "long_array"]:
        loaded_predict = loaded_model.predict(
            ["a", "b"], params={param: np.array([np.int32(1), np.int32(2)])}
        )
        assert (loaded_predict[param] == params[param]).all()
    # 2. long -> float, double
    for param in ["double_array", "float_array"]:
        loaded_predict = loaded_model.predict(["a", "b"], params={param: np.array([1, 2])})
        assert (loaded_predict[param] == params[param]).all()
    # 3. float -> double
    loaded_predict = loaded_model.predict(
        ["a", "b"], params={"double_array": np.array([np.float32(1), np.float32(2)])}
    )
    assert (loaded_predict["double_array"] == params["double_array"]).all()
    # 4. any -> datetime (try conversion)
    loaded_predict = loaded_model.predict(
        ["a", "b"],
        params={"datetime_array": np.array(["2023-06-26 00:00:00", "2023-06-26 00:00:00"])},
    )
    assert (loaded_predict["datetime_array"] == params["datetime_array"]).all()

    # Raise error if failing to convert the type
    with pytest.raises(
        MlflowException, match=r"Failed to convert value 1.0 from type float to DataType.datetime"
    ):
        loaded_model.predict(["a", "b"], params={"datetime_array": [1.0, 2.0]})
    with pytest.raises(MlflowException, match=r"Incompatible types for param 'int_array'"):
        loaded_model.predict(["a", "b"], params={"int_array": np.array([1.0, 2.0])})
    with pytest.raises(MlflowException, match=r"Incompatible types for param 'float_array'"):
        loaded_model.predict(["a", "b"], params={"float_array": [True, False]})
    with pytest.raises(MlflowException, match=r"Incompatible types for param 'double_array'"):
        loaded_model.predict(["a", "b"], params={"double_array": [1.0, "2.0"]})


def test_enforce_schema_with_arrays_in_python_model_serving(sample_params_with_arrays):
    params = sample_params_with_arrays
    signature = infer_signature(["input1"], params=params)
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=PythonModelWithArrayParams(),
            artifact_path="test_model",
            signature=signature,
        )

    with pyfunc_scoring_endpoint(
        model_info.model_uri, extra_args=["--env-manager", "local"]
    ) as endpoint:
        response = endpoint.invoke(
            data=dump_input_data(["a", "b"], params=params),
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        )
        assert response.status_code == 200
        prediction = json.loads(response.content.decode("utf-8"))["predictions"]
        for param, value in params.items():
            if param == "datetime_array":
                assert prediction[param] == list(map(np.datetime_as_string, value))
            else:
                assert (prediction[param] == value).all()

        # Test invalid params for model serving
        response = endpoint.invoke(
            data=dump_input_data(["a", "b"], params={"datetime_array": [1.0, 2.0]}),
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        )
        assert response.status_code == 400
        assert (
            "Failed to convert value 1.0 from type float to DataType.datetime"
            in json.loads(response.content.decode("utf-8"))["message"]
        )

        response = endpoint.invoke(
            data=dump_input_data(["a", "b"], params={"int_array": np.array([1.0, 2.0])}),
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        )
        assert response.status_code == 400
        assert (
            "Incompatible types for param 'int_array'"
            in json.loads(response.content.decode("utf-8"))["message"]
        )

        response = endpoint.invoke(
            data=dump_input_data(["a", "b"], params={"float_array": [True, False]}),
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        )
        assert response.status_code == 400
        assert (
            "Incompatible types for param 'float_array'"
            in json.loads(response.content.decode("utf-8"))["message"]
        )

        response = endpoint.invoke(
            data=dump_input_data(["a", "b"], params={"double_array": [1.0, "2.0"]}),
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        )
        assert response.status_code == 400
        assert (
            "Incompatible types for param 'double_array'"
            in json.loads(response.content.decode("utf-8"))["message"]
        )


@pytest.mark.parametrize(
    ("example", "input_schema", "output_schema"),
    [
        (
            ["input1", "input2", "input3"],
            Schema([ColSpec(DataType.string)]),
            Schema([ColSpec(DataType.string, 0)]),
        ),
        (
            [{"a": "a", "b": "b"}, {"a": "b"}],
            Schema([ColSpec(DataType.string, "a"), ColSpec(DataType.string, "b", required=False)]),
            Schema([ColSpec(DataType.string, "a"), ColSpec(DataType.string, "b", required=False)]),
        ),
        (
            {"a": ["a", "b", "c"], "b": "b"},
            Schema([ColSpec(Array(DataType.string), "a"), ColSpec(DataType.string, "b")]),
            Schema([ColSpec(Array(DataType.string), "a"), ColSpec(DataType.string, "b")]),
        ),
        (
            pd.DataFrame({"a": ["a", "b", "c"], "b": "b"}),
            Schema([ColSpec(DataType.string, "a"), ColSpec(DataType.string, "b")]),
            Schema([ColSpec(DataType.string, "a"), ColSpec(DataType.string, "b")]),
        ),
    ],
)
def test_pyfunc_model_input_example_with_params(
    sample_params_basic, param_schema_basic, tmp_path, example, input_schema, output_schema
):
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=MyModel(),
            artifact_path="test_model",
            input_example=(example, sample_params_basic),
        )

    # Test _infer_signature_from_input_example
    assert model_info.signature.inputs == input_schema
    assert model_info.signature.outputs == output_schema
    assert model_info.signature.params == param_schema_basic

    # Test predict
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    prediction = loaded_model.predict(example)
    expected_df = pd.DataFrame([example] if isinstance(example, dict) else example)
    pd.testing.assert_frame_equal(prediction, expected_df)

    # Test saved example
    local_path = _download_artifact_from_uri(model_info.model_uri, output_path=tmp_path)
    mlflow_model = Model.load(os.path.join(local_path, "MLmodel"))
    loaded_example = mlflow_model.load_input_example(local_path)
    if isinstance(example, list) and all(np.isscalar(x) for x in example):
        np.testing.assert_equal(loaded_example, example)
    else:
        if isinstance(example, pd.DataFrame):
            pd.testing.assert_frame_equal(loaded_example, example)
        else:
            assert loaded_example == example

    for test_example in ["saved_example", "manual_example"]:
        if test_example == "saved_example":
            payload = mlflow_model.get_serving_input(local_path)
        else:
            if isinstance(example, pd.DataFrame):
                payload = json.dumps({"dataframe_split": example.to_dict(orient="split")})
            else:
                payload = json.dumps({"inputs": example})

        response = pyfunc_serve_and_score_model(
            model_info.model_uri,
            data=payload,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
            extra_args=["--env-manager", "local"],
        )
        assert response.status_code == 200, response.content
        result = json.loads(response.content.decode("utf-8"))["predictions"]
        result = pd.DataFrame(result).values.tolist()[0]
        np.testing.assert_equal(result, expected_df.values.tolist()[0])


def test_invalid_input_example_warn_when_model_logging():
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            # List[str] is converted to pandas DataFrame
            # after schema enforcement, so this is invalid
            assert isinstance(model_input, list)
            return "string"

    with mock.patch("mlflow.models.model._logger.warning") as mock_warning:
        with mlflow.start_run():
            mlflow.pyfunc.log_model(
                python_model=MyModel(),
                artifact_path="test_model",
                input_example=["some string"],
            )
        mock_warning.assert_called_once()
        assert "Failed to validate serving input example" in mock_warning.call_args[0][0]


def assert_equal(a, b):
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        np.testing.assert_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    else:
        assert a == b


@pytest.mark.parametrize(
    ("example", "signature", "expected_input", "expected_output"),
    [
        (
            pd.DataFrame({"a": ["input1", "input2", "input3"]}),
            ModelSignature(
                Schema([ColSpec(DataType.string, "a")]), Schema([ColSpec(DataType.string)])
            ),
            pd.DataFrame({"a": ["input1", "input2", "input3"]}),
            "string output",
        ),
        (
            np.array([1, 2, 3]),
            ModelSignature(
                Schema([TensorSpec(np.dtype("int64"), (-1,))]),
                Schema([TensorSpec(np.dtype("float64"), (-1,))]),
            ),
            np.array([1, 2, 3]),
            np.array([1.0, 2.0, 3.0]),
        ),
        (
            np.array([1, 2, 3, np.nan]),
            ModelSignature(
                Schema([TensorSpec(np.dtype("float64"), (-1,))]),
                Schema([TensorSpec(np.dtype("float64"), (-1,))]),
            ),
            np.array([1, 2, 3, np.nan]),
            np.array([1.0, 2.0, 3.0, np.nan]),
        ),
        (
            {"a": np.array([1, 2, 3])},
            ModelSignature(
                Schema([TensorSpec(np.dtype("int64"), (-1,), "a")]),
                Schema([TensorSpec(np.dtype("float64"), (-1,), "b")]),
            ),
            {"a": np.array([1, 2, 3])},
            {"b": np.array([1.0, 2.0, 3.0])},
        ),
        (
            ["input1", "input2", "input3"],
            ModelSignature(Schema([ColSpec(DataType.string)]), Schema([ColSpec(DataType.string)])),
            # This is due to _enforce_schema
            pd.DataFrame(["input1", "input2", "input3"]),
            ["input1", "input2", "input3"],
        ),
        (
            [{"a": ["sentence1", "sentence2"], "b": ["answer1", "answer2"]}],
            ModelSignature(
                Schema(
                    [ColSpec(Array(DataType.string), "a"), ColSpec(Array(DataType.string), "b")]
                ),
                Schema([ColSpec(DataType.string, "output")]),
            ),
            pd.DataFrame([{"a": ["sentence1", "sentence2"], "b": ["answer1", "answer2"]}]),
            {"output": "some prediction"},
        ),
        (
            {"messages": [{"role": "user", "content": "some question"}]},
            ModelSignature(
                Schema(
                    [
                        ColSpec(
                            Array(
                                Object(
                                    [
                                        Property("role", DataType.string),
                                        Property("content", DataType.string),
                                    ]
                                )
                            ),
                            "messages",
                        )
                    ]
                ),
                Schema([ColSpec(DataType.string, "output")]),
            ),
            # we assume the field is array so we need another list wrapper
            pd.DataFrame([{"messages": [{"role": "user", "content": "some question"}]}]),
            {"output": "some prediction"},
        ),
    ],
)
def test_input_example_validation_during_logging(
    tmp_path, example, signature, expected_input, expected_output
):
    from mlflow.models import validate_serving_input

    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            assert_equal(model_input, expected_input)
            return expected_output

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=MyModel(),
            artifact_path="test_model",
            input_example=example,
        )
        assert model_info.signature == signature

    mlflow_model = Model.load(model_info.model_uri)
    local_path = _download_artifact_from_uri(model_info.model_uri, output_path=tmp_path)
    serving_input_example = mlflow_model.get_serving_input(local_path)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=serving_input_example,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert response.status_code == 200, response.content
    if is_unified_llm_input(example):
        result = json.loads(response.content.decode("utf-8"))
    else:
        result = json.loads(response.content.decode("utf-8"))["predictions"]
    assert_equal(result, expected_output)

    # make sure validate_serving_input has the same output
    assert convert_input_example_to_serving_input(example) == serving_input_example
    result = validate_serving_input(model_info.model_uri, serving_input_example)
    assert_equal(result, expected_output)


def test_pyfunc_schema_inference_not_generate_trace():
    # Test that the model logging call does not generate a trace.
    # When input example is provided, we run prediction to infer
    # the model signature, but it should not generate a trace.
    class MyModel(mlflow.pyfunc.PythonModel):
        @mlflow.trace()
        def predict(self, context, model_input):
            return model_input

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=MyModel(),
            artifact_path="test_model",
            input_example=["input"],
        )

    # No trace should be generated
    traces = get_traces()
    assert len(traces) == 0

    # Normal prediction should emit a trace
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    loaded_model.predict("input")
    traces = get_traces()
    assert len(traces) == 1


@pytest.mark.parametrize(
    ("data", "schema"),
    [
        ({"a": np.array([1, 2, 3])}, Schema([ColSpec(DataType.long, name="a")])),
        ({"query": "sentence"}, Schema([ColSpec(DataType.string, name="query")])),
        (
            {"query": ["sentence_1", "sentence_2"]},
            Schema([ColSpec(DataType.string, name="query")]),
        ),
        (
            {"query": ["sentence_1", "sentence_2"], "table": "some_table"},
            Schema(
                [
                    ColSpec(DataType.string, name="query"),
                    ColSpec(DataType.string, name="table"),
                ]
            ),
        ),
        (
            [{"query": "sentence"}, {"query": "sentence"}],
            Schema([ColSpec(DataType.string, name="query")]),
        ),
        (
            [
                {"query": ["sentence_1", "sentence_2"], "table": "some_table"},
                {"query": ["sentence_1", "sentence_2"], "table": "some_table"},
            ],
            Schema(
                [
                    ColSpec(DataType.string, name="query"),
                    ColSpec(DataType.string, name="table"),
                ]
            ),
        ),
    ],
)
def test_pyfunc_model_schema_enforcement_with_dicts_and_lists(data, schema):
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    signature = ModelSignature(schema)
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=MyModel(),
            artifact_path="test_model",
            signature=signature,
        )
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    prediction = loaded_model.predict(data)
    if isinstance(data, dict) and all(
        isinstance(x, str) or (isinstance(x, list) and all(isinstance(y, str) for y in x))
        for x in data.values()
    ):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
    pd.testing.assert_frame_equal(prediction, df)

    # Test pandas DataFrame input
    prediction = loaded_model.predict(df)
    pd.testing.assert_frame_equal(prediction, df)


@pytest.mark.parametrize(
    ("data", "schema"),
    [
        ({"query": "sentence"}, Schema([ColSpec(DataType.string, name="query")])),
        (
            {"query": ["sentence_1", "sentence_2"]},
            Schema([ColSpec(DataType.string, name="query")]),
        ),
        (
            {"query": ["sentence_1", "sentence_2"], "table": "some_table"},
            Schema(
                [
                    ColSpec(DataType.string, name="query"),
                    ColSpec(DataType.string, name="table"),
                ]
            ),
        ),
    ],
)
# `instances` is an invalid key for schema with MLflow < 2.9.0
@pytest.mark.parametrize("format_key", ["inputs", "dataframe_split", "dataframe_records"])
def test_pyfunc_model_serving_with_dicts(data, schema, format_key):
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    signature = ModelSignature(schema)
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=MyModel(),
            artifact_path="test_model",
            signature=signature,
        )

    df = (
        pd.DataFrame([data])
        if all(isinstance(x, str) for x in data.values())
        else pd.DataFrame(data)
    )
    if format_key == "inputs":
        payload = {format_key: data}
    elif format_key in ("dataframe_split", "dataframe_records"):
        payload = {format_key: df.to_dict(orient=format_key[10:])}

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps(payload),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert response.status_code == 200, response.content
    result = json.loads(response.content.decode("utf-8"))["predictions"]
    # This is not consistent with batch inference df
    pd.testing.assert_frame_equal(pd.DataFrame(result), df)


@pytest.mark.parametrize(
    ("data", "schema"),
    [
        (
            [{"query": "sentence"}, {"query": "sentence"}],
            Schema([ColSpec(DataType.string, name="query")]),
        ),
        (
            [
                {"query": ["sentence_1", "sentence_2"], "table": "some_table"},
                {"query": ["sentence_1", "sentence_2"], "table": "some_table"},
            ],
            Schema(
                [
                    ColSpec(DataType.string, name="query"),
                    ColSpec(DataType.string, name="table"),
                ]
            ),
        ),
    ],
)
# `inputs`` is an invalid key for schema with MLflow < 2.9.0
@pytest.mark.parametrize("format_key", ["instances", "dataframe_split", "dataframe_records"])
def test_pyfunc_model_serving_with_lists_of_dicts(data, schema, format_key):
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    signature = ModelSignature(schema)
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=MyModel(),
            artifact_path="test_model",
            signature=signature,
        )

    df = pd.DataFrame(data)
    if format_key == "instances":
        payload = {format_key: data}
    elif format_key in ("dataframe_split", "dataframe_records"):
        payload = {format_key: df.to_dict(orient=format_key[10:])}

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps(payload),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert response.status_code == 200, response.content
    result = json.loads(response.content.decode("utf-8"))["predictions"]
    pd.testing.assert_frame_equal(pd.DataFrame(result), df)


@pytest.mark.parametrize(
    ("data", "schema"),
    [
        ({"query": "sentence"}, Schema([ColSpec(DataType.string, name="query")])),
        (
            {"query": ["sentence_1", "sentence_2"]},
            Schema([ColSpec(Array(DataType.string), name="query")]),
        ),
        (
            {"query": {"a": "a", "b": 1}},
            Schema(
                [
                    ColSpec(
                        Object([Property("a", DataType.string), Property("b", DataType.long)]),
                        "query",
                    )
                ]
            ),
        ),
        (
            {"query": ["sentence_1", "sentence_2"], "table": "some_table"},
            Schema(
                [
                    ColSpec(Array(DataType.string), name="query"),
                    ColSpec(DataType.string, name="table"),
                ]
            ),
        ),
        (
            {"query": [{"name": "value", "age": 10}, {"name": "value"}], "table": ["some_table"]},
            Schema(
                [
                    ColSpec(
                        Array(
                            Object(
                                [
                                    Property("name", DataType.string),
                                    Property("age", DataType.long, required=False),
                                ]
                            )
                        ),
                        name="query",
                    ),
                    ColSpec(Array(DataType.string), name="table"),
                ]
            ),
        ),
        (
            [{"query": "sentence"}, {"query": "sentence"}],
            Schema([ColSpec(DataType.string, name="query")]),
        ),
        (
            [
                {"query": ["sentence_1", "sentence_2"], "table": "some_table"},
                {"query": ["sentence_1", "sentence_2"]},
            ],
            Schema(
                [
                    ColSpec(Array(DataType.string), name="query"),
                    ColSpec(DataType.string, name="table", required=False),
                ]
            ),
        ),
    ],
)
def test_pyfunc_model_schema_enforcement_with_objects_and_arrays(data, schema):
    class MyModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.pipeline = "pipeline"

        def predict(self, context, model_input, params=None):
            assert self.pipeline == "pipeline"
            return model_input

    signature = infer_signature(data)
    assert signature.inputs == schema
    pdf = pd.DataFrame(data if isinstance(data, list) else [data])
    assert infer_signature(pdf).inputs == schema

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=MyModel(),
            artifact_path="test_model",
            signature=signature,
        )
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    prediction = loaded_model.predict(data)
    df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
    pd.testing.assert_frame_equal(prediction, df)

    # Test pandas DataFrame input
    prediction = loaded_model.predict(df)
    pd.testing.assert_frame_equal(prediction, df)


@pytest.mark.parametrize(
    "data",
    [
        {"query": "sentence"},
        {"query": ["sentence_1", "sentence_2"]},
        {"query": ["sentence_1", "sentence_2"], "table": "some_table"},
        {"query": [{"name": "value"}, {"name": "value"}], "table": ["some_table"]},
        [{"query": "sentence"}, {"query": "sentence"}],
        [
            {"query": ["sentence_1", "sentence_2"], "table": "some_table"},
            {"query": ["sentence_1", "sentence_2"]},
        ],
        [
            {"query": [{"name": "value"}, {"name": "value"}], "table": ["some_table"]},
            {"query": [{"name": "value", "age": 10}, {"name": "value"}], "table": ["some_table"]},
        ],
    ],
)
@pytest.mark.parametrize("format_key", ["inputs", "dataframe_split", "dataframe_records"])
def test_pyfunc_model_scoring_with_objects_and_arrays(data, format_key):
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=MyModel(),
            artifact_path="test_model",
            signature=infer_signature(data),
        )

    df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])

    if format_key == "inputs":
        payload = {format_key: data}
    elif format_key == "dataframe_split":
        payload = {format_key: df.to_dict(orient="split")}
    elif format_key == "dataframe_records":
        payload = {format_key: df.to_dict(orient="records")}

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps(payload),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert response.status_code == 200, response.content
    result = json.loads(response.content.decode("utf-8"))["predictions"]
    expected_result = df.to_dict(orient="records")
    np.testing.assert_equal(result, expected_result)


@pytest.mark.parametrize(
    "data",
    [
        {"query": "sentence"},
        {"query": ["sentence_1", "sentence_2"]},
        {"query": ["sentence_1", "sentence_2"], "table": "some_table"},
        {"query": [{"name": "value"}, {"name": "value"}], "table": ["some_table"]},
        [{"query": "sentence"}, {"query": "sentence"}],
    ],
)
def test_pyfunc_model_scoring_with_objects_and_arrays_instances(data):
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=MyModel(),
            artifact_path="test_model",
            signature=infer_signature(data),
        )

    df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"instances": data}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert response.status_code == 200, response.content
    result = json.loads(response.content.decode("utf-8"))["predictions"]
    expected_result = df.to_dict(orient="records")
    np.testing.assert_equal(result, expected_result)


@pytest.mark.parametrize(
    "data",
    [
        [{"query": {"a": "b"}, "name": "A"}, {"query": {"a": "c"}, "name": "B"}],
        [
            {"query": ["sentence_1", "sentence_2"], "table": "some_table"},
            {"query": ["sentence_1", "sentence_2"]},
        ],
        [
            {"query": [{"name": "value"}, {"name": "value"}], "table": ["some_table"]},
            {"query": [{"name": "value", "age": 10}, {"name": "value"}], "table": ["some_table"]},
        ],
    ],
)
def test_pyfunc_model_scoring_with_objects_and_arrays_instances_errors(data):
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=MyModel(),
            artifact_path="test_model",
            signature=infer_signature(data),
        )

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"instances": data}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert response.status_code == 400, response.content
    assert "Failed to enforce schema" in json.loads(response.content.decode("utf-8"))["message"]


@pytest.mark.parametrize(
    ("data", "schema"),
    [
        (
            [{"query": "question1"}, {"query": "question2"}],
            Schema([ColSpec(DataType.string, "query")]),
        ),
        (
            [{"query": ["sentence_1", "sentence_2"]}, {"query": ["sentence_1", "sentence_2"]}],
            Schema([ColSpec(DataType.string, "query")]),
        ),
        (
            [
                {"query": ["sentence_1", "sentence_2"], "table": "some_table"},
                {"query": ["sentence_1", "sentence_2"], "table": "some_table"},
            ],
            Schema([ColSpec(DataType.string, "query"), ColSpec(DataType.string, "table")]),
        ),
    ],
)
def test_pyfunc_model_scoring_instances_backwards_compatibility(data, schema):
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=MyModel(),
            artifact_path="test_model",
            signature=ModelSignature(schema),
        )

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"instances": data}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert response.status_code == 200, response.content
    result = json.loads(response.content.decode("utf-8"))["predictions"]
    np.testing.assert_equal(result, data)


@pytest.mark.parametrize(
    ("data", "schema"),
    [
        (
            {
                "netsed_list": [
                    [["a", "b"], ["c", "d"]],
                    [["e", "f"], ["g"]],
                ]
            },
            Schema([ColSpec(Array(Array(DataType.string)), name="netsed_list")]),
        ),
        (
            {
                "numpy_2d_array": [
                    np.array([[np.int32(1), np.int32(2)], [np.int32(3), np.int32(4)]])
                ]
            },
            Schema([ColSpec(Array(Array(DataType.integer)), name="numpy_2d_array")]),
        ),
        (
            {"list_of_np_array": [[np.array(["a", "b"])], [np.array(["c", "d"])]]},
            Schema([ColSpec(Array(Array(DataType.string)), name="list_of_np_array")]),
        ),
    ],
)
def test_pyfunc_model_schema_enforcement_nested_array(data, schema):
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    df = pd.DataFrame.from_records(data)
    signature = infer_signature(df)
    assert signature.inputs == schema

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=MyModel(),
            artifact_path="test_model",
            signature=signature,
        )
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    prediction = loaded_model.predict(df)
    pd.testing.assert_frame_equal(prediction, df)


@pytest.mark.parametrize(
    ("data", "schema"),
    [
        (
            {
                "simple_map": [
                    {"a": 3, "b": 4},
                    {},
                    {"c": 5},
                ]
            },
            Schema([ColSpec(Map(value_type=DataType.long), name="simple_map")]),
        ),
        (
            {
                "simple_map": [
                    {"a": 3, "b": 4},
                    {},
                    {"c": 5},
                ]
            },
            Schema([ColSpec(Map(value_type=DataType.long))]),  # Unnamed column
        ),
        (
            {
                "nested_map": [
                    {"a": {"a1": 3, "a2": 4}, "b": {"b1": 5}},
                    {},
                    {"c": {}},
                ]
            },
            Schema([ColSpec(Map(value_type=Map(value_type=DataType.long)), name="nested_map")]),
        ),
        (
            {
                "array_in_map": [
                    {"a": [1, 2, 3], "b": [4, 5]},
                    {},
                    {"c": []},
                ]
            },
            Schema([ColSpec(Map(value_type=Array(dtype=DataType.long)), name="array_in_map")]),
        ),
        (
            {
                "object_in_map": [
                    {"a": {"key1": "a1", "key2": 1}, "b": {"key1": "b1"}},
                    {},
                    {"c": {"key1": "c1"}},
                ]
            },
            Schema(
                [
                    ColSpec(
                        Map(
                            value_type=Object(
                                [
                                    Property("key1", DataType.string),
                                    Property("key2", DataType.long, required=False),
                                ]
                            )
                        ),
                        name="object_in_map",
                    )
                ]
            ),
        ),
        (
            {
                "map_in_array": [
                    [{"a": 3, "b": 4}, {"c": 5}],
                    [],
                    [{"d": 6}],
                ]
            },
            Schema([ColSpec(Array(dtype=Map(value_type=DataType.long)), name="map_in_array")]),
        ),
        (
            {
                "map_in_object": [
                    {"key1": {"a": 3, "b": 4}, "key2": {"c": 5}},
                    {"key1": {"d": 6}},
                ]
            },
            Schema(
                [
                    ColSpec(
                        Object(
                            [
                                Property("key1", Map(value_type=DataType.long)),
                                Property("key2", Map(value_type=DataType.long), required=False),
                            ]
                        ),
                        name="map_in_object",
                    )
                ]
            ),
        ),
    ],
)
@pytest.mark.parametrize("format_key", ["dataframe_split", "dataframe_records"])
def test_pyfunc_model_schema_enforcement_map_type(data, schema, format_key):
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    df = pd.DataFrame.from_records(data)

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=MyModel(),
            artifact_path="test_model",
            signature=ModelSignature(inputs=schema, outputs=schema),
        )
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    prediction = loaded_model.predict(df)
    pd.testing.assert_frame_equal(prediction, df)

    if format_key == "dataframe_split":
        payload = {format_key: df.to_dict(orient="split")}
    elif format_key == "dataframe_records":
        payload = {format_key: df.to_dict(orient="records")}

    class CustomJsonEncoder(json.JSONEncoder):
        def default(self, o):
            import numpy as np

            if isinstance(o, np.int64):
                return int(o)

            return super().default(o)

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps(payload, cls=CustomJsonEncoder),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert response.status_code == 200, response.content
    result = json.loads(response.content.decode("utf-8"))["predictions"]
    expected_result = df.to_dict(orient="records")
    np.testing.assert_equal(result, expected_result)


@pytest.mark.parametrize(
    ("data", "schema"),
    [
        (
            [
                {
                    "object_column": {"query": ["sentence_1", "sentence_2"], "table": "some_table"},
                    "string_column": "some_string",
                    "array_column": [{"name": "value"}, {"name": "value"}],
                },
                {
                    "object_column": {"query": ["sentence_1", "sentence_2"]},
                    "string_column": "some_string",
                    "array_column": [{"name": "value"}],
                },
            ],
            Schema(
                [
                    ColSpec(
                        Object(
                            [
                                Property("query", Array(DataType.string)),
                                Property("table", DataType.string, required=False),
                            ]
                        ),
                        "object_column",
                    ),
                    ColSpec(DataType.string, "string_column"),
                    ColSpec(
                        Array(Object([Property("name", DataType.string)])),
                        "array_column",
                    ),
                ]
            ),
        ),
    ],
)
@pytest.mark.parametrize("format_key", ["inputs", "dataframe_split", "dataframe_records"])
def test_pyfunc_model_schema_enforcement_complex(data, schema, format_key):
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    df = pd.DataFrame.from_records(data)
    signature = infer_signature(df)
    assert signature.inputs == schema

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=MyModel(),
            artifact_path="test_model",
            signature=signature,
        )
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    prediction = loaded_model.predict(df)
    pd.testing.assert_frame_equal(prediction, df)

    if format_key == "inputs":
        payload = {format_key: data}
    elif format_key == "dataframe_split":
        payload = {format_key: df.to_dict(orient="split")}
    elif format_key == "dataframe_records":
        payload = {format_key: df.to_dict(orient="records")}

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps(payload),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert response.status_code == 200, response.content
    result = json.loads(response.content.decode("utf-8"))["predictions"]
    expected_result = df.to_dict(orient="records")
    np.testing.assert_equal(result, expected_result)


def test_zero_or_one_longs_convert_to_floats():
    zeros = pd.DataFrame([{"temperature": 0}, {"temperature": 0.9}, {"temperature": 1}, {}])
    schema = Schema([ColSpec(DataType.double, name="temperature", required=False)])
    data = _enforce_schema(zeros, schema)
    pd.testing.assert_series_equal(
        data["temperature"], pd.Series([0.0, 0.9, 1.0, np.nan], dtype=np.float64), check_names=False
    )
