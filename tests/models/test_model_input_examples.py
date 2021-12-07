import json
import math
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, csc_matrix

from mlflow.models.signature import infer_signature
from mlflow.models.utils import (
    _Example,
    _read_tensor_input_from_json,
    _read_sparse_matrix_from_json,
)
from mlflow.types.utils import TensorsNotSupportedException
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import _dataframe_from_json


@pytest.fixture
def pandas_df_with_all_types():
    df = pd.DataFrame(
        {
            "boolean": [True, False, True],
            "integer": np.array([1, 2, 3], np.int32),
            "long": np.array([1, 2, 3], np.int64),
            "float": np.array([math.pi, 2 * math.pi, 3 * math.pi], np.float32),
            "double": [math.pi, 2 * math.pi, 3 * math.pi],
            "binary": [bytes([1, 2, 3]), bytes([4, 5, 6]), bytes([7, 8, 9])],
            "string": ["a", "b", "c"],
            "boolean_ext": [True, False, True],
            "integer_ext": [1, 2, 3],
            "string_ext": ["a", "b", "c"],
            "array": np.array(["a", "b", "c"]),
        }
    )
    df["boolean_ext"] = df["boolean_ext"].astype("boolean")
    df["integer_ext"] = df["integer_ext"].astype("Int64")
    df["string_ext"] = df["string_ext"].astype("string")
    return df


@pytest.fixture
def df_with_nan():
    return pd.DataFrame(
        {
            "boolean": [True, False, True],
            "integer": np.array([1, 2, 3], np.int32),
            "long": np.array([1, 2, 3], np.int64),
            "float": np.array([np.nan, 2 * math.pi, 3 * math.pi], np.float32),
            "double": [math.pi, np.nan, 3 * math.pi],
            "binary": [bytes([1, 2, 3]), bytes([4, 5, 6]), bytes([7, 8, 9])],
            "string": ["a", "b", "c"],
        }
    )


@pytest.fixture
def dict_of_ndarrays():
    return {
        "1D": np.arange(0, 12, 0.5),
        "2D": np.arange(0, 12, 0.5).reshape(3, 8),
        "3D": np.arange(0, 12, 0.5).reshape(2, 3, 4),
        "4D": np.arange(0, 12, 0.5).reshape(3, 2, 2, 2),
    }


@pytest.fixture
def dict_of_ndarrays_with_nans():
    return {
        "1D": np.array([0.5, np.nan, 2.0]),
        "2D": np.array([[0.1, 0.2], [np.nan, 0.5]]),
        "3D": np.array([[[0.1, np.nan], [0.3, 0.4]], [[np.nan, 0.6], [0.7, np.nan]]]),
    }


@pytest.fixture
def dict_of_sparse_matrix():
    return {
        "sparse_matrix_csc": csc_matrix(np.arange(0, 12, 0.5).reshape(3, 8)),
        "sparse_matrix_csr": csr_matrix(np.arange(0, 12, 0.5).reshape(3, 8)),
    }


def test_input_examples(pandas_df_with_all_types, dict_of_ndarrays):
    sig = infer_signature(pandas_df_with_all_types)
    # test setting example with data frame with all supported data types
    with TempDir() as tmp:
        example = _Example(pandas_df_with_all_types)
        example.save(tmp.path())
        filename = example.info["artifact_path"]
        with open(tmp.path(filename), "r") as f:
            data = json.load(f)
            assert set(data.keys()) == set(("columns", "data"))
        parsed_df = _dataframe_from_json(tmp.path(filename), schema=sig.inputs)
        assert (pandas_df_with_all_types == parsed_df).all().all()
        # the frame read without schema should match except for the binary values
        assert (
            (
                parsed_df.drop(columns=["binary"])
                == _dataframe_from_json(tmp.path(filename)).drop(columns=["binary"])
            )
            .all()
            .all()
        )

    # NB: Drop columns that cannot be encoded by proto_json_utils.pyNumpyEncoder
    new_df = pandas_df_with_all_types.drop(columns=["boolean_ext", "integer_ext", "string_ext"])

    # pass the input as dictionary instead
    with TempDir() as tmp:
        d = {name: new_df[name].values for name in new_df.columns}
        example = _Example(d)
        example.save(tmp.path())
        filename = example.info["artifact_path"]
        parsed_dict = _read_tensor_input_from_json(tmp.path(filename))
        assert d.keys() == parsed_dict.keys()
        # Asserting binary will fail since it is converted to base64 encoded strings.
        # The check above suffices that the binary input is stored.
        del d["binary"]
        for key in d:
            assert np.array_equal(d[key], parsed_dict[key])

    # input passed as numpy array
    new_df = pandas_df_with_all_types.drop(columns=["binary"])
    for col in new_df:
        input_example = new_df[col].to_numpy()
        with TempDir() as tmp:
            example = _Example(input_example)
            example.save(tmp.path())
            filename = example.info["artifact_path"]
            parsed_ary = _read_tensor_input_from_json(tmp.path(filename))
            assert np.array_equal(parsed_ary, input_example)

    # pass multidimensional array
    for col in dict_of_ndarrays:
        input_example = dict_of_ndarrays[col]
        with TempDir() as tmp:
            example = _Example(input_example)
            example.save(tmp.path())
            filename = example.info["artifact_path"]
            parsed_ary = _read_tensor_input_from_json(tmp.path(filename))
            assert np.array_equal(parsed_ary, input_example)

    # pass multidimensional array as a list
    example = np.array([[1, 2, 3]])
    with pytest.raises(TensorsNotSupportedException, match=r"Row '0' has shape \(1, 3\)"):
        _Example([example, example])

    # pass dict with scalars
    with TempDir() as tmp:
        example = {"a": 1, "b": "abc"}
        x = _Example(example)
        x.save(tmp.path())
        filename = x.info["artifact_path"]
        parsed_df = _dataframe_from_json(tmp.path(filename))
        assert example == parsed_df.to_dict(orient="records")[0]


def test_sparse_matrix_input_examples(dict_of_sparse_matrix):
    for example_type, input_example in dict_of_sparse_matrix.items():
        with TempDir() as tmp:
            example = _Example(input_example)
            example.save(tmp.path())
            filename = example.info["artifact_path"]
            parsed_matrix = _read_sparse_matrix_from_json(tmp.path(filename), example_type)
            assert np.array_equal(parsed_matrix.toarray(), input_example.toarray())


def test_input_examples_with_nan(df_with_nan, dict_of_ndarrays_with_nans):
    # test setting example with data frame with NaN values in it
    sig = infer_signature(df_with_nan)
    with TempDir() as tmp:
        example = _Example(df_with_nan)
        example.save(tmp.path())
        filename = example.info["artifact_path"]
        with open(tmp.path(filename), "r") as f:
            data = json.load(f)
            assert set(data.keys()) == set(("columns", "data"))
        parsed_df = _dataframe_from_json(tmp.path(filename), schema=sig.inputs)
        # by definition of NaN, NaN == NaN is False but NaN != NaN is True
        assert (
            ((df_with_nan == parsed_df) | ((df_with_nan != df_with_nan) & (parsed_df != parsed_df)))
            .all()
            .all()
        )
        # the frame read without schema should match except for the binary values
        no_schema_df = _dataframe_from_json(tmp.path(filename))
        a = parsed_df.drop(columns=["binary"])
        b = no_schema_df.drop(columns=["binary"])
        assert ((a == b) | ((a != a) & (b != b))).all().all()

    # pass multidimensional array
    for col in dict_of_ndarrays_with_nans:
        input_example = dict_of_ndarrays_with_nans[col]
        sig = infer_signature(input_example)
        with TempDir() as tmp:
            example = _Example(input_example)
            example.save(tmp.path())
            filename = example.info["artifact_path"]
            parsed_ary = _read_tensor_input_from_json(tmp.path(filename), schema=sig.inputs)
            assert np.array_equal(parsed_ary, input_example, equal_nan=True)

            # without a schema/dtype specified, the resulting tensor will keep the None type
            no_schema_df = _read_tensor_input_from_json(tmp.path(filename))
            assert np.array_equal(
                no_schema_df, np.where(np.isnan(input_example), None, input_example)
            )
