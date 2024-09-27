import json
import math
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import sklearn.neighbors as knn
from scipy.sparse import csc_matrix, csr_matrix
from sklearn import datasets
from sklearn.base import BaseEstimator, ClassifierMixin

import mlflow
from mlflow.models import Model
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.models.utils import (
    _Example,
    _read_sparse_matrix_from_json,
    parse_inputs_data,
)
from mlflow.types import DataType
from mlflow.types.schema import ColSpec, Schema, TensorSpec
from mlflow.types.utils import TensorsNotSupportedException
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import dataframe_from_raw_json


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
def df_without_columns():
    return pd.DataFrame({0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]})


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
        with open(tmp.path(filename)) as f:
            data = json.load(f)
            assert set(data.keys()) == {"columns", "data"}
        parsed_df = dataframe_from_raw_json(tmp.path(filename), schema=sig.inputs)
        assert (pandas_df_with_all_types == parsed_df).all().all()
        # the frame read without schema should match except for the binary values
        assert (
            (
                parsed_df.drop(columns=["binary"])
                == dataframe_from_raw_json(tmp.path(filename)).drop(columns=["binary"])
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
        parsed_dict = parse_inputs_data(tmp.path(filename))
        assert d.keys() == parsed_dict.keys()
        # Asserting binary will fail since it is converted to base64 encoded strings.
        # The check above suffices that the binary input is stored.
        del d["binary"]
        for key in d:
            np.testing.assert_array_equal(d[key], parsed_dict[key])

    # input passed as numpy array
    new_df = pandas_df_with_all_types.drop(columns=["binary"])
    for col in new_df:
        input_example = new_df[col].to_numpy()
        with TempDir() as tmp:
            example = _Example(input_example)
            example.save(tmp.path())
            filename = example.info["artifact_path"]
            parsed_ary = parse_inputs_data(tmp.path(filename))
            np.testing.assert_array_equal(parsed_ary, input_example)

    # pass multidimensional array
    for col in dict_of_ndarrays:
        input_example = dict_of_ndarrays[col]
        with TempDir() as tmp:
            example = _Example(input_example)
            example.save(tmp.path())
            filename = example.info["artifact_path"]
            parsed_ary = parse_inputs_data(tmp.path(filename))
            np.testing.assert_array_equal(parsed_ary, input_example)

    # pass multidimensional array as a list
    example = np.array([[1, 2, 3]])
    with pytest.raises(
        TensorsNotSupportedException,
        match=r"Numpy arrays in list are not supported as input examples.",
    ):
        _Example([example, example])

    # pass dict with scalars
    with TempDir() as tmp:
        example = {"a": 1, "b": "abc"}
        x = _Example(example)
        x.save(tmp.path())
        filename = x.info["artifact_path"]
        with open(tmp.path(filename)) as f:
            parsed_data = json.load(f)
        assert example == parsed_data


def test_pandas_orients_for_input_examples(
    pandas_df_with_all_types, df_without_columns, dict_of_ndarrays
):
    # test setting example with data frame with all supported data types
    with TempDir() as tmp:
        example = _Example(pandas_df_with_all_types)
        example.save(tmp.path())
        filename = example.info["artifact_path"]
        assert example.info["type"] == "dataframe"
        assert example.info["pandas_orient"] == "split"
        with open(tmp.path(filename)) as f:
            data = json.load(f)
            dataframe = pd.read_json(
                json.dumps(data), orient=example.info["pandas_orient"], precise_float=True
            )
            assert (
                (
                    pandas_df_with_all_types.drop(columns=["binary"])
                    == dataframe.drop(columns=["binary"])
                )
                .all()
                .all()
            )

    with TempDir() as tmp:
        example = _Example(df_without_columns)
        example.save(tmp.path())
        filename = example.info["artifact_path"]
        assert example.info["type"] == "dataframe"
        assert example.info["pandas_orient"] == "values"
        with open(tmp.path(filename)) as f:
            data = json.load(f)
            assert set(data.keys()) == {"data"}
            # NOTE: when no column names are provided (i.e. values orient),
            # saving an example adds a "data" key rather than directly storing the plain data
            data = data["data"]
            dataframe = pd.read_json(json.dumps(data), orient=example.info["pandas_orient"])
            assert (dataframe == df_without_columns).all().all()

    # pass dict with scalars
    with TempDir() as tmp:
        example = {"a": 1, "b": "abc"}
        x = _Example(example)
        x.save(tmp.path())
        filename = x.info["artifact_path"]
        assert x.info["type"] == "json_object"
        with open(tmp.path(filename)) as f:
            parsed_json = json.load(f)
            assert parsed_json == example


def test_sparse_matrix_input_examples(dict_of_sparse_matrix):
    for example_type, input_example in dict_of_sparse_matrix.items():
        with TempDir() as tmp:
            example = _Example(input_example)
            example.save(tmp.path())
            filename = example.info["artifact_path"]
            assert example.info["type"] == example_type
            parsed_matrix = _read_sparse_matrix_from_json(tmp.path(filename), example_type)
            np.testing.assert_array_equal(parsed_matrix.toarray(), input_example.toarray())


def test_input_examples_with_nan(df_with_nan, dict_of_ndarrays_with_nans):
    # test setting example with data frame with NaN values in it
    sig = infer_signature(df_with_nan)
    with TempDir() as tmp:
        example = _Example(df_with_nan)
        example.save(tmp.path())
        filename = example.info["artifact_path"]
        assert example.info["type"] == "dataframe"
        assert example.info["pandas_orient"] == "split"
        with open(tmp.path(filename)) as f:
            data = json.load(f)
            assert set(data.keys()) == {"columns", "data"}
            pd.read_json(json.dumps(data), orient=example.info["pandas_orient"])

        parsed_df = dataframe_from_raw_json(tmp.path(filename), schema=sig.inputs)

        # by definition of NaN, NaN == NaN is False but NaN != NaN is True
        assert (
            ((df_with_nan == parsed_df) | ((df_with_nan != df_with_nan) & (parsed_df != parsed_df)))
            .all()
            .all()
        )
        # the frame read without schema should match except for the binary values
        no_schema_df = dataframe_from_raw_json(tmp.path(filename))
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
            assert example.info["type"] == "ndarray"
            parsed_ary = parse_inputs_data(tmp.path(filename), schema=sig.inputs)
            assert np.array_equal(parsed_ary, input_example, equal_nan=True)

            # without a schema/dtype specified, the resulting tensor will keep the None type
            no_schema_df = parse_inputs_data(tmp.path(filename))
            np.testing.assert_array_equal(
                no_schema_df, np.where(np.isnan(input_example), None, input_example)
            )


class DummySklearnModel(BaseEstimator, ClassifierMixin):
    def __init__(self, output_shape=(1,)):
        self.output_shape = output_shape

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        full_output_shape = (n_samples,) + self.output_shape
        return np.zeros(full_output_shape, dtype=np.dtype("int64"))


@pytest.mark.parametrize(
    ("input_is_tabular", "output_shape", "expected_signature"),
    [
        # When the input example is column-based, output 1D numpy arrays are interpretted `ColSpec`s
        (
            True,
            (),
            ModelSignature(
                inputs=Schema([ColSpec(name="feature", type=DataType.string)]),
                outputs=Schema([ColSpec(type=DataType.long)]),
            ),
        ),
        # But if the output numpy array has higher dimensions, fallback to interpretting the model
        # output as `TensorSpec`s.
        (
            True,
            (2,),
            ModelSignature(
                inputs=Schema([ColSpec(name="feature", type=DataType.string)]),
                outputs=Schema([TensorSpec(np.dtype("int64"), (-1, 2))]),
            ),
        ),
        # If the input example is tensor-based, intrepret output numpy arrays as `TensorSpec`s
        (
            False,
            (),
            ModelSignature(
                inputs=Schema([TensorSpec(np.dtype("int64"), (-1, 1))]),
                outputs=Schema([TensorSpec(np.dtype("int64"), (-1,))]),
            ),
        ),
    ],
)
def test_infer_signature_with_input_example(input_is_tabular, output_shape, expected_signature):
    model = DummySklearnModel(output_shape=output_shape)
    artifact_path = "model"
    example = pd.DataFrame({"feature": ["value"]}) if input_is_tabular else np.array([[1]])

    with mlflow.start_run():
        mlflow.sklearn.log_model(model, artifact_path=artifact_path, input_example=example)
        model_uri = mlflow.get_artifact_uri(artifact_path)

    mlflow_model = Model.load(model_uri)
    assert mlflow_model.signature == expected_signature


def test_infer_signature_from_example_can_be_disabled():
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            DummySklearnModel(output_shape=()),
            artifact_path=artifact_path,
            input_example=np.array([[1]]),
            signature=False,
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    mlflow_model = Model.load(model_uri)
    assert mlflow_model.signature is None


def test_infer_signature_raises_if_predict_on_input_example_fails(monkeypatch):
    monkeypatch.setenv("MLFLOW_TESTING", "false")

    class ErrorModel(BaseEstimator, ClassifierMixin):
        def fit(self, X, y=None):
            return self

        def predict(self, X):
            raise Exception("oh no!")

    with mock.patch("mlflow.models.model._logger.warning") as mock_warning:
        with mlflow.start_run():
            mlflow.sklearn.log_model(
                ErrorModel(), artifact_path="model", input_example=np.array([[1]])
            )
        mock_warning.assert_called_once()
        assert "Failed to validate serving input example" in mock_warning.call_args[0][0]


@pytest.fixture(scope="module")
def iris_model():
    X, y = datasets.load_iris(return_X_y=True, as_frame=True)
    return knn.KNeighborsClassifier().fit(X, y)


@pytest.mark.parametrize(
    "input_example",
    [
        {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2,
        },
        pd.DataFrame([[5.1, 3.5, 1.4, 0.2]]),
        pd.DataFrame(
            {
                "sepal length (cm)": 5.1,
                "sepal width (cm)": 3.5,
                "petal length (cm)": 1.4,
                "petal width (cm)": 0.2,
            },
            index=[0],
        ),
    ],
)
def test_infer_signature_on_multi_column_input_examples(input_example, iris_model):
    artifact_path = "model"

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            iris_model, artifact_path=artifact_path, input_example=input_example
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    mlflow_model = Model.load(model_uri)
    input_columns = mlflow_model.signature.inputs.inputs
    assert len(input_columns) == 4
    assert all(col.type == DataType.double for col in input_columns)
    assert mlflow_model.signature.outputs == Schema([ColSpec(type=DataType.long)])


@pytest.mark.parametrize(
    "input_example",
    ["some string", bytes([1, 2, 3])],
)
def test_infer_signature_on_scalar_input_examples(input_example):
    class IdentitySklearnModel(BaseEstimator, ClassifierMixin):
        def fit(self, X, y=None):
            return self

        def predict(self, X):
            if isinstance(X, pd.DataFrame):
                return X
            raise Exception("Unsupported input type")

    artifact_path = "model"

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            IdentitySklearnModel(), artifact_path=artifact_path, input_example=input_example
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    mlflow_model = Model.load(model_uri)
    signature = mlflow_model.signature
    assert isinstance(signature, ModelSignature)
    assert signature.inputs.inputs[0].name is None
    t = DataType.string if isinstance(input_example, str) else DataType.binary
    assert signature == ModelSignature(
        inputs=Schema([ColSpec(type=t)]),
        outputs=Schema([ColSpec(name=0, type=t)]),
    )
    # test that a single string still passes pyfunc schema enforcement
    mlflow.pyfunc.load_model(model_uri).predict(input_example)
