import os
import pathlib
import uuid
from datetime import date
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import sklearn.datasets
import sklearn.neighbors
from packaging.version import Version
from scipy.sparse import csc_matrix

import mlflow
from mlflow.models import Model, ModelSignature, infer_signature, set_model, validate_schema
from mlflow.models.model import METADATA_FILES, SET_MODEL_ERROR
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex
from mlflow.models.utils import _read_example, _save_example
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.schema import ColSpec, DataType, ParamSchema, ParamSpec, Schema, TensorSpec
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _validate_and_prepare_target_save_path
from mlflow.utils.proto_json_utils import dataframe_from_raw_json


@pytest.fixture(scope="module")
def iris_data():
    iris = sklearn.datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    return x, y


@pytest.fixture(scope="module")
def sklearn_knn_model(iris_data):
    x, y = iris_data
    knn_model = sklearn.neighbors.KNeighborsClassifier()
    knn_model.fit(x, y)
    return knn_model


def test_model_save_load():
    m = Model(
        artifact_path="some/path",
        run_id="123",
        flavors={"flavor1": {"a": 1, "b": 2}, "flavor2": {"x": 1, "y": 2}},
        signature=ModelSignature(
            inputs=Schema([ColSpec("integer", "x"), ColSpec("integer", "y")]),
            outputs=Schema([ColSpec(name=None, type="double")]),
        ),
        saved_input_example_info={"x": 1, "y": 2},
    )
    assert m.get_input_schema() == m.signature.inputs
    assert m.get_output_schema() == m.signature.outputs
    x = Model(artifact_path="some/other/path", run_id="1234")
    assert x.get_input_schema() is None
    assert x.get_output_schema() is None

    n = Model(
        artifact_path="some/path",
        run_id="123",
        flavors={"flavor1": {"a": 1, "b": 2}, "flavor2": {"x": 1, "y": 2}},
        signature=ModelSignature(
            inputs=Schema([ColSpec("integer", "x"), ColSpec("integer", "y")]),
            outputs=Schema([ColSpec(name=None, type="double")]),
        ),
        saved_input_example_info={"x": 1, "y": 2},
    )
    n.utc_time_created = m.utc_time_created
    n.model_uuid = m.model_uuid
    assert m == n
    n.signature = None
    assert m != n
    with TempDir() as tmp:
        m.save(tmp.path("model"))
        o = Model.load(tmp.path("model"))
    assert m == o
    assert m.to_json() == o.to_json()
    assert m.to_yaml() == o.to_yaml()


def test_model_load_remote(tmp_path, mock_s3_bucket):
    model = Model(
        artifact_path="some/path",
        run_id="123",
        flavors={"flavor1": {"a": 1, "b": 2}, "flavor2": {"x": 1, "y": 2}},
        signature=ModelSignature(
            inputs=Schema([ColSpec("integer", "x"), ColSpec("integer", "y")]),
            outputs=Schema([ColSpec(name=None, type="double")]),
        ),
        saved_input_example_info={"x": 1, "y": 2},
    )
    model_path = tmp_path / "MLmodel"
    model.save(model_path)

    artifact_root = f"s3://{mock_s3_bucket}"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifact(str(model_path))

    model_reloaded_1 = Model.load(f"{artifact_root}/MLmodel")
    assert model_reloaded_1 == model

    model_reloaded_2 = Model.load(artifact_root)
    assert model_reloaded_2 == model


class TestFlavor:
    @classmethod
    def save_model(cls, path, mlflow_model, signature=None, input_example=None):
        mlflow_model.flavors["flavor1"] = {"a": 1, "b": 2}
        mlflow_model.flavors["flavor2"] = {"x": 1, "y": 2}
        _validate_and_prepare_target_save_path(path)
        if signature is not None:
            mlflow_model.signature = signature
        if input_example is not None:
            _save_example(mlflow_model, input_example, path)
        mlflow_model.save(os.path.join(path, "MLmodel"))


def _log_model_with_signature_and_example(
    tmp_path, sig, input_example, metadata=None, resources=None
):
    experiment_id = mlflow.create_experiment("test")

    with mlflow.start_run(experiment_id=experiment_id) as run:
        Model.log(
            "some/path",
            TestFlavor,
            signature=sig,
            input_example=input_example,
            metadata=metadata,
            resources=resources,
        )

    # TODO: remove this after replacing all `with TempDir(chdr=True) as tmp`
    # with tmp_path fixture
    output_path = tmp_path if isinstance(tmp_path, pathlib.PosixPath) else tmp_path.path("")
    local_path = _download_artifact_from_uri(
        f"runs:/{run.info.run_id}/some/path", output_path=output_path
    )

    return local_path, run


def test_model_log():
    with TempDir(chdr=True) as tmp:
        sig = ModelSignature(
            inputs=Schema([ColSpec("integer", "x"), ColSpec("integer", "y")]),
            outputs=Schema([ColSpec(name=None, type="double")]),
        )
        input_example = {"x": 1, "y": 2}
        local_path, r = _log_model_with_signature_and_example(tmp, sig, input_example)

        loaded_model = Model.load(os.path.join(local_path, "MLmodel"))
        assert loaded_model.run_id == r.info.run_id
        assert loaded_model.artifact_path == "some/path"
        assert loaded_model.flavors == {
            "flavor1": {"a": 1, "b": 2},
            "flavor2": {"x": 1, "y": 2},
        }
        assert loaded_model.signature == sig
        x = _read_example(
            Model(saved_input_example_info=loaded_model.saved_input_example_info), local_path
        )
        assert x == input_example
        assert not hasattr(loaded_model, "databricks_runtime")

        loaded_example = loaded_model.load_input_example(local_path)
        assert loaded_example == input_example

        assert Version(loaded_model.mlflow_version) == Version(mlflow.version.VERSION)


def test_model_info():
    with TempDir(chdr=True) as tmp:
        sig = ModelSignature(
            inputs=Schema([ColSpec("integer", "x"), ColSpec("integer", "y")]),
            outputs=Schema([ColSpec(name=None, type="double")]),
        )
        input_example = {"x": 1, "y": 2}

        experiment_id = mlflow.create_experiment("test")
        with mlflow.start_run(experiment_id=experiment_id) as run:
            model_info = Model.log(
                "some/path", TestFlavor, signature=sig, input_example=input_example
            )
        model_uri = f"runs:/{run.info.run_id}/some/path"

        model_info_fetched = mlflow.models.get_model_info(model_uri)
        with pytest.warns(
            FutureWarning,
            match="Field signature_dict is deprecated since v1.28.1. Use signature instead",
        ):
            assert model_info_fetched.signature_dict == sig.to_dict()

        local_path = _download_artifact_from_uri(model_uri, output_path=tmp.path(""))

        assert model_info.run_id == run.info.run_id
        assert model_info_fetched.run_id == run.info.run_id
        assert model_info.artifact_path == "some/path"
        assert model_info_fetched.artifact_path == "some/path"
        assert model_info.model_uri == model_uri
        assert model_info_fetched.model_uri == model_uri

        loaded_model = Model.load(os.path.join(local_path, "MLmodel"))
        assert model_info.utc_time_created == loaded_model.utc_time_created
        assert model_info_fetched.utc_time_created == loaded_model.utc_time_created
        assert model_info.model_uuid == loaded_model.model_uuid
        assert model_info_fetched.model_uuid == loaded_model.model_uuid

        assert model_info.flavors == {
            "flavor1": {"a": 1, "b": 2},
            "flavor2": {"x": 1, "y": 2},
        }

        x = _read_example(
            Model(saved_input_example_info=model_info.saved_input_example_info), local_path
        )
        assert x == input_example

        model_signature = model_info_fetched.signature
        assert model_info.signature_dict == sig.to_dict()
        assert model_signature.to_dict() == sig.to_dict()

        assert model_info.mlflow_version == loaded_model.mlflow_version
        assert model_info_fetched.mlflow_version == loaded_model.mlflow_version


def test_model_info_with_model_version(tmp_path):
    experiment_id = mlflow.create_experiment("test", artifact_location=str(tmp_path))
    with mlflow.start_run(experiment_id=experiment_id):
        model_info = Model.log("some/path", TestFlavor, registered_model_name="model_abc")
        assert model_info.registered_model_version == 1
        model_info = Model.log("some/path", TestFlavor, registered_model_name="model_abc")
        assert model_info.registered_model_version == 2
        model_info = Model.log("some/path", TestFlavor)
        assert model_info.registered_model_version is None


def test_model_metadata():
    with TempDir(chdr=True) as tmp:
        metadata = {"metadata_key": "metadata_value"}
        local_path, _ = _log_model_with_signature_and_example(tmp, None, None, metadata)
        loaded_model = Model.load(os.path.join(local_path, "MLmodel"))
        assert loaded_model.metadata["metadata_key"] == "metadata_value"


def test_load_model_without_mlflow_version():
    with TempDir(chdr=True) as tmp:
        model = Model(artifact_path="some/path", run_id="1234", mlflow_version=None)
        path = tmp.path("model")
        with open(path, "w") as out:
            model.to_yaml(out)
        loaded_model = Model.load(path)

        assert loaded_model.mlflow_version is None


def test_model_log_with_databricks_runtime():
    dbr_version = "8.3.x"
    with TempDir(chdr=True) as tmp, mock.patch(
        "mlflow.models.model.get_databricks_runtime_version", return_value=dbr_version
    ):
        sig = ModelSignature(
            inputs=Schema([ColSpec("integer", "x"), ColSpec("integer", "y")]),
            outputs=Schema([ColSpec(name=None, type="double")]),
        )
        input_example = {"x": 1, "y": 2}
        local_path, r = _log_model_with_signature_and_example(tmp, sig, input_example)

        loaded_model = Model.load(os.path.join(local_path, "MLmodel"))
        assert loaded_model.run_id == r.info.run_id
        assert loaded_model.artifact_path == "some/path"
        assert loaded_model.flavors == {
            "flavor1": {"a": 1, "b": 2},
            "flavor2": {"x": 1, "y": 2},
        }
        assert loaded_model.signature == sig
        x = _read_example(
            Model(saved_input_example_info=loaded_model.saved_input_example_info), local_path
        )
        assert x == input_example
        assert loaded_model.databricks_runtime == dbr_version


def test_model_log_with_input_example_succeeds():
    with TempDir(chdr=True) as tmp:
        sig = ModelSignature(
            inputs=Schema(
                [
                    ColSpec("integer", "a"),
                    ColSpec("string", "b"),
                    ColSpec("boolean", "c"),
                    ColSpec("string", "d"),
                    ColSpec("datetime", "e"),
                ]
            ),
            outputs=Schema([ColSpec(name=None, type="double")]),
        )
        input_example = pd.DataFrame(
            {
                "a": np.int32(1),
                "b": "test string",
                "c": True,
                "d": date.today(),
                "e": np.datetime64("2020-01-01T00:00:00"),
            },
            index=[0],
        )

        local_path, _ = _log_model_with_signature_and_example(tmp, sig, input_example)
        loaded_model = Model.load(os.path.join(local_path, "MLmodel"))
        path = os.path.join(local_path, loaded_model.saved_input_example_info["artifact_path"])
        x = dataframe_from_raw_json(path, schema=sig.inputs)

        # date column will get deserialized into string
        input_example["d"] = input_example["d"].apply(lambda x: x.isoformat())
        # datetime Datatype numpy type is [ns]
        input_example["e"] = input_example["e"].astype(np.dtype("datetime64[ns]"))
        pd.testing.assert_frame_equal(x, input_example)

        loaded_example = loaded_model.load_input_example(local_path)
        assert isinstance(loaded_example, pd.DataFrame)
        pd.testing.assert_frame_equal(loaded_example, input_example)


def test_model_input_example_with_params_log_load_succeeds(tmp_path):
    pdf = pd.DataFrame(
        {
            "a": np.int32(1),
            "b": "test string",
            "c": True,
            "d": date.today(),
            "e": np.datetime64("2020-01-01T00:00:00"),
        },
        index=[0],
    )
    input_example = (pdf, {"a": 1, "b": "string"})

    sig = ModelSignature(
        inputs=Schema(
            [
                ColSpec("integer", "a"),
                ColSpec("string", "b"),
                ColSpec("boolean", "c"),
                ColSpec("string", "d"),
                ColSpec("datetime", "e"),
            ]
        ),
        outputs=Schema([ColSpec(name=None, type="double")]),
        params=ParamSchema(
            [ParamSpec("a", DataType.long, 1), ParamSpec("b", DataType.string, "string")]
        ),
    )

    local_path, _ = _log_model_with_signature_and_example(tmp_path, sig, input_example)
    loaded_model = Model.load(os.path.join(local_path, "MLmodel"))

    # date column will get deserialized into string
    pdf["d"] = pdf["d"].apply(lambda x: x.isoformat())
    loaded_example = loaded_model.load_input_example(local_path)
    assert isinstance(loaded_example, pd.DataFrame)
    # datetime Datatype numpy type is [ns]
    pdf["e"] = pdf["e"].astype(np.dtype("datetime64[ns]"))
    pd.testing.assert_frame_equal(loaded_example, pdf)

    params = loaded_model.load_input_example_params(local_path)
    assert params == input_example[1]


def test_model_load_input_example_numpy():
    with TempDir(chdr=True) as tmp:
        input_example = np.array([[3, 4, 5]], dtype=np.int32)
        sig = ModelSignature(
            inputs=Schema([TensorSpec(type=input_example.dtype, shape=input_example.shape)]),
            outputs=Schema([ColSpec(name=None, type="double")]),
        )

        local_path, _ = _log_model_with_signature_and_example(tmp, sig, input_example)
        loaded_model = Model.load(os.path.join(local_path, "MLmodel"))
        loaded_example = loaded_model.load_input_example(local_path)

        assert isinstance(loaded_example, np.ndarray)
        np.testing.assert_array_equal(input_example, loaded_example)


def test_model_load_input_example_scipy():
    with TempDir(chdr=True) as tmp:
        input_example = csc_matrix(np.arange(0, 12, 0.5).reshape(3, 8))
        sig = ModelSignature(
            inputs=Schema([TensorSpec(type=input_example.data.dtype, shape=input_example.shape)]),
            outputs=Schema([ColSpec(name=None, type="double")]),
        )

        local_path, _ = _log_model_with_signature_and_example(tmp, sig, input_example)
        loaded_model = Model.load(os.path.join(local_path, "MLmodel"))
        loaded_example = loaded_model.load_input_example(local_path)

        assert isinstance(loaded_example, csc_matrix)
        np.testing.assert_array_equal(input_example.data, loaded_example.data)


def test_model_load_input_example_failures():
    with TempDir(chdr=True) as tmp:
        input_example = np.array([[3, 4, 5]], dtype=np.int32)
        sig = ModelSignature(
            inputs=Schema([TensorSpec(type=input_example.dtype, shape=input_example.shape)]),
            outputs=Schema([ColSpec(name=None, type="double")]),
        )

        local_path, _ = _log_model_with_signature_and_example(tmp, sig, input_example)
        loaded_model = Model.load(os.path.join(local_path, "MLmodel"))
        loaded_example = loaded_model.load_input_example(local_path)
        assert loaded_example is not None

        with pytest.raises(FileNotFoundError, match="No such file or directory"):
            loaded_model.load_input_example(os.path.join(local_path, "folder_which_does_not_exist"))

        path = os.path.join(local_path, loaded_model.saved_input_example_info["artifact_path"])
        os.remove(path)
        with pytest.raises(FileNotFoundError, match="No such file or directory"):
            loaded_model.load_input_example(local_path)


def test_model_load_input_example_no_signature():
    with TempDir(chdr=True) as tmp:
        input_example = np.array([[3, 4, 5]], dtype=np.int32)
        sig = ModelSignature(
            inputs=Schema([TensorSpec(type=input_example.dtype, shape=input_example.shape)]),
            outputs=Schema([ColSpec(name=None, type="double")]),
        )

        local_path, _ = _log_model_with_signature_and_example(tmp, sig, input_example=None)
        loaded_model = Model.load(os.path.join(local_path, "MLmodel"))
        loaded_example = loaded_model.load_input_example(local_path)
        assert loaded_example is None


def _is_valid_uuid(val):
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False


def test_model_uuid():
    m = Model()
    assert m.model_uuid is not None
    assert _is_valid_uuid(m.model_uuid)

    m2 = Model()
    assert m.model_uuid != m2.model_uuid

    m_dict = m.to_dict()
    assert m_dict["model_uuid"] == m.model_uuid
    m3 = Model.from_dict(m_dict)
    assert m3.model_uuid == m.model_uuid

    m_dict.pop("model_uuid")
    m4 = Model.from_dict(m_dict)
    assert m4.model_uuid is None


def test_validate_schema(sklearn_knn_model, iris_data, tmp_path):
    sk_model_path = os.path.join(tmp_path, "sk_model")
    X, y = iris_data
    signature = infer_signature(X, y)
    mlflow.sklearn.save_model(
        sklearn_knn_model,
        sk_model_path,
        signature=signature,
    )

    validate_schema(X, signature.inputs)
    prediction = sklearn_knn_model.predict(X)
    reloaded_model = mlflow.sklearn.load_model(sk_model_path)
    np.testing.assert_array_equal(prediction, reloaded_model.predict(X))
    validate_schema(prediction, signature.outputs)


def test_save_load_input_example_without_conversion(tmp_path):
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    input_example = {
        "messages": [
            {"role": "user", "content": "Hello!"},
        ]
    }
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            python_model=MyModel(),
            artifact_path="test_model",
            input_example=input_example,
        )
        local_path = _download_artifact_from_uri(
            f"runs:/{run.info.run_id}/test_model", output_path=tmp_path
        )
    loaded_model = Model.load(os.path.join(local_path, "MLmodel"))
    assert loaded_model.saved_input_example_info["type"] == "json_object"
    loaded_example = loaded_model.load_input_example(local_path)
    assert loaded_example == input_example


def test_model_saved_by_save_model_can_be_loaded(tmp_path, sklearn_knn_model):
    mlflow.sklearn.save_model(sklearn_knn_model, tmp_path)
    info = Model.load(tmp_path).get_model_info()
    assert info.run_id is None
    assert info.artifact_path is None


def test_copy_metadata(mock_is_in_databricks, sklearn_knn_model):
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(sklearn_knn_model, "model")

    artifact_path = mlflow.artifacts.download_artifacts(model_info.model_uri)
    metadata_path = os.path.join(artifact_path, "metadata")
    # Metadata should be copied only in Databricks
    if mock_is_in_databricks.return_value:
        assert set(os.listdir(metadata_path)) == set(METADATA_FILES)
    else:
        assert not os.path.exists(metadata_path)
    mock_is_in_databricks.assert_called_once()


class LegacyTestFlavor:
    @classmethod
    def save_model(cls, path, mlflow_model):
        mlflow_model.flavors["flavor1"] = {"a": 1, "b": 2}
        mlflow_model.flavors["flavor2"] = {"x": 1, "y": 2}
        _validate_and_prepare_target_save_path(path)
        mlflow_model.save(os.path.join(path, "MLmodel"))


def test_legacy_flavor(mock_is_in_databricks):
    with mlflow.start_run():
        model_info = Model.log("some/path", LegacyTestFlavor)

    artifact_path = _download_artifact_from_uri(model_info.model_uri)
    metadata_path = os.path.join(artifact_path, "metadata")
    # Metadata should be copied only in Databricks
    if mock_is_in_databricks.return_value:
        assert set(os.listdir(metadata_path)) == {"MLmodel"}
    else:
        assert not os.path.exists(metadata_path)
    mock_is_in_databricks.assert_called_once()


def test_pyfunc_set_model():
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            return model_input

    set_model(MyModel())
    assert isinstance(mlflow.models.model.__mlflow_model__, mlflow.pyfunc.PythonModel)


def test_langchain_set_model():
    from langchain.chains import LLMChain

    def create_openai_llmchain():
        from langchain.llms import OpenAI
        from langchain.prompts import PromptTemplate

        llm = OpenAI(temperature=0.9, openai_api_key="api_key")
        prompt = PromptTemplate(
            input_variables=["product"],
            template="What is a good name for a company that makes {product}?",
        )
        model = LLMChain(llm=llm, prompt=prompt)
        set_model(model)

    create_openai_llmchain()
    assert isinstance(mlflow.models.model.__mlflow_model__, LLMChain)


def test_error_set_model(sklearn_knn_model):
    with pytest.raises(mlflow.MlflowException, match=SET_MODEL_ERROR):
        set_model(sklearn_knn_model)


def test_model_resources():
    expected_resources = {
        "api_version": "1",
        "databricks": {
            "serving_endpoint": [
                {"name": "databricks-mixtral-8x7b-instruct"},
                {"name": "databricks-bge-large-en"},
                {"name": "azure-eastus-model-serving-2_vs_endpoint"},
            ],
            "vector_search_index": [{"name": "rag.studio_bugbash.databricks_docs_index"}],
        },
    }
    with TempDir(chdr=True) as tmp:
        resources = [
            DatabricksServingEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct"),
            DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
            DatabricksServingEndpoint(endpoint_name="azure-eastus-model-serving-2_vs_endpoint"),
            DatabricksVectorSearchIndex(index_name="rag.studio_bugbash.databricks_docs_index"),
        ]
        local_path, _ = _log_model_with_signature_and_example(tmp, None, None, resources=resources)
        loaded_model = Model.load(os.path.join(local_path, "MLmodel"))
        assert loaded_model.resources == expected_resources
