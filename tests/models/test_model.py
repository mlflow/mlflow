import os
import pytest
from datetime import date

import mlflow
import pandas as pd
import numpy as np

from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.models import Model
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import _save_example
from mlflow.types.schema import Schema, ColSpec, TensorSpec
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import _dataframe_from_json

from unittest import mock
from scipy.sparse import csc_matrix


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


class TestFlavor:
    @classmethod
    def save_model(cls, path, mlflow_model, signature=None, input_example=None):
        mlflow_model.flavors["flavor1"] = {"a": 1, "b": 2}
        mlflow_model.flavors["flavor2"] = {"x": 1, "y": 2}
        os.makedirs(path)
        if signature is not None:
            mlflow_model.signature = signature
        if input_example is not None:
            _save_example(mlflow_model, input_example, path)
        mlflow_model.save(os.path.join(path, "MLmodel"))


def _log_model_with_signature_and_example(tmp_path, sig, input_example):
    experiment_id = mlflow.create_experiment("test")

    with mlflow.start_run(experiment_id=experiment_id) as run:
        Model.log("some/path", TestFlavor, signature=sig, input_example=input_example)

    local_path = _download_artifact_from_uri(
        "runs:/{}/some/path".format(run.info.run_id), output_path=tmp_path.path("")
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
        path = os.path.join(local_path, loaded_model.saved_input_example_info["artifact_path"])
        x = _dataframe_from_json(path)
        assert x.to_dict(orient="records")[0] == input_example
        assert not hasattr(loaded_model, "databricks_runtime")

        loaded_example = loaded_model.load_input_example(local_path)
        assert isinstance(loaded_example, pd.DataFrame)
        assert loaded_example.to_dict(orient="records")[0] == input_example


def test_model_log_with_databricks_runtime():
    dbr = "8.3.x-snapshot-gpu-ml-scala2.12"
    with TempDir(chdr=True) as tmp, mock.patch(
        "mlflow.models.model.get_databricks_runtime", return_value=dbr
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
        path = os.path.join(local_path, loaded_model.saved_input_example_info["artifact_path"])
        x = _dataframe_from_json(path)
        assert x.to_dict(orient="records")[0] == input_example
        assert loaded_model.databricks_runtime == dbr


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
        x = _dataframe_from_json(path, schema=sig.inputs)

        # date column will get deserialized into string
        input_example["d"] = input_example["d"].apply(lambda x: x.isoformat())
        assert x.equals(input_example)

        loaded_example = loaded_model.load_input_example(local_path)
        assert isinstance(loaded_example, pd.DataFrame)
        assert loaded_example.equals(input_example)


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
        assert np.array_equal(input_example, loaded_example)


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
        assert np.array_equal(input_example.data, loaded_example.data)


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
    import uuid

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
