import os

import mlflow
from mlflow.models.utils import _save_example

from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.models import Model
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import _dataframe_from_json


def test_model_save_load():
    m = Model(artifact_path="some/path",
              run_id="123",
              flavors={
                  "flavor1": {"a": 1, "b": 2},
                  "flavor2": {"x": 1, "y": 2},
              },
              signature=ModelSignature(
                  inputs=Schema([ColSpec("integer", "x"), ColSpec("integer", "y")]),
                  outputs=Schema([ColSpec(name=None, type="double")])),
              saved_input_example_info={"x": 1, "y": 2})
    n = Model(artifact_path="some/path",
              run_id="123",
              flavors={
                  "flavor1": {"a": 1, "b": 2},
                  "flavor2": {"x": 1, "y": 2},
              },
              signature=ModelSignature(
                  inputs=Schema([ColSpec("integer", "x"), ColSpec("integer", "y")]),
                  outputs=Schema([ColSpec(name=None, type="double")])),
              saved_input_example_info={"x": 1, "y": 2})
    n.utc_time_created = m.utc_time_created
    assert m == n
    n.run_id = "124"
    assert m != n
    with TempDir() as tmp:
        m.save(tmp.path("model"))
        o = Model.load(tmp.path("model"))
    assert m == o
    assert m.to_json() == o.to_json()
    assert m.to_yaml() == o.to_yaml()


class TestFlavor(object):
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


def test_model_log():
    with TempDir(chdr=True) as tmp:
        experiment_id = mlflow.create_experiment("test")
        sig = ModelSignature(inputs=Schema([ColSpec("integer", "x"), ColSpec("integer", "y")]),
                             outputs=Schema([ColSpec(name=None, type="double")]))
        input_example = {"x": 1, "y": 2}
        with mlflow.start_run(experiment_id=experiment_id) as r:
            Model.log("some/path", TestFlavor,
                      signature=sig,
                      input_example=input_example)

        local_path = _download_artifact_from_uri("runs:/{}/some/path".format(r.info.run_id),
                                                 output_path=tmp.path(""))
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
