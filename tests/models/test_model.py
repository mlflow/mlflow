import json
import os

import mlflow
from build.lib.mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.models import Model, ModelSignature
from mlflow.models.signature import Schema, ColSpec, read_example
from mlflow.utils.file_utils import TempDir


def test_model_save_load():
    m = Model(artifact_path="some/path",
              run_id="123",
              flavors={
                  "flavor1": {"a": 1, "b": 2},
                  "flavor2": {"x": 1, "y": 2},
              },
              signature=ModelSignature(
                  inputs=Schema([ColSpec("x", "integer"), ColSpec("y", "integer")]),
                  outputs=Schema([ColSpec(name=None, type="double")])),
              input_example={"x": 1, "y": 2})
    n = Model(artifact_path="some/path",
              run_id="123",
              flavors={
                  "flavor1": {"a": 1, "b": 2},
                  "flavor2": {"x": 1, "y": 2},
              },
              signature=ModelSignature(
                  inputs=Schema([ColSpec("x", "integer"), ColSpec("y", "integer")]),
                  outputs=Schema([ColSpec(name=None, type="double")])),
              input_example={"x": 1, "y": 2})
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
    def save_model(cls, path, mlflow_model, *args, **kwargs):
        mlflow_model.flavors["flavor1"] = {"a": 1, "b": 2}
        mlflow_model.flavors["flavor2"] = {"x": 1, "y": 2}
        print()
        print("creating dirs for path", path)
        print()
        os.makedirs(path)
        mlflow_model.save(os.path.join(path, "MLmodel"))


def test_model_log():
    with TempDir(chdr=True) as tmp:
        mlflow.create_experiment("test")
        mlflow.set_experiment("test")
        sig = ModelSignature(inputs=Schema([ColSpec("x", "integer"), ColSpec("y", "integer")]),
                             outputs=Schema([ColSpec(name=None, type="double")]))
        input_example = {"x": 1, "y": 2}
        with mlflow.start_run() as r:
            Model.log("some/path", TestFlavor,
                      model_signature=sig,
                      input_example=input_example)

        local_path = _download_artifact_from_uri("runs:/{}/some/path".format(r.info.run_id),
                                                 output_path=tmp.path(""))
        loaded_model = Model.load(os.path.join(local_path, "MLmodel"))
        print()
        print(loaded_model.to_yaml())
        print()
        assert loaded_model.run_id == r.info.run_id
        assert loaded_model.artifact_path == "some/path"
        assert loaded_model.flavors == {
            "flavor1": {"a": 1, "b": 2},
            "flavor2": {"x": 1, "y": 2},
        }
        print()
        print(loaded_model.signature)
        print()
        assert loaded_model.signature == sig
        x = read_example(os.path.join(local_path, loaded_model.input_example))
        assert x.to_dict(orient="records")[0] == input_example
