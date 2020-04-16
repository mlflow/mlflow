from mlflow.models import Model, ModelSignature
from mlflow.models.signature import Schema, ColSpec
from mlflow.utils.file_utils import TempDir


def test_model_save_load():
    m = Model(artifact_path="some/path",
              run_id="123",
              flavors={
                  "flavor1":{"a": 1, "b": 2},
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
    assert m != n
    n.utc_time_created = m.utc_time_created
    assert m == n
    with TempDir() as tmp:
        m.save(tmp.path("model"))
        o = Model.load(tmp.path("model"))
    assert m == o
    assert m.to_json() == o.to_json()
    assert m.to_yaml() == o.to_yaml()

