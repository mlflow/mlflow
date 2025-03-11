import dspy
import pytest
from packaging.version import Version

import mlflow
from mlflow.dspy.util import save_dspy_module_state
from mlflow.tracking import MlflowClient


@pytest.mark.skipif(
    Version(dspy.__version__) < Version("2.5.43"),
    reason="dump_state works differently in older versions",
)
def test_save_dspy_module_state(tmp_path):
    program = dspy.ChainOfThought("question -> answer")

    with mlflow.start_run() as run:
        save_dspy_module_state(program)

    client = MlflowClient()
    artifacts = (x.path for x in client.list_artifacts(run.info.run_id))
    assert "model.json" in artifacts
    client.download_artifacts(run_id=run.info.run_id, path="model.json", dst_path=tmp_path)
    loaded_program = dspy.ChainOfThought("b -> a")
    loaded_program.load(tmp_path / "model.json")
    assert loaded_program.dump_state() == program.dump_state()
