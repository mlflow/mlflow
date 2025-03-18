import importlib.metadata
import json

import dspy
import pytest
from packaging.version import Version

import mlflow
from mlflow.dspy.util import log_dspy_dataset, log_dspy_module_params, save_dspy_module_state
from mlflow.tracking import MlflowClient


@pytest.mark.skipif(
    Version(importlib.metadata.version("dspy")) < Version("2.5.43"),
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


def test_log_dspy_module_state_params():
    program = dspy.Predict("question -> answer")

    with mlflow.start_run() as run:
        log_dspy_module_params(program)

    run = mlflow.last_active_run()
    assert run.data.params == {
        "Predict.signature.fields.0.description": "${question}",
        "Predict.signature.fields.0.prefix": "Question:",
        "Predict.signature.fields.1.description": "${answer}",
        "Predict.signature.fields.1.prefix": "Answer:",
        "Predict.signature.instructions": "Given the fields `question`, produce the fields `answer`.",  # noqa: E501
    }


def test_log_dataset(tmp_path):
    dataset = [
        dspy.Example(question="What is 1 + 1?", answer="2").with_inputs("question"),
        dspy.Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
    ]

    with mlflow.start_run() as run:
        log_dspy_dataset(dataset, "dataset.json")

    client = MlflowClient()
    artifacts = (x.path for x in client.list_artifacts(run.info.run_id))
    assert "dataset.json" in artifacts
    client.download_artifacts(run_id=run.info.run_id, path="dataset.json", dst_path=tmp_path)
    saved_dataset = json.loads((tmp_path / "dataset.json").read_text())
    assert saved_dataset == {
        "columns": ["question", "answer"],
        "data": [
            ["What is 1 + 1?", "2"],
            ["What is 2 + 2?", "4"],
        ],
    }
