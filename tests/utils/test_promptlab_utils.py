import json
import os

import pytest

from mlflow.entities.param import Param
from mlflow.entities.run_status import RunStatus
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.tracking.file_store import FileStore
from mlflow.utils.promptlab_utils import (
    _create_promptlab_run_impl,
    create_eval_results_json,
)

prompt_parameters = [
    Param(key="question", value="my_question"),
    Param(key="context", value="my_context"),
]
model_input = "answer this question: my_question using the following context: my_context"
model_output = "my_answer"
model_output_parameters = [
    Param(key="tokens", value="10"),
    Param(key="latency", value="100"),
]


def test_eval_results_file():
    eval_results_file = create_eval_results_json(
        prompt_parameters, model_input, model_output_parameters, model_output
    )
    expected_eval_results_json = {
        "columns": ["question", "context", "prompt", "output", "tokens", "latency"],
        "data": [
            [
                "my_question",
                "my_context",
                "answer this question: my_question using the following context: my_context",
                "my_answer",
                "10",
                "100",
            ]
        ],
    }
    assert json.loads(eval_results_file) == expected_eval_results_json


@pytest.fixture
def store(tmp_path):
    return FileStore(str(tmp_path.joinpath("mlruns")))


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny does not support the np or pandas dependencies",
)
def test_create_promptlab_run(store):
    exp_id = store.create_experiment("test_create_promptlab_run")
    run = _create_promptlab_run_impl(
        store,
        experiment_id=exp_id,
        run_name="my_promptlab_run",
        tags=[],
        prompt_template="my_prompt_template",
        prompt_parameters=[Param("prompt_param_key", "prompt_param_value")],
        model_route="my_route",
        model_parameters=[Param("temperature", "0.1")],
        model_input="",
        model_output_parameters=[Param("output_param_key", "output_param_value")],
        model_output="my_output",
        mlflow_version="1.0.0",
        user_id="user",
        start_time=1,
    )
    assert run.info.run_id is not None
    assert run.info.status == RunStatus.to_string(RunStatus.FINISHED)

    assert run.data.params["prompt_template"] == "my_prompt_template"
    assert run.data.params["model_route"] == "my_route"
    assert run.data.params["temperature"] == "0.1"

    assert run.data.tags["mlflow.runName"] == "my_promptlab_run"
    assert (
        run.data.tags["mlflow.loggedArtifacts"]
        == '[{"path": "eval_results_table.json", "type": "table"}]'
    )
    assert run.data.tags["mlflow.runSourceType"] == "PROMPT_ENGINEERING"
    assert run.data.tags["mlflow.log-model.history"] is not None

    # list the files in the model folder
    artifact_location = run.info.artifact_uri
    artifact_repo = get_artifact_repository(artifact_location)

    artifact_files = [f.path for f in artifact_repo.list_artifacts()]
    assert "eval_results_table.json" in artifact_files
    assert "model" in artifact_files

    model_files = [f.path for f in artifact_repo.list_artifacts("model")]
    assert "model/MLmodel" in model_files
    assert "model/python_env.yaml" in model_files
    assert "model/conda.yaml" in model_files
    assert "model/requirements.txt" in model_files
    assert "model/input_example.json" in model_files

    # try to load the model
    import mlflow.pyfunc

    mlflow.pyfunc.load_model(f"{artifact_location}/model")
