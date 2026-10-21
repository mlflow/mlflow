import json
import os
import re
import runpy
import shutil
import sys
import uuid
from pathlib import Path
from unittest import mock

import pytest
import requests

import mlflow
from mlflow import cli
from mlflow.utils import process
from mlflow.utils.virtualenv import _get_mlflow_virtualenv_root

from tests.helper_functions import clear_hub_cache, flaky, start_mock_openai_server
from tests.integration.utils import invoke_cli_runner

EXAMPLES_DIR = "examples"


def find_python_env_yaml(directory: Path) -> Path:
    return next(filter(lambda p: p.name == "python_env.yaml", Path(directory).iterdir()))


def replace_mlflow_with_dev_version(yml_path: Path) -> None:
    old_src = yml_path.read_text()
    mlflow_dir = Path(mlflow.__path__[0]).parent
    new_src = re.sub(r"- mlflow.*\n", f"- {mlflow_dir}\n", old_src)
    yml_path.write_text(new_src)


def test_typesafe_judges_example(monkeypatch, db_uri, request):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-typesafe-key")
    monkeypatch.setattr(sys, "argv", ["evaluate.py"])
    previous_tracking_uri = mlflow.get_tracking_uri()
    request.addfinalizer(lambda: mlflow.set_tracking_uri(previous_tracking_uri))
    mlflow.set_tracking_uri(db_uri)

    def respond(**kwargs):
        assert kwargs["url"] == "https://api.typesafe.ai/v1/systemone"
        assert kwargs["headers"] == {"Authorization": "Bearer test-typesafe-key"}
        payload = kwargs["json"]
        assert payload["model"] == "jev-latest"
        state = payload["state"]
        question = payload["questions"]["evaluation"]
        is_account = "password" in state["inputs"]["question"]

        if question["type"] == "noul":
            assert set(state) == {"inputs", "outputs"}
            answer = {"type": "noul", "noul": 0.9}
        else:
            assert question["type"] == "choice"
            assert set(state) == {"inputs"}
            label = "account" if is_account else "billing"
            answer = {
                "type": "choice",
                "choice": label,
                "probabilities": {
                    option: 0.8 if option == label else 0.1 for option in question["criteria"]
                },
                "confidence": 0.7,
            }

        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({
            "model": "jev-1.13.0",
            "answers": {"evaluation": answer},
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }).encode()
        return response

    with mock.patch(
        "mlflow.genai.judges.typesafe._get_http_response_with_retries", side_effect=respond
    ) as http_request:
        namespace = runpy.run_path(
            str(Path(EXAMPLES_DIR, "typesafe_judges", "evaluate.py")), run_name="__main__"
        )

    assert http_request.call_count == 4
    results = namespace["results"]
    assert len(results.result_df) == 2
    assert sorted(results.result_df["answer_helpfulness/value"]) == [True, True]
    assert sorted(results.result_df["request_category/value"]) == ["account", "billing"]

    traces = mlflow.search_traces(run_id=results.run_id, return_type="list")
    assert len(traces) == 2
    for trace in traces:
        feedback = {assessment.name: assessment for assessment in trace.info.assessments}
        assert feedback["answer_helpfulness"].rationale is None
        assert json.loads(feedback["answer_helpfulness"].metadata["typesafe.probability"]) == 0.9
        assert set(json.loads(feedback["request_category"].metadata["typesafe.probabilities"])) == {
            "account",
            "billing",
            "other",
        }


@pytest.fixture(autouse=True)
def clean_up_mlflow_virtual_environments():
    yield

    venv_root = Path(_get_mlflow_virtualenv_root())
    if not venv_root.exists():
        return
    for path in venv_root.iterdir():
        if path.is_dir():
            shutil.rmtree(path)


@pytest.fixture(scope="module", autouse=True)
def mock_openai():
    # Some examples includes OpenAI API calls, so we start a mock server.
    with start_mock_openai_server() as base_url:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("OPENAI_API_BASE", base_url)
            mp.setenv("OPENAI_API_KEY", "test")
            yield


@pytest.mark.notrackingurimock
@flaky()
@pytest.mark.parametrize(
    ("directory", "params"),
    [
        ("h2o", []),
        # TODO: Fix the hyperparam example and re-enable it
        # ("hyperparam", ["-e", "train", "-P", "epochs=1"]),
        # ("hyperparam", ["-e", "random", "-P", "epochs=1"]),
        # ("hyperparam", ["-e", "hyperopt", "-P", "epochs=1"]),
        (
            "lightgbm/lightgbm_native",
            ["-P", "learning_rate=0.1", "-P", "colsample_bytree=0.8", "-P", "subsample=0.9"],
        ),
        ("lightgbm/lightgbm_sklearn", []),
        ("statsmodels", ["-P", "inverse_method=qr"]),
        ("pytorch", ["-P", "epochs=2"]),
        ("sklearn_logistic_regression", []),
        ("sklearn_elasticnet_wine", ["-P", "alpha=0.5"]),
        ("sklearn_elasticnet_diabetes/linux", []),
        ("spacy", []),
        (
            "xgboost/xgboost_native",
            ["-P", "learning_rate=0.3", "-P", "colsample_bytree=0.8", "-P", "subsample=0.9"],
        ),
        ("xgboost/xgboost_sklearn", []),
        ("pytorch/MNIST", ["-P", "max_epochs=1"]),
        ("pytorch/HPOExample", ["-P", "n_trials=2", "-P", "max_epochs=1"]),
        ("pytorch/CaptumExample", ["-P", "max_epochs=50"]),
        ("supply_chain_security", []),
        ("tensorflow", []),
        ("sktime", []),
    ],
)
def test_mlflow_run_example(directory, params, tmp_path):
    # Use tmp_path+uuid as tmp directory to avoid the same
    # directory being reused when re-trying the test since
    # tmp_path is named as the test name
    random_tmp_path = tmp_path / str(uuid.uuid4())
    example_dir = Path(EXAMPLES_DIR, directory)
    tmp_example_dir = random_tmp_path.joinpath(example_dir)
    shutil.copytree(example_dir, tmp_example_dir)
    mlflow.set_tracking_uri(f"sqlite:///{random_tmp_path / 'mlruns.db'}")
    python_env_path = find_python_env_yaml(tmp_example_dir)
    replace_mlflow_with_dev_version(python_env_path)
    cli_run_list = [tmp_example_dir] + params
    invoke_cli_runner(cli.run, list(map(str, cli_run_list)))


@pytest.mark.notrackingurimock
@pytest.mark.parametrize(
    ("directory", "command"),
    [
        ("docker", ["docker", "build", "-t", "mlflow-docker-example", "-f", "Dockerfile", "."]),
        ("keras", [sys.executable, "train.py"]),
        (
            "lightgbm/lightgbm_native",
            [
                sys.executable,
                "train.py",
                "--learning-rate",
                "0.2",
                "--colsample-bytree",
                "0.8",
                "--subsample",
                "0.9",
            ],
        ),
        ("lightgbm/lightgbm_sklearn", [sys.executable, "train.py"]),
        ("statsmodels", [sys.executable, "train.py", "--inverse-method", "qr"]),
        ("quickstart", [sys.executable, "mlflow_tracking.py"]),
        ("remote_store", [sys.executable, "remote_server.py"]),
        (
            "xgboost/xgboost_native",
            [
                sys.executable,
                "train.py",
                "--learning-rate",
                "0.2",
                "--colsample-bytree",
                "0.8",
                "--subsample",
                "0.9",
            ],
        ),
        ("xgboost/xgboost_sklearn", [sys.executable, "train.py"]),
        ("catboost", [sys.executable, "train.py"]),
        ("prophet", [sys.executable, "train.py"]),
        ("sklearn_autolog", [sys.executable, "linear_regression.py"]),
        ("sklearn_autolog", [sys.executable, "pipeline.py"]),
        ("sklearn_autolog", [sys.executable, "grid_search_cv.py"]),
        ("pyspark_ml_autologging", [sys.executable, "logistic_regression.py"]),
        ("pyspark_ml_autologging", [sys.executable, "one_vs_rest.py"]),
        ("pyspark_ml_autologging", [sys.executable, "pipeline.py"]),
        ("shap", [sys.executable, "regression.py"]),
        ("shap", [sys.executable, "binary_classification.py"]),
        ("shap", [sys.executable, "multiclass_classification.py"]),
        ("shap", [sys.executable, "explainer_logging.py"]),
        ("ray_serve", [sys.executable, "train_model.py"]),
        ("pip_requirements", [sys.executable, "pip_requirements.py"]),
        ("pmdarima", [sys.executable, "train.py"]),
        ("evaluation", [sys.executable, "evaluate_on_binary_classifier.py"]),
        ("evaluation", [sys.executable, "evaluate_on_multiclass_classifier.py"]),
        ("evaluation", [sys.executable, "evaluate_on_regressor.py"]),
        ("evaluation", [sys.executable, "evaluate_with_custom_metrics.py"]),
        ("evaluation", [sys.executable, "evaluate_with_custom_metrics_comprehensive.py"]),
        ("evaluation", [sys.executable, "evaluate_with_model_validation.py"]),
        ("spark_udf", [sys.executable, "spark_udf_datetime.py"]),
        ("pyfunc", [sys.executable, "train.py"]),
        ("tensorflow", [sys.executable, "train.py"]),
        ("transformers", [sys.executable, "conversational.py"]),
        ("transformers", [sys.executable, "load_components.py"]),
        ("transformers", [sys.executable, "simple.py"]),
        ("transformers", [sys.executable, "sentence_transformer.py"]),
        ("transformers", [sys.executable, "whisper.py"]),
        ("sentence_transformers", [sys.executable, "simple.py"]),
        ("tracing", [sys.executable, "fluent.py"]),
        ("tracing", [sys.executable, "client.py"]),
        ("llama_index", [sys.executable, "simple_index.py"]),
        ("llama_index", [sys.executable, "autolog.py"]),
    ],
)
def test_command_example(directory, command):
    cwd_dir = Path(EXAMPLES_DIR, directory)
    assert os.environ.get("MLFLOW_HOME") is not None
    if directory == "transformers":
        # NB: Clearing the huggingface_hub cache is to lower the disk storage pressure for CI
        clear_hub_cache()

    process._exec_cmd(command, cwd=cwd_dir, env=os.environ)
