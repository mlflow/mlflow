import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import sklearn
import sklearn.datasets
import sklearn.neighbors
from click.testing import CliRunner
from packaging.requirements import Requirement

import mlflow
import mlflow.models.cli as models_cli
import mlflow.sklearn
from mlflow.environment_variables import MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING
from mlflow.exceptions import MlflowException
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.models.model import get_model_requirements_files, update_model_requirements
from mlflow.models.utils import load_serving_example
from mlflow.protos.databricks_pb2 import BAD_REQUEST, ErrorCode
from mlflow.pyfunc.backend import PyFuncBackend
from mlflow.pyfunc.scoring_server import (
    CONTENT_TYPE_CSV,
    CONTENT_TYPE_JSON,
)
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.utils import PYTHON_VERSION
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.conda import _get_conda_env_name
from mlflow.utils.environment import (
    _get_requirements_from_file,
    _mlflow_conda_env,
)
from mlflow.utils.file_utils import TempDir
from mlflow.utils.process import ShellCommandException

from tests.helper_functions import (
    PROTOBUF_REQUIREMENT,
    RestEndpoint,
    get_safe_port,
    pyfunc_build_image,
    pyfunc_generate_dockerfile,
    pyfunc_serve_and_score_model,
    pyfunc_serve_from_docker_image,
    pyfunc_serve_from_docker_image_with_env_override,
)

# NB: for now, windows tests do not have conda available.
no_conda = ["--env-manager", "local"] if sys.platform == "win32" else []

# NB: need to install mlflow since the pip version does not have mlflow models cli.
install_mlflow = ["--install-mlflow"] if not no_conda else []

extra_options = no_conda + install_mlflow


def env_with_tracking_uri() -> dict[str, str]:
    return {**os.environ, "MLFLOW_TRACKING_URI": mlflow.get_tracking_uri()}


@pytest.fixture(scope="module")
def iris_data() -> tuple[np.ndarray, np.ndarray]:
    iris = sklearn.datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    return x, y


@pytest.fixture(scope="module")
def sk_model(iris_data: tuple[np.ndarray, np.ndarray]) -> sklearn.neighbors.KNeighborsClassifier:
    x, y = iris_data
    knn_model = sklearn.neighbors.KNeighborsClassifier()
    knn_model.fit(x, y)
    return knn_model


@pytest.mark.allow_infer_pip_requirements_fallback
def test_mlflow_is_not_installed_unless_specified():
    if no_conda:
        pytest.skip("This test requires conda.")
    with TempDir(chdr=True) as tmp:
        fake_model_path = tmp.path("fake_model")
        mlflow.pyfunc.save_model(fake_model_path, loader_module=__name__)
        # Overwrite the logged `conda.yaml` to remove mlflow.
        _mlflow_conda_env(path=os.path.join(fake_model_path, "conda.yaml"), install_mlflow=False)
        # The following should fail because there should be no mlflow in the env:
        prc = subprocess.run(
            [
                sys.executable,
                "-m",
                "mlflow",
                "models",
                "predict",
                "-m",
                fake_model_path,
                "--env-manager",
                "conda",
            ],
            stderr=subprocess.PIPE,
            cwd=tmp.path(""),
            check=False,
            text=True,
            env=env_with_tracking_uri(),
        )
        assert prc.returncode != 0
        if PYTHON_VERSION.startswith("3"):
            assert "ModuleNotFoundError: No module named 'mlflow'" in prc.stderr
        else:
            assert "ImportError: No module named mlflow.pyfunc.scoring_server" in prc.stderr


def test_model_with_no_deployable_flavors_fails_pollitely():
    from mlflow.models import Model

    with TempDir(chdr=True) as tmp:
        m = Model(
            artifact_path=None,
            run_id=None,
            utc_time_created="now",
            flavors={"some": {}, "useless": {}, "flavors": {}},
        )
        os.mkdir(tmp.path("model"))
        m.save(tmp.path("model", "MLmodel"))
        # The following should fail because there should be no suitable flavor
        prc = subprocess.run(
            [sys.executable, "-m", "mlflow", "models", "predict", "-m", tmp.path("model")],
            stderr=subprocess.PIPE,
            cwd=tmp.path(""),
            check=False,
            text=True,
            env=env_with_tracking_uri(),
        )
        assert "No suitable flavor backend was found for the model." in prc.stderr


def test_serve_uvicorn_opts(iris_data, sk_model):
    if sys.platform == "win32":
        pytest.skip("This test requires gunicorn which is not available on windows.")
    with mlflow.start_run():
        x, _ = iris_data
        model_info = mlflow.sklearn.log_model(
            sk_model, name="model", registered_model_name="test", input_example=pd.DataFrame(x)
        )

    model_uris = ["models:/test/None", model_info.model_uri]
    for model_uri in model_uris:
        with TempDir() as tpm:
            output_file_path = tpm.path("stdout")
            inference_payload = load_serving_example(model_uri)
            with open(output_file_path, "w") as output_file:
                scoring_response = pyfunc_serve_and_score_model(
                    model_uri,
                    inference_payload,
                    content_type=CONTENT_TYPE_JSON,
                    stdout=output_file,
                    extra_args=["-w", "3", "--env-manager", "local"],
                )
            with open(output_file_path) as output_file:
                stdout = output_file.read()
        actual = pd.read_json(scoring_response.content.decode("utf-8"), orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)
        expected_command_pattern = re.compile(
            r"uvicorn.*--workers 3.*mlflow\.pyfunc\.scoring_server\.app:app"
        )
        assert expected_command_pattern.search(stdout) is not None


@dataclass
class PredictTestData:
    model_uri: str
    model_registry_uri: str
    input_json_path: Path
    input_csv_path: Path
    output_json_path: Path
    x: np.ndarray
    sk_model: sklearn.base.BaseEstimator


@pytest.fixture
def predict_test_setup(
    iris_data: tuple[np.ndarray, np.ndarray],
    sk_model: sklearn.neighbors.KNeighborsClassifier,
    tmp_path: Path,
) -> PredictTestData:
    with mlflow.start_run() as active_run:
        mlflow.sklearn.log_model(sk_model, name="model", registered_model_name="impredicting")
        model_uri = f"runs:/{active_run.info.run_id}/model"

    model_registry_uri = "models:/impredicting/None"
    input_json_path = tmp_path / "input.json"
    input_csv_path = tmp_path / "input.csv"
    output_json_path = tmp_path / "output.json"

    x, _ = iris_data
    with open(input_json_path, "w") as f:
        json.dump({"dataframe_split": pd.DataFrame(x).to_dict(orient="split")}, f)
    pd.DataFrame(x).to_csv(input_csv_path, index=False)

    return PredictTestData(
        model_uri=model_uri,
        model_registry_uri=model_registry_uri,
        input_json_path=input_json_path,
        input_csv_path=input_csv_path,
        output_json_path=output_json_path,
        x=x,
        sk_model=sk_model,
    )


def test_predict_with_model_registry_uri(predict_test_setup: PredictTestData) -> None:
    setup = predict_test_setup
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "predict",
            "-m",
            setup.model_registry_uri,
            "-i",
            setup.input_json_path,
            "-o",
            setup.output_json_path,
            "--env-manager",
            "local",
        ],
        env=env_with_tracking_uri(),
    )
    actual = pd.read_json(setup.output_json_path, orient="records")
    actual = actual[actual.columns[0]].values
    expected = setup.sk_model.predict(setup.x)
    assert all(expected == actual)


def test_predict_with_conda_and_install_mlflow(predict_test_setup: PredictTestData) -> None:
    setup = predict_test_setup
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "predict",
            "-m",
            setup.model_uri,
            "-i",
            setup.input_json_path,
            "-o",
            setup.output_json_path,
            *extra_options,
        ],
        env=env_with_tracking_uri(),
    )
    actual = pd.read_json(setup.output_json_path, orient="records")
    actual = actual[actual.columns[0]].values
    expected = setup.sk_model.predict(setup.x)
    assert all(expected == actual)


def test_predict_explicit_json_format_default_orient(predict_test_setup: PredictTestData) -> None:
    setup = predict_test_setup
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "predict",
            "-m",
            setup.model_uri,
            "-i",
            setup.input_json_path,
            "-o",
            setup.output_json_path,
            "-t",
            "json",
            *extra_options,
        ],
        env=env_with_tracking_uri(),
    )
    actual = pd.read_json(setup.output_json_path, orient="records")
    actual = actual[actual.columns[0]].values
    expected = setup.sk_model.predict(setup.x)
    assert all(expected == actual)


def test_predict_explicit_json_format_split_orient(predict_test_setup: PredictTestData) -> None:
    # Note: This test has the same command as the previous one but tests orient==split
    # The comment in original code mentions this should be split orient
    setup = predict_test_setup
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "predict",
            "-m",
            setup.model_uri,
            "-i",
            setup.input_json_path,
            "-o",
            setup.output_json_path,
            "-t",
            "json",
            *extra_options,
        ],
        env=env_with_tracking_uri(),
    )
    actual = pd.read_json(setup.output_json_path, orient="records")
    actual = actual[actual.columns[0]].values
    expected = setup.sk_model.predict(setup.x)
    assert all(expected == actual)


def test_predict_stdin_stdout(predict_test_setup: PredictTestData) -> None:
    setup = predict_test_setup
    stdout = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "predict",
            "-m",
            setup.model_uri,
            "-t",
            "json",
            *extra_options,
        ],
        input=setup.input_json_path.read_text(),
        env=env_with_tracking_uri(),
        text=True,
    )
    predictions = re.search(r"{\"predictions\": .*}", stdout).group(0)
    actual = pd.read_json(predictions, orient="records")
    actual = actual[actual.columns[0]].values
    expected = setup.sk_model.predict(setup.x)
    assert all(expected == actual)
    # NB: We do not test orient=records here because records may loose column ordering.
    # orient == records is tested in other test with simpler model.


def test_predict_csv_format(predict_test_setup: PredictTestData) -> None:
    setup = predict_test_setup
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "predict",
            "-m",
            setup.model_uri,
            "-i",
            setup.input_csv_path,
            "-o",
            setup.output_json_path,
            "-t",
            "csv",
            *extra_options,
        ],
        env=env_with_tracking_uri(),
    )
    actual = pd.read_json(setup.output_json_path, orient="records")
    actual = actual[actual.columns[0]].values
    expected = setup.sk_model.predict(setup.x)
    assert all(expected == actual)


def test_predict_check_content_type(iris_data, sk_model, tmp_path):
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model, name="model", registered_model_name="impredicting")
    model_registry_uri = "models:/impredicting/None"
    input_json_path = tmp_path / "input.json"
    input_csv_path = tmp_path / "input.csv"
    output_json_path = tmp_path / "output.json"

    x, _ = iris_data
    with input_json_path.open("w") as f:
        json.dump({"dataframe_split": pd.DataFrame(x).to_dict(orient="split")}, f)

    pd.DataFrame(x).to_csv(input_csv_path, index=False)

    # Throw errors for invalid content_type
    prc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "predict",
            "-m",
            model_registry_uri,
            "-i",
            input_json_path,
            "-o",
            output_json_path,
            "-t",
            "invalid",
            "--env-manager",
            "local",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env_with_tracking_uri(),
        check=False,
    )
    assert prc.returncode != 0
    assert "Content type must be one of json or csv." in prc.stderr.decode("utf-8")


def test_predict_check_input_path(iris_data, sk_model, tmp_path):
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model, name="model", registered_model_name="impredicting")
    model_registry_uri = "models:/impredicting/None"
    input_json_path = tmp_path / "input with space.json"
    input_csv_path = tmp_path / "input.csv"
    output_json_path = tmp_path / "output.json"

    x, _ = iris_data
    with input_json_path.open("w") as f:
        json.dump({"dataframe_split": pd.DataFrame(x).to_dict(orient="split")}, f)

    pd.DataFrame(x).to_csv(input_csv_path, index=False)

    # Valid input path with space
    prc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "predict",
            "-m",
            model_registry_uri,
            "-i",
            f"{input_json_path}",
            "-o",
            output_json_path,
            "--env-manager",
            "local",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env_with_tracking_uri(),
        check=False,
        text=True,
    )
    assert prc.returncode == 0

    # Throw errors for invalid input_path
    prc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "predict",
            "-m",
            model_registry_uri,
            "-i",
            f'{input_json_path}"; echo ThisIsABug! "',
            "-o",
            output_json_path,
            "--env-manager",
            "local",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env_with_tracking_uri(),
        check=False,
        text=True,
    )
    assert prc.returncode != 0
    assert "ThisIsABug!" not in prc.stdout
    assert "FileNotFoundError" in prc.stderr

    prc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "predict",
            "-m",
            model_registry_uri,
            "-i",
            f'{input_csv_path}"; echo ThisIsABug! "',
            "-o",
            output_json_path,
            "-t",
            "csv",
            "--env-manager",
            "local",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env_with_tracking_uri(),
        check=False,
        text=True,
    )
    assert prc.returncode != 0
    assert "ThisIsABug!" not in prc.stdout
    assert "FileNotFoundError" in prc.stderr


def test_predict_check_output_path(iris_data, sk_model, tmp_path):
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model, name="model", registered_model_name="impredicting")
    model_registry_uri = "models:/impredicting/None"
    input_json_path = tmp_path / "input.json"
    input_csv_path = tmp_path / "input.csv"
    output_json_path = tmp_path / "output.json"

    x, _ = iris_data
    with input_json_path.open("w") as f:
        json.dump({"dataframe_split": pd.DataFrame(x).to_dict(orient="split")}, f)

    pd.DataFrame(x).to_csv(input_csv_path, index=False)

    prc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "predict",
            "-m",
            model_registry_uri,
            "-i",
            input_json_path,
            "-o",
            f'{output_json_path}"; echo ThisIsABug! "',
            "--env-manager",
            "local",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env_with_tracking_uri(),
        check=False,
        text=True,
    )
    assert prc.returncode == 0
    assert "ThisIsABug!" not in prc.stdout


def test_prepare_env_passes(sk_model):
    if no_conda:
        pytest.skip("This test requires conda.")

    with TempDir(chdr=True):
        with mlflow.start_run() as active_run:
            mlflow.sklearn.log_model(sk_model, name="model")
            model_uri = f"runs:/{active_run.info.run_id}/model"

        # With conda
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlflow",
                "models",
                "prepare-env",
                "-m",
                model_uri,
            ],
            env=env_with_tracking_uri(),
            check=True,
        )

        # Should be idempotent
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlflow",
                "models",
                "prepare-env",
                "-m",
                model_uri,
            ],
            env=env_with_tracking_uri(),
            check=True,
        )


def test_prepare_env_fails(sk_model):
    if no_conda:
        pytest.skip("This test requires conda.")

    with TempDir(chdr=True):
        with mlflow.start_run() as active_run:
            mlflow.sklearn.log_model(
                sk_model, name="model", pip_requirements=["does-not-exist-dep==abc"]
            )
            model_uri = f"runs:/{active_run.info.run_id}/model"

        # With conda - should fail due to bad conda environment.
        prc = subprocess.run(
            [
                sys.executable,
                "-m",
                "mlflow",
                "models",
                "prepare-env",
                "-m",
                model_uri,
            ],
            env=env_with_tracking_uri(),
            check=False,
        )
        assert prc.returncode != 0


@pytest.mark.parametrize("enable_mlserver", [True, False])
def test_generate_dockerfile(sk_model, enable_mlserver, tmp_path):
    with mlflow.start_run() as active_run:
        if enable_mlserver:
            mlflow.sklearn.log_model(
                sk_model, name="model", extra_pip_requirements=["/opt/mlflow", PROTOBUF_REQUIREMENT]
            )
        else:
            mlflow.sklearn.log_model(sk_model, name="model")
        model_uri = f"runs:/{active_run.info.run_id}/model"
    extra_args = ["--install-mlflow"]
    if enable_mlserver:
        extra_args.append("--enable-mlserver")

    output_directory = tmp_path.joinpath("output_directory")
    pyfunc_generate_dockerfile(
        output_directory,
        model_uri,
        extra_args=extra_args,
        env=env_with_tracking_uri(),
    )
    assert output_directory.is_dir()
    assert output_directory.joinpath("Dockerfile").exists()
    assert output_directory.joinpath("model_dir").is_dir()
    # Assert file is not empty
    assert output_directory.joinpath("Dockerfile").stat().st_size != 0


@pytest.mark.parametrize("enable_mlserver", [True, False])
def test_build_docker(iris_data, sk_model, enable_mlserver):
    with mlflow.start_run() as active_run:
        if enable_mlserver:
            mlflow.sklearn.log_model(
                sk_model, name="model", extra_pip_requirements=["/opt/mlflow", PROTOBUF_REQUIREMENT]
            )
        else:
            mlflow.sklearn.log_model(sk_model, name="model", extra_pip_requirements=["/opt/mlflow"])
        model_uri = f"runs:/{active_run.info.run_id}/model"

    x, _ = iris_data
    df = pd.DataFrame(x)

    extra_args = ["--install-mlflow"]
    if enable_mlserver:
        extra_args.append("--enable-mlserver")

    image_name = pyfunc_build_image(
        model_uri,
        extra_args=extra_args,
        env=env_with_tracking_uri(),
    )
    host_port = get_safe_port()
    scoring_proc = pyfunc_serve_from_docker_image(image_name, host_port)
    _validate_with_rest_endpoint(scoring_proc, host_port, df, x, sk_model, enable_mlserver)


def test_build_docker_virtualenv(iris_data, sk_model):
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(
            sk_model, name="model", extra_pip_requirements=["/opt/mlflow"]
        )

    x, _ = iris_data
    df = pd.DataFrame(iris_data[0])

    extra_args = ["--install-mlflow", "--env-manager", "virtualenv"]
    image_name = pyfunc_build_image(
        model_info.model_uri,
        extra_args=extra_args,
        env=env_with_tracking_uri(),
    )
    host_port = get_safe_port()
    scoring_proc = pyfunc_serve_from_docker_image(image_name, host_port)
    _validate_with_rest_endpoint(scoring_proc, host_port, df, x, sk_model)


@pytest.mark.parametrize("enable_mlserver", [True, False])
def test_build_docker_with_env_override(iris_data, sk_model, enable_mlserver):
    with mlflow.start_run() as active_run:
        if enable_mlserver:
            mlflow.sklearn.log_model(
                sk_model, name="model", extra_pip_requirements=["/opt/mlflow", PROTOBUF_REQUIREMENT]
            )
        else:
            mlflow.sklearn.log_model(sk_model, name="model", extra_pip_requirements=["/opt/mlflow"])
        model_uri = f"runs:/{active_run.info.run_id}/model"
    x, _ = iris_data
    df = pd.DataFrame(x)

    extra_args = ["--install-mlflow"]
    if enable_mlserver:
        extra_args.append("--enable-mlserver")

    image_name = pyfunc_build_image(
        model_uri,
        extra_args=extra_args,
        env=env_with_tracking_uri(),
    )
    host_port = get_safe_port()
    scoring_proc = pyfunc_serve_from_docker_image_with_env_override(image_name, host_port)
    _validate_with_rest_endpoint(scoring_proc, host_port, df, x, sk_model, enable_mlserver)


def test_build_docker_without_model_uri(iris_data, sk_model, tmp_path):
    model_path = tmp_path.joinpath("model")
    mlflow.sklearn.save_model(sk_model, model_path, extra_pip_requirements=["/opt/mlflow"])
    image_name = pyfunc_build_image(model_uri=None)
    host_port = get_safe_port()
    scoring_proc = pyfunc_serve_from_docker_image_with_env_override(
        image_name,
        host_port,
        extra_docker_run_options=["-v", f"{model_path}:/opt/ml/model"],
    )
    x = iris_data[0]
    df = pd.DataFrame(x)
    _validate_with_rest_endpoint(scoring_proc, host_port, df, x, sk_model)


def _validate_with_rest_endpoint(scoring_proc, host_port, df, x, sk_model, enable_mlserver=False):
    with RestEndpoint(proc=scoring_proc, port=host_port, validate_version=False) as endpoint:
        for content_type in [CONTENT_TYPE_JSON, CONTENT_TYPE_CSV]:
            scoring_response = endpoint.invoke(df, content_type)
            assert scoring_response.status_code == 200, (
                f"Failed to serve prediction, got response {scoring_response.text}"
            )
            np.testing.assert_array_equal(
                np.array(json.loads(scoring_response.text)["predictions"]), sk_model.predict(x)
            )
        # Try examples of bad input, verify we get a non-200 status code
        for content_type in [CONTENT_TYPE_JSON, CONTENT_TYPE_CSV, CONTENT_TYPE_JSON]:
            scoring_response = endpoint.invoke(data="", content_type=content_type)
            expected_status_code = 500 if enable_mlserver else 400
            assert scoring_response.status_code == expected_status_code, (
                f"Expected server failure with error code {expected_status_code}, "
                f"got response with status code {scoring_response.status_code} "
                f"and body {scoring_response.text}"
            )

            if enable_mlserver:
                # MLServer returns a different set of errors.
                # Skip these assertions until this issue gets tackled:
                # https://github.com/SeldonIO/MLServer/issues/360)
                continue

            scoring_response_dict = json.loads(scoring_response.content)
            assert "error_code" in scoring_response_dict
            assert scoring_response_dict["error_code"] == ErrorCode.Name(BAD_REQUEST)
            assert "message" in scoring_response_dict


def test_env_manager_warning_for_use_of_conda(monkeypatch):
    with mock.patch("mlflow.models.cli.get_flavor_backend") as mock_get_flavor_backend:
        with pytest.warns(UserWarning, match=r"Use of conda is discouraged"):
            CliRunner().invoke(
                models_cli.serve,
                ["--model-uri", "model", "--env-manager", "conda"],
                catch_exceptions=False,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            monkeypatch.setenv(MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING.name, "TRUE")
            CliRunner().invoke(
                models_cli.serve,
                ["--model-uri", "model", "--env-manager", "conda"],
                catch_exceptions=False,
            )

        assert mock_get_flavor_backend.call_count == 2


def test_env_manager_unsupported_value():
    with pytest.raises(MlflowException, match=r"Invalid value for `env_manager`"):
        CliRunner().invoke(
            models_cli.serve,
            ["--model-uri", "model", "--env-manager", "abc"],
            catch_exceptions=False,
        )


def test_host_invalid_value():
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            return model_input

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="test_model", python_model=MyModel(), registered_model_name="model"
        )

    with mock.patch(
        "mlflow.models.cli.get_flavor_backend",
        return_value=PyFuncBackend({}, env_manager=_EnvManager.VIRTUALENV),
    ):
        with pytest.raises(ShellCommandException, match=r"Non-zero exit code: 1"):
            CliRunner().invoke(
                models_cli.serve,
                ["--model-uri", model_info.model_uri, "--host", "localhost & echo BUG"],
                catch_exceptions=False,
            )


def test_change_conda_env_root_location(tmp_path, sk_model):
    def _test_model(env_root_path, model_path, sklearn_ver):
        env_root_path.mkdir(exist_ok=True)

        mlflow.sklearn.save_model(
            sk_model, str(model_path), pip_requirements=[f"scikit-learn=={sklearn_ver}"]
        )

        env = get_flavor_backend(
            str(model_path),
            env_manager=_EnvManager.CONDA,
            install_mlflow=False,
            env_root_dir=str(env_root_path),
        ).prepare_env(model_uri=str(model_path))

        conda_env_name = _get_conda_env_name(
            str(model_path / "conda.yaml"), env_root_dir=env_root_path
        )
        env_path = env_root_path / "conda_envs" / conda_env_name
        assert env_path.exists()

        python_exec_path = str(env_path / "bin" / "python")

        # Test execution of command under the correct activated python env.
        env.execute(
            command=f"python -c \"import sys; assert sys.executable == '{python_exec_path}'; "
            f"import sklearn; assert sklearn.__version__ == '{sklearn_ver}'\"",
        )

        # Cleanup model path and Conda environment to prevent out of space failures on CI
        shutil.rmtree(model_path)
        shutil.rmtree(env_path)

    env_root1_path = tmp_path / "root1"
    env_root2_path = tmp_path / "root2"

    # Test with model1_path
    model1_path = tmp_path / "model1"

    _test_model(env_root1_path, model1_path, "1.4.0")
    _test_model(env_root2_path, model1_path, "1.4.0")

    # Test with model2_path
    model2_path = tmp_path / "model2"
    _test_model(env_root1_path, model2_path, "1.4.2")


@pytest.mark.parametrize(
    ("input_schema", "output_schema", "params_schema"),
    [(True, False, False), (False, True, False), (False, False, True)],
)
def test_signature_enforcement_with_model_serving(input_schema, output_schema, params_schema):
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return ["test"]

    input_data = ["test_input"] if input_schema else None
    output_data = ["test_output"] if output_schema else None
    params = {"test": "test"} if params_schema else None

    signature = mlflow.models.infer_signature(
        model_input=input_data, model_output=output_data, params=params
    )

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="test_model", python_model=MyModel(), signature=signature
        )

    inference_payload = json.dumps({"inputs": ["test"]})

    # Serve and score the model
    scoring_result = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=inference_payload,
        content_type=CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    scoring_result.raise_for_status()

    # Assert the prediction result
    assert json.loads(scoring_result.content)["predictions"] == ["test"]


def assert_base_model_reqs():
    """
    Helper function for testing model requirements. Asserts that the
    contents of requirements.txt and conda.yaml are as expected, then
    returns their filepaths so mutations can be performed.
    """
    import cloudpickle

    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return ["test"]

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(name="model", python_model=MyModel())

    resolved_uri = ModelsArtifactRepository.get_underlying_uri(model_info.model_uri)
    local_paths = get_model_requirements_files(resolved_uri)

    requirements_txt_file = local_paths.requirements
    conda_env_file = local_paths.conda

    reqs = _get_requirements_from_file(requirements_txt_file)
    assert Requirement(f"mlflow=={mlflow.__version__}") in reqs
    assert Requirement(f"cloudpickle=={cloudpickle.__version__}") in reqs

    reqs = _get_requirements_from_file(conda_env_file)
    assert Requirement(f"mlflow=={mlflow.__version__}") in reqs
    assert Requirement(f"cloudpickle=={cloudpickle.__version__}") in reqs

    return model_info.model_uri


def test_update_requirements_cli_adds_reqs_successfully():
    import cloudpickle

    model_uri = assert_base_model_reqs()

    CliRunner().invoke(
        models_cli.update_pip_requirements,
        ["-m", f"{model_uri}", "add", "mlflow>=2.9, !=2.9.0", "coolpackage[extra]==8.8.8"],
        catch_exceptions=False,
    )

    resolved_uri = ModelsArtifactRepository.get_underlying_uri(model_uri)
    local_paths = get_model_requirements_files(resolved_uri)

    # the tool should overwrite mlflow, add coolpackage, and leave cloudpickle alone
    reqs = _get_requirements_from_file(local_paths.requirements)
    assert Requirement("mlflow!=2.9.0,>=2.9") in reqs
    assert Requirement("coolpackage[extra]==8.8.8") in reqs
    assert Requirement(f"cloudpickle=={cloudpickle.__version__}") in reqs

    reqs = _get_requirements_from_file(local_paths.conda)
    assert Requirement("mlflow!=2.9.0,>=2.9") in reqs
    assert Requirement("coolpackage[extra]==8.8.8") in reqs
    assert Requirement(f"cloudpickle=={cloudpickle.__version__}") in reqs


def test_update_requirements_cli_removes_reqs_successfully():
    import cloudpickle

    model_uri = assert_base_model_reqs()

    CliRunner().invoke(
        models_cli.update_pip_requirements,
        ["-m", f"{model_uri}", "remove", "mlflow"],
        catch_exceptions=False,
    )

    resolved_uri = ModelsArtifactRepository.get_underlying_uri(model_uri)
    local_paths = get_model_requirements_files(resolved_uri)

    # the tool should remove mlflow and leave cloudpickle alone
    reqs = _get_requirements_from_file(local_paths.requirements)
    assert reqs == [Requirement(f"cloudpickle=={cloudpickle.__version__}")]

    reqs = _get_requirements_from_file(local_paths.conda)
    assert reqs == [Requirement(f"cloudpickle=={cloudpickle.__version__}")]


def test_update_requirements_cli_throws_on_incompatible_input():
    model_uri = assert_base_model_reqs()

    with pytest.raises(
        MlflowException, match="The specified requirements versions are incompatible"
    ):
        CliRunner().invoke(
            models_cli.update_pip_requirements,
            ["-m", f"{model_uri}", "add", "mlflow<2.6", "mlflow>2.7"],
            catch_exceptions=False,
        )


def test_update_model_requirements_add():
    import cloudpickle

    model_uri = assert_base_model_reqs()
    update_model_requirements(
        model_uri, "add", ["mlflow>=2.9, !=2.9.0", "coolpackage[extra]==8.8.8"]
    )

    resolved_uri = ModelsArtifactRepository.get_underlying_uri(model_uri)
    local_paths = get_model_requirements_files(resolved_uri)

    # the tool should overwrite mlflow, add coolpackage, and leave cloudpickle alone
    reqs = _get_requirements_from_file(local_paths.requirements)
    assert Requirement("mlflow!=2.9.0,>=2.9") in reqs
    assert Requirement("coolpackage[extra]==8.8.8") in reqs
    assert Requirement(f"cloudpickle=={cloudpickle.__version__}") in reqs

    reqs = _get_requirements_from_file(local_paths.conda)
    assert Requirement("mlflow!=2.9.0,>=2.9") in reqs
    assert Requirement("coolpackage[extra]==8.8.8") in reqs
    assert Requirement(f"cloudpickle=={cloudpickle.__version__}") in reqs


def test_update_model_requirements_remove():
    import cloudpickle

    model_uri = assert_base_model_reqs()

    update_model_requirements(model_uri, "remove", ["mlflow"])
    resolved_uri = ModelsArtifactRepository.get_underlying_uri(model_uri)
    local_paths = get_model_requirements_files(resolved_uri)

    # the tool should remove mlflow and leave cloudpickle alone
    reqs = _get_requirements_from_file(local_paths.requirements)
    assert reqs == [Requirement(f"cloudpickle=={cloudpickle.__version__}")]

    reqs = _get_requirements_from_file(local_paths.conda)
    assert reqs == [Requirement(f"cloudpickle=={cloudpickle.__version__}")]
