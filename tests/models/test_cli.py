import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

from click.testing import CliRunner
import numpy as np
import pandas as pd
import pytest
import re
import sklearn
import sklearn.datasets
import sklearn.neighbors

from unittest import mock


import mlflow
import mlflow.sklearn
from mlflow.models.flavor_backend_registry import get_flavor_backend

from mlflow.utils.conda import _get_conda_env_name

import mlflow.models.cli as models_cli

from mlflow.environment_variables import MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode, BAD_REQUEST
from mlflow.pyfunc.scoring_server import (
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_CSV,
)
from mlflow.utils.file_utils import TempDir
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils import PYTHON_VERSION
from tests.helper_functions import (
    pyfunc_build_image,
    pyfunc_serve_from_docker_image,
    pyfunc_serve_from_docker_image_with_env_override,
    RestEndpoint,
    get_safe_port,
    pyfunc_serve_and_score_model,
    PROTOBUF_REQUIREMENT,
    pyfunc_generate_dockerfile,
)

# NB: for now, windows tests do not have conda available.
no_conda = ["--env-manager", "local"] if sys.platform == "win32" else []

# NB: need to install mlflow since the pip version does not have mlflow models cli.
install_mlflow = ["--install-mlflow"] if not no_conda else []

extra_options = no_conda + install_mlflow
gunicorn_options = "--timeout 60 -w 5"


def env_with_tracking_uri():
    return {**os.environ, "MLFLOW_TRACKING_URI": mlflow.get_tracking_uri()}


@pytest.fixture(scope="module")
def iris_data():
    iris = sklearn.datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    return x, y


@pytest.fixture(scope="module")
def sk_model(iris_data):
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
            ["mlflow", "models", "predict", "-m", tmp.path("model")],
            stderr=subprocess.PIPE,
            cwd=tmp.path(""),
            check=False,
            text=True,
            env=env_with_tracking_uri(),
        )
        assert "No suitable flavor backend was found for the model." in prc.stderr


def test_serve_gunicorn_opts(iris_data, sk_model):
    if sys.platform == "win32":
        pytest.skip("This test requires gunicorn which is not available on windows.")
    with mlflow.start_run() as active_run:
        mlflow.sklearn.log_model(sk_model, "model", registered_model_name="imlegit")
        run_id = active_run.info.run_id

    model_uris = [
        "models:/{name}/{stage}".format(name="imlegit", stage="None"),
        f"runs:/{run_id}/model",
    ]
    for model_uri in model_uris:
        with TempDir() as tpm:
            output_file_path = tpm.path("stoudt")
            with open(output_file_path, "w") as output_file:
                x, _ = iris_data
                scoring_response = pyfunc_serve_and_score_model(
                    model_uri,
                    pd.DataFrame(x),
                    content_type=CONTENT_TYPE_JSON,
                    stdout=output_file,
                    extra_args=["-w", "3"],
                )
            with open(output_file_path) as output_file:
                stdout = output_file.read()
        actual = pd.read_json(scoring_response.content.decode("utf-8"), orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)
        expected_command_pattern = re.compile(
            "gunicorn.*-w 3.*mlflow.pyfunc.scoring_server.wsgi:app"
        )
        assert expected_command_pattern.search(stdout) is not None


def test_predict(iris_data, sk_model):
    with TempDir(chdr=True) as tmp:
        with mlflow.start_run() as active_run:
            mlflow.sklearn.log_model(sk_model, "model", registered_model_name="impredicting")
            model_uri = f"runs:/{active_run.info.run_id}/model"
        model_registry_uri = "models:/{name}/{stage}".format(name="impredicting", stage="None")
        input_json_path = tmp.path("input.json")
        input_csv_path = tmp.path("input.csv")
        output_json_path = tmp.path("output.json")
        x, _ = iris_data
        with open(input_json_path, "w") as f:
            json.dump({"dataframe_split": pd.DataFrame(x).to_dict(orient="split")}, f)

        pd.DataFrame(x).to_csv(input_csv_path, index=False)

        # Test with no conda & model registry URI
        subprocess.run(
            [
                "mlflow",
                "models",
                "predict",
                "-m",
                model_registry_uri,
                "-i",
                input_json_path,
                "-o",
                output_json_path,
                "--env-manager",
                "local",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env_with_tracking_uri(),
            check=True,
        )
        actual = pd.read_json(output_json_path, orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)

        # With conda + --install-mlflow
        subprocess.run(
            [
                "mlflow",
                "models",
                "predict",
                "-m",
                model_uri,
                "-i",
                input_json_path,
                "-o",
                output_json_path,
                *extra_options,
            ],
            env=env_with_tracking_uri(),
            check=True,
        )
        actual = pd.read_json(output_json_path, orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)

        # explicit json format with default orient (should be split)
        subprocess.run(
            [
                "mlflow",
                "models",
                "predict",
                "-m",
                model_uri,
                "-i",
                input_json_path,
                "-o",
                output_json_path,
                "-t",
                "json",
                *extra_options,
            ],
            env=env_with_tracking_uri(),
            check=True,
        )

        actual = pd.read_json(output_json_path, orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)

        # explicit json format with orient==split
        subprocess.run(
            [
                "mlflow",
                "models",
                "predict",
                "-m",
                model_uri,
                "-i",
                input_json_path,
                "-o",
                output_json_path,
                "-t",
                "json",
                *extra_options,
            ],
            env=env_with_tracking_uri(),
            check=True,
        )
        actual = pd.read_json(output_json_path, orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)

        # read from stdin, write to stdout.
        prc = subprocess.run(
            [
                "mlflow",
                "models",
                "predict",
                "-m",
                model_uri,
                "-t",
                "json",
                *extra_options,
            ],
            input=Path(input_json_path).read_text(),
            stdout=subprocess.PIPE,
            env=env_with_tracking_uri(),
            text=True,
            check=True,
        )
        actual = pd.read_json(prc.stdout, orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)

        # NB: We do not test orient=records here because records may loose column ordering.
        # orient == records is tested in other test with simpler model.

        # csv
        subprocess.run(
            [
                "mlflow",
                "models",
                "predict",
                "-m",
                model_uri,
                "-i",
                input_csv_path,
                "-o",
                output_json_path,
                "-t",
                "csv",
                *extra_options,
            ],
            env=env_with_tracking_uri(),
            check=True,
        )
        actual = pd.read_json(output_json_path, orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)


def test_prepare_env_passes(sk_model):
    if no_conda:
        pytest.skip("This test requires conda.")

    with TempDir(chdr=True):
        with mlflow.start_run() as active_run:
            mlflow.sklearn.log_model(sk_model, "model")
            model_uri = f"runs:/{active_run.info.run_id}/model"

        # With conda
        subprocess.run(
            [
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
                sk_model, "model", pip_requirements=["does-not-exist-dep==abc"]
            )
            model_uri = f"runs:/{active_run.info.run_id}/model"

        # With conda - should fail due to bad conda environment.
        prc = subprocess.run(
            [
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
                sk_model, "model", extra_pip_requirements=[PROTOBUF_REQUIREMENT]
            )
        else:
            mlflow.sklearn.log_model(sk_model, "model")
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
                sk_model, "model", extra_pip_requirements=[PROTOBUF_REQUIREMENT]
            )
        else:
            mlflow.sklearn.log_model(sk_model, "model")
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
        model_info = mlflow.sklearn.log_model(sk_model, "model")

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
                sk_model, "model", extra_pip_requirements=[PROTOBUF_REQUIREMENT]
            )
        else:
            mlflow.sklearn.log_model(sk_model, "model")
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
    scoring_proc = pyfunc_serve_from_docker_image_with_env_override(
        image_name, host_port, gunicorn_options
    )
    _validate_with_rest_endpoint(scoring_proc, host_port, df, x, sk_model, enable_mlserver)


def test_build_docker_without_model_uri(iris_data, sk_model, tmp_path):
    model_path = tmp_path.joinpath("model")
    mlflow.sklearn.save_model(sk_model, model_path)
    image_name = pyfunc_build_image(model_uri=None)
    host_port = get_safe_port()
    scoring_proc = pyfunc_serve_from_docker_image_with_env_override(
        image_name,
        host_port,
        gunicorn_options,
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
                "Failed to serve prediction, got response %s" % scoring_response.text
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


def test_change_conda_env_root_location(tmp_path, sk_model):
    env_root1_path = tmp_path / "root1"
    env_root1_path.mkdir()

    env_root2_path = tmp_path / "root2"
    env_root2_path.mkdir()

    model1_path = tmp_path / "model1"
    mlflow.sklearn.save_model(sk_model, str(model1_path), pip_requirements=["scikit-learn==1.0.1"])

    model2_path = tmp_path / "model2"
    mlflow.sklearn.save_model(sk_model, str(model2_path), pip_requirements=["scikit-learn==1.0.2"])

    env_path_set = set()
    for env_root_path, model_path, sklearn_ver in [
        (env_root1_path, model1_path, "1.0.1"),
        (
            env_root2_path,
            model1_path,
            "1.0.1",
        ),  # test the same env created in different env root path.
        (
            env_root1_path,
            model2_path,
            "1.0.2",
        ),  # test different env created in the same env root path.
    ]:
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
        env_path_set.add(str(env_path))

        python_exec_path = str(env_path / "bin" / "python")

        # Test execution of command under the correct activated python env.
        env.execute(
            command=f"python -c \"import sys; assert sys.executable == '{python_exec_path}'; "
            f"import sklearn; assert sklearn.__version__ == '{sklearn_ver}'\"",
        )

    assert len(env_path_set) == 3
