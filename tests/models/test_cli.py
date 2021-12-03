import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import re
import sklearn
import sklearn.datasets
import sklearn.neighbors

from unittest import mock

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import mlflow
from mlflow import pyfunc
import mlflow.sklearn

from mlflow.utils.file_utils import TempDir, path_to_local_file_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils import PYTHON_VERSION
from tests.models import test_pyfunc
from tests.helper_functions import (
    pyfunc_build_image,
    pyfunc_serve_from_docker_image,
    pyfunc_serve_from_docker_image_with_env_override,
    RestEndpoint,
    get_safe_port,
    pyfunc_serve_and_score_model,
)
from mlflow.protos.databricks_pb2 import ErrorCode, BAD_REQUEST
from mlflow.pyfunc.scoring_server import (
    CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_CSV,
)

# NB: for now, windows tests do not have conda available.
no_conda = ["--no-conda"] if sys.platform == "win32" else []

# NB: need to install mlflow since the pip version does not have mlflow models cli.
install_mlflow = ["--install-mlflow"] if not no_conda else []

extra_options = no_conda + install_mlflow
gunicorn_options = "--timeout 60 -w 5"


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


@pytest.mark.large
def test_predict_with_old_mlflow_in_conda_and_with_orient_records(iris_data):
    if no_conda:
        pytest.skip("This test needs conda.")
    # TODO: Enable this test after 1.0 is out to ensure we do not break the serve / predict
    # TODO: Also add a test for serve, not just predict.
    pytest.skip("TODO: enable this after 1.0 release is out.")
    x, _ = iris_data
    with TempDir() as tmp:
        input_records_path = tmp.path("input_records.json")
        pd.DataFrame(x).to_json(input_records_path, orient="records")
        output_json_path = tmp.path("output.json")
        test_model_path = tmp.path("test_model")
        test_model_conda_path = tmp.path("conda.yml")
        # create env with old mlflow!
        _mlflow_conda_env(
            path=test_model_conda_path,
            additional_pip_deps=["mlflow=={}".format(test_pyfunc.MLFLOW_VERSION)],
        )
        pyfunc.save_model(
            path=test_model_path,
            loader_module=test_pyfunc.__name__.split(".")[-1],
            code_path=[test_pyfunc.__file__],
            conda_env=test_model_conda_path,
        )
        # explicit json format with orient records
        p = subprocess.Popen(
            [
                "mlflow",
                "models",
                "predict",
                "-m",
                path_to_local_file_uri(test_model_path),
                "-i",
                input_records_path,
                "-o",
                output_json_path,
                "-t",
                "json",
                "--json-format",
                "records",
            ]
            + no_conda
        )
        assert 0 == p.wait()
        actual = pd.read_json(output_json_path, orient="records")
        actual = actual[actual.columns[0]].values
        expected = test_pyfunc.PyFuncTestModel(check_version=False).predict(df=pd.DataFrame(x))
        assert all(expected == actual)


@pytest.mark.large
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
        p = subprocess.Popen(
            ["mlflow", "models", "predict", "-m", fake_model_path],
            stderr=subprocess.PIPE,
            cwd=tmp.path(""),
        )
        _, stderr = p.communicate()
        stderr = stderr.decode("utf-8")
        print(stderr)
        assert p.wait() != 0
        if PYTHON_VERSION.startswith("3"):
            assert "ModuleNotFoundError: No module named 'mlflow'" in stderr
        else:
            assert "ImportError: No module named mlflow.pyfunc.scoring_server" in stderr


@pytest.mark.large
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
        p = subprocess.Popen(
            ["mlflow", "models", "predict", "-m", tmp.path("model")],
            stderr=subprocess.PIPE,
            cwd=tmp.path(""),
        )
        _, stderr = p.communicate()
        stderr = stderr.decode("utf-8")
        print(stderr)
        assert p.wait() != 0
        assert "No suitable flavor backend was found for the model." in stderr


@pytest.mark.large
def test_serve_gunicorn_opts(iris_data, sk_model):
    if sys.platform == "win32":
        pytest.skip("This test requires gunicorn which is not available on windows.")
    with mlflow.start_run() as active_run:
        mlflow.sklearn.log_model(sk_model, "model", registered_model_name="imlegit")
        run_id = active_run.info.run_id

    model_uris = [
        "models:/{name}/{stage}".format(name="imlegit", stage="None"),
        "runs:/{run_id}/model".format(run_id=run_id),
    ]
    for model_uri in model_uris:
        with TempDir() as tpm:
            output_file_path = tpm.path("stoudt")
            with open(output_file_path, "w") as output_file:
                x, _ = iris_data
                scoring_response = pyfunc_serve_and_score_model(
                    model_uri,
                    pd.DataFrame(x),
                    content_type=CONTENT_TYPE_JSON_SPLIT_ORIENTED,
                    stdout=output_file,
                    extra_args=["-w", "3"],
                )
            with open(output_file_path, "r") as output_file:
                stdout = output_file.read()
        actual = pd.read_json(scoring_response.content, orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)
        expected_command_pattern = re.compile(
            ("gunicorn.*-w 3.*mlflow.pyfunc.scoring_server.wsgi:app")
        )
        assert expected_command_pattern.search(stdout) is not None


@pytest.mark.large
def test_predict(iris_data, sk_model):
    with TempDir(chdr=True) as tmp:
        with mlflow.start_run() as active_run:
            mlflow.sklearn.log_model(sk_model, "model", registered_model_name="impredicting")
            model_uri = "runs:/{run_id}/model".format(run_id=active_run.info.run_id)
        model_registry_uri = "models:/{name}/{stage}".format(name="impredicting", stage="None")
        input_json_path = tmp.path("input.json")
        input_csv_path = tmp.path("input.csv")
        output_json_path = tmp.path("output.json")
        x, _ = iris_data
        pd.DataFrame(x).to_json(input_json_path, orient="split")
        pd.DataFrame(x).to_csv(input_csv_path, index=False)

        # Test with no conda & model registry URI
        env_with_tracking_uri = os.environ.copy()
        env_with_tracking_uri.update(MLFLOW_TRACKING_URI=mlflow.get_tracking_uri())
        p = subprocess.Popen(
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
                "--no-conda",
            ],
            stderr=subprocess.PIPE,
            env=env_with_tracking_uri,
        )
        assert p.wait() == 0
        actual = pd.read_json(output_json_path, orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)

        # With conda + --install-mlflow
        p = subprocess.Popen(
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
            ]
            + extra_options,
            env=env_with_tracking_uri,
        )
        assert 0 == p.wait()
        actual = pd.read_json(output_json_path, orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)

        # explicit json format with default orient (should be split)
        p = subprocess.Popen(
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
            ]
            + extra_options,
            env=env_with_tracking_uri,
        )
        assert 0 == p.wait()
        actual = pd.read_json(output_json_path, orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)

        # explicit json format with orient==split
        p = subprocess.Popen(
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
                "--json-format",
                "split",
            ]
            + extra_options,
            env=env_with_tracking_uri,
        )
        assert 0 == p.wait()
        actual = pd.read_json(output_json_path, orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)

        # read from stdin, write to stdout.
        p = subprocess.Popen(
            ["mlflow", "models", "predict", "-m", model_uri, "-t", "json", "--json-format", "split"]
            + extra_options,
            universal_newlines=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            env=env_with_tracking_uri,
        )
        with open(input_json_path, "r") as f:
            stdout, _ = p.communicate(f.read())
        assert 0 == p.wait()
        actual = pd.read_json(StringIO(stdout), orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)

        # NB: We do not test orient=records here because records may loose column ordering.
        # orient == records is tested in other test with simpler model.

        # csv
        p = subprocess.Popen(
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
            ]
            + extra_options,
            env=env_with_tracking_uri,
        )
        assert 0 == p.wait()
        actual = pd.read_json(output_json_path, orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)


@pytest.mark.large
def test_prepare_env_passes(sk_model):
    if no_conda:
        pytest.skip("This test requires conda.")

    with TempDir(chdr=True):
        with mlflow.start_run() as active_run:
            mlflow.sklearn.log_model(sk_model, "model")
            model_uri = "runs:/{run_id}/model".format(run_id=active_run.info.run_id)

        # Test with no conda
        p = subprocess.Popen(
            ["mlflow", "models", "prepare-env", "-m", model_uri, "--no-conda"],
            stderr=subprocess.PIPE,
        )
        assert p.wait() == 0

        # With conda
        p = subprocess.Popen(
            ["mlflow", "models", "prepare-env", "-m", model_uri], stderr=subprocess.PIPE
        )
        assert p.wait() == 0

        # Should be idempotent
        p = subprocess.Popen(
            ["mlflow", "models", "prepare-env", "-m", model_uri], stderr=subprocess.PIPE
        )
        assert p.wait() == 0


@pytest.mark.large
def test_prepare_env_fails(sk_model):
    if no_conda:
        pytest.skip("This test requires conda.")

    with TempDir(chdr=True):
        with mlflow.start_run() as active_run:
            mlflow.sklearn.log_model(
                sk_model, "model", conda_env={"dependencies": ["mlflow-does-not-exist-dep==abc"]}
            )
            model_uri = "runs:/{run_id}/model".format(run_id=active_run.info.run_id)

        # Test with no conda
        p = subprocess.Popen(["mlflow", "models", "prepare-env", "-m", model_uri, "--no-conda"])
        assert p.wait() == 0

        # With conda - should fail due to bad conda environment.
        p = subprocess.Popen(["mlflow", "models", "prepare-env", "-m", model_uri])
        assert p.wait() != 0


@pytest.mark.large
@pytest.mark.parametrize("enable_mlserver", [True, False])
def test_build_docker(iris_data, sk_model, enable_mlserver):
    with mlflow.start_run() as active_run:
        if enable_mlserver:
            # MLServer requires Python 3.7, so we'll force that Python version
            with mock.patch("mlflow.utils.environment.PYTHON_VERSION", "3.7"):
                mlflow.sklearn.log_model(sk_model, "model")
        else:
            mlflow.sklearn.log_model(sk_model, "model")
        model_uri = "runs:/{run_id}/model".format(run_id=active_run.info.run_id)

    x, _ = iris_data
    df = pd.DataFrame(x)

    extra_args = ["--install-mlflow"]
    if enable_mlserver:
        extra_args.append("--enable-mlserver")

    image_name = pyfunc_build_image(model_uri, extra_args=extra_args)
    host_port = get_safe_port()
    scoring_proc = pyfunc_serve_from_docker_image(image_name, host_port)
    _validate_with_rest_endpoint(scoring_proc, host_port, df, x, sk_model, enable_mlserver)


@pytest.mark.large
@pytest.mark.parametrize("enable_mlserver", [True, False])
def test_build_docker_with_env_override(iris_data, sk_model, enable_mlserver):
    with mlflow.start_run() as active_run:
        if enable_mlserver:
            # MLServer requires Python 3.7, so we'll force that Python version
            with mock.patch("mlflow.utils.environment.PYTHON_VERSION", "3.7"):
                mlflow.sklearn.log_model(sk_model, "model")
        else:
            mlflow.sklearn.log_model(sk_model, "model")
        model_uri = "runs:/{run_id}/model".format(run_id=active_run.info.run_id)
    x, _ = iris_data
    df = pd.DataFrame(x)

    extra_args = ["--install-mlflow"]
    if enable_mlserver:
        extra_args.append("--enable-mlserver")

    image_name = pyfunc_build_image(model_uri, extra_args=extra_args)
    host_port = get_safe_port()
    scoring_proc = pyfunc_serve_from_docker_image_with_env_override(
        image_name, host_port, gunicorn_options
    )
    _validate_with_rest_endpoint(scoring_proc, host_port, df, x, sk_model, enable_mlserver)


def _validate_with_rest_endpoint(scoring_proc, host_port, df, x, sk_model, enable_mlserver=False):
    with RestEndpoint(proc=scoring_proc, port=host_port) as endpoint:
        for content_type in [CONTENT_TYPE_JSON_SPLIT_ORIENTED, CONTENT_TYPE_CSV, CONTENT_TYPE_JSON]:
            scoring_response = endpoint.invoke(df, content_type)
            assert scoring_response.status_code == 200, (
                "Failed to serve prediction, got " "response %s" % scoring_response.text
            )
            np.testing.assert_array_equal(
                np.array(json.loads(scoring_response.text)), sk_model.predict(x)
            )
        # Try examples of bad input, verify we get a non-200 status code
        for content_type in [CONTENT_TYPE_JSON_SPLIT_ORIENTED, CONTENT_TYPE_CSV, CONTENT_TYPE_JSON]:
            scoring_response = endpoint.invoke(data="", content_type=content_type)
            expected_status_code = 500 if enable_mlserver else 400
            assert scoring_response.status_code == expected_status_code, (
                "Expected server failure with error code %s, got response with status code %s "
                "and body %s"
                % (expected_status_code, scoring_response.status_code, scoring_response.text)
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
            assert "stack_trace" in scoring_response_dict
