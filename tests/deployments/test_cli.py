import json
import os
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow.deployments import cli
from mlflow.exceptions import MlflowException

f_model_uri = "fake_model_uri"
f_name = "fake_deployment_name"
f_flavor = "fake_flavor"
f_target = "faketarget"
runner = CliRunner()


def test_create():
    res = runner.invoke(
        cli.create_deployment,
        ["--flavor", f_flavor, "--model-uri", f_model_uri, "--target", f_target, "--name", f_name],
    )
    assert f"{f_flavor} deployment {f_name} is created" in res.stdout
    res = runner.invoke(
        cli.create_deployment, ["-f", f_flavor, "-m", f_model_uri, "-t", f_target, "--name", f_name]
    )
    assert f"{f_flavor} deployment {f_name} is created" in res.stdout


def test_update():
    res = runner.invoke(
        cli.update_deployment,
        ["--flavor", f_flavor, "--model-uri", f_model_uri, "--target", f_target, "--name", f_name],
    )
    assert f"Deployment {f_name} is updated (with flavor {f_flavor})" in res.stdout


def test_delete():
    res = runner.invoke(cli.delete_deployment, ["--name", f_name, "--target", f_target])
    assert f"Deployment {f_name} is deleted" in res.stdout


def test_update_no_flavor():
    res = runner.invoke(
        cli.update_deployment, ["--name", f_name, "--target", f_target, "-m", f_model_uri]
    )
    assert f"Deployment {f_name} is updated (with flavor None)" in res.stdout


def test_list():
    res = runner.invoke(cli.list_deployment, ["--target", f_target])
    assert f"[{{'name': '{f_name}'}}]" in res.stdout


def test_create_deployment_with_custom_args():
    res = runner.invoke(
        cli.create_deployment,
        [
            "--model-uri",
            f_model_uri,
            "--target",
            f_target,
            "--name",
            f_name,
            "-C",
            "raiseError=True",
        ],
    )
    assert isinstance(res.exception, RuntimeError)


def test_delete_deployment_with_custom_args():
    res = runner.invoke(
        cli.delete_deployment,
        ["--target", f_target, "--name", f_name, "-C", "raiseError=True"],
    )
    assert isinstance(res.exception, RuntimeError)


def test_get():
    res = runner.invoke(cli.get_deployment, ["--name", f_name, "--target", f_target])
    assert "key1: val1" in res.stdout
    assert "key2: val2" in res.stdout


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny Client does not support predict due to the pandas dependency",
)
def test_predict(tmp_path):
    temp_input_file_path = tmp_path.joinpath("input.json")
    temp_input_file_path.write_text('{"data": [5000]}')

    temp_output_file_path = tmp_path.joinpath("output.json")
    res = runner.invoke(
        cli.predict, ["--target", f_target, "--name", f_name, "--input-path", temp_input_file_path]
    )
    assert '{"predictions": [1, 2, 3]}' in res.stdout

    res = runner.invoke(
        cli.predict,
        [
            "--target",
            f_target,
            "--name",
            f_name,
            "--input-path",
            temp_input_file_path,
            "--output-path",
            temp_output_file_path,
        ],
    )
    with open(temp_output_file_path) as f:
        assert json.load(f) == {"predictions": [1, 2, 3]}


def test_target_help():
    res = runner.invoke(cli.target_help, ["--target", f_target])
    assert "Target help is called" in res.stdout


def test_run_local():
    res = runner.invoke(
        cli.run_local, ["-f", f_flavor, "-m", f_model_uri, "-t", f_target, "--name", f_name]
    )
    assert f"Deployed locally at the key {f_name}" in res.stdout
    assert f"using the model from {f_model_uri}." in res.stdout
    assert f"It's flavor is {f_flavor} and config is {{}}" in res.stdout


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny Client does not support explain due to the pandas dependency",
)
def test_explain(tmp_path):
    temp_input_file_path = tmp_path.joinpath("input.json")
    temp_input_file_path.write_text('{"data": [5000]}')
    res = runner.invoke(
        cli.explain, ["--target", f_target, "--name", f_name, "--input-path", temp_input_file_path]
    )
    assert "1" in res.stdout


def test_explain_with_no_target_implementation(tmp_path):
    file_path = tmp_path.joinpath("input.json")
    file_path.write_text('{"data": [5000]}')
    mock_error = MlflowException("MOCK ERROR")
    with mock.patch.object(CliRunner, "invoke", return_value=mock_error) as mock_explain:
        res = runner.invoke(
            cli.explain, ["--target", f_target, "--name", f_name, "--input-path", file_path]
        )
        assert type(res) == MlflowException
        mock_explain.assert_called_once()
