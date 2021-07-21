import os
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
    assert "{} deployment {} is created".format(f_flavor, f_name) in res.stdout
    res = runner.invoke(
        cli.create_deployment, ["-f", f_flavor, "-m", f_model_uri, "-t", f_target, "--name", f_name]
    )
    assert "{} deployment {} is created".format(f_flavor, f_name) in res.stdout


def test_update():
    res = runner.invoke(
        cli.update_deployment,
        ["--flavor", f_flavor, "--model-uri", f_model_uri, "--target", f_target, "--name", f_name],
    )
    assert "Deployment {} is updated (with flavor {})".format(f_name, f_flavor) in res.stdout


def test_delete():
    res = runner.invoke(cli.delete_deployment, ["--name", f_name, "--target", f_target])
    assert "Deployment {} is deleted".format(f_name) in res.stdout


def test_update_no_flavor():
    res = runner.invoke(
        cli.update_deployment, ["--name", f_name, "--target", f_target, "-m", f_model_uri]
    )
    assert "Deployment {} is updated (with flavor None)".format(f_name) in res.stdout


def test_list():
    res = runner.invoke(cli.list_deployment, ["--target", f_target])
    assert "['{}']".format(f_name) in res.stdout


def test_custom_args():
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


def test_get():
    res = runner.invoke(cli.get_deployment, ["--name", f_name, "--target", f_target])
    assert "key1: val1" in res.stdout
    assert "key2: val2" in res.stdout


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny Client does not support predict due to the pandas dependency",
)
def test_predict(tmpdir):
    temp_input_file_path = tmpdir.join("input.json").strpath
    with open(temp_input_file_path, "w") as temp_input_file:
        temp_input_file.write('{"data": [5000]}')
    res = runner.invoke(
        cli.predict, ["--target", f_target, "--name", f_name, "--input-path", temp_input_file_path]
    )
    assert "1" in res.stdout


def test_target_help():
    res = runner.invoke(cli.target_help, ["--target", f_target])
    assert "Target help is called" in res.stdout


def test_run_local():
    res = runner.invoke(
        cli.run_local, ["-f", f_flavor, "-m", f_model_uri, "-t", f_target, "--name", f_name]
    )
    assert "Deployed locally at the key {}".format(f_name) in res.stdout
    assert "using the model from {}.".format(f_model_uri) in res.stdout
    assert "It's flavor is {} and config is {}".format(f_flavor, str({})) in res.stdout


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny Client does not support explain due to the pandas dependency",
)
def test_explain(tmpdir):
    temp_input_file_path = tmpdir.join("input.json").strpath
    with open(temp_input_file_path, "w") as temp_input_file:
        temp_input_file.write('{"data": [5000]}')
    res = runner.invoke(
        cli.explain, ["--target", f_target, "--name", f_name, "--input-path", temp_input_file_path]
    )
    assert "1" in res.stdout


def test_explain_with_no_target_implementation(tmpdir):
    from unittest import mock

    file_path = tmpdir.join("input.json").strpath
    with open(file_path, "w") as temp_input_file:
        temp_input_file.write('{"data": [5000]}')
    mock_error = MlflowException("MOCK ERROR")
    with mock.patch.object(CliRunner, "invoke", return_value=mock_error) as mock_explain:
        res = runner.invoke(
            cli.explain, ["--target", f_target, "--name", f_name, "--input-path", file_path]
        )
        assert type(res) == MlflowException
        mock_explain.assert_called_once()
