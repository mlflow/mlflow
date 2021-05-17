import os
import pytest

from click.testing import CliRunner
from mlflow.deployments import cli


f_model_uri = "fake_model_uri"
f_name = "fake_deployment_name"
f_flavor = "fake_flavor"
f_target = "faketarget"


def test_create():
    runner = CliRunner()
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
    runner = CliRunner()
    res = runner.invoke(
        cli.update_deployment,
        ["--flavor", f_flavor, "--model-uri", f_model_uri, "--target", f_target, "--name", f_name],
    )
    assert "Deployment {} is updated (with flavor {})".format(f_name, f_flavor) in res.stdout


def test_delete():
    runner = CliRunner()
    res = runner.invoke(cli.delete_deployment, ["--name", f_name, "--target", f_target])
    assert "Deployment {} is deleted".format(f_name) in res.stdout


def test_update_no_flavor():
    runner = CliRunner()
    res = runner.invoke(
        cli.update_deployment, ["--name", f_name, "--target", f_target, "-m", f_model_uri]
    )
    assert "Deployment {} is updated (with flavor None)".format(f_name) in res.stdout


def test_list():
    runner = CliRunner()
    res = runner.invoke(cli.list_deployment, ["--target", f_target])
    assert "['{}']".format(f_name) in res.stdout


def test_custom_args():
    runner = CliRunner()
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
    runner = CliRunner()
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
    runner = CliRunner()
    res = runner.invoke(
        cli.predict, ["--target", f_target, "--name", f_name, "--input-path", temp_input_file_path],
    )
    assert "1" in res.stdout


def test_target_help():
    runner = CliRunner()
    res = runner.invoke(cli.target_help, ["--target", f_target])
    assert "Target help is called" in res.stdout


def test_run_local():
    runner = CliRunner()
    res = runner.invoke(
        cli.run_local, ["-f", f_flavor, "-m", f_model_uri, "-t", f_target, "--name", f_name]
    )
    assert "Deployed locally at the key {}".format(f_name) in res.stdout
    assert "using the model from {}.".format(f_model_uri) in res.stdout
    assert "It's flavor is {} and config is {}".format(f_flavor, str({})) in res.stdout
