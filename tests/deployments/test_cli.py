from click.testing import CliRunner
from mlflow.deployments import cli


f_model_uri = 'fake_model_uri'
f_deployment_id = 'fake_deployment_id'
f_flavor = 'fake_flavor'
f_target = 'fake_target'


def test_create(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.create_cli,
                        ['--flavor', f_flavor, '-m', f_model_uri, '--target', f_target])
    assert '{} deployment {} is created'.format(f_flavor, f_deployment_id) in res.stdout


def test_delete(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.delete_cli,
                        ['--id', f_deployment_id, '--target', f_target])
    assert 'Deployment {} is deleted'.format(f_deployment_id) in res.stdout


def test_update_argument_errors(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.update_cli,
                        ['--id', f_deployment_id, '--target', f_target, '-m', f_model_uri])
    assert 'Deployment {} is updated (with flavor None)'.format(f_deployment_id) in res.stdout


def test_list(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.list_cli,
                        ['--target', f_target])
    assert "['{}']".format(f_deployment_id) in res.stdout


def test_custom_args(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.list_cli,
                        ['--target', f_target, '--raiseError', 'True'])
    assert 'Error: Error requested' in res.stdout


def test_describe(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.describe_cli,
                        ['--id', f_deployment_id, '--target', f_target])
    assert 'key1: val1' in res.stdout
    assert 'key2: val2' in res.stdout
