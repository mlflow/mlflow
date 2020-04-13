from click.testing import CliRunner
from mlflow.deployments import cli


f_model_uri = 'fake_model_uri'
f_deployment_id = 'fake_deployment_id'
f_flavor = 'fake_flavor'
f_target = 'fake_target'


def test_create(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.create_deployment,
                        ['--flavor', f_flavor, '-m', f_model_uri, '--target', f_target])
    assert '{} deployment {} is created'.format(f_flavor, f_deployment_id) in res.stdout


def test_delete(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.delete_deployment,
                        ['--id', f_deployment_id, '--target', f_target])
    assert 'Deployment {} is deleted'.format(f_deployment_id) in res.stdout


def test_update_argument_errors(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.update_deployment,
                        ['--id', f_deployment_id, '--target', f_target, '-m', f_model_uri])
    assert 'Deployment {} is updated (with flavor None)'.format(f_deployment_id) in res.stdout


def test_list(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.list_deployment,
                        ['--target', f_target])
    assert "['{}']".format(f_deployment_id) in res.stdout


def test_custom_args(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.list_deployment,
                        ['--target', f_target, '-C', 'raiseError=True'])
    assert isinstance(res.exception, RuntimeError)


def test_get(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.get_deployment,
                        ['--id', f_deployment_id, '--target', f_target])
    assert 'key1: val1' in res.stdout
    assert 'key2: val2' in res.stdout
