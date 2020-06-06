from click.testing import CliRunner
from mlflow.deployments import cli


f_model_uri = 'fake_model_uri'
f_name = 'fake_deployment_id'
f_flavor = 'fake_flavor'
f_target = 'fake_target'


def test_create(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.create_deployment,
                        ['--flavor', f_flavor, '--model-uri', f_model_uri, '--target-uri', f_target, '--name', f_name])
    assert '{} deployment {} is created'.format(f_flavor, f_name) in res.stdout


def test_update(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.update_deployment,
                        ['--flavor', f_flavor, '--model-uri', f_model_uri, '--target-uri', f_target, '--name', f_name])
    assert 'Deployment {} is updated (with flavor {})'.format(f_name, f_flavor) in res.stdout


def test_delete(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.delete_deployment,
                        ['--name', f_name, '--target-uri', f_target])
    assert 'Deployment {} is deleted'.format(f_name) in res.stdout


def test_update_no_flavor(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.update_deployment,
                        ['--name', f_name, '--target-uri', f_target, '-m', f_model_uri])
    assert 'Deployment {} is updated (with flavor None)'.format(f_name) in res.stdout


def test_list(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.list_deployment,
                        ['--target-uri', f_target])
    assert "['{}']".format(f_name) in res.stdout


def test_custom_args(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.create_deployment,
                        ['--model-uri', f_model_uri, '--target-uri', f_target, '--name', f_name, '-C', 'raiseError=True'])
    assert isinstance(res.exception, RuntimeError)


def test_get(patched_plugin_store):  # pylint: disable=W0613
    runner = CliRunner()
    res = runner.invoke(cli.get_deployment,
                        ['--name', f_name, '--target-uri', f_target])
    assert 'key1: val1' in res.stdout
    assert 'key2: val2' in res.stdout


def test_target_help(patched_plugin_store):
    runner = CliRunner()
    res = runner.invoke(cli.target_help, ["--target-name", f_target])
    assert "Target help is called".format(f_target) in res.stdout


def test_run_local(patched_plugin_store):
    runner = CliRunner()
    res = runner.invoke(cli.run_local, [])
