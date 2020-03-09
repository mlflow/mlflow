from click.testing import CliRunner
from mlflow.runs import list_run
import mlflow


def test_list_run():
    with mlflow.start_run(run_name='apple'):
        pass
    result = CliRunner().invoke(list_run, ["--experiment-id", "0"])
    assert 'apple' in result.output


def test_list_run_experiment_id_required():
    result = CliRunner().invoke(list_run, [])
    assert 'Missing option "--experiment-id"' in result.output
