"""
Small script used to generate mock data to test the UI.
"""
import mlflow
import itertools
from random import random

from mlflow.tracking import MlflowClient

SOURCE_VERSIONS = [
    'f7581541a524f4879794e724a9653eaca2bef1d7',
    '53de5661eb457efa3cb996aa592656c41a888c1d',
    'ccc76efe9ceb633710bbd7acf408bebe0095eb10'
]


def log_metrics(metrics):
    for k, values in metrics.items():
        for v in values:
            mlflow.log_metric(k, v)


def log_params(parameters):
    for k, v in parameters.items():
        mlflow.log_param(k, v)


if __name__ == '__main__':
    client = MlflowClient()
    # Simple run
    for l1, alpha in itertools.product([0, 0.25, 0.5, 0.75, 1], [0, 0.5, 1]):
        with mlflow.start_run(source_name='ipython', source_version=SOURCE_VERSIONS[0]):
            parameters = {
                'l1': str(l1),
                'alpha': str(alpha),
            }
            metrics = {
                'MAE': [random()],
                'R2': [random()],
                'RMSE': [random()],
            }
            log_params(parameters)
            log_metrics(metrics)

    # Big parameter values
    with mlflow.start_run(source_name='ipython', source_version=SOURCE_VERSIONS[1]):
        parameters = {
            'this is a pretty long parameter name': 'NA10921-test_file_2018-08-10.txt',
        }
        metrics = {
            'grower': [i ** 1.2 for i in range(10)]
        }
        log_params(parameters)
        log_metrics(metrics)

    # Nested runs.
    with mlflow.start_run(source_name='multirun.py'):
        l1 = 0.5
        alpha = 0.5
        parameters = {
            'l1': str(l1),
            'alpha': str(alpha),
        }
        metrics = {
            'MAE': [random()],
            'R2': [random()],
            'RMSE': [random()],
        }
        log_params(parameters)
        log_metrics(metrics)

        with mlflow.start_run(source_name='child_params.py', nested=True):
            parameters = {
                'lot': str(random()),
                'of': str(random()),
                'parameters': str(random()),
                'in': str(random()),
                'this': str(random()),
                'experiement': str(random()),
                'run': str(random()),
                'because': str(random()),
                'we': str(random()),
                'need': str(random()),
                'to': str(random()),
                'check': str(random()),
                'how': str(random()),
                'it': str(random()),
                'handles': str(random()),
            }
            log_params(parameters)
            mlflow.log_metric('test_metric', 1)

        with mlflow.start_run(source_name='child_metrics.py', nested=True):
            metrics = {
                'lot': [random()],
                'of': [random()],
                'parameters': [random()],
                'in': [random()],
                'this': [random()],
                'experiement': [random()],
                'run': [random()],
                'because': [random()],
                'we': [random()],
                'need': [random()],
                'to': [random()],
                'check': [random()],
                'how': [random()],
                'it': [random()],
                'handles': [random()],
            }
            log_metrics(metrics)

        with mlflow.start_run(source_name='sort_child.py', nested=True):
            mlflow.log_metric('test_metric', 1)
            mlflow.log_param('test_param', 1)

        with mlflow.start_run(source_name='sort_child.py', nested=True):
            mlflow.log_metric('test_metric', 2)
            mlflow.log_param('test_param', 2)

    # Grandchildren
    with mlflow.start_run(source_name='parent'):
        with mlflow.start_run(source_name='child', nested=True):
            with mlflow.start_run(source_name='grandchild', nested=True):
                pass

    # Loop
    loop_1_run_id = None
    loop_2_run_id = None
    with mlflow.start_run(source_name='loop-1') as run_1:
        with mlflow.start_run(source_name='loop-2', nested=True) as run_2:
            loop_1_run_id = run_1.info.run_uuid
            loop_2_run_id = run_2.info.run_uuid
    client.set_tag(loop_1_run_id, 'mlflow.parentRunId', loop_2_run_id)

    # Lot's of children
    with mlflow.start_run(source_name='parent-with-lots-of-children'):
        for i in range(100):
            with mlflow.start_run(source_name='child-{}'.format(i), nested=True):
                pass
    mlflow.create_experiment("my-empty-experiment")
    mlflow.set_experiment("runs-but-no-metrics-params")
    for i in range(100):
        with mlflow.start_run(source_name="empty-run-{}".format(i)):
            pass
