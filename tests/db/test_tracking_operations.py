import pytest

import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType


pytestmark = pytest.mark.notrackingurimock


class Model(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        pass


def start_run_and_log_data():
    with mlflow.start_run():
        mlflow.log_param("p", "param")
        mlflow.log_metric("m", 1.0)
        mlflow.set_tag("t", "tag")
        mlflow.pyfunc.log_model(
            artifact_path="model", python_model=Model(), registered_model_name="model"
        )


def test_search_runs():
    start_run_and_log_data()
    runs = mlflow.search_runs(experiment_ids=["0"], order_by=["param.start_time DESC"])
    mlflow.get_run(runs["run_id"][0])


def test_list_experiments():
    start_run_and_log_data()
    experiments = mlflow.list_experiments(view_type=ViewType.ALL, max_results=5)
    assert len(experiments) > 0


def test_set_run_status_to_killed():
    """
    This test ensures the following migration scripts work correctly:
    - cfd24bdc0731_update_run_status_constraint_with_killed.py
    - 0a8213491aaa_drop_duplicate_killed_constraint.py
    """
    with mlflow.start_run() as run:
        pass
    client = MlflowClient()
    client.set_terminated(run_id=run.info.run_id, status="KILLED")
