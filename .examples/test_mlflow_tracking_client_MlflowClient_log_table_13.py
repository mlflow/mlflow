# Location: mlflow/mlflow/tracking/client.py:1521
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/client.py:1521 '])
def test(_):
    import mlflow
    from mlflow import MlflowClient

    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "toxicity": [0.0, 0.0],
    }

    client = MlflowClient()
    run = client.create_run(experiment_id="0")
    client.log_table(
        run.info.run_id, data=table_dict, artifact_file="qabot_eval_results.json"
    )


if __name__ == "__main__":
    test()
