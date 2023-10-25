# Location: mlflow/mlflow/tracking/client.py:1539
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/client.py:1539 '])
def test(_):
    import mlflow
    import pandas as pd
    from mlflow import MlflowClient

    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "toxicity": [0.0, 0.0],
    }
    df = pd.DataFrame.from_dict(table_dict)

    client = MlflowClient()
    run = client.create_run(experiment_id="0")
    client.log_table(run.info.run_id, data=df, artifact_file="qabot_eval_results.json")


if __name__ == "__main__":
    test()
