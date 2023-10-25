# Location: mlflow/mlflow/tracking/fluent.py:1090
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:1090 '])
def test(_):
    import mlflow

    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "toxicity": [0.0, 0.0],
    }

    with mlflow.start_run():
        # Log the dictionary as a table
        mlflow.log_table(data=table_dict, artifact_file="qabot_eval_results.json")


if __name__ == "__main__":
    test()
