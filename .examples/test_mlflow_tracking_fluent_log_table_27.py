# Location: mlflow/mlflow/tracking/fluent.py:1105
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:1105 '])
def test(_):
    import mlflow
    import pandas as pd

    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "toxicity": [0.0, 0.0],
    }
    df = pd.DataFrame.from_dict(table_dict)

    with mlflow.start_run():
        # Log the df as a table
        mlflow.log_table(data=df, artifact_file="qabot_eval_results.json")


if __name__ == "__main__":
    test()
