# Location: mlflow/mlflow/tracking/fluent.py:1142
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:1142 '])
def test(_):
    import mlflow

    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "toxicity": [0.0, 0.0],
    }

    with mlflow.start_run() as run:
        # Log the dictionary as a table
        mlflow.log_table(data=table_dict, artifact_file="qabot_eval_results.json")
        run_id = run.info.run_id

    loaded_table = mlflow.load_table(
        artifact_file="qabot_eval_results.json",
        run_ids=[run_id],
        # Append a column containing the associated run ID for each row
        extra_columns=["run_id"],
    )


if __name__ == "__main__":
    test()
