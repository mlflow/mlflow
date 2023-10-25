# Location: mlflow/mlflow/tracking/fluent.py:1165
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:1165 '])
def test(_):
    # Loads the table with the specified name for all runs in the given
    # experiment and joins them together
    import mlflow

    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "toxicity": [0.0, 0.0],
    }

    with mlflow.start_run():
        # Log the dictionary as a table
        mlflow.log_table(data=table_dict, artifact_file="qabot_eval_results.json")

    loaded_table = mlflow.load_table(
        "qabot_eval_results.json",
        # Append the run ID and the parent run ID to the table
        extra_columns=["run_id"],
    )


if __name__ == "__main__":
    test()
