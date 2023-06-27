import csv
import mlflow
import pytest

from mlflow.utils.file_utils import local_file_uri_to_path


def test_llm_predictions_logging():
    inputs = [
        {
            "question": "How do I create a Databricks cluster with UC enabled?",
            "context": "Databricks clusters are amazing",
        }
    ]

    outputs = [
        "<Instructions for cluster creation with UC enabled>",
    ]

    prompts = [
        "Get Databricks documentation to answer all the questions: {input}",
    ]

    artifact_file_name = "llm_predictions.csv"
    with mlflow.start_run():
        with pytest.raises(
            ValueError,
            match="The length of inputs, outputs and prompts must be the same and not empty.",
        ):
            mlflow.llm.log_predictions([], [], [])

        with pytest.raises(
            ValueError,
            match="The length of inputs, outputs and prompts must be the same and not empty.",
        ):
            mlflow.llm.log_predictions(
                [], ["<Instructions for cluster creation with UC enabled>"], []
            )

        mlflow.llm.log_predictions(inputs, outputs, prompts)
        artifact_path = local_file_uri_to_path(mlflow.get_artifact_uri(artifact_file_name))

        with open(artifact_path, newline="") as csvfile:
            predictions = list(csv.reader(csvfile))

        # length of header + length of inputs
        assert len(predictions) == 2
        assert predictions[1][0] == str(inputs[0])
        assert predictions[1][1] == outputs[0]
        assert predictions[1][2] == prompts[0]

        mlflow.llm.log_predictions(inputs, outputs, prompts)

        with open(artifact_path, newline="") as csvfile:
            predictions = list(csv.reader(csvfile))

        assert len(predictions) == 3
