import mlflow
import pytest

from mlflow.utils.file_utils import local_file_uri_to_path


def test_llm_predictions_logging():
    import csv

    inputs = [
        "{'question': 'How do I create a Databricks cluster with UC enabled?',"
        "'context': 'Databricks clusters are amazing'}",
    ]

    outputs = [
        "<Instructions for cluster creation with UC enabled>",
    ]

    intermediate_outputs = [
        "Context: DataBricks clusters are ... Question:"
        " How do I create a Databricks cluster with UC access? Answer:",
    ]

    output_feedbacks = [
        "{feedback: 'thumps up', score: 0.9, toxicity: 0.1}",
    ]

    artifact_file_name = "llm_predictions.csv"
    with mlflow.start_run() as run:
        with pytest.raises(
            ValueError,
            match="The length of inputs, outputs must be the same and not empty.",
        ):
            mlflow.llm.log_predictions([], [], [])

        with pytest.raises(
            ValueError,
            match="The length of inputs, outputs must be the same and not empty.",
        ):
            mlflow.llm.log_predictions(
                [], ["<Instructions for cluster creation with UC enabled>"], []
            )

        mlflow.llm.log_predictions(
            inputs,
            outputs,
            intermediate_outputs=intermediate_outputs,
            output_feedbacks=output_feedbacks,
        )
        artifact_path = local_file_uri_to_path(mlflow.get_artifact_uri(artifact_file_name))

        with open(artifact_path, newline="") as csvfile:
            predictions = list(csv.reader(csvfile))

        # length of header + length of inputs
        assert len(predictions) == 2
        assert predictions[1][0] == str(inputs[0])
        assert predictions[1][1] == outputs[0]
        assert predictions[1][2] == intermediate_outputs[0]
        assert predictions[1][3] == output_feedbacks[0]

        mlflow.llm.log_predictions(inputs, outputs, output_feedbacks)

        with open(artifact_path, newline="") as csvfile:
            predictions = list(csv.reader(csvfile))

        assert len(predictions) == 3

        # print("tags", run.data.tags)
        # current_tag_value = run.data.tags.get("llm_predictions")
        # assert current_tag_value == "['llm_predictions.csv']"
