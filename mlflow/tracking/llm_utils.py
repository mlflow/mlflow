import csv
import logging
import os
import tempfile
from typing import Dict, List, Union

import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.utils.annotations import deprecated

_logger = logging.getLogger(__name__)


@deprecated(
    "mlflow.log_table", since="2.8.1", impact="This method will be removed in MLflow 2.9.0."
)
def log_predictions(
    inputs: List[Union[str, Dict[str, str]]],
    outputs: List[str],
    prompts: List[Union[str, Dict[str, str]]],
) -> None:
    """
    Log a batch of inputs, outputs and prompts for the current evaluation run.
    If no run is active, this method will create a new active run.

    :param inputs: Union of either List of input strings or List of input dictionary
    :param outputs: List of output strings
    :param prompts: Union of either List of prompt strings or List of prompt dictionary
    :returns: None

    .. testcode:: python
        :caption: Example

        import mlflow

        inputs = [
            {
                "question": "How do I create a Databricks cluster with UC access?",
                "context": "Databricks clusters are ...",
            },
        ]

        outputs = [
            "<Instructions for cluster creation with UC enabled>",
        ]

        prompts = [
            "Get Databricks documentation to answer all the questions: {input}",
        ]


        with mlflow.start_run():
            # Log llm predictions
            mlflow.llm.log_predictions(inputs, outputs, prompts)
    """
    if len(inputs) <= 0 or len(inputs) != len(outputs) or len(inputs) != len(prompts):
        raise ValueError(
            "The length of inputs, outputs and prompts must be the same and not empty."
        )

    artifact_path = None
    predictions = []
    run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
    LLM_ARTIFACT_NAME = "llm_predictions.csv"

    for row in zip(inputs, outputs, prompts):
        predictions.append(row)

    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts = [f.path for f in MlflowClient().list_artifacts(run_id)]
        if LLM_ARTIFACT_NAME in artifacts:
            artifact_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=LLM_ARTIFACT_NAME, dst_path=tmpdir
            )
            _logger.info(
                "Appending new inputs to already existing artifact "
                f"{LLM_ARTIFACT_NAME} for run {run_id}."
            )
        else:
            # If the artifact doesn't exist, we need to write the header.
            predictions.insert(0, ["inputs", "outputs", "prompts"])
            artifact_path = os.path.join(tmpdir, LLM_ARTIFACT_NAME)
            _logger.info(f"Creating a new {LLM_ARTIFACT_NAME} for run {run_id}.")

        if os.path.exists(artifact_path):
            with open(artifact_path, newline="") as llm_prediction:
                num_existing_predictions = sum(1 for _ in csv.reader(llm_prediction))
                if num_existing_predictions + len(predictions) > 1000:
                    _logger.warning(
                        f"Trying to log a {LLM_ARTIFACT_NAME} with length "
                        "more than 1000 records. It might slow down performance."
                    )

        with open(artifact_path, "a", encoding="UTF8", newline="") as llm_prediction:
            writer = csv.writer(llm_prediction)
            writer.writerows(predictions)
        mlflow.tracking.fluent.log_artifact(artifact_path)
