import pandas as pd

from mlflow.evaluation.evaluation import EvaluationEntity as EvaluationEntity
from mlflow.evaluation.utils import (
    _get_assessments_dataframe_schema,
    _get_evaluations_dataframe_schema,
    _get_metrics_dataframe_schema,
    _get_tags_dataframe_schema,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST
from mlflow.tracking.client import MlflowClient


def get_evaluation(*, run_id: str, evaluation_id: str) -> EvaluationEntity:
    """
    Retrieves an Evaluation object from an MLflow Run.

    Args:
        run_id (str): ID of the MLflow Run containing the evaluation.
        evaluation_id (str): The ID of the evaluation.

    Returns:
        Evaluation: The Evaluation object.
    """
    client = MlflowClient()
    if not _contains_evaluation_artifacts(client=client, run_id=run_id):
        raise MlflowException(
            "The specified run does not contain any evaluations. "
            "Please log evaluations to the run before retrieving them.",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    evaluations_file = client.download_artifacts(run_id=run_id, path="_evaluations.json")
    evaluations_df = _read_evaluations_dataframe(evaluations_file)

    assessments_file = client.download_artifacts(run_id=run_id, path="_assessments.json")
    assessments_df = _read_assessments_dataframe(assessments_file)

    metrics_file = client.download_artifacts(run_id=run_id, path="_metrics.json")
    metrics_df = _read_metrics_dataframe(metrics_file)

    tags_file = client.download_artifacts(run_id=run_id, path="_tags.json")
    tags_df = _read_tags_dataframe(tags_file)

    return _get_evaluation_from_dataframes(
        run_id=run_id,
        evaluation_id=evaluation_id,
        evaluations_df=evaluations_df,
        metrics_df=metrics_df,
        assessments_df=assessments_df,
        tags_df=tags_df,
    )


def _contains_evaluation_artifacts(*, client: MlflowClient, run_id: str) -> bool:
    return {"_evaluations.json", "_metrics.json", "_assessments.json", "_tags.json"}.issubset(
        {file.path for file in client.list_artifacts(run_id)}
    )


def _read_evaluations_dataframe(path: str) -> pd.DataFrame:
    """
    Reads an evaluations DataFrame from a file.

    Args:
        path (str): Path to the file.

    Returns:
        pd.DataFrame: The evaluations DataFrame.
    """
    schema = _get_evaluations_dataframe_schema()
    return pd.read_json(path, orient="split", dtype=schema, convert_dates=False).replace(
        pd.NA, None
    )


def _read_assessments_dataframe(path: str) -> pd.DataFrame:
    """
    Reads an assessments DataFrame from a file.

    Args:
        path (str): Path to the file.

    Returns:
        pd.DataFrame: The assessments DataFrame.
    """
    schema = _get_assessments_dataframe_schema()
    return pd.read_json(path, orient="split", dtype=schema, convert_dates=False).replace(
        pd.NA, None
    )


def _read_metrics_dataframe(path: str) -> pd.DataFrame:
    """
    Reads a metrics DataFrame from a file.

    Args:
        path (str): Path to the file.

    Returns:
        pd.DataFrame: The metrics DataFrame.
    """
    schema = _get_metrics_dataframe_schema()
    return pd.read_json(path, orient="split", dtype=schema, convert_dates=False).replace(
        pd.NA, None
    )


def _read_tags_dataframe(path: str) -> pd.DataFrame:
    """
    Reads a tags DataFrame from a file.

    Args:
        path (str): Path to the file.

    Returns:
        pd.DataFrame: The tags DataFrame.
    """
    schema = _get_tags_dataframe_schema()
    return pd.read_json(path, orient="split", dtype=schema, convert_dates=False).replace(
        pd.NA, None
    )


def _get_evaluation_from_dataframes(
    *,
    run_id: str,
    evaluation_id: str,
    evaluations_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    assessments_df: pd.DataFrame,
    tags_df: pd.DataFrame,
) -> EvaluationEntity:
    """
    Parses an Evaluation object with the specified evaluation ID from the specified DataFrames.
    """
    evaluation_row = evaluations_df[evaluations_df["evaluation_id"] == evaluation_id]
    if evaluation_row.empty:
        raise MlflowException(
            f"The specified evaluation ID '{evaluation_id}' does not exist in the run '{run_id}'.",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    evaluations: list[EvaluationEntity] = _dataframes_to_evaluations(
        evaluations_df=evaluation_row,
        metrics_df=metrics_df,
        assessments_df=assessments_df,
        tags_df=tags_df,
    )
    if len(evaluations) != 1:
        raise MlflowException(
            f"Expected to find a single evaluation with ID '{evaluation_id}', but found "
            f"{len(evaluations)} evaluations.",
            error_code=INTERNAL_ERROR,
        )

    return evaluations[0]


def _dataframes_to_evaluations(
    evaluations_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    assessments_df: pd.DataFrame,
    tags_df: pd.DataFrame,
) -> list[EvaluationEntity]:
    """
    Converts four separate DataFrames (main evaluation data, metrics, assessments, and tags) back
    into a list of Evaluation entities.

    Args:
        evaluations_df (pd.DataFrame): DataFrame with the main evaluation data
            (excluding assessment and metrics).
        metrics_df (pd.DataFrame): DataFrame with metrics.
        assessments_df (pd.DataFrame): DataFrame with assessments.
        tags_df (pd.DataFrame): DataFrame with tags.

    Returns:
        List[EvaluationEntity]: A list of Evaluation entities created from the DataFrames.
    """
    # Group metrics and assessment by evaluation_id
    metrics_by_eval = _group_dataframe_by_evaluation_id(metrics_df)
    assessments_by_eval = _group_dataframe_by_evaluation_id(assessments_df)
    tags_by_eval = _group_dataframe_by_evaluation_id(tags_df)

    # Convert main DataFrame to list of dictionaries and create Evaluation objects
    evaluations = []
    for eval_dict in evaluations_df.to_dict(orient="records"):
        evaluation_id = eval_dict["evaluation_id"]
        eval_dict["metrics"] = [
            {
                "key": metric["key"],
                "value": metric["value"],
                "timestamp": metric["timestamp"],
                # Evaluation metrics don't have steps, but we're reusing the MLflow Metric
                # class to represent Evaluation metrics as entities in Python for now. Accordingly,
                # we set the step to 0 in order to parse the evaluation metric as an MLflow Metric
                # Python entity
                "step": 0,
                # Also discard the evaluation_id field from the evaluation metric, since this
                # field is not part of the MLflow Metric Python entity
            }
            for metric in metrics_by_eval.get(evaluation_id, [])
        ]
        eval_dict["assessments"] = assessments_by_eval.get(evaluation_id, [])
        eval_dict["tags"] = tags_by_eval.get(evaluation_id, [])
        evaluations.append(EvaluationEntity.from_dictionary(eval_dict))

    return evaluations


def _group_dataframe_by_evaluation_id(df: pd.DataFrame):
    """
    Groups evaluation dataframe rows by 'evaluation_id'.

    Args:
        df (pd.DataFrame): DataFrame to group.

    Returns:
        Dict[str, List]: A dictionary with 'evaluation_id' as keys and lists of entity
            dictionaries as values.
    """
    grouped = df.groupby("evaluation_id", group_keys=False).apply(
        lambda x: x.to_dict(orient="records")
    )
    return grouped.to_dict()
