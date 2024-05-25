from typing import Any, Dict, List, Tuple

import pandas as pd

from mlflow.entities.evaluation import Evaluation
from mlflow.entities.feedback import Feedback
from mlflow.entities.metric import Metric


def evaluations_to_dataframes(
    evaluations: List[Evaluation],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Converts a list of Evaluation objects to three separate DataFrames: one for main evaluation
    data (excluding feedback and metrics), one for metrics, and one for feedback.

    Args:
        evaluations (List[Evaluation]): List of Evaluation objects.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing three DataFrames:
            1. DataFrame with the main evaluation data (excluding feedback and metrics).
            2. DataFrame with metrics.
            3. DataFrame with feedback.
    """
    evaluations_data = []
    metrics_data = []
    feedback_data = []

    for evaluation in evaluations:
        eval_dict = evaluation.to_dictionary()

        # Extract feedback and metrics
        feedback_list = eval_dict.pop("feedback", [])
        metrics_list = eval_dict.pop("metrics", [])

        evaluations_data.append(eval_dict)

        for metric_dict in metrics_list:
            metric_dict["evaluation_id"] = eval_dict["evaluation_id"]
            # Remove 'step' key if it exists, since it is not valid for evaluation metrics
            metric_dict.pop("step", None)
            metrics_data.append(metric_dict)

        for fb_dict in feedback_list:
            fb_dict["evaluation_id"] = eval_dict["evaluation_id"]
            feedback_data.append(fb_dict)

    evaluations_df = _apply_schema_to_dataframe(
        pd.DataFrame(evaluations_data), _get_evaluation_dataframe_schema()
    )
    metrics_df = (
        _apply_schema_to_dataframe(pd.DataFrame(metrics_data), _get_metrics_dataframe_schema())
        if metrics_data
        else _get_empty_metrics_dataframe()
    )
    feedback_df = (
        _apply_schema_to_dataframe(pd.DataFrame(feedback_data), _get_feedback_dataframe_schema())
        if feedback_data
        else _get_empty_feedback_dataframe()
    )

    return evaluations_df, metrics_df, feedback_df


def dataframes_to_evaluations(
    evaluations_df: pd.DataFrame, metrics_df: pd.DataFrame, feedback_df: pd.DataFrame
) -> List[Evaluation]:
    """
    Converts three separate DataFrames (main evaluation data, metrics, and feedback) back to a
    list of Evaluation objects.

    Args:
        evaluations_df (pd.DataFrame): DataFrame with the main evaluation data
            (excluding feedback and metrics).
        metrics_df (pd.DataFrame): DataFrame with metrics.
        feedback_df (pd.DataFrame): DataFrame with feedback.

    Returns:
        List[Evaluation]: A list of Evaluation objects created from the DataFrames.
    """

    def group_by_evaluation_id(df: pd.DataFrame, entity_cls):
        """
        Groups rows by 'evaluation_id' and converts them into entities using the provided class.

        Args:
            df (pd.DataFrame): DataFrame to group.
            entity_cls: Class used to create entities from dictionary rows.

        Returns:
            Dict[str, List]: A dictionary with 'evaluation_id' as keys and lists of entity
                instances as values.
        """
        grouped = df.groupby("evaluation_id").apply(
            lambda x: [entity_cls.from_dictionary(row) for row in x.to_dict(orient="records")]
        )
        return grouped.to_dict()

    # Group metrics and feedback by evaluation_id
    metrics_by_eval = group_by_evaluation_id(metrics_df, Metric)
    feedback_by_eval = group_by_evaluation_id(feedback_df, Feedback)

    # Convert main DataFrame to list of dictionaries and create Evaluation objects
    evaluations = []
    for eval_dict in evaluations_df.to_dict(orient="records"):
        evaluation_id = eval_dict["evaluation_id"]
        eval_dict["metrics"] = metrics_by_eval.get(evaluation_id, [])
        eval_dict["feedback"] = feedback_by_eval.get(evaluation_id, [])
        evaluations.append(Evaluation.from_dictionary(eval_dict))

    return evaluations


def read_evaluations_dataframe(path: str) -> pd.DataFrame:
    """
    Reads an evaluations DataFrame from a file.

    Args:
        path (str): Path to the file.

    Returns:
        pd.DataFrame: The evaluations DataFrame.
    """
    schema = _get_evaluation_dataframe_schema()
    return pd.read_json(path, orient="split", dtype=schema)


def read_feedback_dataframe(path: str) -> pd.DataFrame:
    """
    Reads a feedback DataFrame from a file.

    Args:
        path (str): Path to the file.

    Returns:
        pd.DataFrame: The feedback DataFrame.
    """
    schema = _get_feedback_dataframe_schema()
    return pd.read_json(path, orient="split", dtype=schema)


def read_metrics_dataframe(path: str) -> pd.DataFrame:
    """
    Reads a metrics DataFrame from a file.

    Args:
        path (str): Path to the file.

    Returns:
        pd.DataFrame: The metrics DataFrame.
    """
    schema = _get_metrics_dataframe_schema()
    return pd.read_json(path, orient="split", dtype=schema)


def _get_evaluation_dataframe_schema() -> Dict[str, Any]:
    """
    Returns the schema for the feedback DataFrame.
    """
    return {
        "evaluation_id": "string",
        "run_id": "string",
        "inputs_id": "string",
        "inputs": "object",
        "outputs": "object",
        "request_id": "object",
        "ground_truths": "object",
    }


def _apply_schema_to_dataframe(df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies a schema to a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to apply the schema to.
        schema (Dict[str, Any]): Schema to apply.

    Returns:
        pd.DataFrame: DataFrame with schema applied.
    """
    for column in df.columns:
        df[column] = df[column].astype(schema[column])
    return df


def _get_feedback_dataframe_schema() -> Dict[str, Any]:
    """
    Returns the schema for the feedback DataFrame.
    """
    return {
        "evaluation_id": "string",
        "name": "string",
        "source": "object",
        "timestamp": "int",
        "boolean_value": "object",
        "numeric_value": "object",
        "string_value": "object",
        "rationale": "str",
        "metadata": "object",
    }


def _get_empty_feedback_dataframe() -> pd.DataFrame:
    """
    Creates an empty DataFrame with columns for evaluation feedback data.
    """
    schema = _get_feedback_dataframe_schema()
    df = pd.DataFrame(columns=schema.keys())
    return _apply_schema_to_dataframe(df, schema)


def _get_metrics_dataframe_schema() -> Dict[str, Any]:
    """
    Returns the schema for the metrics DataFrame.
    """
    return {
        "evaluation_id": "string",
        "key": "string",
        "value": "float",
        "timestamp": "int",
    }


def _get_empty_metrics_dataframe() -> pd.DataFrame:
    """
    Creates an empty DataFrame with columns for evaluation metric data.
    """
    schema = _get_metrics_dataframe_schema()
    df = pd.DataFrame(columns=schema.keys())
    return _apply_schema_to_dataframe(df, schema)
