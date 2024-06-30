from typing import Any, Dict, List, Tuple

import pandas as pd

from mlflow.entities.evaluation import Evaluation


def evaluations_to_dataframes(
    evaluations: List[Evaluation],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Converts a list of Evaluation objects to four separate DataFrames: one for main evaluation
    data (excluding assessments and metrics), one for metrics, one for assessments, and one for
    tags.

    Args:
        evaluations (List[Evaluation]): List of Evaluation objects.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple of four DataFrames
        containing evaluation data, metrics data, assessments data, and tags data.
    """
    evaluations_data = []
    metrics_data = []
    assessments_data = []
    tags_data = []

    for evaluation in evaluations:
        eval_dict = evaluation.to_dictionary()

        # Extract assessment and metrics
        assessments_list = eval_dict.pop("assessments", [])
        metrics_list = eval_dict.pop("metrics", [])
        tags_list = eval_dict.pop("tags", [])

        evaluations_data.append(eval_dict)

        for metric_dict in metrics_list:
            metric_dict["evaluation_id"] = eval_dict["evaluation_id"]
            # Remove 'step' key if it exists, since it is not valid for evaluation metrics
            metric_dict.pop("step", None)
            metrics_data.append(metric_dict)

        for assess_dict in assessments_list:
            assess_dict["evaluation_id"] = eval_dict["evaluation_id"]
            assessments_data.append(assess_dict)

        for tag_dict in tags_list:
            tag_dict["evaluation_id"] = eval_dict["evaluation_id"]
            tags_data.append(tag_dict)

    evaluations_df = _apply_schema_to_dataframe(
        pd.DataFrame(evaluations_data), _get_evaluation_dataframe_schema()
    )
    metrics_df = (
        _apply_schema_to_dataframe(pd.DataFrame(metrics_data), _get_metrics_dataframe_schema())
        if metrics_data
        else _get_empty_metrics_dataframe()
    )
    assessments_df = (
        _apply_schema_to_dataframe(
            pd.DataFrame(assessments_data), _get_assessments_dataframe_schema()
        )
        if assessments_data
        else _get_empty_assessments_dataframe()
    )
    tags_df = (
        _apply_schema_to_dataframe(pd.DataFrame(tags_data), _get_tags_dataframe_schema())
        if tags_data
        else _get_empty_tags_dataframe()
    )

    return evaluations_df, metrics_df, assessments_df, tags_df


def _get_evaluation_dataframe_schema() -> Dict[str, Any]:
    """
    Returns the schema for the evaluation DataFrame.
    """
    return {
        "evaluation_id": "string",
        "run_id": "string",
        "inputs_id": "string",
        "inputs": "object",
        "outputs": "object",
        "request_id": "object",
        "targets": "object",
        "error_code": "object",
        "error_message": "object",
    }


def _get_assessments_dataframe_schema() -> Dict[str, Any]:
    """
    Returns the schema for the assessments DataFrame.
    """
    return {
        "evaluation_id": "string",
        "name": "string",
        "source": "object",
        "timestamp": "int64",
        "boolean_value": "object",
        "numeric_value": "object",
        "string_value": "object",
        "rationale": "object",
        "metadata": "object",
        "error_code": "object",
        "error_message": "object",
    }


def _get_empty_assessments_dataframe() -> pd.DataFrame:
    """
    Creates an empty DataFrame with columns for evaluation assessments data.
    """
    schema = _get_assessments_dataframe_schema()
    df = pd.DataFrame(columns=schema.keys())
    return _apply_schema_to_dataframe(df, schema)


def _get_metrics_dataframe_schema() -> Dict[str, Any]:
    """
    Returns the schema for the metrics DataFrame.
    """
    return {
        "evaluation_id": "string",
        "key": "string",
        "value": "float64",
        "timestamp": "int64",
    }


def _get_empty_metrics_dataframe() -> pd.DataFrame:
    """
    Creates an empty DataFrame with columns for evaluation metric data.
    """
    schema = _get_metrics_dataframe_schema()
    df = pd.DataFrame(columns=schema.keys())
    return _apply_schema_to_dataframe(df, schema)


def _get_tags_dataframe_schema() -> Dict[str, Any]:
    """
    Returns the schema for the tags DataFrame.
    """
    return {
        "evaluation_id": "string",
        "key": "string",
        "value": "string",
    }


def _get_empty_tags_dataframe() -> pd.DataFrame:
    """
    Creates an empty DataFrame with columns for evaluation tags data.
    """
    schema = _get_tags_dataframe_schema()
    df = pd.DataFrame(columns=schema.keys())
    return _apply_schema_to_dataframe(df, schema)


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
    return df.replace(pd.NA, None)
