import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from mlflow.entities.assessment import Assessment
from mlflow.entities.evaluation import Evaluation
from mlflow.entities.metric import Metric
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


def evaluations_to_dataframes(
    evaluations: List[Evaluation],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Converts a list of Evaluation objects to three separate DataFrames: one for main evaluation
    data (excluding assessments and metrics), one for metrics, and one for assessments.

    Args:
        evaluations (List[Evaluation]): List of Evaluation objects.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing three DataFrames:
            1. DataFrame with the main evaluation data (excluding assessments and metrics).
            2. DataFrame with metrics.
            3. DataFrame with assessments.
    """
    evaluations_data = []
    metrics_data = []
    assessments_data = []

    for evaluation in evaluations:
        eval_dict = evaluation.to_dictionary()

        # Extract assessment and metrics
        assessments_list = eval_dict.pop("assessments", [])
        metrics_list = eval_dict.pop("metrics", [])

        evaluations_data.append(eval_dict)

        for metric_dict in metrics_list:
            metric_dict["evaluation_id"] = eval_dict["evaluation_id"]
            # Remove 'step' key if it exists, since it is not valid for evaluation metrics
            metric_dict.pop("step", None)
            metrics_data.append(metric_dict)

        for assess_dict in assessments_list:
            assess_dict["evaluation_id"] = eval_dict["evaluation_id"]
            assessments_data.append(assess_dict)

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

    return evaluations_df, metrics_df, assessments_df


def dataframes_to_evaluations(
    evaluations_df: pd.DataFrame, metrics_df: pd.DataFrame, assessments_df: pd.DataFrame
) -> List[Evaluation]:
    """
    Converts three separate DataFrames (main evaluation data, metrics, and assessment) back to a
    list of Evaluation objects.

    Args:
        evaluations_df (pd.DataFrame): DataFrame with the main evaluation data
            (excluding assessment and metrics).
        metrics_df (pd.DataFrame): DataFrame with metrics.
        assessments_df (pd.DataFrame): DataFrame with assessments.

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
        grouped = df.groupby("evaluation_id").apply(lambda x: x.to_dict(orient="records"))
        return grouped.to_dict()

    # Group metrics and assessment by evaluation_id
    metrics_by_eval = group_by_evaluation_id(metrics_df, Metric)
    assessments_by_eval = group_by_evaluation_id(assessments_df, Assessment)

    # Convert main DataFrame to list of dictionaries and create Evaluation objects
    evaluations = []
    for eval_dict in evaluations_df.to_dict(orient="records"):
        evaluation_id = eval_dict["evaluation_id"]
        eval_dict["metrics"] = metrics_by_eval.get(evaluation_id, [])
        eval_dict["assessments"] = assessments_by_eval.get(evaluation_id, [])
        evaluations.append(Evaluation.from_dictionary(eval_dict))

    return evaluations


def verify_assessments_have_same_value_type(assessments: Optional[List[Assessment]]):
    """
    Verifies that all assessments with the same name have the same value type.
    """
    if assessments is None:
        return

    assessment_value_types_by_key = defaultdict(list)

    for assessment in assessments:
        assessment_value_types_by_key[assessment.name].append(assessment.get_value_type())

    for assessment_name, value_types in assessment_value_types_by_key.items():
        if len(set(value_types)) > 1:
            raise MlflowException(
                f"Assessments with name '{assessment_name}' have different value"
                f" types: {set(value_types)}",
                error_code=INVALID_PARAMETER_VALUE,
            )


def read_evaluations_dataframe(path: str) -> pd.DataFrame:
    """
    Reads an evaluations DataFrame from a file.

    Args:
        path (str): Path to the file.

    Returns:
        pd.DataFrame: The evaluations DataFrame.
    """
    schema = _get_evaluation_dataframe_schema()
    return pd.read_json(path, orient="split", dtype=schema, convert_dates=False).replace(
        pd.NA, None
    )


def read_assessments_dataframe(path: str) -> pd.DataFrame:
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


def append_to_assessments_dataframe(
    assessments_df: pd.DataFrame, assessments: List[Assessment]
) -> pd.DataFrame:
    """
    Appends assessments to an assessments DataFrame.

    Args:
        assessments_df (pd.DataFrame): Assessments DataFrame to append assessments to.
    """
    new_assessments_data = [assess.to_dictionary() for assess in assessments]
    new_assessments_df = pd.DataFrame(new_assessments_data)
    return pd.concat([assessments_df, new_assessments_df])


def read_metrics_dataframe(path: str) -> pd.DataFrame:
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


def read_tags_dataframe(path: str) -> pd.DataFrame:
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


@dataclass
class _BaseStats:
    """
    Base class for statistics.
    """

    assessment_name: str
    assessment_source: str

    def _get_metric(self, stat_name) -> Metric:
        """
        Get metrics for the statistics.
        """
        key = f"{self.assessment_name}_{stat_name}_{self.assessment_source}"
        return Metric(
            key=key, value=getattr(self, stat_name), timestamp=int(time.time() * 1000), step=0
        )

    def to_metrics(self) -> List[Metric]:
        """
        Get loggable metrics for the statistics.
        """
        raise NotImplementedError


@dataclass
class BooleanAssessmentStats(_BaseStats):
    """
    Statistics for boolean assessments.
    """

    true_count: int
    false_count: int
    ratio: float

    def to_metrics(self) -> List[Metric]:
        return [
            self._get_metric("true_count"),
            self._get_metric("false_count"),
            self._get_metric("ratio"),
        ]


@dataclass
class NumericAssessmentStats(_BaseStats):
    """
    Statistics for numeric assessments.
    """

    min: float
    mean: float
    p50: float
    p90: float
    max: float

    def to_metrics(self) -> List[Metric]:
        return [
            self._get_metric("min"),
            self._get_metric("mean"),
            self._get_metric("p50"),
            self._get_metric("p90"),
            self._get_metric("max"),
        ]


@dataclass
class StringAssessmentStats:
    """
    Statistics for string assessments.
    """

    assessment_name: str
    assessment_source: str
    distinct_values_count: int

    def to_metrics(self) -> List[Metric]:
        return []


def compute_assessment_stats_by_source(
    assessments_df: pd.DataFrame, assessment_name: str
) -> Dict[str, Union[BooleanAssessmentStats, NumericAssessmentStats, StringAssessmentStats]]:
    """
    Computes statistics for a given assessment by source.

    Args:
        assessments_df (pd.DataFrame): DataFrame with assessments.
        assessment_name (str): Name of the assessment.

    Returns:
        Dict[str, Union[BooleanAssessmentStats, NumericAssessmentStats, StringAssessmentStats]]:
            A dictionary with statistics for the assessment by source name.
    """
    matching_assessments_df = assessments_df[assessments_df["name"] == assessment_name]
    if matching_assessments_df.empty:
        raise ValueError(f"No assessments found for '{assessment_name}'.")

    matching_assessments = [
        Assessment.from_dictionary(assess)
        for assess in matching_assessments_df.to_dict(orient="records")
    ]

    def compute_stats(assessment_name: str, assessment_source: str, assessments: List[Assessment]):
        assessments = [
            assessment for assessment in assessments if assessment.get_value_type() is not None
        ]
        if not assessments:
            return

        if assessments[0].get_value_type() == "boolean":
            return compute_boolean_assessment_stats(assessment_name, assessment_source, assessments)
        elif assessments[0].get_value_type() == "numeric":
            return compute_numeric_assessment_stats(assessment_name, assessment_source, assessments)
        elif assessments[0].get_value_type() == "string":
            return compute_string_assessment_stats(assessment_name, assessment_source, assessments)
        else:
            raise ValueError(
                f"Unsupported assessment value type: {assessments[0].get_value_type()}."
            )

    matching_assessments_by_source = defaultdict(list)
    for assessment in matching_assessments:
        matching_assessments_by_source[assessment.source.source_id].append(assessment)

    assessment_stats_by_source = {}
    for source, assessments in matching_assessments_by_source.items():
        stats_or_none = compute_stats(assessment_name, source, assessments)
        if stats_or_none is not None:
            assessment_stats_by_source[source] = stats_or_none
    return assessment_stats_by_source


def compute_boolean_assessment_stats(
    assessment_name: str, assessment_source: str, assessments: List[Assessment]
) -> BooleanAssessmentStats:
    """
    Computes statistics for boolean assessments.

    Args:
        assessment_name (str): Name of the assessment.
        assessment_source (str): Source of the assessment.
        assessments (List[Assessment]): List of boolean assessments.
    """
    true_count = sum(assess.boolean_value for assess in assessments)
    false_count = len(assessments) - true_count
    ratio = float(true_count) / len(assessments)
    return BooleanAssessmentStats(
        assessment_name=assessment_name,
        assessment_source=assessment_source,
        true_count=true_count,
        false_count=false_count,
        ratio=ratio,
    )


def compute_numeric_assessment_stats(
    assessment_name: str, assessment_source: str, assessments: List[Assessment]
) -> NumericAssessmentStats:
    """
    Computes statistics for numeric assessments.

    Args:
        assessment_name (str): Name of the assessment.
        assessment_source (str): Source of the assessment.
        assessments (List[Assessment]): List of numeric assessments.
    """
    min_value = min(assess.numeric_value for assess in assessments)
    mean = np.mean([assess.numeric_value for assess in assessments])
    p50 = np.percentile([assess.numeric_value for assess in assessments], 50)
    p90 = np.percentile([assess.numeric_value for assess in assessments], 90)
    max_value = max(assess.numeric_value for assess in assessments)
    return NumericAssessmentStats(
        assessment_name=assessment_name,
        assessment_source=assessment_source,
        min=min_value,
        mean=mean,
        p50=p50,
        p90=p90,
        max=max_value,
    )


def compute_string_assessment_stats(
    assessment_name: str, assessment_source: str, assessments: List[Assessment]
) -> StringAssessmentStats:
    """
    Computes statistics for string assessments.

    Args:
        assessment_name (str): Name of the assessment.
        assessment_source (str): Source of the assessment.
        assessments (List[Assessment]): List of string assessments.
    """
    distinct_values_count = len({assess.string_value for assess in assessments})
    return StringAssessmentStats(
        assessment_name=assessment_name,
        assessment_source=assessment_source,
        distinct_values_count=distinct_values_count,
    )
