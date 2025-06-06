import json
import logging
from typing import TYPE_CHECKING, Any, Optional, Union

from mlflow.entities import Assessment, Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.constant import (
    AgentEvaluationReserverKey,
)
from mlflow.genai.scorers import Scorer
from mlflow.models import EvaluationMetric

try:
    # `pandas` is not required for `mlflow-skinny`.
    import pandas as pd
except ImportError:
    pass

if TYPE_CHECKING:
    from mlflow.genai.datasets import EvaluationDataset

    try:
        import pyspark.sql.dataframe

        EvaluationDatasetTypes = Union[
            pd.DataFrame, pyspark.sql.dataframe.DataFrame, list[dict], EvaluationDataset
        ]
    except ImportError:
        EvaluationDatasetTypes = Union[pd.DataFrame, list[dict], EvaluationDataset]


_logger = logging.getLogger(__name__)


def _convert_eval_set_to_df(data: "EvaluationDatasetTypes") -> "pd.DataFrame":
    """
    Takes in a dataset in the format that `mlflow.genai.evaluate()` expects and
    converts it into a pandas DataFrame.
    """
    if isinstance(data, list):
        # validate that every item in the list is a dict and has inputs as key
        for item in data:
            if not isinstance(item, dict):
                raise MlflowException.invalid_parameter_value(
                    "Every item in the list must be a dictionary."
                )
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        # Data is already a pd DataFrame, just copy it
        df = data.copy()
    else:
        try:
            from mlflow.utils.spark_utils import get_spark_dataframe_type

            if isinstance(data, get_spark_dataframe_type()):
                df = _deserialize_inputs_and_expectations_column(data.toPandas())
            else:
                raise MlflowException.invalid_parameter_value(
                    "Invalid type for parameter `data`. Expected a list of dictionaries, "
                    f"a pandas DataFrame, or a Spark DataFrame. Got: {type(data)}"
                )
        except ImportError:
            raise ImportError(
                "The `pyspark` package is required to use mlflow.genai.evaluate() "
                "Please install it with `pip install pyspark`."
            )

    if len(df) == 0:
        raise MlflowException.invalid_parameter_value(
            "The dataset is empty. Please provide a non-empty dataset."
        )

    if not any(col in df.columns for col in ("trace", "inputs")):
        raise MlflowException.invalid_parameter_value(
            "Either `inputs` or `trace` column is required in the dataset. Please provide inputs "
            "for every datapoint or provide a trace."
        )

    return df


def _convert_to_legacy_eval_set(data: "EvaluationDatasetTypes") -> "pd.DataFrame":
    """
    Takes in a dataset in the format that mlflow.genai.evaluate() expects and converts it into
    to the current eval-set schema that Agent Evaluation takes in. The transformed schema should
    be accepted by mlflow.evaluate().
    The expected schema can be found at:
    https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/evaluation-schema

    NB: The harness secretly support 'expectations' column as well. It accepts a dictionary of
        expectations, which is same as the schema that mlflow.genai.evaluate() expects.
        Therefore, we can simply pass through expectations column.
    """
    column_mapping = {
        "inputs": "request",
        "outputs": "response",
    }

    df = _convert_eval_set_to_df(data)

    return (
        df.rename(columns=column_mapping)
        .pipe(_extract_request_from_trace)
        .pipe(_extract_expectations_from_trace)
    )


def _deserialize_inputs_and_expectations_column(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Deserialize the `inputs` and `expectations` string columns from the dataframe.

    When managed datasets are read as Spark DataFrames, the `inputs` and `expectations` columns
    are loaded as string columns of JSON strings. This function deserializes these columns into
    dictionaries expected by mlflow.genai.evaluate().
    """
    target_columns = ["inputs", "expectations"]
    for col in target_columns:
        if col not in df.columns or not isinstance(df[col][0], str):
            continue

        try:
            df[col] = df[col].apply(json.loads)
        except json.JSONDecodeError as e:
            if col == "inputs":
                msg = (
                    "The `inputs` column must be a valid JSON string of field names and values. "
                    "For example, `{'question': 'What is the capital of France?'}`"
                )
            else:
                msg = (
                    "The `expectations` column must be a valid JSON string of assessment names and "
                    "values. For example, `{'expected_facts': ['fact1', 'fact2']}`"
                )
            raise MlflowException.invalid_parameter_value(
                f"Failed to parse `{col}` column. Error: {e}\nHint: {msg}"
            )

    return df


def _extract_request_from_trace(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Add `request` columns to the dataframe if it is not already present.
    This is for compatibility with mlflow.evaluate() that requires `request` column.
    """
    if "trace" not in df.columns:
        return df

    if "request" not in df.columns:
        df["request"] = df["trace"].apply(lambda trace: json.loads(trace.data.request))
    return df


def _extract_expectations_from_trace(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Add `expectations` columns to the dataframe from assessments
    stored in the traces, if the "expectations" column is not already present.
    """
    if "trace" not in df.columns:
        return df

    expectations_column = []
    for trace in df["trace"]:
        expectations = {}
        for assessment in trace.info.assessments or []:
            if assessment.expectation is not None:
                expectations[assessment.name] = assessment.expectation.value
        expectations_column.append(expectations)
    # If no trace has assessments, not add the column
    if all(len(expectations) == 0 for expectations in expectations_column):
        return df

    df["expectations"] = expectations_column
    return df


def _convert_scorer_to_legacy_metric(scorer: Scorer) -> EvaluationMetric:
    """
    Takes in a Scorer object and converts it into a legacy MLflow 2.x
    Metric object.
    """
    try:
        from databricks.agents.evals import metric
    except ImportError:
        raise ImportError(
            "The `databricks-agents` package is required to use mlflow.genai.evaluate() "
            "Please install it with `pip install databricks-agents`."
        )

    from mlflow.types.llm import ChatCompletionRequest

    def eval_fn(
        request_id: str,
        request: Union[ChatCompletionRequest, str],
        response: Optional[Any],
        expected_response: Optional[Any],
        trace: Optional[Trace],
        guidelines: Optional[Union[list[str], dict[str, list[str]]]],
        expected_facts: Optional[list[str]],
        expected_retrieved_context: Optional[list[dict[str, str]]],
        custom_expected: Optional[dict[str, Any]],
        **kwargs,
    ) -> Union[int, float, bool, str, Assessment, list[Assessment]]:
        # Condense all expectations into a single dict
        expectations = {}
        if expected_response is not None:
            expectations[AgentEvaluationReserverKey.EXPECTED_RESPONSE] = expected_response
        if expected_facts is not None:
            expectations[AgentEvaluationReserverKey.EXPECTED_FACTS] = expected_facts
        if expected_retrieved_context is not None:
            expectations[AgentEvaluationReserverKey.EXPECTED_RETRIEVED_CONTEXT] = (
                expected_retrieved_context
            )
        if guidelines is not None:
            expectations[AgentEvaluationReserverKey.GUIDELINES] = guidelines
        if custom_expected is not None:
            expectations.update(custom_expected)

        merged = {
            "inputs": request,
            "outputs": response,
            "expectations": expectations,
            "trace": trace,
        }
        return scorer.run(**merged)

    return metric(
        eval_fn=eval_fn,
        name=scorer.name,
    )
