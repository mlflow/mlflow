import json
import logging
import math
from concurrent.futures import Future, as_completed
from typing import TYPE_CHECKING, Any, Collection

from mlflow.entities import Assessment, Trace
from mlflow.entities.assessment import DEFAULT_FEEDBACK_NAME, Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
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
    from mlflow.genai.evaluation.entities import EvalResult

    try:
        import pyspark.sql.dataframe

        EvaluationDatasetTypes = (
            pd.DataFrame | pyspark.sql.dataframe.DataFrame | list[dict] | EvaluationDataset
        )
    except ImportError:
        EvaluationDatasetTypes = pd.DataFrame | list[dict] | EvaluationDataset


_logger = logging.getLogger(__name__)

USER_DEFINED_ASSESSMENT_NAME_KEY = "_user_defined_assessment_name"
PGBAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed}, Remaining: {remaining}] {postfix}"
)


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


def _convert_to_eval_set(data: "EvaluationDatasetTypes") -> "pd.DataFrame":
    """
    Takes in a dataset in the multiple format that mlflow.genai.evaluate() expects and converts it
    into standardized Pandas DataFrame.
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
        .pipe(_deserialize_trace_column_if_needed)
        .pipe(_extract_request_response_from_trace)
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


def _deserialize_trace_column_if_needed(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Deserialize the `trace` column from the dataframe if it is a string.

    Since MLflow 3.2.0, mlflow.search_traces() returns a pandas DataFrame with a `trace`
    column that is a trace json representation rather than the Trace object itself. This
    function deserializes the `trace` column into a Trace object.
    """
    if "trace" in df.columns:
        df["trace"] = df["trace"].apply(lambda t: Trace.from_json(t) if isinstance(t, str) else t)
    return df


def _extract_request_response_from_trace(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Add `request` and `response`columns from traces if it is not already present.
    """
    if "trace" not in df.columns:
        return df

    if "request" not in df.columns:
        df["request"] = df["trace"].apply(lambda trace: json.loads(trace.data.request))
    if "response" not in df.columns:
        df["response"] = df["trace"].apply(lambda trace: json.loads(trace.data.response))
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

    from mlflow.genai.scorers.builtin_scorers import BuiltInScorer
    from mlflow.types.llm import ChatCompletionRequest

    def eval_fn(
        request_id: str,
        request: ChatCompletionRequest | str,
        response: Any | None,
        expected_response: Any | None,
        trace: Trace | None,
        guidelines: list[str] | dict[str, list[str]] | None,
        expected_facts: list[str] | None,
        expected_retrieved_context: list[dict[str, str]] | None,
        custom_expected: dict[str, Any] | None,
        **kwargs,
    ) -> int | float | bool | str | Assessment | list[Assessment]:
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

    metric_instance = metric(
        eval_fn=eval_fn,
        name=scorer.name,
        aggregations=scorer.aggregations,
    )
    # Add attribute to indicate if this is a built-in scorer
    metric_instance._is_builtin_scorer = isinstance(scorer, BuiltInScorer)

    return metric_instance


def standardize_scorer_value(scorer_name: str, value: Any) -> list[Feedback]:
    """
    Convert the scorer return value to a list of MLflow Assessment (Feedback) objects.

    Scorer can return:
    - A number, boolean, or string, a list of them.
    - An Feedback object
    - A list of Feedback objects

    All of the above will be converted to a list of Feedback objects.
    """
    # None is a valid metric value, return an empty list
    if value is None:
        return []

    # Primitives are valid metric values
    if isinstance(value, (int, float, bool, str)):
        return [
            Feedback(
                name=scorer_name,
                source=make_code_type_assessment_source(scorer_name),
                value=value,
            )
        ]

    if isinstance(value, Feedback):
        value.name = _get_custom_assessment_name(value, scorer_name)
        return [value]

    if isinstance(value, Collection):
        assessments = []
        for item in value:
            if isinstance(item, Feedback):
                # Scorer returns multiple assessments as a list.
                item.name = _get_custom_assessment_name(item, scorer_name)
                assessments.append(item)
            else:
                # If the item is not assessment, the list represents a single assessment
                # value of list type. Convert it to a Feedback object.
                assessments.append(
                    Feedback(
                        name=scorer_name,
                        source=make_code_type_assessment_source(scorer_name),
                        value=item,
                    )
                )
        return assessments

    raise MlflowException.invalid_parameter_value(
        f"Got unsupported result from scorer '{scorer_name}'. "
        f"Expected the metric value to be a number, or a boolean, or a string, "
        "or an Feedback, or a list of Feedbacks. "
        f"Got {value}.",
    )


def _get_custom_assessment_name(assessment: Feedback, scorer_name: str) -> str:
    """Get the name of the custom assessment. Use assessment name if present and not a builtin judge
    name, otherwise use the scorer name.

    Args:
        assessment: The assessment to get the name for.
        scorer_name: The name of the scprer.
    """
    # If the user didn't provide a name, use the scorer name
    if assessment.name == DEFAULT_FEEDBACK_NAME or (
        assessment.metadata is not None
        and assessment.metadata.get(USER_DEFINED_ASSESSMENT_NAME_KEY) == "false"
    ):
        return scorer_name
    return assessment.name


def make_code_type_assessment_source(scorer_name: str) -> AssessmentSource:
    return AssessmentSource(source_type=AssessmentSourceType.CODE, source_id=scorer_name)


def is_none_or_nan(value: Any) -> bool:
    """
    Checks whether a value is None or NaN.

    NB: This function does not handle pandas.NA.
    """
    # isinstance(value, float) check is needed to ensure that math.isnan is not called on an array.
    return value is None or (isinstance(value, float) and math.isnan(value))


def complete_eval_futures_with_progress_base(futures: list[Future]) -> list["EvalResult"]:
    """Wraps the as_completed function with a progress bar."""
    futures_as_completed = as_completed(futures)

    try:
        from tqdm.auto import tqdm

        futures_as_completed = tqdm(
            futures_as_completed,
            total=len(futures),
            disable=False,
            desc="Evaluating",
            smoothing=0,  # 0 means using average speed for remaining time estimates
            bar_format=PGBAR_FORMAT,
        )
    except ImportError:
        # If tqdm is not installed, we don't show a progress bar
        pass

    return [future.result() for future in futures_as_completed]
