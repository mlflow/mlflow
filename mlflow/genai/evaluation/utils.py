import inspect
from typing import TYPE_CHECKING, Any, Optional, Union

from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.entities import Assessment, Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import Scorer
from mlflow.models import EvaluationMetric

try:
    # `pandas` is not required for `mlflow-skinny`.
    import pandas as pd
except ImportError:
    pass

if TYPE_CHECKING:
    try:
        import pyspark.sql.dataframe

        EvaluationDatasetTypes = Union[
            pd.DataFrame, pyspark.sql.dataframe.DataFrame, list[dict], EvaluationDataset
        ]
    except ImportError:
        EvaluationDatasetTypes = Union[pd.DataFrame, list[dict], EvaluationDataset]


def _convert_to_legacy_eval_set(data: "EvaluationDatasetTypes") -> "pd.DataFrame":
    """
    Takes in a dataset in the format that mlflow.genai.evaluate() expects and converts it into
    to the current eval-set schema that Agent Evaluation takes in. The transformed schema should
    be accepted by mlflow.evaluate().
    The expected schema can be found at:
    https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/evaluation-schema
    """
    column_mapping = {
        "inputs": "request",
        "outputs": "response",
    }

    if isinstance(data, list):
        # validate that every item in the list is a dict and has inputs as key
        for item in data:
            if not isinstance(item, dict):
                raise MlflowException.invalid_parameter_value(
                    "Every item in the list must be a dictionary."
                )
            if "inputs" not in item:
                raise MlflowException.invalid_parameter_value(
                    "Every item in the list must have an 'inputs' key."
                )

        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        # Data is already a pd DataFrame, just copy it
        df = data.copy()
    else:
        try:
            import pyspark.sql.dataframe

            if isinstance(data, pyspark.sql.dataframe.DataFrame):
                df = data.toPandas()
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

    renamed_df = df.rename(columns=column_mapping)

    # expand out expectations into separate columns
    if "expectations" in df.columns:
        for field in ["expected_response", "expected_retrieved_context", "expected_facts"]:
            renamed_df[field] = None

        # Process each row individually to handle mixed types
        for idx, value in df["expectations"].items():
            if isinstance(value, dict):
                for field in ["expected_response", "expected_retrieved_context", "expected_facts"]:
                    if field in value:
                        renamed_df.at[idx, field] = value[field]
            # Non-dictionary values go to expected_response
            elif value is not None:
                renamed_df.at[idx, "expected_response"] = value

        renamed_df.drop(columns=["expectations"], inplace=True)

    return renamed_df


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
        retrieved_context: Optional[list[dict[str, str]]],
        guidelines: Optional[Union[list[str], dict[str, list[str]]]],
        expected_facts: Optional[list[str]],
        expected_retrieved_context: Optional[list[dict[str, str]]],
        custom_expected: Optional[dict[str, Any]],
        custom_inputs: Optional[dict[str, Any]],
        custom_outputs: Optional[dict[str, Any]],
        tool_calls: Optional[list[Any]],
        **kwargs,
    ) -> Union[int, float, bool, str, Assessment, list[Assessment]]:
        # TODO: scorer.aggregations require a refactor on the agents side
        merged = {
            "inputs": request,
            "outputs": response,
            "expectations": expected_response,
            "trace": trace,
            "guidelines": guidelines,
            "retrieved_context": retrieved_context,
            "expected_facts": expected_facts,
            "expected_retrieved_context": expected_retrieved_context,
            "custom_expected": custom_expected,
            "custom_inputs": custom_inputs,
            "custom_outputs": custom_outputs,
            "tool_calls": tool_calls,
            **kwargs,
        }
        # Filter to only the parameters the scorer actually expects
        sig = inspect.signature(scorer)
        filtered = {k: v for k, v in merged.items() if k in sig.parameters}
        return scorer(**filtered)

    return metric(
        eval_fn=eval_fn,
        name=scorer.name,
    )
