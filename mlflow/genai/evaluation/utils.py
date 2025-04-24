from typing import TYPE_CHECKING, Any, Optional, Union

from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.entities import Trace
from mlflow.evaluation import Assessment
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import Scorer
from mlflow.models import EvaluationMetric
from mlflow.types.llm import ChatCompletionRequest

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


def _convert_to_legacy_eval_set(data: "EvaluationDatasetTypes") -> dict:
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
        "trace": "trace",
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

        df = pd.DataFrame(data) if 1 < len(data) else pd.DataFrame(data[0])
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

    renamed_df = df.filter(items=column_mapping.keys()).rename(columns=column_mapping)

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

    def eval_fn(
        request_id: str,
        request: Union[ChatCompletionRequest, str],
        response: Optional[Any],
        expected_response: Optional[Any],
        trace: Optional[Trace],
        **kwargs,
    ) -> Union[int, float, bool, str, Assessment, list[Assessment]]:
        # TODO: scorer.aggregations require a refactor on the agents side
        return scorer(
            inputs=request,
            outputs=response,
            expectations=expected_response,
            trace=trace,
        )

    return metric(
        eval_fn=eval_fn,
        name=scorer.name,
    )
