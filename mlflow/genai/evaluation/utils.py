from typing import TYPE_CHECKING, Any, Optional, Union

from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.entities import Trace
from mlflow.evaluation import Assessment
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


# TODO: ML-52299
def _convert_to_legacy_eval_set(data: "EvaluationDatasetTypes") -> dict:
    """
    Takes in a dataset in the format that mlflow.genai.evaluate() expects and converts it into
    to the current eval-set schema that Agent Evaluation takes in. The transformed schema should
    be accepted by mlflow.evaluate().
    """
    column_mapping = {
        "inputs": "request",
        "outputs": "response",
        "expectations": "expected_response",
        "trace": "trace",
    }

    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, spark.DataFrame):
        df = data.toPandas()
    else:  # data is already a pandas DataFrame
        df = data.copy()

    # Rename columns according to the mapping
    renamed_df = pd.DataFrame()
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            renamed_df[new_col] = df[old_col]

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
