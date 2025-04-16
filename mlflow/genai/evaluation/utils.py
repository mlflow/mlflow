from typing import Any, Optional, Union

from databricks.agents.evals import metric
from pyspark import sql as spark

from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.entities import Trace
from mlflow.evaluation import Assessment
from mlflow.genai.scorers import Scorer
from mlflow.metrics.base import MetricValue
from mlflow.metrics.genai.genai_metric import _get_aggregate_results
from mlflow.models import EvaluationMetric
from mlflow.types.llm import ChatCompletionRequest

try:
    # `pandas` is not required for `mlflow-skinny`.
    import pandas as pd
except ImportError:
    pass


# TODO: ML-52299
def _convert_to_legacy_eval_set(
    data: Union[pd.DataFrame, spark.DataFrame, list[dict], EvaluationDataset],
) -> dict:
    """
    Takes in different types of inputs and converts it into to the current eval-set schema that
    Agent Evaluation takes in. The transformed schema should be accepted by mlflow.evaluate()
    """
    raise NotImplementedError("This function is not implemented yet.")


def _convert_scorer_to_legacy_metric(scorer: Scorer) -> EvaluationMetric:
    """
    Takes in a Scorer object and converts it into a legacy MLflow 2.x
    Metric object.
    """

    def eval_fn(
        request_id: str,
        request: Union[ChatCompletionRequest, str],
        response: Optional[Any],
        expected_response: Optional[Any],
        trace: Optional[Trace],
        **kwargs,
    ) -> MetricValue:
        # TODO: predictions and targets are out of order
        scores = scorer(
            inputs=request,
            outputs=response,
            expectations=expected_response,
            trace=trace,
        )

        scores = scores if isinstance(scores, list) else [scores]

        processed_scores = []
        for score in scores:
            if isinstance(score, Assessment):
                processed_scores.append(score.value)
            else:
                processed_scores.append(score)

        try:
            aggregated_results = _get_aggregate_results(scores, scorer.aggregations)
        except Exception as e:
            raise ValueError(
                f"Failed to aggregate results for scorer `{scorer.name}` with ",
                f"aggregations {scorer.aggregations}. Please check that the outputs ",
                "of the scorer are compatible with the aggregations.",
            ) from e

        return MetricValue(
            scores=processed_scores,
            aggregate_results=aggregated_results,
        )

    return metric(
        eval_fn=eval_fn,
        name=scorer.name,
    )
