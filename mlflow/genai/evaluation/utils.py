from databricks.agents.evals import metric
from typing import Union

from pyspark import sql as spark

from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.evaluation import Assessment
from mlflow.genai.scorers import Scorer
from mlflow.metrics.base import MetricValue
from mlflow.metrics.genai.genai_metric import _get_aggregate_results
from mlflow.models import EvaluationMetric, make_metric

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
    # TODO: complete implementation
    metric_metadata = {"assessment_type": "ANSWER"}
    genai_metric_args = {
        "name": scorer.name,
        "aggregations": scorer.aggregations,
        "greater_is_better": scorer.greater_is_better,
        "metric_metadata": metric_metadata,
    }

    def eval_fn(
        inputs,
        predictions,
        targets,
        *args,
        **kwargs,
    ) -> MetricValue:
        # TODO: how do we get the trace?
        # TODO: predictions and targets are out of order
        scores = scorer(
            inputs=inputs,
            outputs=predictions,
            expectations=targets,
            trace=None,
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

    return make_metric(
        eval_fn=eval_fn,
        name=scorer.name,
        greater_is_better=scorer.greater_is_better,
        metric_metadata=metric_metadata,
        genai_metric_args=genai_metric_args,
    )
