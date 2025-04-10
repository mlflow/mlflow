from databricks.agents.evals import metric
from pyspark import sql as spark

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
    data: pd.DataFrame | spark.DataFrame | list[dict], EvaluationDataset
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
    genai_metric_args = {
        "name": scorer.name,
        "aggregations": scorer.aggregations,
        "greater_is_better": scorer.greater_is_better,
    }

    def eval_fn(
        predictions: "pd.Series",
        metrics: dict[str, MetricValue],
        inputs: "pd.Series",
        *args,
    ) -> MetricValue:
        # TODO: figure out what the inputs of the eval_fn are going to be after
        # the legacy df conversion. Then, deconvert the legacy conversion if needed
        # to actually use the scorer and get the scores.
        scores = scorer(
            inputs=inputs,
            outputs=predictions,
            expectations=None,
            trace=None,
        )
        # TODO: the output of the scorer could be one of
        # float | bool | str | Assessment so we need to make sure that
        # _get_aggregate results can handle that. Especially for Assessment objects.
        return MetricValue(
            scores=scores,
            aggregate_results=_get_aggregate_results(scores, scorer.aggregations),
        )

    return make_metric(
        eval_fn=eval_fn,
        greater_is_better=scorer.greater_is_better,
        genai_metric_args=genai_metric_args,
    )
