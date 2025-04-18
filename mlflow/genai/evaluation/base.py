import logging
from typing import Any, Callable, Optional

from pyspark import sql as spark

import mlflow
from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.genai.evaluation.utils import (
    _convert_scorer_to_legacy_metric,
    _convert_to_legacy_eval_set,
)
from mlflow.genai.scorers import BuiltInScorer, Scorer
from mlflow.tracing.utils import is_model_traced

try:
    # `pandas` is not required for `mlflow-skinny`.
    import pandas as pd
except ImportError:
    pass

logger = logging.getLogger(__name__)


class EvaluationResult:
    run_id: str
    metrics: dict[str, float]


def evaluate(
    data: pd.DataFrame | spark.DataFrame | list[dict] | EvaluationDataset,
    predict_fn: Optional[Callable[..., Any]] = None,
    scorers: Optional[list[Scorer]] = None,
    model_id: Optional[str] = None,
) -> mlflow.genai.EvaluationResult:
    """
    TODO: updating docstring with real examples and API links
    Args:
        data: Dataset for the evaluation. It must be one of the following format:
            * A EvaluationDataset entity
            * Pandas DataFrame
            * Spark DataFrame
            * List of dictionary

            If a dataframe is specified, it must contain the following schema:
              - inputs (required): A column that contains a single input.
              - outputs (optional): A column that contains a single output from the
                   target model/app. If the predict_fn is provided, this is generated
                   by MLflow so not required.
              - expectations (optional): A column that contains a ground truth, or a
                   dictionary of ground truths for individual output fields.
              - traces (optional): A column that contains a single trace object
                   corresponding to the prediction for the row. Only required when
                   any of scorers requires a trace in order to compute
                   assessments/metrics.

            If a list of dictionary is passed, each dictionary should contain keys
            following the above schema.

        predict_fn: The target function to be evaluated. The specified function will be
            executed for each row in the input dataset, and outputs will be used for
            scoring.

            The function must emit a single trace per call. If it doesn't, decorate
            the function with @mlflow.trace decorator to ensure a trace to be emitted.

        scorers: A list of Scorer objects that produces evaluation scores from
            inputs, outputs, and other additional contexts. MLflow provides pre-defined
            scorers, but you can also define custom ones.

        model_id: Optional. Specify an ID of the model e.g. models:/my-model/1 to
            associate the evaluation result with. There are several ways to associate
            model with association.

            1. Use the `model_id` parameters.
            2. Use the mlflow.set_active_model() function to set model ID to global context.
               ```python
               mlflow.set_active_model(model_id="xyz")

               mlflow.evaluate(data, ...)
               ```
    """
    if mlflow.get_tracking_uri() != "databricks":
        raise ValueError(
            (
                "The genai evaluation function is only supported on Databricks. ",
                "Please set the tracking URI to Databricks.",
            )
        )

    builtin_scorers = []
    custom_scorers = []

    for scorer in scorers:
        if isinstance(scorer, BuiltInScorer):
            builtin_scorers.append(scorer)
        elif isinstance(scorer, Scorer):
            custom_scorers.append(scorer)
        else:
            raise TypeError(
                (
                    f"Scorer {scorer} is not a valid scorer. Please use the @scorer decorator ",
                    "to convert a function into a scorer or inherit from the Scorer class",
                )
            )

    evaluation_config = {
        "databricks-agents": {
            "metrics": [],
        }
    }
    for _scorer in builtin_scorers:
        evaluation_config = _scorer.update_evaluation_config(evaluation_config)

    extra_metrics = []
    for _scorer in custom_scorers:
        extra_metrics.append(_convert_scorer_to_legacy_metric(_scorer))

    if not is_model_traced(predict_fn):
        logger.info("Annotating predict_fn with tracing since it is not already traced.")
        predict_fn = mlflow.trace(predict_fn)

    mlflow.evaluate(
        model=predict_fn,
        # convert into a pandas dataframe with current evaluation set schema
        data=_convert_to_legacy_eval_set(data),
        evaluator_config=evaluation_config,
        extra_metrics=extra_metrics,
        model_type="databricks-agent",
    )
