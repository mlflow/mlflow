import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import mlflow
from mlflow.genai.evaluation.utils import (
    _convert_scorer_to_legacy_metric,
    _convert_to_legacy_eval_set,
)
from mlflow.genai.scorers import BuiltInScorer, Scorer
from mlflow.genai.scorers.builtin_scorers import GENAI_CONFIG_NAME
from mlflow.genai.utils.trace_utils import is_model_traced
from mlflow.models.evaluation.base import (
    _get_model_from_deployment_endpoint_uri,
    _is_model_deployment_endpoint_uri,
)
from mlflow.utils.uri import is_databricks_uri

if TYPE_CHECKING:
    from genai.evaluation.utils import EvaluationDatasetTypes

try:
    # `pandas` is not required for `mlflow-skinny`.
    import pandas as pd
except ImportError:
    pass


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    run_id: str
    metrics: dict[str, float]
    result_df: "pd.DataFrame"


def evaluate(
    data: "EvaluationDatasetTypes",
    predict_fn: Optional[Callable[..., Any]] = None,
    scorers: Optional[list[Scorer]] = None,
    model_id: Optional[str] = None,
) -> EvaluationResult:
    """
    TODO: updating docstring with real examples and API links

    .. warning::

        This function is not thread-safe. Please do not use it in multi-threaded
        environments.

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
              - trace (optional): A column that contains a single trace object
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
    try:
        from databricks.rag_eval.evaluation.metrics import Metric as DBAgentsMetric
    except ImportError:
        raise ImportError(
            "The `databricks-agents` package is required to use mlflow.genai.evaluate() "
            "Please install it with `pip install databricks-agents`."
        )

    if not is_databricks_uri(mlflow.get_tracking_uri()):
        raise ValueError(
            "The genai evaluation function is only supported on Databricks. "
            "Please set the tracking URI to Databricks."
        )

    builtin_scorers = []
    custom_scorers = []

    for scorer in scorers or []:
        if isinstance(scorer, BuiltInScorer):
            builtin_scorers.append(scorer)
        elif isinstance(scorer, Scorer):
            custom_scorers.append(scorer)
        elif isinstance(scorer, DBAgentsMetric):
            logger.warning(
                f"{scorer} is a legacy metric and will soon be deprecated in future releases. "
                "Please use the @scorer decorator or use builtin scorers instead."
            )
            custom_scorers.append(scorer)
        else:
            raise TypeError(
                (
                    f"Scorer {scorer} is not a valid scorer. Please use the @scorer decorator ",
                    "to convert a function into a scorer or inherit from the Scorer class",
                )
            )

    evaluation_config = {}
    for _scorer in builtin_scorers:
        evaluation_config = _scorer.update_evaluation_config(evaluation_config)

    extra_metrics = []
    for _scorer in custom_scorers:
        extra_metrics.append(_convert_scorer_to_legacy_metric(_scorer))

    # convert into a pandas dataframe with current evaluation set schema
    data = _convert_to_legacy_eval_set(data)

    if predict_fn:
        sample_input = data.iloc[0]["request"]
        if not is_model_traced(predict_fn, sample_input):
            logger.info("Annotating predict_fn with tracing since it is not already traced.")
            predict_fn = mlflow.trace(predict_fn)

    result = mlflow.evaluate(
        model=predict_fn,
        data=data,
        evaluator_config=evaluation_config,
        extra_metrics=extra_metrics,
        model_type=GENAI_CONFIG_NAME,
        model_id=model_id,
    )

    return EvaluationResult(
        run_id=result._run_id,
        metrics=result.metrics,
        result_df=result.tables["eval_results"],
    )


def to_predict_fn(endpoint_uri: str) -> Callable:
    """
    Convert an endpoint URI to a predict function.

    Args:
        endpoint_uri: The endpoint URI to convert.

    Returns:
        A predict function that can be used to make predictions.

    Example:
        .. code-block:: python

            data = (
                pd.DataFrame(
                    {
                        "inputs": ["What is MLflow?", "What is Spark?"],
                    }
                ),
            )
            predict_fn = mlflow.genai.to_predict_fn("endpoints:/chat")
            mlflow.genai.evaluate(
                data=data,
                predict_fn=predict_fn,
            )
    """
    if not _is_model_deployment_endpoint_uri(endpoint_uri):
        raise ValueError(
            f"Invalid endpoint URI: {endpoint_uri}. The endpoint URI must be a valid model "
            f"deployment endpoint URI."
        )

    model = _get_model_from_deployment_endpoint_uri(endpoint_uri)
    if model is None:
        raise ValueError(f"Model not found for endpoint URI: {endpoint_uri}")

    return model.predict
