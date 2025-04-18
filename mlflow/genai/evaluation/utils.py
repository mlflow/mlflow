from databricks.agents.evals import metric
from pyspark import sql as spark

from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.genai.scorers import Scorer

try:
    # `pandas` is not required for `mlflow-skinny`.
    import pandas as pd
except ImportError:
    pass


# TODO: ML-52299
def _convert_to_legacy_eval_set(
    data: pd.DataFrame | spark.DataFrame | list[dict] | EvaluationDataset,
) -> dict:
    """
    Takes in different types of inputs and converts it into to the current eval-set schema that
    Agent Evaluation takes in. The transformed schema should be accepted by mlflow.evaluate()
    """
    raise NotImplementedError("This function is not implemented yet.")


# TODO: ML-52297
def _convert_scorer_to_legacy_metric(scorer: Scorer):
    """
    Takes in a Scorer object and converts it into a legacy MLflow 2.x
    Metric object.
    """
    # TODO: complete implementation
    return metric(
        eval_fn=scorer,
    )
