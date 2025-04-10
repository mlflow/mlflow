from pyspark import sql as spark

try:
    # `pandas` is not required for `mlflow-skinny`.
    import pandas as pd
except ImportError:
    pass

# TODO: ML-52299
def _convert_to_legacy_eval_set(data: pd.DataFrame | spark.DataFrame | list[dict], EvaluationDataset) -> dict:
    """
    Takes in different types of inputs and converts it into to the current eval-set schema that
    Agent Evaluation takes in. The transformed schema should be accepted by mlflow.evaluate()
    """
    raise NotImplementedError(
        "This function is not implemented yet."
    )