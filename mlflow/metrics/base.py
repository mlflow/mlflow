from types import FunctionType
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.exceptions import MlflowException


class MetricValue:
    """
    The value of a metric.
    :param scores: The value of the metric per row
    :param justifications: The justification (if applicable) for the respective score
    :param aggregate_results: A dictionary mapping the name of the aggregation to its value
    """

    def __init__(
        self,
        scores: list[float] = None,
        justifications: list[str] = None,
        aggregate_results: dict[str, float] = None,
    ):
        self.scores = scores
        self.justifications = justifications
        self.aggregate_results = aggregate_results


class EvaluationMetric:
    '''
    An evaluation metric.
    :param eval_fn:
        A function that computes the metric with the following signature:
        .. code-block:: python
            def eval_fn(
                eval_df: Union[pandas.Dataframe, pyspark.sql.DataFrame],
                builtin_metrics: Dict[str, MetricValue],
            ) -> MetricValue:
                """
                :param eval_df:
                    A Pandas or Spark DataFrame containing ``prediction`` and ``target`` column.
                    The ``prediction`` column contains the predictions made by the model.
                    The ``target`` column contains the corresponding labels to the predictions made
                    on that row.
                :param builtin_metrics:
                    A dictionary containing the metrics calculated by the default evaluator.
                    The keys are the names of the metrics and the values are the metric values.
                    Refer to the DefaultEvaluator behavior section for what metrics
                    will be returned based on the type of model (i.e. classifier or regressor).
                :return:
                    
                """
                ...
    :param name: The name of the metric.
    :param greater_is_better: Whether a higher value of the metric is better.
    :param long_name: (Optional) The long name of the metric. For example,
        ``"root_mean_squared_error"`` for ``"mse"``.
    :param version: (Optional) The metric version. For example ``v1``.
    '''

    def __init__(self, eval_fn, name, greater_is_better, long_name=None, version=None):
        self.eval_fn = eval_fn
        self.name = name
        self.greater_is_better = greater_is_better
        self.long_name = long_name or name
        self.version = version

    def __str__(self):
        if self.long_name:
            return (
                f"EvaluationMetric(name={self.name}, long_name={self.long_name}, "
                f"greater_is_better={self.greater_is_better})"
            )
        else:
            return f"EvaluationMetric(name={self.name}, greater_is_better={self.greater_is_better})"


def make_metric(
    *,
    eval_fn,
    greater_is_better,
    name=None,
    long_name=None,
    version=None,
):
    if name is None:
        if isinstance(eval_fn, FunctionType) and eval_fn.__name__ == "<lambda>":
            raise MlflowException(
                "`name` must be specified if `eval_fn` is a lambda function.",
                INVALID_PARAMETER_VALUE,
            )
        if not hasattr(eval_fn, "__name__"):
            raise MlflowException(
                "`name` must be specified if `eval_fn` does not have a `__name__` attribute.",
                INVALID_PARAMETER_VALUE,
            )
        name = eval_fn.__name__

    return EvaluationMetric(eval_fn, name, greater_is_better, long_name, version)
