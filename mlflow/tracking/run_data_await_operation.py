from mlflow.exceptions import MlflowException


class RunDataAwaitOperation:
    """
    Represents run data (metrics, tags, params) ingestion operation that can be awaited. Eg. Awaiting logging of metrics.
    """

    """
    Blocks on completion of the MLflow Run log data operations.
    """

    def __init__(self, run_id, operation_future):
        self._run_id = run_id
        self._operation_future = operation_future

    def await_completion(self):
        """
        Blocks on completion of the MLflow Run log data operations.
        """
        if self._operation_future:
            try:
                self._operation_future.result()
            except Exception as e:
                print(e)
                raise
