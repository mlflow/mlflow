from mlflow.exceptions import MlflowException


class RunDataAwaitOperation:
    """
    Represents run data (metrics, tags, params) ingestion operation that can be awaited. Eg. Awaiting logging of metrics.
    """

    """
    Blocks on completion of the MLflow Run log data operations.
    """

    def __init__(self, run_id, operation_futures):
        self._run_id = run_id
        self._operation_futures = operation_futures

    def await_completion(self):
        """
        Blocks on completion of the MLflow Run log data operations.
        """
        failed_operations = []
        for future in self._operation_futures:
            try:
                future.result()
            except Exception as e:
                failed_operations.append(e)

        if len(failed_operations) > 0:
            raise MlflowException(
                message=(
                    "The following failures occurred while performing one or more logging"
                    f" operations: {failed_operations}"
                )
            )
