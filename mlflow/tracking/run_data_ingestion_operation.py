class RunDataIngestionOperation:
    """
    Represents run data (metrics, tags, params) ingestion operation that can be awaited. Eg. Awaiting logging of metrics.
    """

    def __init__(self, run_id, operation_future):
        self._run_id = run_id
        self._operation_future = operation_future

    def await_completion(self):
        """
        Blocks on completion of the MLflow Run operations.
        """
