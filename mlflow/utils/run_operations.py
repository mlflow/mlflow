class RunOperations:
    """
    Represents a collection of operations on one or more MLflow Runs, such as run creation
    or metric logging.
    """

    def __init__(self, operation_futures):
        self._operation_futures = operation_futures or []

    def await_completion(self):
        """
        Blocks on completion of the MLflow Run operations.
        """
        failed_operations = []
        for future in self._operation_futures:
            try:
                future.result()
            except Exception as e:
                failed_operations.append(e)

        if len(failed_operations) > 0:
            raise Exception(
                message=(
                    "The following failures occurred while performing one or more logging"
                    f" operations: {failed_operations}"
                )
            )
