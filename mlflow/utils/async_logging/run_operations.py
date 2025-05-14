class RunOperations:
    """Class that helps manage the futures of MLflow async logging."""

    def __init__(self, operation_futures):
        self._operation_futures = operation_futures or []

    def wait(self):
        """Blocks on completion of all futures."""
        from mlflow.exceptions import MlflowException

        failed_operations = []
        for future in self._operation_futures:
            try:
                future.result()
            except Exception as e:
                failed_operations.append(e)

        if len(failed_operations) > 0:
            raise MlflowException(
                "The following failures occurred while performing one or more async logging "
                f"operations: {failed_operations}"
            )


def get_combined_run_operations(run_operations_list: list[RunOperations]) -> RunOperations:
    """Combine a list of RunOperations objects into a single RunOperations object.

    Given a list of `RunOperations`, returns a single `RunOperations` object that represents the
    combined set of operations. If the input list is empty, returns None. If the input list
    contains only one element, returns that element. Otherwise, creates a new `RunOperations`
    object that combines the operation futures from each input RunOperations object.

    Args:
        run_operations_list: A list of `RunOperations` objects to combine.

    Returns:
        A single `RunOperations` object that represents the combined set of operations.
    """
    if not run_operations_list:
        return None
    if len(run_operations_list) == 1:
        return run_operations_list[0]

    if len(run_operations_list) > 1:
        operation_futures = []
        for run_operations in run_operations_list:
            if run_operations and run_operations._operation_futures:
                operation_futures.extend(run_operations._operation_futures)
        return RunOperations(operation_futures)
