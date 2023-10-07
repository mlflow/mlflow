def run_operations_status_check(run_operations_list=[]):
    for run_operations in run_operations_list:
        run_operations.await_completion()
