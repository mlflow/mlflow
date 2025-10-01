from mlflow.server.jobs import job_function


@job_function(max_workers=1)
def job_func1(x, y=1):
    pass
