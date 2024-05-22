import mlflow

mlflow.set_experiment("tracing")


@mlflow.trace()
def f1(x: int) -> int:
    return x + 1


@mlflow.trace()
def f2(x: int) -> int:
    return f1(x) + 2


assert f2(1) == 4

traces = mlflow.search_traces()
print(traces)
