MLFLOW_VERSION = "0.8.0"  # we expect this model to be bound to this mlflwo version.


class TestModel:
    def __init__(self, check_version=True):
        self._check_version = check_version

    def predict(self, df):
        from mlflow.version import VERSION
        if self._check_version:
            assert VERSION == MLFLOW_VERSION
        mu = df.mean().mean()
        return [mu for _ in range(len(df))]


def _load_pyfunc(_):
    return TestModel()
