import mlflow


def _load_pyfunc(path):
    """
    Redirect `mlflow.keras._load_pyfunc` to `mlflow.tensorflow._load_pyfunc`,
    For backwards compatibility on loading keras model saved by old mlflow versions.
    """
    return mlflow.tensorflow._load_pyfunc(path)
