import mlflow.keras


def _load_pyfunc(path):
    import tensorflow.keras
    return mlflow.keras._load_pyfunc(path, keras_module=tensorflow.keras)
