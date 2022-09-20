import mlflow


# Redirect `mlflow.keras._load_pyfunc` to `mlflow.tensorflow._load_pyfunc`,
# For backwards compatibility on loading keras model saved by old mlflow versions.
_load_pyfunc = mlflow.tensorflow._load_pyfunc

load_model = mlflow.tensorflow.load_model

log_model = mlflow.tensorflow.log_model

save_model = mlflow.tensorflow.save_model

autolog = mlflow.tensorflow.autolog
