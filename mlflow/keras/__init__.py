# MLflow Keras 3 flavor.
import keras
from packaging.version import Version

if Version(keras.__version__) < Version("3.0.0"):
    from mlflow.tensorflow import (
        # Redirect `mlflow.keras._load_pyfunc` to `mlflow.tensorflow._load_pyfunc`,
        # For backwards compatibility on loading keras model saved by old mlflow versions.
        _load_pyfunc,
        autolog,
        load_model,
        log_model,
        save_model,
    )

    __all__ = [
        "_load_pyfunc",
        "autolog",
        "load_model",
        "save_model",
        "log_model",
    ]
else:
    from mlflow.keras.autologging import autolog
    from mlflow.keras.callback import MlflowCallback
    from mlflow.keras.load import _load_pyfunc, load_model
    from mlflow.keras.save import (
        get_default_conda_env,
        get_default_pip_requirements,
        log_model,
        save_model,
    )

    FLAVOR_NAME = "keras"

    MLflowCallback = MlflowCallback  # for backwards compatibility

    __all__ = [
        "_load_pyfunc",
        "MlflowCallback",
        "MLflowCallback",
        "autolog",
        "load_model",
        "save_model",
        "log_model",
        "get_default_pip_requirements",
        "get_default_conda_env",
    ]
