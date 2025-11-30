from mlflow.dspy.autolog import autolog
from mlflow.version import IS_TRACING_SDK_ONLY

__all__ = ["autolog"]

# Import model logging APIs only if mlflow skinny or full package is installed,
# i.e., skip if only mlflow-tracing package is installed.
if not IS_TRACING_SDK_ONLY:
    from mlflow.dspy.load import _load_pyfunc, load_model
    from mlflow.dspy.save import log_model, save_model

    __all__ += [
        "save_model",
        "log_model",
        "load_model",
        "_load_pyfunc",
    ]
