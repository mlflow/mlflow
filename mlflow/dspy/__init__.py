from mlflow.dspy.autolog import autolog
from mlflow.version import HAS_MODEL_LOGGING_DEPS

__all__ = ["autolog"]

# Import model logging APIs only if their dependencies are installed, i.e., skip if only
# mlflow-tracing, or mlflow-skinny without numpy and pandas, is installed.
if HAS_MODEL_LOGGING_DEPS:
    from mlflow.dspy.load import _load_pyfunc, load_model
    from mlflow.dspy.save import log_model, save_model

    __all__ += [
        "save_model",
        "log_model",
        "load_model",
        "_load_pyfunc",
    ]
