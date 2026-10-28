from mlflow.langchain.autolog import autolog
from mlflow.langchain.constants import FLAVOR_NAME
from mlflow.version import HAS_MODEL_LOGGING_DEPS

__all__ = ["autolog", "FLAVOR_NAME"]

# Import model logging APIs only if their dependencies are installed, i.e., skip if only
# mlflow-tracing, or mlflow-skinny without numpy and pandas, is installed.
if HAS_MODEL_LOGGING_DEPS:
    from mlflow.langchain.model import (
        _LangChainModelWrapper,
        _load_pyfunc,
        load_model,
        log_model,
        save_model,
    )

    __all__ += [
        "_LangChainModelWrapper",
        "_load_pyfunc",
        "load_model",
        "log_model",
        "save_model",
    ]
