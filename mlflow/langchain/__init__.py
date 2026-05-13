from mlflow.langchain.autolog import autolog
from mlflow.langchain.constants import FLAVOR_NAME
from mlflow.version import IS_TRACING_SDK_ONLY

__all__ = ["autolog", "FLAVOR_NAME"]

# Import model logging APIs only if mlflow skinny or full package is installed,
# i.e., skip if only mlflow-tracing package is installed.
if not IS_TRACING_SDK_ONLY:
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
