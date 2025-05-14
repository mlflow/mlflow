from mlflow.llama_index.autolog import autolog
from mlflow.llama_index.constant import FLAVOR_NAME
from mlflow.version import IS_TRACING_SDK_ONLY

__all__ = ["autolog", "FLAVOR_NAME"]

# Import model logging APIs only if mlflow skinny or full package is installed,
# i.e., skip if only mlflow-tracing package is installed.
if not IS_TRACING_SDK_ONLY:
    from mlflow.llama_index.model import (
        _load_pyfunc,
        load_model,
        log_model,
        save_model,
    )

    __all__ += [
        "load_model",
        "log_model",
        "save_model",
        "_load_pyfunc",
    ]
