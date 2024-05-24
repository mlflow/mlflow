from typing import Any, Dict, Optional

from mlflow.pyfunc.loaders.chat_model import _ChatModelPyfuncWrapper
from mlflow.
from mlflow.pyfunc.model import (
    ChatModel,
    _load_context_model_and_signature,
    _PythonModelPyfuncWrapper,
)


def _load_pyfunc(local_path: str, model_config: Optional[Dict[str, Any]] = None):
    context, model, signature = _load_context_model_and_signature(local_path, model_config)
    if isinstance(model, ChatModel):
        return _ChatModelPyfuncWrapper(model, context, signature)
    else:
        # TODO: replace this with something like _PythonModelPyfuncWrapper that does not enforce dataframe
        return _PythonModelPyfuncWrapper(model, context, signature)
