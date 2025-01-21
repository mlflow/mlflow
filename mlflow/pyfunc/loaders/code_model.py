from typing import Any, Optional

from mlflow.pyfunc.loaders.chat_agent import _ChatAgentPyfuncWrapper
from mlflow.pyfunc.loaders.chat_model import _ChatModelPyfuncWrapper
from mlflow.pyfunc.model import (
    ChatAgent,
    ChatModel,
    _load_context_model_and_signature,
    _PythonModelPyfuncWrapper,
)


def _load_pyfunc(local_path: str, model_config: Optional[dict[str, Any]] = None):
    context, model, signature = _load_context_model_and_signature(local_path, model_config)
    if isinstance(model, ChatModel):
        return _ChatModelPyfuncWrapper(model, context, signature)
    if isinstance(model, ChatAgent):
        return _ChatAgentPyfuncWrapper(model)
    else:
        return _PythonModelPyfuncWrapper(model, context, signature)
