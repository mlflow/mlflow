from typing import Any

from mlflow.pyfunc.loaders.chat_agent import _ChatAgentPyfuncWrapper
from mlflow.pyfunc.loaders.chat_model import _ChatModelPyfuncWrapper
from mlflow.pyfunc.model import (
    ChatAgent,
    ChatModel,
    _load_context_model_and_signature,
    _PythonModelPyfuncWrapper,
)

try:
    from mlflow.pyfunc.model import ResponsesAgent

    IS_RESPONSES_AGENT_AVAILABLE = True
except ImportError:
    IS_RESPONSES_AGENT_AVAILABLE = False


def _load_pyfunc(local_path: str, model_config: dict[str, Any] | None = None):
    context, model, signature = _load_context_model_and_signature(local_path, model_config)
    if isinstance(model, ChatModel):
        return _ChatModelPyfuncWrapper(model, context, signature)
    elif isinstance(model, ChatAgent):
        return _ChatAgentPyfuncWrapper(model)
    elif IS_RESPONSES_AGENT_AVAILABLE and isinstance(model, ResponsesAgent):
        from mlflow.pyfunc.loaders.responses_agent import _ResponsesAgentPyfuncWrapper

        return _ResponsesAgentPyfuncWrapper(model, context)
    else:
        return _PythonModelPyfuncWrapper(model, context, signature)
