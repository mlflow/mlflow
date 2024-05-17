from typing import Any, Dict, Optional

from mlflow.pyfunc.model import (
    _load_context_model_and_signature,
    _PythonModelPyfuncWrapper,
)


def _load_pyfunc(local_path: str, model_config: Optional[Dict[str, Any]] = None):
    context, model, signature = _load_context_model_and_signature(local_path, model_config)
    return _PythonModelPyfuncWrapper(model=model, context=context, signature=signature)
