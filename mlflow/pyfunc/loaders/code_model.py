import inspect
import logging
from typing import Any, Dict, Optional

from mlflow.pyfunc.model import (
    _load_context_model_and_signature,
    _log_warning_if_params_not_in_predict_signature,
)
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


def _load_pyfunc(local_path: str, model_config: Optional[Dict[str, Any]] = None):
    context, model, signature = _load_context_model_and_signature(local_path, model_config)
    return _CodeModelPyfuncWrapper(model=model, context=context, signature=signature)


@experimental
class _CodeModelPyfuncWrapper:
    def __init__(self, model, context, signature):
        """
        Args:
            model: An instance of a subclass of :class:`~PythonModel`.
            context: A :class:`~PythonModelContext` instance containing artifacts that
                        ``chat_model`` may use when performing inference.
            signature: :class:`~ModelSignature` instance describing model input and output.
        """
        self.python_model = model
        self.context = context
        self.signature = signature

    def predict(self, model_input, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            model_input: Model input data as one of dict, str, bool, bytes, float, int, str type.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions as an iterator of chunks. The chunks in the iterator must be type of
            dict or string. Chunk dict fields are determined by the model implementation.
        """
        if inspect.signature(self.python_model.predict).parameters.get("params"):
            return self.python_model.predict(self.context, model_input, params=params)
        _log_warning_if_params_not_in_predict_signature(_logger, params)
        return self.python_model.predict(self.context, model_input)

    def predict_stream(self, model_input, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            model_input: LLM Model single input.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Streaming predictions.
        """
        if inspect.signature(self.python_model.predict_stream).parameters.get("params"):
            return self.python_model.predict_stream(
                self.context, self._convert_input(model_input), params=params
            )
        _log_warning_if_params_not_in_predict_signature(_logger, params)
        return self.python_model.predict_stream(self.context, model_input)
