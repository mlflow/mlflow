import inspect
import logging
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.pyfunc.model import (
    _get_first_string_column,
    _log_warning_if_params_not_in_predict_signature,
)

CONFIG_KEY_ARTIFACTS = "artifacts"
CONFIG_KEY_ARTIFACT_RELATIVE_PATH = "path"
CONFIG_KEY_ARTIFACT_URI = "uri"
CONFIG_KEY_PYTHON_MODEL = "python_model"
CONFIG_KEY_CLOUDPICKLE_VERSION = "cloudpickle_version"
_SAVED_PYTHON_MODEL_SUBPATH = "python_model.pkl"


_logger = logging.getLogger(__name__)


class _OompaLoopaModelPyfuncWrapper:
    """
    Flexible wrapper class
    """

    def __init__(self, python_model, context, signature):
        """
        Args:
            python_model: An instance of a subclass of :class:`~PythonModel`.
            context: A :class:`~PythonModelContext` instance containing artifacts that
                     ``python_model`` may use when performing inference.
            signature: :class:`~ModelSignature` instance describing model input and output.
        """
        self.python_model = python_model
        self.context = context
        self.signature = signature

    # def _convert_input(self, model_input):
    #     """
    #     Convert the input back into an object which the model can understand.
    #     """
    #     import pandas

    #     if isinstance(model_input, dict):
    #         dict_input = model_input
    #     elif isinstance(model_input, pandas.DataFrame):
    #         dict_input = {
    #             key: value[0] for key, value in model_input.to_dict(orient="list").items()
    #         }
    #     else:
    #         raise MlflowException(
    #             "Unsupported model input type. Expected a dict or pandas.DataFrame, "
    #             f"but got {type(model_input)} instead.",
    #             error_code=INTERNAL_ERROR,
    #         )

    #     return dict_input

    def _convert_input(self, model_input):
        # this will NEVER return a dataframe
        # this will only return an object
        import pandas as pd

        hints = self.python_model._get_type_hints()
        if hints.input == List[str]:
            if isinstance(model_input, pd.DataFrame):
                first_string_column = _get_first_string_column(model_input)
                if first_string_column is None:
                    raise MlflowException.invalid_parameter_value(
                        "Expected model input to contain at least one string column"
                    )
                return model_input[first_string_column].tolist()
            elif isinstance(model_input, list):
                if all(isinstance(x, dict) for x in model_input):
                    return [next(iter(d.values())) for d in model_input]
                elif all(isinstance(x, str) for x in model_input):
                    return model_input
        elif hints.input == List[Dict[str, str]]:
            if isinstance(model_input, pd.DataFrame):
                if (
                    len(self.signature.inputs) == 1
                    and next(iter(self.signature.inputs)).name is None
                ):
                    first_string_column = _get_first_string_column(model_input)
                    return model_input[[first_string_column]].to_dict(orient="records")
                columns = [x.name for x in self.signature.inputs]
                return model_input[columns].to_dict(orient="records")
            elif isinstance(model_input, list) and all(isinstance(x, dict) for x in model_input):
                keys = [x.name for x in self.signature.inputs]
                return [{k: d[k] for k in keys} for d in model_input]
        # TODO: handle the List[Dict[str, Any]] case
        # handle the Dict[str, Any] case
        # cc
        elif hints.input == Dict[str,List[Dict[str,str]]]:
            # TODO: fill in
        # split completions
        elif hints.input == mlflow.models.rag_signatures.ChatCompletionsInput:
            if isinstance(model_input, pd.DataFrame):
                # use ChatCompletionsInput dataclass to convert to the right format
                return mlflow.models.rag_signatures.ChatCompletionsInput(
                    completions=model_input.to_dict(orient="records")
                )
            # Turn into ChatCompletionsInput
        elif hints.input == Dict[str, Any]:
            if isinstance(model_input, pd.DataFrame):
                dict_input = {
                    key: value[0] for key, value in model_input.to_dict(orient="list").items()
                }
                return dict_input
            elif isinstance(model_input, dict):
                return model_input
            
        # next quarter we add ssmth

        raise MlflowException

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
            return self.python_model.predict(
                self.context, self._convert_input(model_input), params=params
            )
        _log_warning_if_params_not_in_predict_signature(_logger, params)
        return self.python_model.predict(self.context, self._convert_input(model_input))

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
        return self.python_model.predict_stream(self.context, self._convert_input(model_input))
