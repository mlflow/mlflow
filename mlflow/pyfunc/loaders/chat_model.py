import os
from typing import Any, Dict, Optional

import cloudpickle

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.pyfunc.model import (
    CONFIG_KEY_ARTIFACT_RELATIVE_PATH,
    CONFIG_KEY_ARTIFACTS,
    CONFIG_KEY_CLOUDPICKLE_VERSION,
    CONFIG_KEY_PYTHON_MODEL,
    PythonModelContext,
)
from mlflow.types.llm import ChatMessage, ChatParams, ChatRequest, ChatResponse
from mlflow.utils.annotations import experimental
from mlflow.utils.model_utils import _get_flavor_configuration


def _load_pyfunc(model_path: str, model_config: Optional[Dict[str, Any]] = None):
    pyfunc_config = _get_flavor_configuration(
        model_path=model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME
    )

    python_model_cloudpickle_version = pyfunc_config.get(CONFIG_KEY_CLOUDPICKLE_VERSION, None)
    if python_model_cloudpickle_version is None:
        mlflow.pyfunc._logger.warning(
            "The version of CloudPickle used to save the model could not be found in the MLmodel"
            " configuration"
        )
    elif python_model_cloudpickle_version != cloudpickle.__version__:
        # CloudPickle does not have a well-defined cross-version compatibility policy. Micro version
        # releases have been known to cause incompatibilities. Therefore, we match on the full
        # library version
        mlflow.pyfunc._logger.warning(
            "The version of CloudPickle that was used to save the model, `CloudPickle %s`, differs"
            " from the version of CloudPickle that is currently running, `CloudPickle %s`, and may"
            " be incompatible",
            python_model_cloudpickle_version,
            cloudpickle.__version__,
        )

    python_model_subpath = pyfunc_config.get(CONFIG_KEY_PYTHON_MODEL, None)
    if python_model_subpath is None:
        raise MlflowException("Python model path was not specified in the model configuration")
    with open(os.path.join(model_path, python_model_subpath), "rb") as f:
        chat_model = cloudpickle.load(f)

    artifacts = {}
    for saved_artifact_name, saved_artifact_info in pyfunc_config.get(
        CONFIG_KEY_ARTIFACTS, {}
    ).items():
        artifacts[saved_artifact_name] = os.path.join(
            model_path, saved_artifact_info[CONFIG_KEY_ARTIFACT_RELATIVE_PATH]
        )

    context = PythonModelContext(artifacts=artifacts, model_config=model_config)
    chat_model.load_context(context=context)
    signature = mlflow.models.Model.load(model_path).signature
    return _ChatModelPyfuncWrapper(chat_model=chat_model, context=context, signature=signature)


@experimental
class _ChatModelPyfuncWrapper:
    """
    Wrapper class that converts dict inputs to pydantic objects accepted by :class:`~ChatModel`.
    """

    def __init__(self, chat_model, context, signature):
        """
        :param chat_model: An instance of a subclass of :class:`~ChatModel`.
        :param context: A :class:`~PythonModelContext` instance containing artifacts that
                        ``python_model`` may use when performing inference.
        :param signature: :class:`~ModelSignature` instance describing model input and output.
        """
        self.chat_model = chat_model
        self.context = context
        self.signature = signature

    def _convert_input(self, model_input):
        # model_input should be correct from signature validation, so just convert it to dict here
        dict_input = {key: value[0] for key, value in model_input.to_dict(orient="list").items()}

        messages = [ChatMessage(**message) for message in dict_input.pop("messages", None)]
        params = ChatParams(**dict_input)

        return messages, params

    def predict(self, model_input: ChatRequest, params: Optional[Dict[str, Any]] = None):
        """
        :param model_input: Model input data.
        :param params: Additional parameters to pass to the model for inference.
                       Unused in this implementation, as the params are handled
                       via ``self._convert_input()``.
        :return: Model predictions.
        """
        messages, params = self._convert_input(model_input)
        response = self.chat_model.predict(self.context, messages, params)

        if isinstance(response, ChatResponse):
            return response.to_dict()

        return response
