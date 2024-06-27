from typing import Any, Callable, Dict, List, Optional, Union

from llama_index.core import QueryBundle
from llama_index.core.llms import ChatMessage

from mlflow.models.utils import _convert_llm_input_data

CHAT_ENGINE_NAME = "chat"
QUERY_ENGINE_NAME = "query"
RETRIEVER_ENGINE_NAME = "retriever"
SUPPORTED_ENGINES = {CHAT_ENGINE_NAME, QUERY_ENGINE_NAME, RETRIEVER_ENGINE_NAME}


class _LlamaIndexModelWrapperBase:
    def __init__(
        self,
        index,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        self.index = index
        self.model_config = model_config or {}
        self.predict_callable = self._build_engine_method()

    def _build_engine_method(self) -> Callable:
        raise NotImplementedError

    def _format_predict_input(self, data):
        raise NotImplementedError

    def _do_inference(self, input, params: Optional[Dict[str, Any]]) -> Dict:
        """
        Perform engine inference on a single engine input e.g. not an iterable of
        engine inputs. The engine inputs must already be preprocessed/cleaned.
        """
        if isinstance(input, dict):
            return self.predict_callable(**input, **(params or {}))
        else:
            return self.predict_callable(input, **(params or {}))

    def predict(self, data, params: Optional[Dict[str, Any]] = None) -> Union[List[str], str]:
        data = self._format_predict_input(data)

        if isinstance(data, list):
            return [self._do_inference(x, params) for x in data]
        else:
            return self._do_inference(data, params)


class ChatEngineWrapper(_LlamaIndexModelWrapperBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine_type = CHAT_ENGINE_NAME

    def _build_engine_method(self) -> Callable:
        return self.index.as_chat_engine(**self.model_config).chat

    def _safe_convert_to_chat_messages(self, data) -> Union[List[ChatMessage], str]:
        # NB: Chat messages that have been converted to a string via str() are formatted as
        # "`role`: `content`", for example "system: You are a helpful bot".
        if isinstance(data, (list, tuple)) and all(":" in message for message in data):
            return [ChatMessage.from_str(*(message.split(": ", 1)[::-1])) for message in data]

        return data

    def _format_predict_input(self, data):
        """
        Chat engines can have a variety of signatures. The two primary standards are:
        1. chat(str, Sequence[ChatMessage])
        2. chat(Sequence[ChatMessage])

        There are not consistent naming conventions for either of these standards, so
        we'll need to leverage positional arguments.
        """
        data = _convert_llm_input_data(data)

        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            if len(data) == 1:
                # Assume this value should be treated as a positional argument
                payload = next(iter(data.values()))
                return self._safe_convert_to_chat_messages(payload)

            elif len(data) > 1:
                # Assume that any kwarg value that's a sequence of strings with a colon
                # should be converted to a list of ChatMessage objects.
                return {k: self._safe_convert_to_chat_messages(v) for k, v in data.items()}
        elif isinstance(data, list):
            prediction_input = [self._format_predict_input(d) for d in data]
            return prediction_input if len(prediction_input) > 1 else prediction_input[0]
        else:
            raise ValueError(
                f"Unsupported input type: {type(data)}. It must be one of "
                "[str, dict, list, numpy.ndarray, pandas.DataFrame]"
            )


class QueryEngineWrapper(_LlamaIndexModelWrapperBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine_type = QUERY_ENGINE_NAME

    def _build_engine_method(self) -> Callable:
        return self.index.as_query_engine(**self.model_config).query

    def _format_predict_input(self, data):
        """Convert pyfunc input to a QueryBundle."""
        data = _convert_llm_input_data(data)

        if isinstance(data, str):
            return QueryBundle.from_dict({"query_str": data})
        elif isinstance(data, dict):
            try:
                return QueryBundle.from_dict(data)
            except KeyError:
                raise ValueError(
                    "The input dictionary did not have the correct schema. It must support the "
                    "supported in a llama_index QueryBundle: "
                    "['query_str', 'image_path', 'custom_embedding_strs', 'embedding']"
                )
        elif isinstance(data, list):
            prediction_input = [self._format_predict_input(d) for d in data]
            return prediction_input if len(prediction_input) > 1 else prediction_input[0]
        else:
            raise ValueError(
                f"Unsupported input type: {type(data)}. It must be one of "
                "[str, dict, list, numpy.ndarray, pandas.DataFrame]"
            )


class RetrieverEngineWrapper(QueryEngineWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine_type = RETRIEVER_ENGINE_NAME

    def _build_engine_method(self) -> Callable:
        return self.index.as_retriever(**self.model_config).retrieve


def create_engine_wrapper(index, engine_type: str, model_config: Optional[Dict[str, Any]] = None):
    if engine_type == QUERY_ENGINE_NAME:
        return QueryEngineWrapper(index, model_config)
    elif engine_type == CHAT_ENGINE_NAME:
        return ChatEngineWrapper(index, model_config)
    elif engine_type == RETRIEVER_ENGINE_NAME:
        return RetrieverEngineWrapper(index, model_config)
    else:
        raise ValueError(
            f"Unsupported engine type: {engine_type}. It must be one of {SUPPORTED_ENGINES}"
        )
