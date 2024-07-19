from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from llama_index.core import QueryBundle

from mlflow.models.utils import _convert_llm_input_data

CHAT_ENGINE_NAME = "chat"
QUERY_ENGINE_NAME = "query"
RETRIEVER_ENGINE_NAME = "retriever"
SUPPORTED_ENGINES = {CHAT_ENGINE_NAME, QUERY_ENGINE_NAME, RETRIEVER_ENGINE_NAME}

_CHAT_MESSAGE_HISTORY_PARAMETER_NAME = "chat_history"


def _format_predict_input_query_engine_and_retriever(data) -> "QueryBundle":
    """Convert pyfunc input to a QueryBundle."""
    from llama_index.core import QueryBundle

    data = _convert_llm_input_data(data)

    if isinstance(data, str):
        return QueryBundle(query_str=data)
    elif isinstance(data, dict):
        return QueryBundle(**data)
    elif isinstance(data, list):
        # NB: handle pandas returning lists when there is a single row
        prediction_input = [_format_predict_input_query_engine_and_retriever(d) for d in data]
        return prediction_input if len(prediction_input) > 1 else prediction_input[0]
    else:
        raise ValueError(
            f"Unsupported input type: {type(data)}. It must be one of "
            "[str, dict, list, numpy.ndarray, pandas.DataFrame]"
        )


class _LlamaIndexModelWrapperBase:
    def __init__(
        self,
        index,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        self.index = index
        self.model_config = model_config or {}

    def _predict_single(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def _format_predict_input(self, data):
        raise NotImplementedError

    def _do_inference(self, input, params: Optional[Dict[str, Any]]) -> Dict:
        """
        Perform engine inference on a single engine input e.g. not an iterable of
        engine inputs. The engine inputs must already be preprocessed/cleaned.
        """

        if isinstance(input, Dict):
            return self._predict_single(**input, **(params or {}))
        else:
            return self._predict_single(input, **(params or {}))

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
        self.engine = self.index.as_chat_engine(**self.model_config)

    def _predict_single(self, *args, **kwargs) -> str:
        return self.engine.chat(*args, **kwargs).response

    @staticmethod
    def _convert_chat_message_history_to_chat_message_objects(data: Dict) -> Dict:
        from llama_index.core.llms import ChatMessage

        if chat_message_history := data.get(_CHAT_MESSAGE_HISTORY_PARAMETER_NAME):
            if isinstance(chat_message_history, list):
                if all(isinstance(message, dict) for message in chat_message_history):
                    data[_CHAT_MESSAGE_HISTORY_PARAMETER_NAME] = [
                        ChatMessage(**message) for message in chat_message_history
                    ]
                else:
                    raise ValueError(
                        f"Unsupported input type: {type(chat_message_history)}. "
                        "It must be a list of dicts."
                    )

        return data

    def _format_predict_input(self, data) -> Union[str, Dict, List]:
        data = _convert_llm_input_data(data)

        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            return self._convert_chat_message_history_to_chat_message_objects(data)
        elif isinstance(data, list):
            # NB: handle pandas returning lists when there is a single row
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
        self.engine = self.index.as_query_engine(**self.model_config)

    def _predict_single(self, *args, **kwargs) -> str:
        return self.engine.query(*args, **kwargs).response

    def _format_predict_input(self, data) -> "QueryBundle":
        return _format_predict_input_query_engine_and_retriever(data)


class RetrieverEngineWrapper(_LlamaIndexModelWrapperBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine_type = RETRIEVER_ENGINE_NAME
        self.engine = self.index.as_retriever(**self.model_config)

    def _predict_single(self, *args, **kwargs) -> List[Dict]:
        response = self.engine.retrieve(*args, **kwargs)
        return [node.dict() for node in response]

    def _format_predict_input(self, data) -> "QueryBundle":
        return _format_predict_input_query_engine_and_retriever(data)


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
