from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from llama_index.core import QueryBundle

from mlflow.models.utils import _convert_llm_input_data

CHAT_ENGINE_NAME = "chat"
QUERY_ENGINE_NAME = "query"
RETRIEVER_ENGINE_NAME = "retriever"
SUPPORTED_ENGINES = {CHAT_ENGINE_NAME, QUERY_ENGINE_NAME, RETRIEVER_ENGINE_NAME}

_CHAT_MESSAGE_HISTORY_PARAMETER_NAME = "chat_history"


def _convert_llm_input_data_with_unwrapping(data):
    """
    Transforms the input data to the format expected by the LlamaIndex engine.

    TODO: Migrate the unwrapping logic to mlflow.evaluate() function or _convert_llm_input_data,
    # because it is not specific to LlamaIndex.
    """
    data = _convert_llm_input_data(data)

    # For mlflow.evaluate() call, the input dataset will be a pandas DataFrame. The DF should have
    # a column named "inputs" which contains the actual query data. After the preprocessing, the
    # each row will be passed here as a dictionary with the key "inputs". Therefore, we need to
    # extract the actual query data from the dictionary.
    if isinstance(data, dict) and ("inputs" in data):
        data = data["inputs"]

    return data


def _format_predict_input_query_engine_and_retriever(data) -> "QueryBundle":
    """Convert pyfunc input to a QueryBundle."""
    from llama_index.core import QueryBundle

    data = _convert_llm_input_data_with_unwrapping(data)

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
        engine,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        self.engine = engine
        self.model_config = model_config or {}

    @property
    def index(self):
        return self.engine.index

    def get_raw_model(self):
        return self.engine

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
    @property
    def engine_type(self):
        return CHAT_ENGINE_NAME

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
        data = _convert_llm_input_data_with_unwrapping(data)

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
    @property
    def engine_type(self):
        return QUERY_ENGINE_NAME

    def _predict_single(self, *args, **kwargs) -> str:
        return self.engine.query(*args, **kwargs).response

    def _format_predict_input(self, data) -> "QueryBundle":
        return _format_predict_input_query_engine_and_retriever(data)


class RetrieverEngineWrapper(_LlamaIndexModelWrapperBase):
    @property
    def engine_type(self):
        return RETRIEVER_ENGINE_NAME

    def _predict_single(self, *args, **kwargs) -> List[Dict]:
        response = self.engine.retrieve(*args, **kwargs)
        return [node.dict() for node in response]

    def _format_predict_input(self, data) -> "QueryBundle":
        return _format_predict_input_query_engine_and_retriever(data)


def create_engine_wrapper(
    index_or_engine: Any,
    engine_type: Optional[str] = None,
    model_config: Optional[Dict[str, Any]] = None,
):
    """
    A factory function that creates a Pyfunc wrapper around a LlamaIndex index or engine.
    """
    from llama_index.core.indices.base import BaseIndex

    if isinstance(index_or_engine, BaseIndex):
        return _create_wrapper_from_index(index_or_engine, engine_type, model_config)
    else:
        return _create_wrapper_from_engine(index_or_engine, model_config)


def _create_wrapper_from_index(
    index, engine_type: str, model_config: Optional[Dict[str, Any]] = None
):
    model_config = model_config or {}
    if engine_type == QUERY_ENGINE_NAME:
        engine = index.as_query_engine(**model_config)
        return QueryEngineWrapper(engine, model_config)
    elif engine_type == CHAT_ENGINE_NAME:
        engine = index.as_chat_engine(**model_config)
        return ChatEngineWrapper(engine, model_config)
    elif engine_type == RETRIEVER_ENGINE_NAME:
        engine = index.as_retriever(**model_config)
        return RetrieverEngineWrapper(engine, model_config)
    else:
        raise ValueError(
            f"Unsupported engine type: {engine_type}. It must be one of {SUPPORTED_ENGINES}"
        )


def _create_wrapper_from_engine(engine: Any, model_config: Optional[Dict[str, Any]] = None):
    from llama_index.core.base.base_query_engine import BaseQueryEngine
    from llama_index.core.chat_engine.types import BaseChatEngine
    from llama_index.core.retrievers import BaseRetriever

    if isinstance(engine, BaseChatEngine):
        return ChatEngineWrapper(engine, model_config)
    elif isinstance(engine, BaseQueryEngine):
        return QueryEngineWrapper(engine, model_config)
    elif isinstance(engine, BaseRetriever):
        return RetrieverEngineWrapper(engine, model_config)
    else:
        raise ValueError(
            f"Unsupported engine type: {type(engine)}. It must be one of {SUPPORTED_ENGINES}"
        )
