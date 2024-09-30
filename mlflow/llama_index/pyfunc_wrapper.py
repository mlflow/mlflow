import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from mlflow import MlflowException

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
        llama_model,  # Engine or Workflow
        model_config: Optional[Dict[str, Any]] = None,
    ):
        self._llama_model = llama_model
        self.model_config = model_config or {}

    @property
    def index(self):
        return self._llama_model.index

    def get_raw_model(self):
        return self._llama_model

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
        return self._llama_model.chat(*args, **kwargs).response

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
        return self._llama_model.query(*args, **kwargs).response

    def _format_predict_input(self, data) -> "QueryBundle":
        return _format_predict_input_query_engine_and_retriever(data)


class RetrieverEngineWrapper(_LlamaIndexModelWrapperBase):
    @property
    def engine_type(self):
        return RETRIEVER_ENGINE_NAME

    def _predict_single(self, *args, **kwargs) -> List[Dict]:
        response = self._llama_model.retrieve(*args, **kwargs)
        return [node.dict() for node in response]

    def _format_predict_input(self, data) -> "QueryBundle":
        return _format_predict_input_query_engine_and_retriever(data)


class WorkflowWrapper(_LlamaIndexModelWrapperBase):
    @property
    def index(self):
        raise NotImplementedError("LlamaIndex Workflow does not have an index")

    @property
    def engine_type(self):
        raise NotImplementedError("LlamaIndex Workflow is not an engine")

    def predict(self, data, params: Optional[Dict[str, Any]] = None) -> Union[List[str], str]:
        data = _convert_llm_input_data_with_unwrapping(data)
        params = params or {}
        inputs = [{**data, **params}] if isinstance(data, dict) else [{**x, **params} for x in data]

        # NB: LlamaIndex Workflow runs asynchronusly but MLflow doesn't support async inference
        # in pyfunc. As a workaround, we run an event loop and block until the result is available.
        tasks = [self._predict_single(x) for x in inputs]
        return self._run_async_task(tasks)

    async def _predict_single(self, x: Dict[str, Any]) -> Any:
        if not isinstance(x, dict):
            raise ValueError(f"Unsupported input type: {type(x)}. It must be a dictionary.")
        return await self._llama_model.run(**x)

    def _run_async_task(self, tasks: List[asyncio.Future]) -> List[Any]:
        """
        An utility function to run async tasks in a blocking manner.
        """
        try:
            asyncio.get_event_loop()
            is_loop_running = True
        except RuntimeError:
            is_loop_running = False

        # In notebook environment, the event loop is already running so we cannot create
        # a new one. Instead, use nest_asyncio to allow running tasks in the existing loop.
        if is_loop_running:
            try:
                import nest_asyncio
            except ImportError:
                raise MlflowException(
                    "Running LlamaIndex Workflow as a pyfunc model in your "
                    "environment requires the nest_asyncio package. Please "
                    "install it using `pip install nest_asyncio`."
                )

            nest_asyncio.apply()
            return asyncio.run(asyncio.gather(*tasks))
        else:
            loop = asyncio.new_event_loop()
            return loop.run_until_complete(*tasks)


def create_pyfunc_wrapper(
    model: Any,
    engine_type: Optional[str] = None,
    model_config: Optional[Dict[str, Any]] = None,
):
    """
    A factory function that creates a Pyfunc wrapper around a LlamaIndex index/engine/workflow.

    Args:
        model: A LlamaIndex index/engine/workflow.
        engine_type: The type of the engine. Only required if `model` is an index
            and must be one of [chat, query, retriever].
        model_config: A dictionary of model configuration parameters.
    """
    from llama_index.core.indices.base import BaseIndex
    from llama_index.core.workflow import Workflow

    if isinstance(model, BaseIndex):
        return _create_wrapper_from_index(model, engine_type, model_config)
    elif isinstance(model, Workflow):
        return _create_wrapper_from_workflow(model, model_config)
    else:
        # Engine does not have a common base class so we assume
        # everything else is an engine
        return _create_wrapper_from_engine(model, model_config)


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


def _create_wrapper_from_workflow(workflow: Any, model_config: Optional[Dict[str, Any]] = None):
    return WorkflowWrapper(workflow, model_config)
