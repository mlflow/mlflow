import asyncio
import threading
import uuid
from typing import TYPE_CHECKING, Any

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
        model_config: dict[str, Any] | None = None,
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

    def _do_inference(self, input, params: dict[str, Any] | None) -> dict[str, Any]:
        """
        Perform engine inference on a single engine input e.g. not an iterable of
        engine inputs. The engine inputs must already be preprocessed/cleaned.
        """

        if isinstance(input, dict):
            return self._predict_single(**input, **(params or {}))
        else:
            return self._predict_single(input, **(params or {}))

    def predict(self, data, params: dict[str, Any] | None = None) -> list[str] | str:
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
    def _convert_chat_message_history_to_chat_message_objects(
        data: dict[str, Any],
    ) -> dict[str, Any]:
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

    def _format_predict_input(self, data) -> str | dict[str, Any] | list[Any]:
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

    def _predict_single(self, *args, **kwargs) -> list[dict[str, Any]]:
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

    def predict(self, data, params: dict[str, Any] | None = None) -> list[str] | str:
        inputs = self._format_predict_input(data, params)

        # LlamaIndex Workflow runs async but MLflow pyfunc doesn't support async inference yet.
        predictions = self._wait_async_task(self._run_predictions(inputs))

        # Even if the input is single instance, the signature enforcement convert it to a Pandas
        # DataFrame with a single row. In this case, we should unwrap the result (list) so it
        # won't be inconsistent with the output without signature enforcement.
        should_unwrap = len(data) == 1 and isinstance(predictions, list)
        return predictions[0] if should_unwrap else predictions

    def _format_predict_input(
        self, data, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        inputs = _convert_llm_input_data_with_unwrapping(data)
        params = params or {}
        if isinstance(inputs, dict):
            return [{**inputs, **params}]
        return [{**x, **params} for x in inputs]

    async def _run_predictions(self, inputs: list[dict[str, Any]]) -> asyncio.Future:
        tasks = [self._predict_single(x) for x in inputs]
        return await asyncio.gather(*tasks)

    async def _predict_single(self, x: dict[str, Any]) -> Any:
        if not isinstance(x, dict):
            raise ValueError(f"Unsupported input type: {type(x)}. It must be a dictionary.")
        return await self._llama_model.run(**x)

    def _wait_async_task(self, task: asyncio.Future) -> Any:
        """
        A utility function to run async tasks in a blocking manner.

        If there is no event loop running already, for example, in a model serving endpoint,
        we can simply create a new event loop and run the task there. However, in a notebook
        environment (or pytest with asyncio decoration), there is already an event loop running
        at the root level and we cannot start a new one.
        """
        if not self._is_event_loop_running():
            return asyncio.new_event_loop().run_until_complete(task)
        else:
            # NB: The popular way to run async task where an event loop is already running is to
            # use nest_asyncio. However, nest_asyncio.apply() breaks the async OpenAI client
            # somehow, which is used for the most of LLM calls in LlamaIndex including Databricks
            # LLMs. Therefore, we use a hacky workaround that creates a new thread and run the
            # new event loop there. This may degrade the performance compared to the native
            # asyncio, but it should be fine because this is only used in the notebook env.
            results = None
            exception = None

            def _run():
                nonlocal results, exception

                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(task)
                except Exception as e:
                    exception = e
                finally:
                    loop.close()

            thread = threading.Thread(
                target=_run, name=f"mlflow_llamaindex_async_task_runner_{uuid.uuid4().hex[:8]}"
            )
            thread.start()
            thread.join()

            if exception:
                raise exception

            return results

    def _is_event_loop_running(self) -> bool:
        try:
            loop = asyncio.get_running_loop()
            return loop is not None
        except Exception:
            return False


def create_pyfunc_wrapper(
    model: Any,
    engine_type: str | None = None,
    model_config: dict[str, Any] | None = None,
):
    """
    A factory function that creates a Pyfunc wrapper around a LlamaIndex index/engine/workflow.

    Args:
        model: A LlamaIndex index/engine/workflow.
        engine_type: The type of the engine. Only required if `model` is an index
            and must be one of [chat, query, retriever].
        model_config: A dictionary of model configuration parameters.
    """
    try:
        from llama_index.core.workflow import Workflow

        if isinstance(model, Workflow):
            return _create_wrapper_from_workflow(model, model_config)
    except ImportError:
        pass

    from llama_index.core.indices.base import BaseIndex

    if isinstance(model, BaseIndex):
        return _create_wrapper_from_index(model, engine_type, model_config)
    else:
        # Engine does not have a common base class so we assume
        # everything else is an engine
        return _create_wrapper_from_engine(model, model_config)


def _create_wrapper_from_index(index, engine_type: str, model_config: dict[str, Any] | None = None):
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


def _create_wrapper_from_engine(engine: Any, model_config: dict[str, Any] | None = None):
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


def _create_wrapper_from_workflow(workflow: Any, model_config: dict[str, Any] | None = None):
    return WorkflowWrapper(workflow, model_config)
