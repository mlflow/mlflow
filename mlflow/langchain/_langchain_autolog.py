import contextlib
import inspect
import logging
import uuid
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union, cast
from uuid import UUID

from packaging.version import Version
from typing_extensions import override

import mlflow
from mlflow import MlflowClient
from mlflow.entities import RunTag
from mlflow.environment_variables import _MLFLOW_TESTING
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.autologging_utils import (
    ExceptionSafeAbstractClass,
    disable_autologging,
    get_autologging_config,
)
from mlflow.utils.autologging_utils.safety import _resolve_extra_tags

MIN_REQ_VERSION = Version(_ML_PACKAGE_VERSIONS["langchain"]["autologging"]["minimum"])
MAX_REQ_VERSION = Version(_ML_PACKAGE_VERSIONS["langchain"]["autologging"]["maximum"])

_logger = logging.getLogger(__name__)

UNSUPPORT_LOG_MODEL_MESSAGE = (
    "MLflow autologging does not support logging models containing BaseRetriever because "
    "logging the model requires `loader_fn` and `persist_dir`. Please log the model manually "
    "using `mlflow.langchain.log_model(model, artifact_path, loader_fn=..., persist_dir=...)`"
)
INFERENCE_FILE_NAME = "inference_inputs_outputs.json"


def _get_input_data_from_function(func_name, model, args, kwargs):
    func_param_name_mapping = {
        "__call__": "inputs",
        "invoke": "input",
        "get_relevant_documents": "query",
    }
    input_example_exc = None
    if param_name := func_param_name_mapping.get(func_name):
        inference_func = getattr(model, func_name)
        # A guard to make sure `param_name` is the first argument of inference function
        if next(iter(inspect.signature(inference_func).parameters.keys())) != param_name:
            input_example_exc = MlflowException(
                "Inference function signature changes, please contact MLflow team to "
                "fix langchain autologging.",
            )
        else:
            return args[0] if len(args) > 0 else kwargs.get(param_name)
    else:
        input_example_exc = MlflowException(
            f"Unsupported inference function. Only support {list(func_param_name_mapping.keys())}."
        )
    _logger.warning(
        f"Failed to gather input example of model {model.__class__.__name__} "
        f"due to {input_example_exc}."
    )


def _convert_data_to_dict(data, key):
    if isinstance(data, dict):
        return {f"{key}-{k}": v for k, v in data.items()}
    if isinstance(data, list):
        return {key: data}
    if isinstance(data, str):
        return {key: [data]}
    raise MlflowException("Unsupported data type.")


def _combine_input_and_output(input, output, session_id, func_name):
    """
    Combine input and output into a single dictionary
    """
    if func_name == "get_relevant_documents" and output is not None:
        output = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in output]
        # to make sure output is inside a single row when converted into pandas DataFrame
        output = [output]
    result = {"session_id": [session_id]}
    if input:
        result.update(_convert_data_to_dict(input, "input"))
    if output:
        result.update(_convert_data_to_dict(output, "output"))
    return result


def _update_langchain_model_config(model):
    try:
        from langchain_core.pydantic_v1 import Extra
    except ImportError as e:
        warnings.warn(
            "MLflow langchain autologging might log model several "
            "times due to the pydantic.config.Extra import error. "
            f"Error: {e}"
        )
        return False
    else:
        # Langchain models are Pydantic models, and the value for extra is
        # ignored, we need to set it to allow so as to set attributes on
        # the model to keep track of logging status
        if hasattr(model, "__config__"):
            model.__config__.extra = Extra.allow
        return True


def _inject_mlflow_callbacks(func_name, mlflow_callbacks, args, kwargs):
    if func_name == "invoke":
        from langchain.schema.runnable.config import RunnableConfig

        in_args = False
        # `config` is the second positional argument of runnable.invoke function
        # https://github.com/langchain-ai/langchain/blob/7d444724d7582386de347fb928619c2243bd0e55/libs/core/langchain_core/runnables/base.py#L468
        if len(args) >= 2:
            config = args[1]
            in_args = True
        else:
            config = kwargs.get("config", None)
        if config is None:
            callbacks = mlflow_callbacks
            config = RunnableConfig(callbacks=callbacks)
        else:
            callbacks = config.get("callbacks") or []
            callbacks.extend(mlflow_callbacks)
            config["callbacks"] = callbacks
        if in_args:
            args = (args[0], config) + args[2:]
        else:
            kwargs["config"] = config
        return args, kwargs

    if func_name == "__call__":
        # `callbacks` is the third positional argument of chain.__call__ function
        # https://github.com/langchain-ai/langchain/blob/7d444724d7582386de347fb928619c2243bd0e55/libs/langchain/langchain/chains/base.py#L320
        if len(args) >= 3:
            callbacks = args[2] or []
            callbacks.extend(mlflow_callbacks)
            args = args[:2] + (callbacks,) + args[3:]
        else:
            callbacks = kwargs.get("callbacks") or []
            callbacks.extend(mlflow_callbacks)
            kwargs["callbacks"] = callbacks
        return args, kwargs

    # https://github.com/langchain-ai/langchain/blob/7d444724d7582386de347fb928619c2243bd0e55/libs/core/langchain_core/retrievers.py#L173
    if func_name == "get_relevant_documents":
        callbacks = kwargs.get("callbacks") or []
        callbacks.extend(mlflow_callbacks)
        kwargs["callbacks"] = callbacks
        return args, kwargs


def _runnable_with_retriever(model):
    from langchain.schema import BaseRetriever

    with contextlib.suppress(ImportError):
        from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnableSequence
        from langchain.schema.runnable.passthrough import RunnableAssign

        if isinstance(model, RunnableBranch):
            return any(_runnable_with_retriever(runnable) for _, runnable in model.branches)

        if isinstance(model, RunnableParallel):
            return any(_runnable_with_retriever(runnable) for runnable in model.steps.values())

        if isinstance(model, RunnableSequence):
            return any(_runnable_with_retriever(runnable) for runnable in model.steps)

        if isinstance(model, RunnableAssign):
            return _runnable_with_retriever(model.mapper)

    return isinstance(model, BaseRetriever)


def _chain_with_retriever(model):
    with contextlib.suppress(ImportError):
        from langchain.chains import RetrievalQA

        return isinstance(model, RetrievalQA)
    return False


def get_mlflow_langchain_tracer():
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.documents import Document
    from langchain_core.load import dumpd
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import (
        ChatGeneration,
        ChatGenerationChunk,
        GenerationChunk,
        LLMResult,
    )
    from tenacity import RetryCallState

    from mlflow.entities import ExceptionEvent, SpanEvent, SpanType
    from mlflow.entities.span_status import SpanStatus
    from mlflow.entities.trace_status import TraceStatus
    from mlflow.tracing.types.wrapper import MLflowSpanWrapper

    class MlflowLangchainTracer(BaseCallbackHandler, metaclass=ExceptionSafeAbstractClass):
        """
        Callback for auto-logging artifacts.
        We need to inherit ExceptionSafeAbstractClass to avoid invalid new
        input arguments added to original function call.
        Args:
            tracking_uri: MLflow tracking server uri.
            experiment_name: Name of the experiment.
            run_id: Id of the run to log the artifacts.
        """

        def __init__(self):
            super().__init__()
            self._mlflow_client = MlflowClient()
            self._run_span_mapping: Dict[str, MLflowSpanWrapper] = {}

        def _get_span_by_run_id(self, run_id: UUID) -> Optional[MLflowSpanWrapper]:
            if span := self._run_span_mapping.get(str(run_id)):
                return span
            raise MlflowException(f"Span for run_id {run_id!s} not found.")

        def _start_span(
            self,
            span_name: str,
            parent_run_id: Optional[UUID],
            span_type: str,
            run_id: UUID,
            inputs: Optional[Dict[str, Any]] = None,
            attributes: Optional[Dict[str, Any]] = None,
        ) -> MLflowSpanWrapper:
            """Start MLflow Span (or Trace if it is root component)"""
            parent = self._get_span_by_run_id(parent_run_id) if parent_run_id else None
            if parent:
                span = self._mlflow_client.start_span(
                    name=span_name,
                    request_id=parent.request_id,
                    parent_span_id=parent.span_id,
                    span_type=span_type,
                    inputs=inputs,
                    attributes=attributes,
                )
            else:
                # When parent_run_id is None, this is root component so start trace
                span = self._mlflow_client.start_trace(
                    name=span_name, inputs=inputs, attributes=attributes
                )
            self._run_span_mapping[str(run_id)] = span
            return span

        def _end_span(
            self,
            span: MLflowSpanWrapper,
            outputs=None,
            attributes=None,
            status=SpanStatus(TraceStatus.OK),
        ):
            """Close MLflow Span (or Trace if it is root component)"""
            self._mlflow_client.end_span(
                request_id=span.request_id,
                span_id=span.span_id,
                outputs=outputs,
                attributes=attributes,
                status=status,
            )

        def _reset(self):
            self._run_span_mapping = {}

        @override
        def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages: List[List[BaseMessage]],
            *,
            run_id: UUID,
            tags: Optional[List[str]] = None,
            parent_run_id: Optional[UUID] = None,
            metadata: Optional[Dict[str, Any]] = None,
            name: Optional[str] = None,
            **kwargs: Any,
        ):
            """Run when a chat model starts running."""
            if metadata:
                kwargs.update({"metadata": metadata})
            llm_inputs = {"messages": [[dumpd(msg) for msg in batch] for batch in messages]}
            self._start_span(
                span_name=name or "chat model",
                parent_run_id=parent_run_id,
                # we use LLM for chat models as well
                span_type=SpanType.LLM,
                run_id=run_id,
                inputs=llm_inputs,
                attributes=kwargs,
            )

        @override
        def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: List[str],
            *,
            run_id: UUID,
            tags: Optional[List[str]] = None,
            parent_run_id: Optional[UUID] = None,
            metadata: Optional[Dict[str, Any]] = None,
            name: Optional[str] = None,
            **kwargs: Any,
        ) -> None:
            """Run when LLM (non-chat models) starts running."""
            inputs = {"prompts": prompts}
            if metadata:
                kwargs.update({"metadata": metadata})
            self._start_span(
                span_name=name or "llm",
                parent_run_id=parent_run_id,
                span_type=SpanType.LLM,
                run_id=run_id,
                inputs=inputs,
                attributes=kwargs,
            )

        @override
        def on_llm_new_token(
            self,
            token: str,
            *,
            chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ):
            """Run on new LLM token. Only available when streaming is enabled."""
            llm_span = self._get_span_by_run_id(run_id)
            event_kwargs = {"token": token}
            if chunk:
                event_kwargs["chunk"] = chunk
            llm_span.add_event(
                SpanEvent(
                    name="new_token",
                    attributes=event_kwargs,
                )
            )

        @override
        def on_retry(
            self,
            retry_state: RetryCallState,
            *,
            run_id: UUID,
            **kwargs: Any,
        ):
            """Run on a retry event."""
            span = self._get_span_by_run_id(run_id)
            retry_d: Dict[str, Any] = {
                "slept": retry_state.idle_for,
                "attempt": retry_state.attempt_number,
            }
            if retry_state.outcome is None:
                retry_d["outcome"] = "N/A"
            elif retry_state.outcome.failed:
                retry_d["outcome"] = "failed"
                exception = retry_state.outcome.exception()
                retry_d["exception"] = str(exception)
                retry_d["exception_type"] = exception.__class__.__name__
            else:
                retry_d["outcome"] = "success"
                retry_d["result"] = str(retry_state.outcome.result())
            span.add_event(
                SpanEvent(
                    name="retry",
                    attributes=retry_d,
                )
            )

        @override
        def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any):
            """End the span for an LLM run."""
            llm_span = self._get_span_by_run_id(run_id)
            outputs = response.dict()
            for i, generations in enumerate(response.generations):
                for j, generation in enumerate(generations):
                    output_generation = outputs["generations"][i][j]
                    if "message" in output_generation:
                        output_generation["message"] = dumpd(
                            cast(ChatGeneration, generation).message
                        )
            self._end_span(llm_span, outputs=outputs)

        @override
        def on_llm_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            **kwargs: Any,
        ):
            """Handle an error for an LLM run."""
            llm_span = self._get_span_by_run_id(run_id)
            llm_span.add_event(ExceptionEvent(error))
            self._end_span(llm_span, status=SpanStatus(TraceStatus.ERROR, str(error)))

        def _get_chain_inputs(self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
            return inputs if isinstance(inputs, dict) else {"input": inputs}

        @override
        def on_chain_start(
            self,
            serialized: Dict[str, Any],
            inputs: Union[Dict[str, Any], Any],
            *,
            run_id: UUID,
            tags: Optional[List[str]] = None,
            parent_run_id: Optional[UUID] = None,
            metadata: Optional[Dict[str, Any]] = None,
            run_type: Optional[str] = None,
            name: Optional[str] = None,
            **kwargs: Any,
        ):
            """Start span for a chain run."""
            if metadata:
                kwargs.update({"metadata": metadata})
            # not considering streaming events for now
            self._start_span(
                span_name=name or "chain",
                parent_run_id=parent_run_id,
                span_type=SpanType.CHAIN,
                run_id=run_id,
                inputs=self._get_chain_inputs(inputs),
                attributes=kwargs,
            )

        @override
        def on_chain_end(
            self,
            outputs: Dict[str, Any],
            *,
            run_id: UUID,
            inputs: Optional[Union[Dict[str, Any], Any]] = None,
            **kwargs: Any,
        ):
            """Run when chain ends running."""
            chain_span = self._get_span_by_run_id(run_id)
            if inputs:
                chain_span.set_inputs(self._get_chain_inputs(inputs))
            self._end_span(chain_span, outputs=outputs)

        @override
        def on_chain_error(
            self,
            error: BaseException,
            *,
            inputs: Optional[Union[Dict[str, Any], Any]] = None,
            run_id: UUID,
            **kwargs: Any,
        ):
            """Run when chain errors."""
            chain_span = self._get_span_by_run_id(run_id)
            if inputs:
                chain_span.set_inputs(self._get_chain_inputs(inputs))
            chain_span.add_event(ExceptionEvent(error))
            self._end_span(chain_span, status=SpanStatus(TraceStatus.ERROR, str(error)))

        @override
        def on_tool_start(
            self,
            serialized: Dict[str, Any],
            input_str: str,
            *,
            run_id: UUID,
            tags: Optional[List[str]] = None,
            parent_run_id: Optional[UUID] = None,
            metadata: Optional[Dict[str, Any]] = None,
            name: Optional[str] = None,
            inputs: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ):
            """Start span for a tool run."""
            if metadata:
                kwargs.update({"metadata": metadata})
            self._start_span(
                span_name=name or "tool",
                parent_run_id=parent_run_id,
                span_type=SpanType.TOOL,
                run_id=run_id,
                inputs={"input_str": input_str},
                attributes=kwargs,
            )

        @override
        def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any):
            """Run when tool ends running."""
            tool_span = self._get_span_by_run_id(run_id)
            self._end_span(tool_span, outputs={"output": str(output)})

        @override
        def on_tool_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            **kwargs: Any,
        ):
            """Run when tool errors."""
            tool_span = self._get_span_by_run_id(run_id)
            tool_span.add_event(ExceptionEvent(error))
            self._end_span(tool_span, status=SpanStatus(TraceStatus.ERROR, str(error)))

        @override
        def on_retriever_start(
            self,
            serialized: Dict[str, Any],
            query: str,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            name: Optional[str] = None,
            **kwargs: Any,
        ):
            """Run when Retriever starts running."""
            if metadata:
                kwargs.update({"metadata": metadata})
            self._start_span(
                span_name=name or "retriever",
                parent_run_id=parent_run_id,
                span_type=SpanType.RETRIEVER,
                run_id=run_id,
                inputs={"query": query},
                attributes=kwargs,
            )

        @override
        def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, **kwargs: Any):
            """Run when Retriever ends running."""
            retriever_span = self._get_span_by_run_id(run_id)
            self._end_span(retriever_span, outputs={"documents": documents})

        @override
        def on_retriever_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            **kwargs: Any,
        ):
            """Run when Retriever errors."""
            retriever_span = self._get_span_by_run_id(run_id)
            retriever_span.add_event(ExceptionEvent(error))
            self._end_span(retriever_span, status=SpanStatus(TraceStatus.ERROR, str(error)))

        @override
        def on_agent_action(
            self,
            action: AgentAction,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> Any:
            """Run on agent action."""
            kwargs.update({"log": action.log})
            self._start_span(
                span_name=action.tool,
                parent_run_id=parent_run_id,
                span_type=SpanType.AGENT,
                run_id=run_id,
                inputs={"tool_input": action.tool_input},
                attributes=kwargs,
            )

        @override
        def on_agent_finish(
            self,
            finish: AgentFinish,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> Any:
            """Run on agent end."""
            agent_span = self._get_span_by_run_id(run_id)
            kwargs.update({"log": finish.log})
            self._end_span(agent_span, outputs=finish.return_values, attributes=kwargs)

        @override
        def on_text(
            self,
            text: str,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> Any:
            """Run on arbitrary text."""
            try:
                span = self._get_span_by_run_id(run_id)
            except MlflowException:
                _logger.warning("Span not found for text event. Skipping text event logging.")
            else:
                span.add_event(
                    SpanEvent(
                        "text",
                        attributes={"text": text},
                    )
                )

        def flush_tracker(self):
            mlflow.get_traces()
            self._reset()

    return MlflowLangchainTracer


def patched_inference(func_name, original, self, *args, **kwargs):
    """
    A patched implementation of langchain models inference process which enables logging the
    following parameters, metrics and artifacts:

    - model
    - metrics
    - data

    We patch either `invoke` or `__call__` function for different models
    based on their usage.
    """

    import langchain
    from langchain_community.callbacks import MlflowCallbackHandler

    class _MlflowLangchainCallback(MlflowCallbackHandler, metaclass=ExceptionSafeAbstractClass):
        """
        Callback for auto-logging metrics and parameters.
        We need to inherit ExceptionSafeAbstractClass to avoid invalid new
        input arguments added to original function call.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    _lc_version = Version(langchain.__version__)
    if not MIN_REQ_VERSION <= _lc_version <= MAX_REQ_VERSION:
        warnings.warn(
            "Autologging is known to be compatible with langchain versions between "
            f"{MIN_REQ_VERSION} and {MAX_REQ_VERSION} and may not succeed with packages "
            "outside this range."
        )

    run_id = getattr(self, "run_id", None)
    active_run = mlflow.active_run()
    if run_id is None:
        # only log the tags once
        extra_tags = get_autologging_config(mlflow.langchain.FLAVOR_NAME, "extra_tags", None)
        # include run context tags
        resolved_tags = context_registry.resolve_tags(extra_tags)
        tags = _resolve_extra_tags(mlflow.langchain.FLAVOR_NAME, resolved_tags)
        if active_run:
            run_id = active_run.info.run_id
            mlflow.MlflowClient().log_batch(
                run_id=run_id,
                tags=[RunTag(key, str(value)) for key, value in tags.items()],
            )
    else:
        tags = None
    # TODO: test adding callbacks works
    # Use session_id-inference_id as artifact directory where mlflow
    # callback logs artifacts into, to avoid overriding artifacts
    session_id = getattr(self, "session_id", uuid.uuid4().hex)
    inference_id = getattr(self, "inference_id", 0)
    mlflow_callback = _MlflowLangchainCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        run_id=run_id,
        artifacts_dir=f"artifacts-{session_id}-{inference_id}",
        tags=tags,
    )
    mlflow_tracer = get_mlflow_langchain_tracer()()
    args, kwargs = _inject_mlflow_callbacks(
        func_name, [mlflow_callback, mlflow_tracer], args, kwargs
    )
    with disable_autologging():
        result = original(self, *args, **kwargs)

    try:
        mlflow_callback.flush_tracker()
    except Exception as e:
        if _MLFLOW_TESTING.get():
            raise
        _logger.warning(f"Failed to flush mlflow callback due to error {e}.")
    finally:
        # Terminate the run if it is not managed by the user
        if active_run is None or active_run.info.run_id != mlflow_callback.run_id:
            mlflow.MlflowClient().set_terminated(mlflow_callback.mlflg.run_id)
    try:
        mlflow_tracer.flush_tracker()
    except Exception as e:
        if _MLFLOW_TESTING.get():
            raise
        _logger.warning(f"Failed to flush mlflow tracer due to error {e}.")

    log_models = get_autologging_config(mlflow.langchain.FLAVOR_NAME, "log_models", False)
    log_input_examples = get_autologging_config(
        mlflow.langchain.FLAVOR_NAME, "log_input_examples", False
    )
    log_model_signatures = get_autologging_config(
        mlflow.langchain.FLAVOR_NAME, "log_model_signatures", False
    )
    input_example = None
    if log_models and not hasattr(self, "model_logged"):
        if (
            (func_name == "get_relevant_documents")
            or _runnable_with_retriever(self)
            or _chain_with_retriever(self)
        ):
            _logger.info(UNSUPPORT_LOG_MODEL_MESSAGE)
        else:
            # warn user in case we did't capture some cases where retriever is used
            warnings.warn(UNSUPPORT_LOG_MODEL_MESSAGE)
            if log_input_examples:
                input_example = deepcopy(
                    _get_input_data_from_function(func_name, self, args, kwargs)
                )
                if not log_model_signatures:
                    _logger.info(
                        "Signature is automatically generated for logged model if "
                        "input_example is provided. To disable log_model_signatures, "
                        "please also disable log_input_examples."
                    )

            registered_model_name = get_autologging_config(
                mlflow.langchain.FLAVOR_NAME, "registered_model_name", None
            )
            try:
                with disable_autologging():
                    mlflow.langchain.log_model(
                        self,
                        "model",
                        input_example=input_example,
                        registered_model_name=registered_model_name,
                        run_id=mlflow_callback.mlflg.run_id,
                    )
            except Exception as e:
                _logger.warning(f"Failed to log model due to error {e}.")
            if _update_langchain_model_config(self):
                self.model_logged = True

    # Even if the model is not logged, we keep a single run per model
    if _update_langchain_model_config(self):
        if not hasattr(self, "run_id"):
            self.run_id = mlflow_callback.mlflg.run_id
        if not hasattr(self, "session_id"):
            self.session_id = session_id
        self.inference_id = inference_id + 1

    log_inputs_outputs = get_autologging_config(
        mlflow.langchain.FLAVOR_NAME, "log_inputs_outputs", False
    )
    if log_inputs_outputs:
        if input_example is None:
            input_data = deepcopy(_get_input_data_from_function(func_name, self, args, kwargs))
            if input_data is None:
                _logger.info("Input data gathering failed, only log inference results.")
        else:
            input_data = input_example
        try:
            data_dict = _combine_input_and_output(input_data, result, self.session_id, func_name)
        except Exception as e:
            _logger.warning(
                f"Failed to log inputs and outputs into `{INFERENCE_FILE_NAME}` "
                f"file due to error {e}."
            )
        mlflow.log_table(data_dict, INFERENCE_FILE_NAME, run_id=mlflow_callback.mlflg.run_id)

    return result
