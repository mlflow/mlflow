import contextlib
import inspect
import logging
import os
import random
import string
import sys
import traceback
import uuid
import warnings
from copy import deepcopy
from datetime import datetime, timezone
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
from mlflow.tracking.fluent import _get_experiment_id
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
BASIC_METRICS = [
    "step",
    "starts",
    "ends",
    "errors",
    "text_counts",
    "chain_starts",
    "chain_ends",
    "llm_starts",
    "llm_ends",
    "llm_new_tokens",
    "tool_starts",
    "tool_ends",
    "agent_starts",
    "agent_ends",
    "retriever_starts",
    "retriever_ends",
]

TEXT_COMPLEXITY_METRICS = [
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "smog_index",
    "coleman_liau_index",
    "automated_readability_index",
    "dale_chall_readability_score",
    "difficult_words",
    "linsear_write_formula",
    "gunning_fog",
    "fernandez_huerta",
    "szigriszt_pazos",
    "gutierrez_polini",
    "crawford",
    "gulpease_index",
    "osman",
]


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


def _inject_mlflow_callback(func_name, mlflow_callback, args, kwargs):
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
            callbacks = [mlflow_callback]
            config = RunnableConfig(callbacks=callbacks)
        else:
            callbacks = config.get("callbacks") or []
            callbacks.append(mlflow_callback)
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
            callbacks.append(mlflow_callback)
            args = args[:2] + (callbacks,) + args[3:]
        else:
            callbacks = kwargs.get("callbacks") or []
            callbacks.append(mlflow_callback)
            kwargs["callbacks"] = callbacks
        return args, kwargs

    # https://github.com/langchain-ai/langchain/blob/7d444724d7582386de347fb928619c2243bd0e55/libs/core/langchain_core/retrievers.py#L173
    if func_name == "get_relevant_documents":
        callbacks = kwargs.get("callbacks") or []
        callbacks.append(mlflow_callback)
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


def get_mlflow_langchain_callback():
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

    from mlflow.tracing.types.model import Trace
    from mlflow.tracing.types.wrapper import MLflowSpanWrapper, SpanType, Status, StatusCode

    class MLflowLangchainCallback(BaseCallbackHandler, metaclass=ExceptionSafeAbstractClass):
        """
        Callback for auto-logging artifacts.
        We need to inherit ExceptionSafeAbstractClass to avoid invalid new
        input arguments added to original function call.
        Args:
            tracking_uri: MLflow tracking server uri.
            experiment_name: Name of the experiment.
            run_id: Id of the run to log the artifacts.
        """

        def __init__(
            self,
            experiment_name="langchain_autologging",
            run_id=None,
            *,
            tags: Optional[Dict[str, Any]] = None,
            artifacts_dir: str = "",
        ):
            super().__init__()
            self._mlflow_client = MlflowClient()
            # create experiment if not exists
            if "DATABRICKS_RUNTIME_VERSION" in os.environ:
                mlflow.set_tracking_uri("databricks")
                self._experiment_id = _get_experiment_id()
            else:
                if experiment := self._mlflow_client.get_experiment_by_name(experiment_name):
                    self._experiment_id = experiment.experiment_id
                else:
                    self._experiment_id = self._mlflow_client.create_experiment(experiment_name)
            # create run if run_id is None
            if run_id is None:
                rname = "".join(random.choices(string.ascii_uppercase + string.digits, k=7))
                run = self._mlflow_client.create_run(
                    self._experiment_id, run_name=f"langchain-{rname}", tags=tags
                )
                run_id = run.info.run_id
            self.run_id = run_id
            self._run_span_mapping: Dict[str, MLflowSpanWrapper] = {}
            self.artifacts_dir = artifacts_dir
            self.metrics = {key: 0 for key in BASIC_METRICS}

        def _get_span_by_run_id(self, run_id: Union[str, UUID]) -> Optional[MLflowSpanWrapper]:
            if span := self._run_span_mapping.get(str(run_id)):
                return span
            raise MlflowException(f"Span for run_id {run_id!s} not found.")

        def _start_span(
            self,
            span_name: str,
            parent_run_id: Optional[Union[str, UUID]],
            span_type: str,
            run_id: Union[str, UUID],
            inputs: Optional[Dict[str, Any]] = None,
            attributes: Optional[Dict[str, Any]] = None,
        ) -> MLflowSpanWrapper:
            """Start MLflow Span (or Trace if it is root component)"""
            parent = self._get_span_by_run_id(parent_run_id) if parent_run_id else None
            if parent:
                span = self._mlflow_client.start_span(
                    name=span_name,
                    trace_id=parent.trace_id,
                    parent_span_id=parent.span_id,
                    span_type=span_type,
                    inputs=inputs,
                    attributes=attributes,
                )
            else:
                # When parent_run_id is None, this is root component so start trace
                span = self._mlflow_client.start_trace(name=span_name)
            self._run_span_mapping[str(run_id)] = span
            return span

        def _end_span(
            self,
            span: MLflowSpanWrapper,
            outputs=None,
            attributes=None,
            status=Status(StatusCode.OK, ""),
        ):
            """Close MLflow Span (or Trace if it is root component)"""
            self._mlflow_client.end_span(
                trace_id=span.trace_id,
                span_id=span.span_id,
                outputs=outputs,
                attributes=attributes,
                status=status,
            )

        def _reset(self):
            self._run_span_mapping = {}

        def _log_artifact(self, data: Dict[str, Any], filename: str):
            """Log an artifact to the run."""
            mlflow.log_dict(
                data, os.path.join(self.artifacts_dir, f"{filename}.json"), run_id=self.run_id
            )

        def _analyze_text(self, text: str):
            try:
                import textstat
            except ImportError:
                warnings.warn(
                    "Please install `textstat` for generating text complexity metrics "
                    "for the text."
                )
                return
            else:
                return {key: getattr(textstat, key)(text) for key in TEXT_COMPLEXITY_METRICS}

        @staticmethod
        def _get_stacktrace(error: BaseException) -> str:
            """Get the stacktrace of the parent error."""
            msg = repr(error)
            try:
                if sys.version_info < (3, 10):
                    tb = traceback.format_exception(error.__class__, error, error.__traceback__)
                else:
                    tb = traceback.format_exception(error)
                return (msg + "\n\n".join(tb)).strip()
            except Exception:
                return msg

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
            kwargs.update({"serialized": serialized})
            llm_inputs = {"messages": [[dumpd(msg) for msg in batch] for batch in messages]}
            llm_span = self._start_span(
                span_name=name,
                parent_run_id=parent_run_id,
                # we use LLM for chat models as well
                span_type=SpanType.LLM,
                run_id=run_id,
                inputs=llm_inputs,
                attributes=kwargs,
            )
            llm_span.add_event("start", timestamp=int(datetime.now(timezone.utc).timestamp()))
            self._log_artifact({**llm_inputs, "kwargs": kwargs}, f"llm_start_{llm_span.span_id}")

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
            kwargs.update({"serialized": serialized})
            llm_span = self._start_span(
                span_name=name,
                parent_run_id=parent_run_id,
                span_type=SpanType.LLM,
                run_id=run_id,
                inputs=inputs,
                attributes=kwargs,
            )
            llm_span.add_event("start", timestamp=int(datetime.now(timezone.utc).timestamp()))
            self._log_artifact(
                {"prompts": prompts, "kwargs": kwargs}, f"llm_start_{llm_span.span_id}"
            )
            self.metrics["step"] += 1
            self.metrics["llm_starts"] += 1
            self.metrics["starts"] += 1

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
                name="new_token",
                timestamp=int(datetime.now(timezone.utc).timestamp()),
                attributes=event_kwargs,
            )
            self.metrics["step"] += 1
            self.metrics["llm_new_tokens"] += 1

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
                name="retry",
                timestamp=int(datetime.now(timezone.utc).timestamp()),
                attributes=retry_d,
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
            llm_span.add_event("end", timestamp=int(datetime.now(timezone.utc).timestamp()))
            self._end_span(llm_span, outputs=outputs)
            for i, generations in enumerate(response.generations):
                for j, generation in enumerate(generations):
                    self._log_artifact(
                        {"generation": generation.dict(), **self._analyze_text(generation.text)},
                        f"llm_end_{llm_span.span_id}_generation_{i}_{j}",
                    )
            self.metrics["step"] += 1
            self.metrics["llm_ends"] += 1
            self.metrics["ends"] += 1

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
            llm_span.add_event(
                "error",
                timestamp=int(datetime.now(timezone.utc).timestamp()),
                attributes={"error": str(error)},
            )
            error_message = self._get_stacktrace(error)
            self._end_span(llm_span, status=Status(StatusCode.ERROR, error_message))
            self.metrics["step"] += 1
            self.metrics["errors"] += 1

        def _get_chain_inputs(self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
            return inputs if isinstance(inputs, dict) else {"input": inputs}

        def _format_data_to_string(self, data: Any) -> str:
            if isinstance(data, dict):
                return ",".join([f"{k}={v}" for k, v in data.items()])
            if isinstance(data, list):
                return ",".join([str(x) for x in data])
            return str(data)

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
            kwargs.update({"serialized": serialized})
            # not considering streaming events for now
            chain_span = self._start_span(
                span_name=name,
                parent_run_id=parent_run_id,
                span_type=SpanType.CHAIN,
                run_id=run_id,
                inputs=self._get_chain_inputs(inputs),
                attributes=kwargs,
            )
            chain_span.add_event("start", timestamp=int(datetime.now(timezone.utc).timestamp()))
            self._log_artifact(
                {"inputs": self._format_data_to_string(inputs), "kwargs": kwargs},
                f"chain_start_{chain_span.span_id}",
            )
            self.metrics["step"] += 1
            self.metrics["chain_starts"] += 1
            self.metrics["starts"] += 1

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
            chain_span.add_event("end", timestamp=int(datetime.now(timezone.utc).timestamp()))
            if inputs:
                chain_span.set_inputs(self._get_chain_inputs(inputs))
            self._end_span(chain_span, outputs=outputs)
            self._log_artifact(
                {"outputs": self._format_data_to_string(outputs), "kwargs": kwargs},
                f"chain_end_{chain_span.span_id}",
            )
            self.metrics["step"] += 1
            self.metrics["chain_ends"] += 1
            self.metrics["ends"] += 1

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
            chain_span.add_event(
                "error",
                timestamp=int(datetime.now(timezone.utc).timestamp()),
                attributes={"error": str(error)},
            )
            if inputs:
                chain_span.set_inputs(self._get_chain_inputs(inputs))
            error_message = self._get_stacktrace(error)
            self._end_span(chain_span, status=Status(StatusCode.ERROR, error_message))
            self.metrics["step"] += 1
            self.metrics["errors"] += 1

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
            kwargs.update({"serialized": serialized})
            tool_span = self._start_span(
                span_name=name,
                parent_run_id=parent_run_id,
                span_type=SpanType.TOOL,
                run_id=run_id,
                inputs={"input_str": input_str},
                attributes=kwargs,
            )
            tool_span.add_event("start", timestamp=int(datetime.now(timezone.utc).timestamp()))
            self._log_artifact(
                {"input_str": input_str, "kwargs": kwargs}, f"tool_start_{tool_span.span_id}"
            )
            self.metrics["step"] += 1
            self.metrics["tool_starts"] += 1
            self.metrics["starts"] += 1

        @override
        def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any):
            """Run when tool ends running."""
            tool_span = self._get_span_by_run_id(run_id)
            tool_span.add_event("end", timestamp=int(datetime.now(timezone.utc).timestamp()))
            self._end_span(tool_span, outputs={"output": str(output)})
            self._log_artifact(
                {"output": str(output), "kwargs": kwargs}, f"tool_end_{tool_span.span_id}"
            )
            self.metrics["step"] += 1
            self.metrics["tool_ends"] += 1
            self.metrics["ends"] += 1

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
            tool_span.add_event(
                "error",
                timestamp=int(datetime.now(timezone.utc).timestamp()),
                attributes={"error": str(error)},
            )
            error_message = self._get_stacktrace(error)
            self._end_span(tool_span, status=Status(StatusCode.ERROR, error_message))
            self.metrics["step"] += 1
            self.metrics["errors"] += 1

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
            kwargs.update({"serialized": serialized})
            retriever_span = self._start_span(
                span_name=name,
                parent_run_id=parent_run_id,
                span_type=SpanType.RETRIEVER,
                run_id=run_id,
                inputs={"query": query},
                attributes=kwargs,
            )
            retriever_span.add_event("start", timestamp=int(datetime.now(timezone.utc).timestamp()))
            self._log_artifact(
                {"query": query, "kwargs": kwargs}, f"retriever_start_{retriever_span.span_id}"
            )
            self.metrics["step"] += 1
            self.metrics["retriever_starts"] += 1
            self.metrics["starts"] += 1

        @override
        def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, **kwargs: Any):
            """Run when Retriever ends running."""
            retriever_span = self._get_span_by_run_id(run_id)
            retriever_span.add_event("end", timestamp=int(datetime.now(timezone.utc).timestamp()))
            self._end_span(retriever_span, outputs={"documents": documents})
            retriever_documents = [
                {
                    "page_content": doc.page_content,
                    "metadata": {
                        k: (str(v) if not isinstance(v, list) else ",".join(str(x) for x in v))
                        for k, v in doc.metadata.items()
                    },
                }
                for doc in documents
            ]
            self._log_artifact(
                {"documents": retriever_documents}, f"retriever_end_{retriever_span.span_id}"
            )
            self.metrics["step"] += 1
            self.metrics["retriever_ends"] += 1
            self.metrics["ends"] += 1

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
            retriever_span.add_event(
                "error",
                timestamp=int(datetime.now(timezone.utc).timestamp()),
                attributes={"error": str(error)},
            )
            error_message = self._get_stacktrace(error)
            self._end_span(retriever_span, status=Status(StatusCode.ERROR, error_message))
            self.metrics["step"] += 1
            self.metrics["errors"] += 1

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
            agent_action_span = self._start_span(
                span_name=action.tool,
                parent_run_id=parent_run_id,
                span_type=SpanType.AGENT,
                run_id=run_id,
                inputs={"tool_input": action.tool_input},
                attributes=kwargs,
            )
            agent_action_span.add_event(
                "start", timestamp=int(datetime.now(timezone.utc).timestamp())
            )
            self._log_artifact(
                {
                    "tool": action.tool,
                    "tool_input": action.tool_input,
                    "log": action.log,
                    "kwargs": kwargs,
                },
                f"agent_action_{agent_action_span.span_id}",
            )
            self.metrics["step"] += 1
            self.metrics["agent_starts"] += 1
            self.metrics["starts"] += 1

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
            agent_span.add_event("end", timestamp=int(datetime.now(timezone.utc).timestamp()))
            kwargs.update({"log": finish.log})
            self._end_span(agent_span, outputs=finish.return_values, attributes=kwargs)
            self._log_artifact(
                {
                    "outputs": finish.return_values,
                    "log": finish.log,
                    "kwargs": kwargs,
                },
                f"agent_finish_{agent_span.span_id}",
            )
            self.metrics["step"] += 1
            self.metrics["agent_ends"] += 1
            self.metrics["ends"] += 1

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
                    "text",
                    timestamp=int(datetime.now(timezone.utc).timestamp()),
                    attributes={"text": text},
                )
            self.metrics["step"] += 1
            self.metrics["text_counts"] += 1

        # TODO: update this method to log traces to mlflow
        def _log_trace(self, trace: Trace):
            self._log_artifact(
                {"trace": trace.to_json()},
                f"trace_{trace.trace_info.trace_id}",
            )

        def flush_tracker(self):
            mlflow.log_metrics(self.metrics, run_id=self.run_id)
            # for now we only display the traces
            traces = mlflow.get_traces(1)
            if len(traces) == 0:
                _logger.warning("No valid traces found.")
            else:
                self._log_trace(traces[0])
            self._reset()

    return MLflowLangchainCallback


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
    args, kwargs = _inject_mlflow_callback(func_name, mlflow_callback, args, kwargs)
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
            mlflow.MlflowClient().set_terminated(mlflow_callback.run_id)

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
                        run_id=mlflow_callback.run_id,
                    )
            except Exception as e:
                _logger.warning(f"Failed to log model due to error {e}.")
            if _update_langchain_model_config(self):
                self.model_logged = True

    # Even if the model is not logged, we keep a single run per model
    if _update_langchain_model_config(self):
        if not hasattr(self, "run_id"):
            self.run_id = mlflow_callback.run_id
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
        mlflow.log_table(data_dict, INFERENCE_FILE_NAME, run_id=mlflow_callback.run_id)

    return result
