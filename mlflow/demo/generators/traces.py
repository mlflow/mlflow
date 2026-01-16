"""Demo generator for LLM traces."""

from __future__ import annotations

import logging
import random
import time
from typing import Literal

import mlflow
from mlflow.demo.base import (
    DEMO_EXPERIMENT_NAME,
    BaseDemoGenerator,
    DemoFeature,
    DemoResult,
)
from mlflow.demo.data import (
    AGENT_TRACES,
    RAG_TRACES,
    SESSION_TRACES,
    DemoTrace,
)
from mlflow.entities import SpanType
from mlflow.tracing.constant import TraceMetadataKey

_logger = logging.getLogger(__name__)

DEMO_VERSION_TAG = "mlflow.demo.version"
DEMO_TRACE_TYPE_TAG = "mlflow.demo.trace_type"


def _simulate_latency(min_ms: int, max_ms: int) -> None:
    time.sleep(random.uniform(min_ms, max_ms) / 1000)


class TracesDemoGenerator(BaseDemoGenerator):
    """Generates demo traces for the MLflow UI.

    Creates two sets of traces showing agent improvement:
    - V1 traces: Initial/baseline agent (uses v1_response)
    - V2 traces: Improved agent after updates (uses v2_response)

    Both versions use the same inputs but produce different outputs,
    simulating an agent improvement workflow.
    """

    name = DemoFeature.TRACES
    version = 1

    def generate(self) -> DemoResult:
        experiment = mlflow.set_experiment(DEMO_EXPERIMENT_NAME)

        # Generate V1 traces (baseline)
        v1_trace_ids = self._generate_trace_set("v1")

        # Generate V2 traces (improved)
        v2_trace_ids = self._generate_trace_set("v2")

        all_trace_ids = v1_trace_ids + v2_trace_ids

        return DemoResult(
            feature=self.name,
            entity_ids=all_trace_ids,
            navigation_url=f"#/experiments/{experiment.experiment_id}/traces",
        )

    def _generate_trace_set(self, version: Literal["v1", "v2"]) -> list[str]:
        """Generate a complete set of traces for the given version."""
        trace_ids = [
            trace_id
            for trace_def in RAG_TRACES
            if (trace_id := self._create_rag_trace(trace_def, version))
        ]

        trace_ids.extend(
            trace_id
            for trace_def in AGENT_TRACES
            if (trace_id := self._create_agent_trace(trace_def, version))
        )

        trace_ids.extend(self._create_session_traces(version))

        return trace_ids

    def _data_exists(self) -> bool:
        from mlflow.tracking._tracking_service.utils import _get_store

        store = _get_store()
        try:
            experiment = store.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
            if experiment is None or experiment.lifecycle_stage != "active":
                return False
            client = mlflow.MlflowClient()
            traces = client.search_traces(
                locations=[experiment.experiment_id],
                max_results=1,
            )
            return len(traces) > 0
        except Exception:
            _logger.debug("Failed to check if demo data exists", exc_info=True)
            return False

    def delete_demo(self) -> None:
        from mlflow.tracking._tracking_service.utils import _get_store

        store = _get_store()
        try:
            experiment = store.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
            if experiment is None:
                return
            client = mlflow.MlflowClient()
            traces = client.search_traces(
                locations=[experiment.experiment_id],
                max_results=100,
            )
            if trace_ids := [trace.info.trace_id for trace in traces]:
                try:
                    client.delete_traces(
                        experiment_id=experiment.experiment_id,
                        trace_ids=trace_ids,
                    )
                except Exception:
                    pass
        except Exception:
            _logger.debug("Failed to delete demo traces", exc_info=True)

    def _get_response(self, trace_def: DemoTrace, version: Literal["v1", "v2"]) -> str:
        """Get the appropriate response based on version."""
        return trace_def.v1_response if version == "v1" else trace_def.v2_response

    def _create_rag_trace(self, trace_def: DemoTrace, version: Literal["v1", "v2"]) -> str | None:
        """Create a RAG pipeline trace: embed → retrieve → generate."""
        response = self._get_response(trace_def, version)

        with mlflow.start_span("rag_pipeline", span_type=SpanType.CHAIN) as root:
            root.set_inputs({"query": trace_def.query})

            trace_type = "baseline" if version == "v1" else "improved"
            mlflow.update_current_trace(
                metadata={DEMO_VERSION_TAG: version, DEMO_TRACE_TYPE_TAG: trace_type},
                tags={"demo": "true", "trace_type": trace_type},
            )

            with mlflow.start_span("embed_query", span_type=SpanType.EMBEDDING) as embed:
                embed.set_inputs({"text": trace_def.query})
                _simulate_latency(3, 8)
                embedding = [random.uniform(-1, 1) for _ in range(384)]
                embed.set_outputs({"embedding": embedding[:5], "dimensions": 384})

            with mlflow.start_span("retrieve_docs", span_type=SpanType.RETRIEVER) as retrieve:
                retrieve.set_inputs({"embedding": embedding[:5], "top_k": 3})
                _simulate_latency(10, 30)
                docs = [
                    {"id": f"doc_{i}", "score": round(random.uniform(0.7, 0.95), 2)}
                    for i in range(3)
                ]
                retrieve.set_outputs({"documents": docs})

            with mlflow.start_span("generate_response", span_type=SpanType.LLM) as llm:
                llm.set_inputs(
                    {
                        "messages": [
                            {"role": "system", "content": "You are an MLflow assistant."},
                            {"role": "user", "content": trace_def.query},
                        ],
                        "context": docs,
                    }
                )
                _simulate_latency(50, 200)
                llm.set_outputs(
                    {
                        "response": response,
                        "usage": {"prompt_tokens": 150, "completion_tokens": 80},
                    }
                )

            root.set_outputs({"response": response})

        return mlflow.get_last_active_trace_id()

    def _create_agent_trace(self, trace_def: DemoTrace, version: Literal["v1", "v2"]) -> str | None:
        """Create an agent trace with tool calls."""
        response = self._get_response(trace_def, version)

        with mlflow.start_span("agent", span_type=SpanType.AGENT) as root:
            root.set_inputs({"query": trace_def.query})

            trace_type = "baseline" if version == "v1" else "improved"
            mlflow.update_current_trace(
                metadata={DEMO_VERSION_TAG: version, DEMO_TRACE_TYPE_TAG: trace_type},
                tags={"demo": "true", "trace_type": trace_type},
            )

            for tool in trace_def.tools:
                with mlflow.start_span(tool.name, span_type=SpanType.TOOL) as tool_span:
                    tool_span.set_inputs(tool.input)
                    _simulate_latency(20, 100)
                    tool_span.set_outputs(tool.output)

            with mlflow.start_span("generate_response", span_type=SpanType.LLM) as llm:
                llm.set_inputs(
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant with tools.",
                            },
                            {"role": "user", "content": trace_def.query},
                        ],
                        "tool_results": [t.output for t in trace_def.tools],
                    }
                )
                _simulate_latency(50, 150)
                llm.set_outputs({"response": response})

            root.set_outputs({"response": response})

        return mlflow.get_last_active_trace_id()

    def _create_session_traces(self, version: Literal["v1", "v2"]) -> list[str]:
        """Create multi-turn conversation session traces."""
        trace_ids = []
        current_session = None
        turn_counter = 0

        for trace_def in SESSION_TRACES:
            if trace_def.session_id != current_session:
                current_session = trace_def.session_id
                turn_counter = 0

            turn_counter += 1
            # Append version suffix to session_id to keep v1 and v2 sessions separate
            versioned_session_id = f"{trace_def.session_id}-{version}"
            if trace_id := self._create_session_turn_trace(
                trace_def, turn_counter, version, versioned_session_id
            ):
                trace_ids.append(trace_id)

        return trace_ids

    def _create_session_turn_trace(
        self,
        trace_def: DemoTrace,
        turn: int,
        version: Literal["v1", "v2"],
        versioned_session_id: str,
    ) -> str | None:
        """Create a single turn in a conversation session."""
        response = self._get_response(trace_def, version)

        with mlflow.start_span("chat_agent", span_type=SpanType.AGENT) as root:
            root.set_inputs({"message": trace_def.query, "turn": turn})

            trace_type = "baseline" if version == "v1" else "improved"
            mlflow.update_current_trace(
                metadata={
                    TraceMetadataKey.TRACE_SESSION: versioned_session_id,
                    TraceMetadataKey.TRACE_USER: trace_def.session_user or "user",
                    DEMO_VERSION_TAG: version,
                    DEMO_TRACE_TYPE_TAG: trace_type,
                },
                tags={"demo": "true", "trace_type": trace_type},
            )

            for tool in trace_def.tools:
                with mlflow.start_span(tool.name, span_type=SpanType.TOOL) as tool_span:
                    tool_span.set_inputs(tool.input)
                    _simulate_latency(15, 50)
                    tool_span.set_outputs(tool.output)

            with mlflow.start_span("generate_response", span_type=SpanType.LLM) as llm:
                llm.set_inputs(
                    {
                        "messages": [
                            {"role": "system", "content": "You are an MLflow assistant."},
                            {"role": "user", "content": trace_def.query},
                        ],
                        "model": "gpt-4o-mini",
                    }
                )
                _simulate_latency(30, 120)
                llm.set_outputs({"role": "assistant", "content": response})

            root.set_outputs({"response": response})

        return mlflow.get_last_active_trace_id()
