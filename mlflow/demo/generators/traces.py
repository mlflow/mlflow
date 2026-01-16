"""Demo generator for LLM traces."""

from __future__ import annotations

import hashlib
import logging
import random
from datetime import datetime, timedelta, timezone
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
    PROMPT_TRACES,
    RAG_TRACES,
    SESSION_TRACES,
    DemoTrace,
)
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey

_logger = logging.getLogger(__name__)

DEMO_VERSION_TAG = "mlflow.demo.version"
DEMO_TRACE_TYPE_TAG = "mlflow.demo.trace_type"

# Total number of traces for timestamp distribution
# 2 RAG + 2 agent + 6 prompt + 7 session = 17 per version = 34 total
_TOTAL_TRACES_PER_VERSION = 17


def _get_trace_timestamp(trace_index: int, version: str) -> tuple[int, int]:
    """Get deterministic start and end timestamps for a trace.

    Distributes traces over the last 7 days with a deterministic pattern
    based on the trace index and version. This ensures the demo dashboard
    shows activity across the time range.

    Args:
        trace_index: Index of the trace (0-based) within its version set.
        version: "v1" or "v2" - v1 traces are earlier, v2 traces are later.

    Returns:
        Tuple of (start_time_ns, end_time_ns).
    """
    now = datetime.now(timezone.utc)
    seven_days_ago = now - timedelta(days=7)

    # v1 traces are distributed in the first half of the week (days 0-3)
    # v2 traces are distributed in the second half (days 3-7)
    if version == "v1":
        day_offset = (trace_index * 3.5) / _TOTAL_TRACES_PER_VERSION
    else:
        day_offset = 3.5 + (trace_index * 3.5) / _TOTAL_TRACES_PER_VERSION

    # Add some deterministic "randomness" based on trace index for realistic spread
    hash_input = f"{trace_index}:{version}"
    hash_val = int(hashlib.md5(hash_input.encode(), usedforsecurity=False).hexdigest()[:8], 16)
    hour_offset = (hash_val % 24) / 24  # Spread across hours of the day
    minute_offset = ((hash_val >> 8) % 60) / (60 * 24)  # Spread across minutes

    trace_time = seven_days_ago + timedelta(days=day_offset + hour_offset + minute_offset)

    # Duration based on trace type (50ms to 2s)
    duration_ms = 50 + (hash_val % 1950)

    start_ns = int(trace_time.timestamp() * 1_000_000_000)
    end_ns = start_ns + (duration_ms * 1_000_000)

    return start_ns, end_ns


def _estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation: ~4 chars per token)."""
    return max(1, len(text) // 4)


class TracesDemoGenerator(BaseDemoGenerator):
    """Generates demo traces for the MLflow UI.

    Creates two sets of traces showing agent improvement:
    - V1 traces: Initial/baseline agent (uses v1_response)
    - V2 traces: Improved agent after updates (uses v2_response)

    Both versions use the same inputs but produce different outputs,
    simulating an agent improvement workflow.

    Trace types generated:
    - RAG: Document retrieval and generation pipeline
    - Agent: Tool-using agent with function calls
    - Prompt: Prompt template-based generation
    - Session: Multi-turn conversation sessions
    """

    name = DemoFeature.TRACES
    version = 2  # Bumped for timestamp and token count changes

    def generate(self) -> DemoResult:
        experiment = mlflow.set_experiment(DEMO_EXPERIMENT_NAME)

        v1_trace_ids = self._generate_trace_set("v1")
        v2_trace_ids = self._generate_trace_set("v2")

        all_trace_ids = v1_trace_ids + v2_trace_ids

        return DemoResult(
            feature=self.name,
            entity_ids=all_trace_ids,
            navigation_url=f"#/experiments/{experiment.experiment_id}/traces",
        )

    def _generate_trace_set(self, version: Literal["v1", "v2"]) -> list[str]:
        """Generate a complete set of traces for the given version."""
        trace_ids = []
        trace_index = 0

        # RAG traces (2)
        for trace_def in RAG_TRACES:
            start_ns, end_ns = _get_trace_timestamp(trace_index, version)
            if trace_id := self._create_rag_trace(trace_def, version, start_ns, end_ns):
                trace_ids.append(trace_id)
            trace_index += 1

        # Agent traces (2)
        for trace_def in AGENT_TRACES:
            start_ns, end_ns = _get_trace_timestamp(trace_index, version)
            if trace_id := self._create_agent_trace(trace_def, version, start_ns, end_ns):
                trace_ids.append(trace_id)
            trace_index += 1

        # Prompt traces (6)
        for trace_def in PROMPT_TRACES:
            start_ns, end_ns = _get_trace_timestamp(trace_index, version)
            if trace_id := self._create_prompt_trace(trace_def, version, start_ns, end_ns):
                trace_ids.append(trace_id)
            trace_index += 1

        # Session traces (7 across 3 sessions)
        trace_ids.extend(self._create_session_traces(version, trace_index))

        return trace_ids

    def _data_exists(self) -> bool:
        from mlflow.tracking._tracking_service.utils import _get_store

        store = _get_store()
        try:
            experiment = store.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
            if experiment is None or experiment.lifecycle_stage != "active":
                return False
            traces = mlflow.search_traces(
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
                max_results=200,
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

    def _create_rag_trace(
        self,
        trace_def: DemoTrace,
        version: Literal["v1", "v2"],
        start_ns: int,
        end_ns: int,
    ) -> str | None:
        """Create a RAG pipeline trace: embed -> retrieve -> generate."""
        response = self._get_response(trace_def, version)
        prompt_tokens = _estimate_tokens(trace_def.query) + 50  # query + system prompt
        completion_tokens = _estimate_tokens(response)

        # Calculate intermediate timestamps
        total_duration = end_ns - start_ns
        embed_end = start_ns + int(total_duration * 0.1)
        retrieve_end = embed_end + int(total_duration * 0.2)
        llm_start = retrieve_end
        llm_end = end_ns - int(total_duration * 0.05)

        root = mlflow.start_span_no_context(
            name="rag_pipeline",
            span_type=SpanType.CHAIN,
            inputs={"query": trace_def.query},
            metadata={DEMO_VERSION_TAG: version, DEMO_TRACE_TYPE_TAG: "rag"},
            start_time_ns=start_ns,
        )

        # Embedding span
        embed = mlflow.start_span_no_context(
            name="embed_query",
            span_type=SpanType.EMBEDDING,
            parent_span=root,
            inputs={"text": trace_def.query},
            start_time_ns=start_ns + 1000,
        )
        embedding = [random.uniform(-1, 1) for _ in range(384)]
        embed.set_outputs({"embedding": embedding[:5], "dimensions": 384})
        embed.end(end_time_ns=embed_end)

        # Retrieval span
        retrieve = mlflow.start_span_no_context(
            name="retrieve_docs",
            span_type=SpanType.RETRIEVER,
            parent_span=root,
            inputs={"embedding": embedding[:5], "top_k": 3},
            start_time_ns=embed_end + 1000,
        )
        docs = [
            {"id": f"doc_{i}", "score": round(0.7 + random.uniform(0, 0.25), 2)} for i in range(3)
        ]
        retrieve.set_outputs({"documents": docs})
        retrieve.end(end_time_ns=retrieve_end)

        # LLM generation span
        llm = mlflow.start_span_no_context(
            name="generate_response",
            span_type=SpanType.LLM,
            parent_span=root,
            inputs={
                "messages": [
                    {"role": "system", "content": "You are an MLflow assistant."},
                    {"role": "user", "content": trace_def.query},
                ],
                "context": docs,
            },
            attributes={
                SpanAttributeKey.CHAT_USAGE: {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            },
            start_time_ns=llm_start,
        )
        llm.set_outputs({"response": response})
        llm.end(end_time_ns=llm_end)

        root.set_outputs({"response": response})
        root.end(end_time_ns=end_ns)

        return root.trace_id

    def _create_agent_trace(
        self,
        trace_def: DemoTrace,
        version: Literal["v1", "v2"],
        start_ns: int,
        end_ns: int,
    ) -> str | None:
        """Create an agent trace with tool calls."""
        response = self._get_response(trace_def, version)
        prompt_tokens = _estimate_tokens(trace_def.query) + 100
        completion_tokens = _estimate_tokens(response)

        total_duration = end_ns - start_ns
        tool_duration = int(total_duration * 0.3)
        llm_start = start_ns + tool_duration + 10000

        root = mlflow.start_span_no_context(
            name="agent",
            span_type=SpanType.AGENT,
            inputs={"query": trace_def.query},
            metadata={DEMO_VERSION_TAG: version, DEMO_TRACE_TYPE_TAG: "agent"},
            start_time_ns=start_ns,
        )

        # Tool spans
        tool_start = start_ns + 5000
        for i, tool in enumerate(trace_def.tools):
            tool_span = mlflow.start_span_no_context(
                name=tool.name,
                span_type=SpanType.TOOL,
                parent_span=root,
                inputs=tool.input,
                start_time_ns=tool_start,
            )
            tool_span.set_outputs(tool.output)
            tool_span.end(end_time_ns=tool_start + tool_duration // len(trace_def.tools))
            tool_start += tool_duration // len(trace_def.tools) + 1000

        # LLM span
        llm = mlflow.start_span_no_context(
            name="generate_response",
            span_type=SpanType.LLM,
            parent_span=root,
            inputs={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant with tools."},
                    {"role": "user", "content": trace_def.query},
                ],
                "tool_results": [t.output for t in trace_def.tools],
            },
            attributes={
                SpanAttributeKey.CHAT_USAGE: {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            },
            start_time_ns=llm_start,
        )
        llm.set_outputs({"response": response})
        llm.end(end_time_ns=end_ns - 5000)

        root.set_outputs({"response": response})
        root.end(end_time_ns=end_ns)

        return root.trace_id

    def _create_prompt_trace(
        self,
        trace_def: DemoTrace,
        version: Literal["v1", "v2"],
        start_ns: int,
        end_ns: int,
    ) -> str | None:
        """Create a prompt-based trace showing template rendering and generation."""
        response = self._get_response(trace_def, version)

        if trace_def.prompt_template is None:
            return None

        rendered_prompt = trace_def.prompt_template.render()
        prompt_tokens = _estimate_tokens(rendered_prompt) + 20
        completion_tokens = _estimate_tokens(response)

        total_duration = end_ns - start_ns
        render_end = start_ns + int(total_duration * 0.1)
        llm_start = render_end + 1000

        root = mlflow.start_span_no_context(
            name="prompt_chain",
            span_type=SpanType.CHAIN,
            inputs={
                "query": trace_def.query,
                "template_variables": trace_def.prompt_template.variables,
            },
            metadata={DEMO_VERSION_TAG: version, DEMO_TRACE_TYPE_TAG: "prompt"},
            start_time_ns=start_ns,
        )

        # Prompt rendering span
        render = mlflow.start_span_no_context(
            name="render_prompt",
            span_type=SpanType.CHAIN,
            parent_span=root,
            inputs={
                "template": trace_def.prompt_template.template,
                "variables": trace_def.prompt_template.variables,
            },
            start_time_ns=start_ns + 1000,
        )
        render.set_outputs({"rendered_prompt": rendered_prompt})
        render.end(end_time_ns=render_end)

        # LLM generation span
        llm = mlflow.start_span_no_context(
            name="generate_response",
            span_type=SpanType.LLM,
            parent_span=root,
            inputs={
                "messages": [
                    {"role": "user", "content": rendered_prompt},
                ],
                "model": "gpt-4o-mini",
            },
            attributes={
                SpanAttributeKey.CHAT_USAGE: {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            },
            start_time_ns=llm_start,
        )
        llm.set_outputs({"response": response})
        llm.end(end_time_ns=end_ns - 5000)

        root.set_outputs({"response": response})
        root.end(end_time_ns=end_ns)

        return root.trace_id

    def _create_session_traces(self, version: Literal["v1", "v2"], start_index: int) -> list[str]:
        """Create multi-turn conversation session traces."""
        trace_ids = []
        current_session = None
        turn_counter = 0
        trace_index = start_index

        for trace_def in SESSION_TRACES:
            if trace_def.session_id != current_session:
                current_session = trace_def.session_id
                turn_counter = 0

            turn_counter += 1
            versioned_session_id = f"{trace_def.session_id}-{version}"

            start_ns, end_ns = _get_trace_timestamp(trace_index, version)
            if trace_id := self._create_session_turn_trace(
                trace_def, turn_counter, version, versioned_session_id, start_ns, end_ns
            ):
                trace_ids.append(trace_id)
            trace_index += 1

        return trace_ids

    def _create_session_turn_trace(
        self,
        trace_def: DemoTrace,
        turn: int,
        version: Literal["v1", "v2"],
        versioned_session_id: str,
        start_ns: int,
        end_ns: int,
    ) -> str | None:
        """Create a single turn in a conversation session."""
        response = self._get_response(trace_def, version)
        prompt_tokens = _estimate_tokens(trace_def.query) + 80
        completion_tokens = _estimate_tokens(response)

        total_duration = end_ns - start_ns
        tool_end = start_ns + int(total_duration * 0.3)
        llm_start = tool_end + 1000

        root = mlflow.start_span_no_context(
            name="chat_agent",
            span_type=SpanType.AGENT,
            inputs={"message": trace_def.query, "turn": turn},
            metadata={
                TraceMetadataKey.TRACE_SESSION: versioned_session_id,
                TraceMetadataKey.TRACE_USER: trace_def.session_user or "user",
                DEMO_VERSION_TAG: version,
                DEMO_TRACE_TYPE_TAG: "session",
            },
            start_time_ns=start_ns,
        )

        # Tool spans if any
        tool_start = start_ns + 5000
        for tool in trace_def.tools:
            tool_span = mlflow.start_span_no_context(
                name=tool.name,
                span_type=SpanType.TOOL,
                parent_span=root,
                inputs=tool.input,
                start_time_ns=tool_start,
            )
            tool_span.set_outputs(tool.output)
            tool_span.end(end_time_ns=tool_end)
            tool_start = tool_end + 1000

        # LLM span
        llm = mlflow.start_span_no_context(
            name="generate_response",
            span_type=SpanType.LLM,
            parent_span=root,
            inputs={
                "messages": [
                    {"role": "system", "content": "You are an MLflow assistant."},
                    {"role": "user", "content": trace_def.query},
                ],
                "model": "gpt-4o-mini",
            },
            attributes={
                SpanAttributeKey.CHAT_USAGE: {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            },
            start_time_ns=llm_start,
        )
        llm.set_outputs({"role": "assistant", "content": response})
        llm.end(end_time_ns=end_ns - 5000)

        root.set_outputs({"response": response})
        root.end(end_time_ns=end_ns)

        return root.trace_id
