from __future__ import annotations

import hashlib
import logging
import random
import re
from datetime import datetime, timedelta, timezone
from typing import Literal

import mlflow
from mlflow.demo.base import (
    DEMO_EXPERIMENT_NAME,
    DEMO_PROMPT_PREFIX,
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
from mlflow.tracking._tracking_service.utils import _get_store

_logger = logging.getLogger(__name__)

DEMO_VERSION_TAG = "mlflow.demo.version"
DEMO_TRACE_TYPE_TAG = "mlflow.demo.trace_type"

_TOTAL_TRACES_PER_VERSION = 17


def _get_trace_timestamps(trace_index: int, version: str) -> tuple[int, int]:
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

    if version == "v1":
        day_offset = (trace_index * 3.5) / _TOTAL_TRACES_PER_VERSION
    else:
        day_offset = 3.5 + (trace_index * 3.5) / _TOTAL_TRACES_PER_VERSION

    hash_input = f"{trace_index}:{version}"
    hash_val = int(hashlib.md5(hash_input.encode(), usedforsecurity=False).hexdigest()[:8], 16)
    hour_offset = (hash_val % 24) / 24
    minute_offset = ((hash_val >> 8) % 60) / (60 * 24)

    trace_time = seven_days_ago + timedelta(days=day_offset + hour_offset + minute_offset)

    duration_ms = 50 + (hash_val % 1950)

    start_ns = int(trace_time.timestamp() * 1_000_000_000)
    end_ns = start_ns + (duration_ms * 1_000_000)

    return start_ns, end_ns


def _estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation: ~4 chars per token)."""
    return max(1, len(text) // 4)


# Model names and approximate per-token pricing (USD per 1M tokens).
# Using three distinct models so the cost breakdown chart shows a nice distribution.
_DEMO_MODELS = ("gpt-5.2", "claude-sonnet-4-5", "gemini-3-pro")
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input $/1M tokens, output $/1M tokens)
    "gpt-5.2": (1.75, 14.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "gemini-3-pro": (2.00, 12.00),
}


def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> dict[str, float]:
    """Compute synthetic cost using approximate per-model pricing."""
    input_rate, output_rate = _MODEL_PRICING[model]
    input_cost = prompt_tokens * input_rate / 1_000_000
    output_cost = completion_tokens * output_rate / 1_000_000
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


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
    version = 1

    def generate(self) -> DemoResult:
        self._restore_experiment_if_deleted()
        experiment = mlflow.set_experiment(DEMO_EXPERIMENT_NAME)
        mlflow.MlflowClient().set_experiment_tag(
            experiment.experiment_id, "mlflow.experimentKind", "genai_development"
        )
        mlflow.set_experiment_tag(
            "mlflow.note.content",
            "Sample experiment with pre-populated demo data including traces, evaluations, "
            "and prompts. Explore MLflow's GenAI features with this experiment.",
        )

        v1_trace_ids = self._generate_trace_set("v1")
        v2_trace_ids = self._generate_trace_set("v2")

        all_trace_ids = v1_trace_ids + v2_trace_ids

        return DemoResult(
            feature=self.name,
            entity_ids=all_trace_ids,
            navigation_url=f"#/experiments/{experiment.experiment_id}",
        )

    def _generate_trace_set(self, version: Literal["v1", "v2"]) -> list[str]:
        """Generate a complete set of traces for the given version."""
        trace_ids = []
        trace_index = 0

        for trace_def in RAG_TRACES:
            start_ns, end_ns = _get_trace_timestamps(trace_index, version)
            if trace_id := self._create_rag_trace(trace_def, version, start_ns, end_ns):
                trace_ids.append(trace_id)
            trace_index += 1

        for trace_def in AGENT_TRACES:
            start_ns, end_ns = _get_trace_timestamps(trace_index, version)
            if trace_id := self._create_agent_trace(trace_def, version, start_ns, end_ns):
                trace_ids.append(trace_id)
            trace_index += 1

        for idx, trace_def in enumerate(PROMPT_TRACES):
            start_ns, end_ns = _get_trace_timestamps(trace_index, version)
            prompt_version_num = str(idx % 2 + 1) if version == "v1" else str(idx % 2 + 3)
            if trace_id := self._create_prompt_trace(
                trace_def, version, start_ns, end_ns, prompt_version_num
            ):
                trace_ids.append(trace_id)
            trace_index += 1

        trace_ids.extend(self._create_session_traces(version, trace_index))

        return trace_ids

    def _data_exists(self) -> bool:
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

    def _restore_experiment_if_deleted(self) -> None:
        """Restore the demo experiment if it was soft-deleted."""
        store = _get_store()
        try:
            experiment = store.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
            if experiment is not None and experiment.lifecycle_stage == "deleted":
                _logger.info("Restoring soft-deleted demo experiment")
                client = mlflow.MlflowClient()
                client.restore_experiment(experiment.experiment_id)
        except Exception:
            _logger.debug("Failed to check/restore demo experiment", exc_info=True)

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
        prompt_tokens = _estimate_tokens(trace_def.query) + 50
        completion_tokens = _estimate_tokens(response)

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

        model = "gpt-5.2"
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
                "model": model,
            },
            attributes={
                SpanAttributeKey.CHAT_USAGE: {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                SpanAttributeKey.MODEL: model,
                SpanAttributeKey.LLM_COST: _compute_cost(model, prompt_tokens, completion_tokens),
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

        model = "claude-sonnet-4-5"
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
                "model": model,
            },
            attributes={
                SpanAttributeKey.CHAT_USAGE: {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                SpanAttributeKey.MODEL: model,
                SpanAttributeKey.LLM_COST: _compute_cost(model, prompt_tokens, completion_tokens),
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
        prompt_version: str = "1",
    ) -> str | None:
        """Create a prompt-based trace showing template rendering and generation.

        Fetches the actual registered prompt template and renders it with appropriate
        variables to ensure trace contents match the linked prompt version.
        """
        response = self._get_response(trace_def, version)

        if trace_def.prompt_template is None:
            return None

        full_prompt_name = f"{DEMO_PROMPT_PREFIX}.prompts.{trace_def.prompt_template.prompt_name}"
        try:
            client = mlflow.MlflowClient()
            prompt_version_obj = client.get_prompt_version(
                name=full_prompt_name,
                version=prompt_version,
            )
            actual_template = prompt_version_obj.template
        except Exception:
            actual_template = trace_def.prompt_template.template

        variables = self._get_prompt_variables(
            trace_def.prompt_template.prompt_name,
            trace_def.query,
            trace_def.prompt_template.variables,
        )

        rendered_prompt = self._render_template(actual_template, variables)
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
                "template_variables": variables,
            },
            metadata={DEMO_VERSION_TAG: version, DEMO_TRACE_TYPE_TAG: "prompt"},
            start_time_ns=start_ns,
        )

        render = mlflow.start_span_no_context(
            name="render_prompt",
            span_type=SpanType.CHAIN,
            parent_span=root,
            inputs={
                "template": actual_template,
                "variables": variables,
            },
            start_time_ns=start_ns + 1000,
        )
        render.set_outputs({"rendered_prompt": rendered_prompt})
        render.end(end_time_ns=render_end)

        model = "gemini-3-pro"
        llm = mlflow.start_span_no_context(
            name="generate_response",
            span_type=SpanType.LLM,
            parent_span=root,
            inputs={
                "messages": [
                    {"role": "user", "content": rendered_prompt},
                ],
                "model": model,
            },
            attributes={
                SpanAttributeKey.CHAT_USAGE: {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                SpanAttributeKey.MODEL: model,
                SpanAttributeKey.LLM_COST: _compute_cost(model, prompt_tokens, completion_tokens),
            },
            start_time_ns=llm_start,
        )
        llm.set_outputs({"response": response})
        llm.end(end_time_ns=end_ns - 5000)

        root.set_outputs({"response": response})
        root.end(end_time_ns=end_ns)

        trace_id = root.trace_id

        self._link_prompt_to_trace(trace_def.prompt_template.prompt_name, trace_id, prompt_version)

        return trace_id

    def _link_prompt_to_trace(
        self, short_prompt_name: str, trace_id: str, prompt_version: str = "1"
    ) -> None:
        full_prompt_name = f"{DEMO_PROMPT_PREFIX}.prompts.{short_prompt_name}"
        try:
            client = mlflow.MlflowClient()
            prompt_version_obj = client.get_prompt_version(
                name=full_prompt_name,
                version=prompt_version,
            )
            client.link_prompt_versions_to_trace(
                prompt_versions=[prompt_version_obj],
                trace_id=trace_id,
            )
        except Exception:
            _logger.debug(
                "Failed to link prompt %s v%s to trace %s",
                full_prompt_name,
                prompt_version,
                trace_id,
                exc_info=True,
            )

    def _get_prompt_variables(
        self, prompt_name: str, query: str, base_variables: dict[str, str]
    ) -> dict[str, str]:
        """Get complete variable set for a prompt type.

        Combines base variables from the trace definition with additional
        variables that may be needed for more advanced prompt versions.
        """
        variables = dict(base_variables)

        if "query" not in variables:
            variables["query"] = query

        if prompt_name == "customer-support":
            variables.setdefault("company_name", "TechCorp")
            variables.setdefault("context", "Customer has been with us for 2 years, premium tier.")
        elif prompt_name == "document-summarizer":
            variables.setdefault("max_words", "150")
            variables.setdefault("audience", "technical professionals")
            variables.setdefault(
                "document",
                variables.get("query", "Sample document content for summarization."),
            )
        elif prompt_name == "code-reviewer":
            variables.setdefault("language", "python")
            variables.setdefault("focus_areas", "security, performance, readability")
            variables.setdefault("severity_levels", "critical, warning, suggestion")
            variables.setdefault("code", variables.get("query", "def example(): pass"))

        return variables

    def _render_template(
        self, template: str | list[dict[str, str]], variables: dict[str, str]
    ) -> str:
        """Render a prompt template with variables.

        Handles both string templates and chat-format templates (list of messages).
        """

        def substitute(text: str, vars_dict: dict[str, str]) -> str:
            for key, value in vars_dict.items():
                text = re.sub(r"\{\{\s*" + key + r"\s*\}\}", str(value), text)
            return text

        if isinstance(template, str):
            return substitute(template, variables)
        elif isinstance(template, list):
            rendered_parts = []
            for msg in template:
                role = msg.get("role", "user")
                content = substitute(msg.get("content", ""), variables)
                rendered_parts.append(f"[{role}]: {content}")
            return "\n\n".join(rendered_parts)
        else:
            return str(template)

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

            start_ns, end_ns = _get_trace_timestamps(trace_index, version)
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

        model = _DEMO_MODELS[turn % len(_DEMO_MODELS)]
        llm = mlflow.start_span_no_context(
            name="generate_response",
            span_type=SpanType.LLM,
            parent_span=root,
            inputs={
                "messages": [
                    {"role": "system", "content": "You are an MLflow assistant."},
                    {"role": "user", "content": trace_def.query},
                ],
                "model": model,
            },
            attributes={
                SpanAttributeKey.CHAT_USAGE: {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                SpanAttributeKey.MODEL: model,
                SpanAttributeKey.LLM_COST: _compute_cost(model, prompt_tokens, completion_tokens),
            },
            start_time_ns=llm_start,
        )
        llm.set_outputs({"role": "assistant", "content": response})
        llm.end(end_time_ns=end_ns - 5000)

        root.set_outputs({"response": response})
        root.end(end_time_ns=end_ns)

        return root.trace_id
