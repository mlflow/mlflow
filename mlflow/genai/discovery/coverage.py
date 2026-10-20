"""
Whether a trace carries the content its judgments would need.

A trace can be structurally complete -- the right span tree, ``OK`` status, real timings --
and still record none of the signal quality analysis depends on, because the instrumentation
truncated the payload or replaced it with a count. Judging such a trace produces
confident-sounding findings that nothing in the trace supports, so the pipeline separates
"inspected and sound" from "inspected and unjudgeable" instead of letting the second silently
become a pass.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from mlflow.entities.span import Span, SpanType
from mlflow.entities.trace import Trace
from mlflow.tracing.constant import TokenUsageKey

_logger = logging.getLogger(__name__)

# Span types whose payload is the evidence for a quality judgment: what was retrieved, what a
# tool returned, what the model was asked and answered.
_PAYLOAD_SPAN_TYPES = {SpanType.RETRIEVER, SpanType.TOOL}
_MODEL_SPAN_TYPES = {SpanType.LLM, SpanType.CHAT_MODEL}

# A retriever or tool output that is only a tally of what it found: "20 matches", "14 docs",
# "retrieved 8 rows", "3 results". The count is not the content, so nothing about relevance or
# correctness can be read from it.
_COUNT_ONLY_OUTPUT = re.compile(
    r"^\W*(?:retrieved|returned|found|got)?\s*\d[\d,.]*\s*"
    r"(?:match|matches|doc|docs|document|documents|row|rows|result|results|record|records|"
    r"item|items|chunk|chunks|hit|hits|file|files)\b\W*$",
    re.IGNORECASE,
)

# Serialized graph/state objects a framework dumps in place of the real request or response.
_OPAQUE_DUMP = re.compile(
    r"<[A-Za-z_][\w.]*\s+object\s+at\s+0x[0-9a-fA-F]+>|^[A-Za-z_][\w.]*\((?:\.{3}|)\)$"
)

# Placeholders instrumentation substitutes for content it chose not to record.
_ELIDED_MARKERS = (
    "[truncated]",
    "<truncated>",
    "...truncated",
    "[elided]",
    "<elided>",
    "[redacted]",
    "<redacted>",
    "[omitted]",
    "<content omitted>",
)

# Reason codes. Stable strings: they are reported to users and asserted in tests.
GAP_RETRIEVAL_COUNT_ONLY = "retrieval_payload_is_a_count"
GAP_TOOL_COUNT_ONLY = "tool_payload_is_a_count"
GAP_MODEL_PROMPT_MISSING = "model_prompt_missing"
GAP_MODEL_COMPLETION_MISSING = "model_completion_missing"
GAP_CONTENT_ELIDED = "span_content_elided"
GAP_ROOT_OPAQUE = "root_payload_is_an_opaque_dump"
GAP_ROOT_MISSING = "root_payload_missing"
GAP_CACHE_USAGE_MISSING = "cache_token_usage_missing"

# Human-readable text for each reason, so a report can say what to instrument.
GAP_DESCRIPTIONS = {
    GAP_RETRIEVAL_COUNT_ONLY: "a RETRIEVER span records how many documents it found, not the text",
    GAP_TOOL_COUNT_ONLY: "a TOOL span records how many results it returned, not the results",
    GAP_MODEL_PROMPT_MISSING: "an LLM span records no prompt",
    GAP_MODEL_COMPLETION_MISSING: "an LLM span records no completion",
    GAP_CONTENT_ELIDED: "span content was replaced with a truncation placeholder",
    GAP_ROOT_OPAQUE: (
        "the root span's request or response is a serialized object, not the real text"
    ),
    GAP_ROOT_MISSING: "the root span records no request or response",
    GAP_CACHE_USAGE_MISSING: (
        "token counts are present but cache-read/cache-write counts and cost are not"
    ),
}


@dataclass
class TraceDataAdequacy:
    """
    What a trace can and cannot support a judgment about.

    Args:
        trace_id: The trace this describes.
        gaps: Reason codes for content the trace does not carry, in a stable order.
        quality_judgeable: Whether the payload needed for a correctness, relevance, adherence
            or safety judgment is present. False when the content a judge would have to read
            is missing, which is distinct from the trace passing.
        cost_judgeable: Whether usage information is complete enough to compare spend. A trace
            with tokens but no cache or cost figures can be counted but not priced.
    """

    trace_id: str
    gaps: list[str] = field(default_factory=list)
    quality_judgeable: bool = True
    cost_judgeable: bool = True

    @property
    def is_thin(self) -> bool:
        """Whether any required content is missing."""
        return bool(self.gaps)

    def describe(self) -> str:
        """The gaps as one human-readable clause, for a coverage report."""
        return "; ".join(GAP_DESCRIPTIONS.get(gap, gap) for gap in self.gaps)


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _is_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _is_count_only(value: object) -> bool:
    """Whether a payload is a tally of its content rather than the content."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    text = _as_text(value).strip()
    if not text:
        return False
    # A bare number, or a number with the unit it counts.
    return bool(_COUNT_ONLY_OUTPUT.match(text)) or text.replace(",", "").isdigit()


def _is_elided(value: object) -> bool:
    text = _as_text(value).lower()
    return any(marker in text for marker in _ELIDED_MARKERS)


def _is_opaque_dump(value: object) -> bool:
    text = _as_text(value).strip()
    if not text:
        return False
    return bool(_OPAQUE_DUMP.search(text))


def _model_span_gaps(span: Span) -> list[str]:
    gaps = []
    if _is_empty(span.inputs):
        gaps.append(GAP_MODEL_PROMPT_MISSING)
    if _is_empty(span.outputs):
        gaps.append(GAP_MODEL_COMPLETION_MISSING)
    if _is_elided(span.inputs) or _is_elided(span.outputs):
        gaps.append(GAP_CONTENT_ELIDED)
    return gaps


def _payload_span_gaps(span: Span) -> list[str]:
    gaps = []
    if _is_count_only(span.outputs):
        gaps.append(
            GAP_RETRIEVAL_COUNT_ONLY
            if span.span_type == SpanType.RETRIEVER
            else GAP_TOOL_COUNT_ONLY
        )
    elif _is_elided(span.outputs):
        gaps.append(GAP_CONTENT_ELIDED)
    return gaps


def _cache_usage_gap(trace: Trace) -> list[str]:
    """
    Token counts without cache or cost figures.

    Reported only when tokens are present: a trace with no usage information at all is not
    making a claim about its own spend, whereas one that counts tokens and omits the cache
    breakdown invites a cost comparison it cannot support.
    """
    usage = trace.info.token_usage or {}
    if not usage.get(TokenUsageKey.TOTAL_TOKENS) and not usage.get(TokenUsageKey.INPUT_TOKENS):
        return []
    has_cache = any(
        usage.get(key)
        for key in (
            TokenUsageKey.CACHE_READ_INPUT_TOKENS,
            TokenUsageKey.CACHE_CREATION_INPUT_TOKENS,
        )
    )
    has_cost = bool((trace.info.cost or {}).get("total_cost"))
    return [] if (has_cache or has_cost) else [GAP_CACHE_USAGE_MISSING]


def assess_trace_data_adequacy(trace: Trace) -> TraceDataAdequacy:
    """
    Check that a trace records the content its judgments would need.

    Args:
        trace: The trace to inspect. Only recorded content is read; nothing is fetched.

    Returns:
        A :class:`TraceDataAdequacy` naming the missing content. An adequate trace returns
        no gaps; a thin one is reported rather than judged, because analyzing a content-free
        trace produces findings the trace cannot support.
    """
    trace_id = trace.info.trace_id
    gaps: list[str] = []

    request = trace.data.request
    response = trace.data.response
    if _is_empty(request) and _is_empty(response):
        gaps.append(GAP_ROOT_MISSING)
    elif _is_opaque_dump(request) or _is_opaque_dump(response):
        gaps.append(GAP_ROOT_OPAQUE)
    elif _is_elided(request) or _is_elided(response):
        gaps.append(GAP_CONTENT_ELIDED)

    for span in trace.data.spans or []:
        if span.span_type in _MODEL_SPAN_TYPES:
            gaps.extend(_model_span_gaps(span))
        elif span.span_type in _PAYLOAD_SPAN_TYPES:
            gaps.extend(_payload_span_gaps(span))

    cache_gaps = _cache_usage_gap(trace)

    # Deduplicate while keeping first-seen order, so one reason code is reported once however
    # many spans exhibit it.
    ordered = list(dict.fromkeys([*gaps, *cache_gaps]))
    quality_gaps = [gap for gap in ordered if gap != GAP_CACHE_USAGE_MISSING]
    return TraceDataAdequacy(
        trace_id=trace_id,
        gaps=ordered,
        quality_judgeable=not quality_gaps,
        cost_judgeable=GAP_CACHE_USAGE_MISSING not in ordered,
    )


def assess_sample_data_adequacy(traces: list[Trace]) -> dict[str, TraceDataAdequacy]:
    """Data adequacy for every trace in a sample, keyed by trace ID."""
    adequacy = {}
    for trace in traces:
        try:
            adequacy[trace.info.trace_id] = assess_trace_data_adequacy(trace)
        except Exception:
            # A trace whose shape cannot be read is reported as unknown rather than dropped:
            # silently omitting it would shrink the denominator a failure rate is quoted against.
            _logger.debug("Could not assess trace %s", trace.info.trace_id, exc_info=True)
            adequacy[trace.info.trace_id] = TraceDataAdequacy(
                trace_id=trace.info.trace_id,
                gaps=[GAP_ROOT_MISSING],
                quality_judgeable=False,
                cost_judgeable=False,
            )
    return adequacy
