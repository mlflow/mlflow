"""
Structured evidence for why one trace demonstrates one issue.

Detection used to annotate an affected trace with a paragraph of prose. The Issues tab can
render an ordered, span-anchored walkthrough instead -- each step citing a real span and
quoting the part of its own text that carries the finding -- which is what makes the evidence
navigable rather than merely readable.

The walkthrough is generated, so it is validated before it is stored: a step may only cite a
span the trace actually recorded, and its highlights must be exact substrings of its own text.
Evidence that fails validation degrades to the summary rather than discarding the finding,
because the annotation is the record that this trace exhibits the issue.
"""

from __future__ import annotations

import json
import logging

import pydantic

from mlflow.entities.issue_evidence import (
    HIGHLIGHT_TONES,
    IssueEvidence,
    IssueEvidenceStep,
)
from mlflow.entities.span import Span
from mlflow.entities.trace import Trace
from mlflow.genai.discovery.constants import (
    EVIDENCE_GENERATION_SYSTEM_PROMPT,
    SPAN_CONTENT_TRUNCATION,
)
from mlflow.genai.utils.llm_utils import _call_llm, _TokenCounter

_logger = logging.getLogger(__name__)

# How many spans of a trace are offered to the generator. A walkthrough cites a handful of
# steps, and a very wide trace would otherwise crowd out the content of each one.
MAX_SPANS_IN_PROMPT = 40


class _GeneratedStep(pydantic.BaseModel):
    span_id: str = pydantic.Field(
        description="The span_id of the span this step is about, copied exactly from the list"
    )
    text: str = pydantic.Field(
        description=(
            "One claim about this span in one or two short sentences, leading with the "
            "conclusion. Quote only the smallest useful excerpt; never copy a whole payload."
        )
    )
    highlights: list[str] = pydantic.Field(
        description=(
            "One or more EXACT substrings of this step's own text to emphasize. Each must "
            "appear in text character for character."
        )
    )
    highlight_tone: str = pydantic.Field(
        description=(
            "'error' for a failure, hallucination or extreme cost/latency; 'warning' for an "
            "elevated but non-failing condition; 'success' for correct or expected behavior."
        )
    )
    # No default, deliberately. A pydantic default keeps the field out of the schema's
    # ``required`` list, and OpenAI's strict structured-output mode requires every property to
    # be listed there -- so a default here makes the request invalid rather than optional, the
    # generator exhausts its retries, and evidence silently falls back to prose.
    mono: bool = pydantic.Field(
        description=(
            "True only when text is preformatted, such as a payload or stack trace; "
            "false otherwise."
        ),
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _tolerate_omitted_mono(cls, data):
        """
        Accept a step that leaves ``mono`` out, without giving the field a schema default.

        Strict mode obliges the model to send every property, so this is unreachable there.
        It matters for a provider that ignores strict mode: the field has to stay in
        ``required`` to keep the request valid, but a response missing a presentational flag
        should not cost the reader the whole walkthrough.
        """
        if isinstance(data, dict) and "mono" not in data:
            return {**data, "mono": False}
        return data


class _GeneratedEvidence(pydantic.BaseModel):
    summary: str = pydantic.Field(
        description="One or two sentences naming what went wrong on this trace."
    )
    steps: list[_GeneratedStep] = pydantic.Field(
        description=(
            "An ordered causal walkthrough: what the input required, what a step did, why the "
            "output therefore failed. One step per span worth opening."
        )
    )
    # Required but nullable, for the reason given at ``mono``: the field must be present in
    # the response, and absence is expressed as an explicit null rather than omission.
    primary_span_id: str | None = pydantic.Field(
        description=(
            "The one span to open first, which must be cited by a step. Null when no single "
            "span is the meaningful locus, such as a trace-wide latency or cost finding."
        ),
    )


def _span_excerpt(value: object) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:SPAN_CONTENT_TRUNCATION]


def format_span_inventory(trace: Trace, max_spans: int = MAX_SPANS_IN_PROMPT) -> str:
    """
    The trace's spans as the generator sees them: identity plus bounded content.

    Span IDs are listed because the walkthrough has to cite them exactly; a span name would be
    ambiguous in a trace that calls the same tool twice. Content is truncated per span so a
    wide trace still leaves room for every span's identity.
    """
    spans: list[Span] = list(trace.data.spans or [])[:max_spans]
    if not spans:
        return "(this trace recorded no spans)"

    entries = []
    for span in spans:
        parts = [f"span_id={span.span_id} name={span.name!r} type={span.span_type}"]
        if status := getattr(span.status, "status_code", None):
            parts.append(f"status={status}")
        if inputs := _span_excerpt(span.inputs):
            parts.append(f"inputs: {inputs}")
        if outputs := _span_excerpt(span.outputs):
            parts.append(f"outputs: {outputs}")
        entries.append("\n  ".join(parts))
    inventory = "\n".join(f"- {entry}" for entry in entries)

    total = len(trace.data.spans or [])
    if total > max_spans:
        inventory += f"\n- ({total - max_spans} further spans omitted)"
    return inventory


def _coerce_step(step: _GeneratedStep, known_span_ids: set[str]) -> IssueEvidenceStep | None:
    """
    One generated step as a validated step, or None when it cannot be trusted.

    Highlights are dropped individually when they are not exact substrings -- a model often
    gets most of a quote right -- but a step left with none is discarded, since a step that
    emphasizes nothing points at no part of its own text.
    """
    if step.span_id not in known_span_ids:
        _logger.debug("Dropping evidence step citing unknown span %s", step.span_id)
        return None
    text = (step.text or "").strip()
    if not text:
        return None
    highlights = [h for h in (step.highlights or []) if h and h in text]
    if not highlights:
        _logger.debug("Dropping evidence step with no exact highlight for span %s", step.span_id)
        return None
    tone = step.highlight_tone if step.highlight_tone in HIGHLIGHT_TONES else "warning"
    try:
        return IssueEvidenceStep(
            span_id=step.span_id,
            text=text,
            highlights=highlights,
            highlight_tone=tone,
            mono=bool(step.mono),
        )
    except Exception:
        _logger.debug("Dropping evidence step that failed validation", exc_info=True)
        return None


def build_evidence(
    generated: _GeneratedEvidence,
    trace: Trace,
    *,
    fallback_summary: str = "",
) -> IssueEvidence | None:
    """
    A validated :class:`IssueEvidence` from a generated walkthrough, or None.

    Returns None when nothing survives validation, which the caller treats as "store the
    summary alone" rather than as a failure to annotate.
    """
    known_span_ids = {span.span_id for span in trace.data.spans or []}
    steps = [
        coerced
        for coerced in (_coerce_step(step, known_span_ids) for step in generated.steps or [])
        if coerced is not None
    ]
    if not steps:
        return None

    summary = (generated.summary or "").strip() or fallback_summary.strip()
    if not summary:
        # The summary becomes the reference's rationale, which is what a reader without the
        # structured schema sees. Without one there is nothing to degrade to.
        return None

    cited = {step.span_id for step in steps}
    primary = generated.primary_span_id if generated.primary_span_id in cited else None
    try:
        return IssueEvidence(summary=summary, steps=steps, primary_span_id=primary)
    except Exception:
        _logger.debug("Generated evidence failed validation as a whole", exc_info=True)
        return None


def generate_trace_evidence(
    *,
    issue_name: str,
    issue_description: str,
    root_causes: str,
    trace: Trace,
    trace_content: str,
    triage_rationale: str,
    categories: list[str] | None,
    model: str,
    token_counter: _TokenCounter | None = None,
) -> tuple[IssueEvidence | None, str]:
    """
    Generate a span-anchored walkthrough for one trace and issue.

    Returns:
        A tuple of ``(evidence, summary)``. ``evidence`` is None when generation or validation
        did not produce a usable walkthrough; ``summary`` is always the best available prose,
        so the caller can annotate the trace either way.
    """
    user_content = (
        f"=== ISSUE ===\n"
        f"Name: {issue_name}\n"
        f"Description: {issue_description}\n"
        f"Root causes: {root_causes or '(not available)'}\n\n"
        f"=== TRACE SUMMARY ===\n{trace_content}\n\n"
        f"=== SPANS (cite span_id values from this list only) ===\n"
        f"{format_span_inventory(trace)}\n\n"
        f"=== TRIAGE JUDGE RATIONALE ===\n{triage_rationale or '(not available)'}"
    )
    if categories:
        user_content += f"\n\n=== RELEVANT CATEGORIES ===\n{', '.join(categories)}"

    response = _call_llm(
        model,
        [
            {"role": "system", "content": EVIDENCE_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=_GeneratedEvidence,
        token_counter=token_counter,
    )
    content = (response.choices[0].message.content or "").strip()
    generated = _GeneratedEvidence(**json.loads(content))

    summary = (generated.summary or "").strip()
    evidence = build_evidence(generated, trace, fallback_summary=summary)
    if evidence is not None:
        return evidence, evidence.summary
    return None, summary
