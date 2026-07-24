"""
Get span image tool for MLflow GenAI judges.

This module provides a tool that resolves an ``mlflow-attachment://`` image
reference inside a span, downloads the real bytes, and returns them as a
base64 data URL so a multimodal judge model can actually view the image.

When autolog extracts an image from a span it rewrites the inline content to an
``mlflow-attachment://<id>?content_type=...&trace_id=...&size=...`` reference.
The text-only judge tools only surface that reference token, so a ``{{ trace }}``
judge is structurally blind to the pixels. This tool bridges that gap.
"""

import base64
import json
import re
from dataclasses import dataclass

from mlflow.entities.trace import Trace
from mlflow.environment_variables import MLFLOW_TRACE_MAX_ATTACHMENT_SIZE
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.constants import ToolNames
from mlflow.tracing.attachments import Attachment
from mlflow.types.llm import FunctionToolDefinition, ToolDefinition, ToolParamsSchema

_ATTACHMENT_REF_RE = re.compile(r"mlflow-attachment://[^\s\"'\\]+")


@dataclass
class SpanImageResult:
    """The image bytes fetched for a span, as a base64 data URL."""

    span_id: str
    content_type: str
    data_url: str


class GetSpanImageTool(JudgeTool):
    """
    Tool for fetching the actual image bytes of an attachment referenced in a span.

    Resolves an ``mlflow-attachment://`` reference, downloads the bytes, and
    returns them as a base64 data URL.
    """

    @property
    def name(self) -> str:
        return ToolNames.GET_SPAN_IMAGE

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name=ToolNames.GET_SPAN_IMAGE,
                description=(
                    "Fetch the actual image content of an attachment referenced in a span "
                    "(an mlflow-attachment:// URL) so you can view it. Use this whenever a "
                    "span input/output contains an image reference and you need to see the "
                    "image to answer. After calling this, the image is delivered to you as a "
                    "viewable image in the following user message. If a span contains multiple "
                    "image attachments, use attachment_index to select which one to fetch."
                ),
                parameters=ToolParamsSchema(
                    type="object",
                    properties={
                        "span_id": {
                            "type": "string",
                            "description": "The ID of the span whose image attachment to fetch",
                        },
                        "attachment_index": {
                            "type": "integer",
                            "description": (
                                "Zero-based index selecting which attachment reference to fetch "
                                "when the span contains more than one. If omitted, the first image "
                                "attachment is used (non-image references are skipped)."
                            ),
                        },
                    },
                    required=["span_id"],
                ),
            ),
            type="function",
        )

    def invoke(
        self, trace: Trace, span_id: str, attachment_index: int | None = None
    ) -> SpanImageResult | str:
        """
        Fetch an image attachment referenced in a span.

        Args:
            trace: The MLflow trace object to analyze.
            span_id: The ID of the span whose image attachment to fetch.
            attachment_index: Zero-based index selecting which attachment reference to
                fetch when the span has multiple. If ``None`` (the default), the first
                image attachment is used, skipping any non-image references.

        Returns:
            A SpanImageResult with the base64 data URL on success, or a descriptive
            error string on any not-found / parse / non-image / out-of-range condition.
        """
        trace_id = trace.info.trace_id if trace and trace.info else None

        if not trace or not trace.data or not trace.data.spans:
            return f"Error: trace '{trace_id}' has no spans"

        target = next((s for s in trace.data.spans if s.span_id == span_id), None)
        if target is None:
            return f"Error: span '{span_id}' not found in trace '{trace_id}'"

        # Autolog rewrites an inline image data URL to an mlflow-attachment:// token,
        # which survives only inside the serialized span, so scan that JSON for it.
        serialized = json.dumps(target.to_dict(), default=str)
        refs = _ATTACHMENT_REF_RE.findall(serialized)
        if not refs:
            return (
                f"Error: no mlflow-attachment:// image reference found in span '{span_id}' "
                f"of trace '{trace_id}'"
            )

        if attachment_index is None:
            # Default: pick the first image ref so a non-image attachment sitting before
            # an image one doesn't shadow it.
            parsed = next(
                (
                    p
                    for ref in refs
                    if (p := Attachment.parse_ref(ref)) is not None
                    and p["content_type"].startswith("image/")
                ),
                None,
            )
            if parsed is None:
                return f"Error: no image attachment found in span '{span_id}' of trace '{trace_id}'"
        else:
            if attachment_index < 0 or attachment_index >= len(refs):
                return (
                    f"Error: attachment_index {attachment_index} is out of range for span "
                    f"'{span_id}' of trace '{trace_id}', which has {len(refs)} "
                    f"attachment reference(s)"
                )

            parsed = Attachment.parse_ref(refs[attachment_index])
            if parsed is None:
                return (
                    f"Error: could not parse attachment reference in span '{span_id}' "
                    f"of trace '{trace_id}'"
                )

            if not parsed["content_type"].startswith("image/"):
                return (
                    f"Error: attachment in span '{span_id}' of trace '{trace_id}' is not an "
                    f"image (content_type='{parsed['content_type']}')"
                )

        content_type = parsed["content_type"]

        # NB: _get_artifact_repo_for_trace / download_trace_attachment is the current
        # internal accessor for trace attachment bytes; there is no public API for it yet.
        from mlflow.tracing.client import TracingClient

        repo = TracingClient()._get_artifact_repo_for_trace(trace.info)
        content_bytes = repo.download_trace_attachment(parsed["attachment_id"])

        # Cap the inlined image so a large attachment can't blow up the judge's context.
        # The ref's parsed size isn't reliably populated, so check the actual downloaded
        # length. A partial image is worse than a clear error, so reject rather than truncate.
        max_size = MLFLOW_TRACE_MAX_ATTACHMENT_SIZE.get()
        size = len(content_bytes)
        if max_size is not None and max_size > 0 and size > max_size:
            return (
                f"Error: image attachment in span '{span_id}' of trace '{trace_id}' is "
                f"{size} bytes, exceeding the {max_size} byte limit "
                f"(MLFLOW_TRACE_MAX_ATTACHMENT_SIZE)"
            )

        b64 = base64.b64encode(content_bytes).decode()
        return SpanImageResult(
            span_id=span_id,
            content_type=content_type,
            data_url=f"data:{content_type};base64,{b64}",
        )
