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
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.constants import ToolNames
from mlflow.tracing.attachments import Attachment
from mlflow.types.llm import FunctionToolDefinition, ToolDefinition, ToolParamsSchema

# Matches any mlflow-attachment:// reference embedded in the serialized span.
_ATTACHMENT_REF_RE = re.compile(r"mlflow-attachment://[^\s\"'\\]+")


@dataclass
class SpanImageResult:
    """Result carrying real image bytes fetched for a span.

    The data URL is carried out of the tool so the tool-calling loop can deliver
    it on a follow-up ``role="user"`` turn (OpenAI-format endpoints reject image
    blocks inside ``role="tool"`` messages) rather than flattening it to text.
    """

    span_id: str
    content_type: str
    data_url: str


class GetSpanImageTool(JudgeTool):
    """
    Tool for fetching the actual image bytes of an attachment referenced in a span.

    Resolves an ``mlflow-attachment://`` reference, downloads the bytes, and
    returns them as a base64 data URL. The image is delivered to the model as a
    viewable image in the following user message.
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
                                "Zero-based index selecting which image attachment to fetch "
                                "when the span contains more than one. Defaults to 0 (the first)."
                            ),
                        },
                    },
                    required=["span_id"],
                ),
            ),
            type="function",
        )

    def invoke(
        self, trace: Trace, span_id: str, attachment_index: int = 0
    ) -> SpanImageResult | str:
        """
        Fetch an image attachment referenced in a span.

        Args:
            trace: The MLflow trace object to analyze.
            span_id: The ID of the span whose image attachment to fetch.
            attachment_index: Zero-based index selecting which attachment reference to
                fetch when the span has multiple. Defaults to 0.

        Returns:
            A SpanImageResult with the base64 data URL on success, or a descriptive
            error string on any not-found / parse / non-image / out-of-range condition.
        """
        if not trace or not trace.data or not trace.data.spans:
            return "Error: trace has no spans"

        target = next((s for s in trace.data.spans if s.span_id == span_id), None)
        if target is None:
            return f"Error: span '{span_id}' not found in trace"

        # The image reference survives only inside the serialized span (autolog
        # rewrites the inline data URL to an mlflow-attachment:// token), so scan
        # the same JSON the text tools expose to the model.
        serialized = json.dumps(target.to_dict(), default=str)
        refs = _ATTACHMENT_REF_RE.findall(serialized)
        if not refs:
            return f"Error: no mlflow-attachment:// image reference found in span '{span_id}'"

        if attachment_index < 0 or attachment_index >= len(refs):
            return (
                f"Error: attachment_index {attachment_index} is out of range for span "
                f"'{span_id}', which has {len(refs)} attachment reference(s)"
            )

        parsed = Attachment.parse_ref(refs[attachment_index])
        if parsed is None:
            return f"Error: could not parse attachment reference in span '{span_id}'"

        content_type = parsed["content_type"]
        if not content_type.startswith("image/"):
            return (
                f"Error: attachment in span '{span_id}' is not an image "
                f"(content_type='{content_type}')"
            )

        # NB: _get_artifact_repo_for_trace / download_trace_attachment is the current
        # internal accessor for trace attachment bytes; there is no public API for it
        # yet (the tracing client and server handler use the same internal path).
        from mlflow.tracing.client import TracingClient

        repo = TracingClient()._get_artifact_repo_for_trace(trace.info)
        content_bytes = repo.download_trace_attachment(parsed["attachment_id"])
        b64 = base64.b64encode(content_bytes).decode()
        return SpanImageResult(
            span_id=span_id,
            content_type=content_type,
            data_url=f"data:{content_type};base64,{b64}",
        )
