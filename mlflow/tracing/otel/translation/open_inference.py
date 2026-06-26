"""
Translation utilities for OpenInference semantic conventions.

Reference: https://github.com/Arize-ai/openinference/blob/main/python/openinference-semantic-conventions/
"""

import json
import re
from typing import Any

from mlflow.entities.span import SpanType
from mlflow.tracing.otel.translation.base import OtelSchemaTranslator
from mlflow.tracing.utils import dump_span_attribute_value


class OpenInferenceTranslator(OtelSchemaTranslator):
    """
    Translator for OpenInference semantic conventions.

    Only defines the attribute keys and mappings. All translation logic
    is inherited from the base class.
    """

    # OpenInference span kind attribute key
    # Reference: https://github.com/Arize-ai/openinference/blob/50eaf3c943d818f12fdc8e37b7c305c763c82050/python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py#L356
    SPAN_KIND_ATTRIBUTE_KEY = "openinference.span.kind"

    # Mapping from OpenInference span kinds to MLflow span types
    SPAN_KIND_TO_MLFLOW_TYPE = {
        "TOOL": SpanType.TOOL,
        "CHAIN": SpanType.CHAIN,
        "LLM": SpanType.LLM,
        "RETRIEVER": SpanType.RETRIEVER,
        "EMBEDDING": SpanType.EMBEDDING,
        "AGENT": SpanType.AGENT,
        "RERANKER": SpanType.RERANKER,
        "UNKNOWN": SpanType.UNKNOWN,
        "GUARDRAIL": SpanType.GUARDRAIL,
        "EVALUATOR": SpanType.EVALUATOR,
    }

    # Token count attribute keys
    # Reference: https://github.com/Arize-ai/openinference/blob/c80c81b8d6fa564598bd359cdd7313f4472ceca8/python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py
    INPUT_TOKEN_KEY = "llm.token_count.prompt"
    OUTPUT_TOKEN_KEY = "llm.token_count.completion"
    TOTAL_TOKEN_KEY = "llm.token_count.total"

    # Input/Output attribute keys
    # Reference: https://github.com/Arize-ai/openinference/blob/c80c81b8d6fa564598bd359cdd7313f4472ceca8/python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py
    INPUT_VALUE_KEYS = ["input.value"]
    OUTPUT_VALUE_KEYS = ["output.value"]

    # Model name attribute key
    # Reference: https://github.com/Arize-ai/openinference/blob/c80c81b8d6fa564598bd359cdd7313f4472ceca8/python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py#L45
    MODEL_NAME_KEYS = ["llm.model_name", "embedding.model_name"]
    # https://github.com/Arize-ai/openinference/blob/c80c81b8d6fa564598bd359cdd7313f4472ceca8/python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py#L49
    LLM_PROVIDER_KEY = "llm.provider"

    # A retriever span's documents are emitted as flattened, indexed attributes
    # ``retrieval.documents.{i}.document.{content|id|score|metadata}`` (from the
    # SpanAttributes.RETRIEVAL_DOCUMENTS + DocumentAttributes conventions), not a single value.
    # Reference: https://github.com/Arize-ai/openinference/blob/main/python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py
    _RETRIEVAL_DOCUMENT_ATTRIBUTE = re.compile(
        r"^retrieval\.documents\.(\d+)\.document\.(content|id|score|metadata)$"
    )

    def get_retrieval_documents(self, attributes: dict[str, Any]) -> Any:
        """
        Reassemble OpenInference retriever documents into MLflow ``Document``-shaped dicts.

        OpenInference flattens each retrieved document into ``retrieval.documents.{i}.document.*``
        attributes. Without this, only the span-level ``output.value`` blob is captured, so
        ``extract_retrieval_context_from_trace`` finds no per-document text and retrieval-grounded
        scorers silently see empty context.
        """
        documents_by_index: dict[int, dict[str, Any]] = {}
        for key, value in attributes.items():
            if match := self._RETRIEVAL_DOCUMENT_ATTRIBUTE.match(key):
                index = int(match.group(1))
                field = match.group(2)
                documents_by_index.setdefault(index, {})[field] = self._try_decode_if_json(value)

        if not documents_by_index:
            return None

        documents = []
        for index in sorted(documents_by_index):
            fields = documents_by_index[index]
            content = fields.get("content")
            if not isinstance(content, str):
                # ``document.content`` is spec'd as a plain string; coerce defensively so a
                # non-string never reaches the scorers that string-join the context.
                content = "" if content is None else str(content)
            metadata = self._coerce_document_metadata(fields.get("metadata"))
            if (score := fields.get("score")) is not None:
                # MLflow's Document has no score field; preserve it under metadata. The dedicated
                # ``document.score`` attribute is authoritative and wins over any metadata["score"].
                metadata = {**metadata, "score": score}
            documents.append({
                "page_content": content,
                "id": fields.get("id"),
                "metadata": metadata,
            })
        # Use MLflow's canonical span-attribute serializer (TraceJSONEncoder) rather than a raw
        # ``json.dumps``: ``document.metadata`` may carry non-JSON-native values, and this keeps
        # serialization consistent with how every other OUTPUTS attribute is stored and read back.
        return dump_span_attribute_value(documents)

    @staticmethod
    def _coerce_document_metadata(value: Any) -> dict[str, Any]:
        # ``document.metadata`` is spec'd as a JSON-string dict; tolerate already-decoded dicts
        # and malformed/non-dict payloads (return ``{}`` rather than raising).
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value:
            try:
                loaded = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return {}
            if isinstance(loaded, dict):
                return loaded
        return {}
