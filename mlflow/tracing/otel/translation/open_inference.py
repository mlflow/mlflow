"""
Translation utilities for OpenInference semantic conventions.

Reference: https://github.com/Arize-ai/openinference/blob/main/python/openinference-semantic-conventions/
"""

from mlflow.entities.span import SpanType
from mlflow.tracing.otel.translation.base import OtelSchemaTranslator


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
