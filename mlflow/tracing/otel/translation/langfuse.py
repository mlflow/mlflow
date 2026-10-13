"""
Translation utilities for Langfuse observation attributes.

Maps ``langfuse.observation.*`` attributes to MLflow span semantics so that
spans forwarded from Langfuse via the generic OTEL processor are stored with
correct span types, inputs, and outputs.
"""

from mlflow.entities.span import SpanType
from mlflow.tracing.otel.translation.base import OtelSchemaTranslator


class LangfuseTranslator(OtelSchemaTranslator):
    SPAN_KIND_ATTRIBUTE_KEY = "langfuse.observation.type"

    SPAN_KIND_TO_MLFLOW_TYPE = {
        "generation": SpanType.LLM,
        "embedding": SpanType.EMBEDDING,
        "tool": SpanType.TOOL,
        "retriever": SpanType.RETRIEVER,
        "agent": SpanType.AGENT,
        "chain": SpanType.CHAIN,
        "evaluator": SpanType.EVALUATOR,
        "guardrail": SpanType.GUARDRAIL,
        "span": SpanType.UNKNOWN,
    }

    INPUT_VALUE_KEYS = ["langfuse.observation.input"]
    OUTPUT_VALUE_KEYS = ["langfuse.observation.output"]
