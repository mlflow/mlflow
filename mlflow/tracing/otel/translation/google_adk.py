from mlflow.tracing.otel.translation.base import OtelSchemaTranslator


class GoogleADKTranslator(OtelSchemaTranslator):
    """
    Translator for Google ADK semantic conventions.
    Google ADK mostly uses OpenTelemetry semantic conventions, but with some custom
    inputs and outputs attributes.
    """

    # Input/Output attribute keys
    # Reference: https://github.com/google/adk-python/blob/d2888a3766b87df2baaaa1a67a2235b1b80f138f/src/google/adk/telemetry/tracing.py#L264
    INPUT_VALUE_KEYS = ["gcp.vertex.agent.llm_request", "gcp.vertex.agent.tool_call_args"]
    OUTPUT_VALUE_KEYS = ["gcp.vertex.agent.llm_response", "gcp.vertex.agent.tool_response"]
