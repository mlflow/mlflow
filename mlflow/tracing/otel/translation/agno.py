from typing import Any

from mlflow.tracing.otel.translation.open_inference import OpenInferenceTranslator


class AgnoTranslator(OpenInferenceTranslator):
    """
    Translator for Agno traces via OpenInference semantic conventions.

    Detection: Only uses `agno.*` prefixed attributes which are guaranteed
    to be set by Agno's OpenInference instrumentation:
    - agno.agent.id
    - agno.team.id
    - agno.run.id
    - agno.model.id

    This is the only reliable detection method. Message structure-based
    detection is not used as it could cause false positives with other frameworks.
    """

    MESSAGE_FORMAT = "agno"
    AGNO_ATTRIBUTE_PREFIX = "agno."

    def get_message_format(self, attributes: dict[str, Any]) -> str | None:
        """Return 'agno' if this is an Agno trace, None otherwise."""
        # Only check for agno.* prefixed attributes - this is the only reliable detection
        # Examples: agno.agent.id, agno.team.id, agno.run.id, agno.model.id
        if any(key.startswith(self.AGNO_ATTRIBUTE_PREFIX) for key in attributes):
            return self.MESSAGE_FORMAT
        return None
