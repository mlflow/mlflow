"""Gateway provider and model definitions for secrets management."""

from dataclasses import asdict, dataclass
from typing import List


@dataclass
class GatewayModel:
    """Represents an LLM model for a specific provider."""

    model_id: str
    display_name: str


@dataclass
class GatewayProvider:
    """Represents an LLM provider with its supported models."""

    provider_id: str
    display_name: str
    default_key_name: str
    models: List[GatewayModel]

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "provider_id": self.provider_id,
            "display_name": self.display_name,
            "default_key_name": self.default_key_name,
            "models": [asdict(m) for m in self.models],
        }


GATEWAY_PROVIDERS = [
    GatewayProvider(
        provider_id="anthropic",
        display_name="Anthropic",
        default_key_name="ANTHROPIC_API_KEY",
        models=[
            GatewayModel("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5 (Latest)"),
            GatewayModel("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
            GatewayModel("claude-opus-4-1-20250805", "Claude Opus 4.1"),
        ],
    ),
    GatewayProvider(
        provider_id="openai",
        display_name="OpenAI",
        default_key_name="OPENAI_API_KEY",
        models=[
            GatewayModel("gpt-4o", "GPT-4o"),
            GatewayModel("gpt-4o-mini", "GPT-4o Mini"),
            GatewayModel("gpt-4-turbo", "GPT-4 Turbo"),
        ],
    ),
    GatewayProvider(
        provider_id="vertex_ai",
        display_name="Google Vertex AI",
        default_key_name="GOOGLE_APPLICATION_CREDENTIALS",
        models=[
            GatewayModel("gemini-2.5-pro", "Gemini 2.5 Pro"),
            GatewayModel("gemini-2.5-flash", "Gemini 2.5 Flash"),
            GatewayModel("gemini-2.0-flash", "Gemini 2.0 Flash"),
        ],
    ),
    GatewayProvider(
        provider_id="bedrock",
        display_name="AWS Bedrock",
        default_key_name="AWS_ACCESS_KEY_ID",
        models=[
            GatewayModel("anthropic.claude-sonnet-4-5-20250929-v1:0", "Claude Sonnet 4.5"),
            GatewayModel("anthropic.claude-opus-4-1-20250805-v1:0", "Claude Opus 4.1"),
            GatewayModel("meta.llama3-3-70b-instruct-v1:0", "Llama 3.3 70B"),
        ],
    ),
    GatewayProvider(
        provider_id="databricks",
        display_name="Databricks",
        default_key_name="DATABRICKS_TOKEN",
        models=[
            GatewayModel("databricks-claude-sonnet-4-5", "Claude Sonnet 4.5"),
            GatewayModel("databricks-llama-4-maverick", "Llama 4 Maverick"),
            GatewayModel("databricks-gemini-2.5-pro", "Gemini 2.5 Pro"),
        ],
    ),
]


def get_gateway_providers() -> List[GatewayProvider]:
    """Return the list of supported gateway providers."""
    return GATEWAY_PROVIDERS
