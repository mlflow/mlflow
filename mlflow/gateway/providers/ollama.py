from mlflow.gateway.config import _OpenAICompatibleConfig
from mlflow.gateway.providers.openai_compatible import OpenAICompatibleProvider


class OllamaConfig(_OpenAICompatibleConfig):
    # Ollama runs locally and doesn't require an API key by default
    api_key: str = "ollama"


class OllamaProvider(OpenAICompatibleProvider):
    NAME = "Ollama"
    CONFIG_TYPE = OllamaConfig
    DEFAULT_API_BASE = "http://localhost:11434/v1"

    @property
    def headers(self) -> dict[str, str]:
        # Ollama doesn't require auth — only send Authorization if a real key was set
        if self._provider_config.api_key and self._provider_config.api_key != "ollama":
            return {"Authorization": f"Bearer {self._provider_config.api_key}"}
        return {}
