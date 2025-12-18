from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncIterable

import numpy as np

from mlflow.exceptions import MlflowException
from mlflow.gateway.base_models import ConfigModel
from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.utils.annotations import developer_stable


class PassthroughAction(str, Enum):
    """
    Enum for passthrough endpoint actions.
    """

    OPENAI_CHAT = "openai_chat"
    OPENAI_EMBEDDINGS = "openai_embeddings"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    GEMINI_GENERATE_CONTENT = "gemini_generate_content"
    GEMINI_STREAM_GENERATE_CONTENT = "gemini_stream_generate_content"


# Mapping of passthrough actions to their gateway API routes
PASSTHROUGH_ROUTES = {
    PassthroughAction.OPENAI_CHAT: "/openai/v1/chat/completions",
    PassthroughAction.OPENAI_EMBEDDINGS: "/openai/v1/embeddings",
    PassthroughAction.OPENAI_RESPONSES: "/openai/v1/responses",
    PassthroughAction.ANTHROPIC_MESSAGES: "/anthropic/v1/messages",
    PassthroughAction.GEMINI_GENERATE_CONTENT: "/gemini/v1beta/models/{endpoint_name}:generateContent",  # noqa: E501
    PassthroughAction.GEMINI_STREAM_GENERATE_CONTENT: "/gemini/v1beta/models/{endpoint_name}:streamGenerateContent",  # noqa: E501
}


@developer_stable
class BaseProvider(ABC):
    """
    Base class for MLflow Gateway providers.
    """

    NAME: str = ""
    SUPPORTED_ROUTE_TYPES: tuple[str, ...]
    CONFIG_TYPE: type[ConfigModel]
    PASSTHROUGH_PROVIDER_PATHS: dict[PassthroughAction, str] = {}

    def __init__(self, config: EndpointConfig):
        if self.NAME == "":
            raise ValueError(
                f"{self.__class__.__name__} is a subclass of BaseProvider and must "
                f"override 'NAME' attribute as a non-empty string."
            )

        if not hasattr(self, "CONFIG_TYPE") or not issubclass(self.CONFIG_TYPE, ConfigModel):
            raise ValueError(
                f"{self.__class__.__name__} is a subclass of BaseProvider and must "
                f"override 'CONFIG_TYPE' attribute as a subclass of ConfigModel."
            )

        self.config = config

    async def chat_stream(
        self, payload: chat.RequestPayload
    ) -> AsyncIterable[chat.StreamResponsePayload]:
        raise AIGatewayException(
            status_code=501,
            detail=f"The chat streaming route is not implemented for {self.NAME} models.",
        )

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        raise AIGatewayException(
            status_code=501,
            detail=f"The chat route is not implemented for {self.NAME} models.",
        )

    async def completions_stream(
        self, payload: completions.RequestPayload
    ) -> AsyncIterable[completions.StreamResponsePayload]:
        raise AIGatewayException(
            status_code=501,
            detail=f"The completions streaming route is not implemented for {self.NAME} models.",
        )

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        raise AIGatewayException(
            status_code=501,
            detail=f"The completions route is not implemented for {self.NAME} models.",
        )

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        raise AIGatewayException(
            status_code=501,
            detail=f"The embeddings route is not implemented for {self.NAME} models.",
        )

    def _validate_passthrough_action(self, action: PassthroughAction) -> str:
        """
        Validates that the passthrough action is supported by this provider
        and returns the provider path.

        Args:
            action: The passthrough action to validate

        Returns:
            The provider path for the action
        """
        provider_path = self.PASSTHROUGH_PROVIDER_PATHS.get(action)
        if provider_path is None:
            route = PASSTHROUGH_ROUTES.get(action, action.value)
            supported_routes = ", ".join(
                f"/gateway{route} (provider_path: {path})"
                for act in self.PASSTHROUGH_PROVIDER_PATHS.keys()
                if (route := PASSTHROUGH_ROUTES.get(act))
                and (path := self.PASSTHROUGH_PROVIDER_PATHS.get(act))
            )
            raise AIGatewayException(
                status_code=400,
                detail=f"Unsupported passthrough endpoint '{route}' for {self.NAME} provider. "
                f"Supported endpoints: {supported_routes}",
            )
        return provider_path

    async def passthrough(
        self, action: PassthroughAction, payload: dict[str, Any]
    ) -> dict[str, Any] | AsyncIterable[bytes]:
        """
        Unified passthrough endpoint for raw API requests.

        Args:
            action: The passthrough action to perform (e.g., OPENAI_CHAT, OPENAI_EMBEDDINGS)
            payload: Raw request payload in the format expected by the target API

        Returns:
            Raw response from the target API, optionally as an async iterable for streaming

        Raises:
            AIGatewayException: If the passthrough action is not implemented for this provider
        """
        route = PASSTHROUGH_ROUTES.get(action)
        raise AIGatewayException(
            status_code=501,
            detail=f"The passthrough route '{route}' is not implemented for {self.NAME} models.",
        )

    @staticmethod
    def check_for_model_field(payload):
        if "model" in payload:
            raise AIGatewayException(
                status_code=422,
                detail="The parameter 'model' is not permitted to be passed. The route being "
                "queried already defines a model instance.",
            )


class TrafficRouteProvider(BaseProvider):
    """
    A provider that split traffic and forward to multiple providers
    """

    NAME: str = "TrafficRoute"

    def __init__(
        self,
        configs: list[EndpointConfig],
        traffic_splits: list[int],
        routing_strategy: str,
    ):
        from mlflow.gateway.providers import get_provider

        if len(configs) != len(traffic_splits):
            raise MlflowException.invalid_parameter_value(
                "'configs' and 'traffic_splits' should have the same length."
            )

        if routing_strategy != "TRAFFIC_SPLIT":
            raise MlflowException.invalid_parameter_value(
                "'routing_strategy' must be 'TRAFFIC_SPLIT'."
            )

        self._providers = [get_provider(config.model.provider)(config) for config in configs]

        self._weights = np.array(traffic_splits, dtype=np.float32) / 100
        self._indices = np.arange(len(self._providers))

    def _get_provider(self):
        chosen_index = np.random.choice(self._indices, p=self._weights)
        return self._providers[chosen_index]

    async def chat_stream(
        self, payload: chat.RequestPayload
    ) -> AsyncIterable[chat.StreamResponsePayload]:
        prov = self._get_provider()
        async for i in prov.chat_stream(payload):
            yield i

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        prov = self._get_provider()
        return await prov.chat(payload)

    async def completions_stream(
        self, payload: completions.RequestPayload
    ) -> AsyncIterable[completions.StreamResponsePayload]:
        prov = self._get_provider()
        async for i in prov.completions_stream(payload):
            yield i

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        prov = self._get_provider()
        return await prov.completions(payload)

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        prov = self._get_provider()
        return await prov.embeddings(payload)

    async def passthrough(
        self, action: PassthroughAction, payload: dict[str, Any]
    ) -> dict[str, Any] | AsyncIterable[bytes]:
        prov = self._get_provider()
        return await prov.passthrough(action, payload)


class ProviderAdapter(ABC):
    @classmethod
    @abstractmethod
    def model_to_embeddings(cls, resp, config): ...

    @classmethod
    @abstractmethod
    def model_to_completions(cls, resp, config): ...

    @classmethod
    def model_to_completions_streaming(cls, resp, config):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def completions_to_model(cls, payload, config): ...

    @classmethod
    def completions_streaming_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def model_to_chat(cls, resp, config):
        raise NotImplementedError

    @classmethod
    def model_to_chat_streaming(cls, resp, config):
        raise NotImplementedError

    @classmethod
    def chat_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def chat_streaming_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def embeddings_to_model(cls, payload, config): ...

    @classmethod
    def check_keys_against_mapping(cls, mapping, payload):
        for k1, k2 in mapping.items():
            if k2 in payload:
                raise AIGatewayException(
                    status_code=400, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
