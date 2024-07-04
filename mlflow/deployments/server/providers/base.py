from abc import ABC, abstractmethod
from typing import AsyncIterable, Tuple

from fastapi import HTTPException

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.schemas import chat, completions, embeddings


class BaseProvider(ABC):
    """
    Base class for MLflow Gateway providers.
    """

    NAME: str = ""
    SUPPORTED_ROUTE_TYPES: Tuple[str, ...]

    def __init__(self, config: RouteConfig):
        if self.NAME == "":
            raise ValueError(
                f"{self.__class__.__name__} is a subclass of BaseProvider and must "
                f"override 'NAME' attribute as a non-empty string."
            )

        self.config = config

    async def chat_stream(
        self, payload: chat.RequestPayload
    ) -> AsyncIterable[chat.StreamResponsePayload]:
        raise HTTPException(
            status_code=501,
            detail=f"The chat streaming route is not implemented for {self.NAME} models.",
        )

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        raise HTTPException(
            status_code=501,
            detail=f"The chat route is not implemented for {self.NAME} models.",
        )

    async def completions_stream(
        self, payload: completions.RequestPayload
    ) -> AsyncIterable[completions.StreamResponsePayload]:
        raise HTTPException(
            status_code=501,
            detail=f"The completions streaming route is not implemented for {self.NAME} models.",
        )

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        raise HTTPException(
            status_code=501,
            detail=f"The completions route is not implemented for {self.NAME} models.",
        )

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        raise HTTPException(
            status_code=501,
            detail=f"The embeddings route is not implemented for {self.NAME} models.",
        )

    @staticmethod
    def check_for_model_field(payload):
        from fastapi import HTTPException

        if "model" in payload:
            raise HTTPException(
                status_code=422,
                detail="The parameter 'model' is not permitted to be passed. The route being "
                "queried already defines a model instance.",
            )


class ProviderAdapter(ABC):
    """ """

    @classmethod
    @abstractmethod
    def model_to_embeddings(cls, resp, config):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def model_to_completions(cls, resp, config):
        raise NotImplementedError

    @classmethod
    def model_to_completions_streaming(cls, resp, config):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def completions_to_model(cls, payload, config):
        raise NotImplementedError

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
    def embeddings_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def check_keys_against_mapping(cls, mapping, payload):
        for k1, k2 in mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=400, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
