from abc import ABC
from typing import Tuple

from fastapi import HTTPException

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.schemas import chat, completions, embeddings


class BaseProvider(ABC):
    """
    Base class for MLflow Gateway providers.
    """

    NAME: str
    SUPPORTED_ROUTE_TYPES: Tuple[str, ...]

    def __init__(self, config: RouteConfig):
        self.config = config

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        raise NotImplementedError

    from mlflow.gateway.schemas.chat import RequestPayload as ChatRequestPayload

    def translate_payload_for_chat(self, payload: ChatRequestPayload) -> ChatRequestPayload:
        return payload

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        raise NotImplementedError

    from mlflow.gateway.schemas.completions import RequestPayload as CompletionsRequestPayload

    def translate_payload_for_completions(
        self, payload: CompletionsRequestPayload
    ) -> CompletionsRequestPayload:
        return payload

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        raise NotImplementedError

    from mlflow.gateway.schemas.embeddings import RequestPayload as EmbeddingsRequestPayload

    def translate_payload_for_embeddings(
        self, payload: EmbeddingsRequestPayload
    ) -> EmbeddingsRequestPayload:
        return payload

    @staticmethod
    def check_for_model_field(payload):
        if "model" in payload:
            raise HTTPException(
                status_code=422,
                detail="The parameter 'model' is not permitted to be passed. The route being "
                "queried already defines a model instance.",
            )
