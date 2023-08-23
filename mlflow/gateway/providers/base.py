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

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        raise NotImplementedError

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        raise NotImplementedError

    @staticmethod
    def check_for_model_field(payload):
        if "model" in payload:
            raise HTTPException(
                status_code=422,
                detail="The parameter 'model' is not permitted to be passed. The route being "
                "queried already defines a model instance.",
            )
