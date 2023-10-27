from abc import ABC, abstractclassmethod
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
        from fastapi import HTTPException

        if "model" in payload:
            raise HTTPException(
                status_code=422,
                detail="The parameter 'model' is not permitted to be passed. The route being "
                "queried already defines a model instance.",
            )


class ProviderAdapter(ABC):
    """ """

    @abstractclassmethod
    def model_to_embeddings(cls, resp, config):
        raise NotImplementedError

    @abstractclassmethod
    def model_to_completions(cls, resp, config):
        raise NotImplementedError

    @abstractclassmethod
    def completions_to_model(cls, payload, config):
        raise NotImplementedError

    @abstractclassmethod
    def embeddings_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def check_keys_against_mapping(cls, mapping, payload):
        for k1, k2 in mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=400, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
