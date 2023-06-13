from abc import ABC
from typing import Tuple

from ..schemas import chat, completions, embeddings
from ..config import RouteConfig


class BaseProvider(ABC):
    """
    Abstract base class for MLflow gateway providers.
    """

    NAME: str
    SUPPORTED_TASKS: Tuple[str, ...]

    def __init__(self, config: RouteConfig):
        self.config = config

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        raise NotImplementedError

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        raise NotImplementedError

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        raise NotImplementedError
