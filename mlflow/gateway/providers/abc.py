from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Tuple
from ..schemas import chat, embeddings


class AbstractGatewayClient(ABC):
    PROVIDER: str
    SUPPORTED_TASKS: Tuple[str, ...]

    def __init__(self, config: Any) -> None:
        self.config = config

    async def chat(self, json: chat.Request) -> chat.Response:
        raise NotImplementedError

    async def completions(self, json: chat.Request) -> chat.Response:
        raise NotImplementedError

    async def embeddings(self, json: embeddings.Request) -> embeddings.Response:
        raise NotImplementedError
