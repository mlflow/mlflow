from abc import ABC, abstractmethod
from typing import Tuple


class AbstractProvider(ABC):
    NAME: str
    SUPPORTED_TASKS: Tuple[str, ...]

    def __init__(self, config):
        self.config = config

    async def chat(self, payload):
        raise NotImplementedError

    async def completions(self, payload):
        raise NotImplementedError

    async def embeddings(self, payload):
        raise NotImplementedError
