import time

from foo.config import FooConfig
from mlflow.gateway.config import RouteConfig
from mlflow.gateway.providers import BaseProvider
from mlflow.gateway.schemas import chat


class FooProvider(BaseProvider):
    NAME = "foo"

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, FooConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.foo_config: FooConfig = config.model.config

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        return chat.ResponsePayload(
            id="id-123",
            created=int(time.time()),
            model=self.config.model.name,
            choices=[
                chat.Choice(
                    index=0,
                    message=chat.ResponseMessage(
                        role="assistant", content="This is a response from FooProvider"
                    ),
                )
            ],
            usage=chat.ChatUsage(
                prompt_tokens=10,
                completion_tokens=18,
                total_tokens=28,
            ),
        )