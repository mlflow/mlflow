import time

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.providers import BaseProvider
from mlflow.gateway.schemas import completions

from foo.config import FooConfig


class FooProvider(BaseProvider):
    NAME = "foo"

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, FooConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.foo_config: FooConfig = config.model.config

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        return completions.ResponsePayload(
            id="id-123",
            created=int(time.time()),
            model=self.config.model.name,
            choices=[completions.Choice(index=0, text="This is a response from FooProvider")],
            usage=completions.CompletionsUsage(
                prompt_tokens=12,
                completion_tokens=34,
                total_tokens=46,
            ),
        )
