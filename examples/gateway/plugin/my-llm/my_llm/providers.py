import time

from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.providers import BaseProvider
from mlflow.gateway.schemas import chat
from my_llm.config import MyLLMConfig


class MyLLMProvider(BaseProvider):
    NAME = "MyLLM"
    CONFIG_TYPE = MyLLMConfig

    def __init__(self, config: EndpointConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, MyLLMConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.my_llm_config: MyLLMConfig = config.model.config

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        return chat.ResponsePayload(
            id="id-123",
            created=int(time.time()),
            model=self.config.model.name,
            choices=[
                chat.Choice(
                    index=0,
                    message=chat.ResponseMessage(
                        role="assistant", content="This is a response from MyLLMProvider"
                    ),
                )
            ],
            usage=chat.ChatUsage(
                prompt_tokens=10,
                completion_tokens=18,
                total_tokens=28,
            ),
        )
