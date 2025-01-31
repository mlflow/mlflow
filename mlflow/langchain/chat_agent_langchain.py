from typing import Generator

from langchain_core.runnables.base import Runnable

from mlflow.pyfunc.model import ChatAgent
from mlflow.types.agent import (
    ChatAgentRequest,
    ChatAgentResponse,
)
from mlflow.utils.annotations import experimental


@experimental
class LangChainChatAgent(ChatAgent):
    """
    Helper class to wrap a LangChain runnable as a ChatAgent. Use this with :py:class:`mlflow.
    langchain.output_parsers.ChatAgentOutputParser`.
    """

    def __init__(self, agent: Runnable):
        self.agent = agent

    # TODO trace this by default once manual tracing of predict_stream is supported
    def predict(self, model_input: ChatAgentRequest) -> ChatAgentResponse:
        response = self.agent.invoke(model_input.model_dump_compat(exclude_none=True))
        return ChatAgentResponse(**response)

    # TODO trace this by default once manual tracing of predict_stream is supported
    def predict_stream(
        self, model_input: ChatAgentRequest
    ) -> Generator[ChatAgentResponse, None, None]:
        for event in self.agent.stream(model_input.model_dump_compat(exclude_none=True)):
            yield ChatAgentResponse(**event)
