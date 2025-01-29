from typing import Generator, Optional

from langchain_core.runnables.base import Runnable

from mlflow.pyfunc.model import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentParams, ChatAgentResponse
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
    def predict(
        self, messages: list[ChatAgentMessage], params: Optional[ChatAgentParams] = None
    ) -> ChatAgentResponse:
        response = self.agent.invoke({"messages": self._convert_messages_to_dict(messages)})
        return ChatAgentResponse(**response)

    # TODO trace this by default once manual tracing of predict_stream is supported
    def predict_stream(
        self, messages: list[ChatAgentMessage], params: Optional[ChatAgentParams] = None
    ) -> Generator[ChatAgentResponse, None, None]:
        for event in self.agent.stream({"messages": self._convert_messages_to_dict(messages)}):
            yield ChatAgentResponse(**event)
