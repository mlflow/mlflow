from typing import Any, Generator, Optional

from langchain_core.runnables.base import Runnable

from mlflow.pyfunc.model import ChatAgent
from mlflow.types.agent import ChatAgentChunk, ChatAgentMessage, ChatAgentResponse, ChatContext
from mlflow.utils.annotations import experimental


@experimental
class LangChainChatAgent(ChatAgent):
    """
    Helper class to wrap a LangChain runnable as a :py:class:`ChatAgent <mlflow.pyfunc.ChatAgent>`. Use this with :py:class:`ChatAgentOutputParser <mlflow.langchain.output_parsers.ChatAgentOutputParser>`.
    """

    def __init__(self, agent: Runnable):
        self.agent = agent

    # TODO trace this by default once manual tracing of predict_stream is supported
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        response = self.agent.invoke({"messages": self._convert_messages_to_dict(messages)})
        return ChatAgentResponse(**response)

    # TODO trace this by default once manual tracing of predict_stream is supported
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        for event in self.agent.stream({"messages": self._convert_messages_to_dict(messages)}):
            yield ChatAgentChunk(**event)
