from typing import Optional

from mlflow.pyfunc.model import ChatAgent, ChatAgentMessage, ChatAgentParams, ChatAgentResponse


class LangChainChatAgent(ChatAgent):
    def __init__(self, agent):
        self.agent = agent

    def predict(self, messages: list[ChatAgentMessage], params: Optional[ChatAgentParams] = None):
        response = self.agent.invoke({"messages": self._convert_messages_to_dict(messages)})
        return ChatAgentResponse(**response)

    def predict_stream(
        self, messages: list[ChatAgentMessage], params: Optional[ChatAgentParams] = None
    ):
        for event in self.agent.stream({"messages": self._convert_messages_to_dict(messages)}):
            yield ChatAgentResponse(**event)
