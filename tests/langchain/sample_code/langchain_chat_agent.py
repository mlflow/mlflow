from operator import itemgetter
from typing import Any, Generator

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import Runnable
from langchain_openai import ChatOpenAI

import mlflow
from mlflow.langchain.output_parsers import ChatAgentOutputParser
from mlflow.pyfunc.model import ChatAgent
from mlflow.types.agent import ChatAgentChunk, ChatAgentMessage, ChatAgentResponse, ChatContext


class FakeOpenAI(ChatOpenAI, extra="allow"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._responses = iter([AIMessage(content="1")])
        self._stream_responses = iter(
            [
                AIMessageChunk(content="1"),
                AIMessageChunk(content="2"),
                AIMessageChunk(content="3"),
            ]
        )

    def _generate(self, *args, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=next(self._responses))])

    def _stream(self, *args, **kwargs):
        for r in self._stream_responses:
            yield ChatGenerationChunk(message=r)


mlflow.langchain.autolog()


# Helper functions
def extract_user_query_string(messages):
    return messages[-1]["content"]


def extract_chat_history(messages):
    return messages[:-1]


# Define components
prompt = ChatPromptTemplate.from_template(
    """Previous conversation:
{chat_history}

User's question:
{question}"""
)

model = FakeOpenAI()
output_parser = ChatAgentOutputParser()

# Chain definition
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
    }
    | prompt
    | model
    | output_parser
)


class LangChainChatAgent(ChatAgent):
    """
    Helper class to wrap a LangChain runnable as a :py:class:`ChatAgent <mlflow.pyfunc.ChatAgent>`.
    Use this class with
    :py:class:`ChatAgentOutputParser <mlflow.langchain.output_parsers.ChatAgentOutputParser>`.
    """

    def __init__(self, agent: Runnable):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: ChatContext | None = None,
        custom_inputs: dict[str, Any] | None = None,
    ) -> ChatAgentResponse:
        response = self.agent.invoke({"messages": self._convert_messages_to_dict(messages)})
        return ChatAgentResponse(**response)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: ChatContext | None = None,
        custom_inputs: dict[str, Any] | None = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        for event in self.agent.stream({"messages": self._convert_messages_to_dict(messages)}):
            yield ChatAgentChunk(**event)


chat_agent = LangChainChatAgent(chain)

mlflow.models.set_model(chat_agent)
