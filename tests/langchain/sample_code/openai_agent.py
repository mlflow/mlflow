import itertools
import os

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.messages import AIMessageChunk, ToolCall
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

import mlflow


class FakeOpenAI(ChatOpenAI, extra="allow"):
    # In normal LangChain tests, we use the fake OpenAI server to mock the OpenAI REST API.
    # The fake server returns the input payload as it is. However, for agent tests, the
    # response should be a specific format so that the agent can parse it correctly.
    # Also, mocking with mock.patch does not work for testing model serving (as the server
    # will run in a separate process).
    # Therefore, we mock the OpenAI client in the model definition here.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Using itertools.cycle to create an infinite iterator
        self._responses = itertools.cycle(
            [
                AIMessageChunk(
                    content="",
                    tool_calls=[ToolCall(name="multiply", args={"a": 2, "b": 3}, id="123")],
                ),
                AIMessageChunk(content="The result of 2 * 3 is 6."),
            ]
        )

    def _stream(self, *args, **kwargs):
        yield ChatGenerationChunk(message=next(self._responses))


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def create_openai_agent():
    llm = FakeOpenAI()
    tools = [add, multiply]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    return_immediate = os.environ.get("RETURN_INTERMEDIATE_STEPS", "false").lower() == "true"
    return AgentExecutor(
        agent=agent,
        tools=tools,
        # Use env var to switch model configuration during testing
        return_intermediate_steps=return_immediate,
    )


agent = create_openai_agent()
mlflow.models.set_model(agent)
