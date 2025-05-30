from typing import Literal

from langchain_core.messages import AIMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

import mlflow


class FakeOpenAI(ChatOpenAI, extra="allow"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._responses = iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(name="get_weather", args={"city": "sf"}, id="123")],
                    usage_metadata={"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
                ),
                AIMessage(
                    content="The weather in San Francisco is always sunny!",
                    usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
                ),
            ]
        )

    def _generate(self, *args, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=next(self._responses))])


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"


llm = FakeOpenAI()
tools = [get_weather]
graph = create_react_agent(llm, tools)

mlflow.models.set_model(graph)
