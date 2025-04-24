from typing import Literal

from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

import mlflow
from mlflow.entities.span import SpanType


class FakeOpenAI(ChatOpenAI, extra="allow"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._responses = iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(name="get_weather", args={"city": "sf"}, id="123")],
                ),
                AIMessage(content="The weather in San Francisco is always sunny!"),
            ]
        )

    def _generate(self, *args, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=next(self._responses))])


def get_inner_runnable():
    llm = ChatOpenAI()
    prompt = PromptTemplate.from_template("what is the weather in {city}?")
    return prompt | llm | StrOutputParser()


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    with mlflow.start_span(name="get_weather_inner", span_type=SpanType.CHAIN) as span:
        span.set_inputs(city)

        # Call another LangChain module
        inner_runnable = get_inner_runnable()
        inner_runnable.invoke({"city": city})

        if city == "nyc":
            output = "It might be cloudy in nyc"
        elif city == "sf":
            output = "It's always sunny in sf"
        span.set_outputs(output)
    return output


llm = FakeOpenAI()
tools = [get_weather]
graph = create_react_agent(llm, tools)

mlflow.models.set_model(graph)
