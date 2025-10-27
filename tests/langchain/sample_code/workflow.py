import json
import os
from typing import Any, Sequence

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

import mlflow
from mlflow.langchain.chat_agent_langgraph import (
    ChatAgentState,
    ChatAgentToolNode,
)

os.environ["OPENAI_API_KEY"] = "test"


class FakeOpenAI(ChatOpenAI, extra="allow"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._responses = iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(name="uc_tool_format", args={}, id="123")],
                ),
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(name="lc_tool_format", args={}, id="456")],
                ),
                AIMessage(content="Successfully generated", id="789"),
            ]
        )

    def _generate(self, *args, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=next(self._responses))])


@tool
def uc_tool_format() -> str:
    """Returns uc tool format"""
    return json.dumps(
        {
            "format": "SCALAR",
            "value": '{"content":"hi","attachments":{"a":"b"},"custom_outputs":{"c":"d"}}',
            "truncated": False,
        }
    )


@tool
def lc_tool_format() -> dict[str, Any]:
    """Returns lc tool format"""
    nums = [1, 2]
    return {
        "content": f"Successfully generated array of 2 random ints: {nums}.",
        "attachments": {"key1": "attach1", "key2": "attach2"},
        "custom_outputs": {"random_nums": nums},
    }


tools = [uc_tool_format, lc_tool_format]


def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: ToolNode | Sequence[BaseTool],
    agent_prompt: str | None = None,
) -> CompiledStateGraph:
    model = model.bind_tools(tools)

    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are function calls, continue. else, end
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    @mlflow.trace
    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


mlflow.langchain.autolog()
llm = FakeOpenAI()
graph = create_tool_calling_agent(llm, tools)

mlflow.models.set_model(graph)
