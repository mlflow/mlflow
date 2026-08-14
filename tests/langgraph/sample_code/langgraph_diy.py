# Sample code that contains custom python nodes
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import mlflow


def generate(state):
    messages = state["messages"]
    llm = ChatOpenAI()
    response = llm.invoke(messages[-1].content)
    return {"messages": response}


def should_continue(state):
    if len(state["messages"]) > 3:
        return "no"
    else:
        return "yes"


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


workflow = StateGraph(AgentState)
workflow.add_node("generate", generate)
workflow.add_edge(START, "generate")
workflow.add_conditional_edges(
    "generate",
    should_continue,
    {
        "yes": "generate",
        "no": END,
    },
)

graph = workflow.compile()

mlflow.models.set_model(graph)
