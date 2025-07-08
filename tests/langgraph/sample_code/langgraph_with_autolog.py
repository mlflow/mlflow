from dataclasses import dataclass

from langchain.tools import tool
from langgraph.graph import END, StateGraph

import mlflow

mlflow.langchain.autolog()


@dataclass
class OverallState:
    name: str = "LangChain"  # add whatever fields you need


@tool
def my_tool():
    """
    Called as the very first node.
    Side-effect: add an MLflow tag to the *current* trace.
    Must return a dict of state-field updates.
    """
    mlflow.update_current_trace(tags={"order_total": "hello"})
    return {"status": "done"}


builder = StateGraph(dict)
builder.add_node("test_tool", my_tool)  # ← calls your tool
builder.set_entry_point("test_tool")  # start here
builder.add_edge("test_tool", END)  # nothing else to do

graph = builder.compile()
mlflow.models.set_model(graph)
