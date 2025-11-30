"""
This is an example for leveraging MLflow's auto tracing capabilities for AutoGen.

For more information about MLflow Tracing, see: https://mlflow.org/docs/latest/llms/tracing/index.html
"""

import os
from typing import Annotated, Literal

from autogen import ConversableAgent

import mlflow

# Turn on auto tracing for AutoGen by calling mlflow.autogen.autolog()
mlflow.autogen.autolog()


config_list = [
    {
        "model": "gpt-4o-mini",
        # Please set your OpenAI API Key to the OPENAI_API_KEY env var before running this example
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }
]

Operator = Literal["+", "-", "*", "/"]


def calculator(a: int, b: int, operator: Annotated[Operator, "operator"]) -> int:
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        return int(a / b)
    else:
        raise ValueError("Invalid operator")


# First define the assistant agent that suggests tool calls.
assistant = ConversableAgent(
    name="Assistant",
    system_message="You are a helpful AI assistant. "
    "You can help with simple calculations. "
    "Return 'TERMINATE' when the task is done.",
    llm_config={"config_list": config_list},
)

# The user proxy agent is used for interacting with the assistant agent
# and executes tool calls.
user_proxy = ConversableAgent(
    name="Tool Agent",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
)

# Register the tool signature with the assistant agent.
assistant.register_for_llm(name="calculator", description="A simple calculator")(calculator)
user_proxy.register_for_execution(name="calculator")(calculator)
response = user_proxy.initiate_chat(assistant, message="What is (44231 + 13312 / (230 - 20)) * 4?")
