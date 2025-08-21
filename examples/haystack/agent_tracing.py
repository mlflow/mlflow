"""
Example demonstrating MLflow tracing for Haystack Agents.

This example shows how MLflow captures the full execution flow of an Agent,
including its internal calls to chat generators and tool invokers.
"""

import asyncio
import datetime

from haystack.components.agents import Agent
from haystack.components.generators import OpenAIGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

import mlflow

# Turn on auto tracing for Haystack
mlflow.haystack.autolog()


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


# Example 1: Simple Agent without tools (behaves like ChatGenerator)
def simple_agent_example():
    print("=== Simple Agent Example (No Tools) ===")

    # Create a chat generator
    llm = OpenAIGenerator(model="gpt-4o-mini")

    # Create an agent without tools
    agent = Agent(chat_generator=llm, system_prompt="You are a helpful assistant.")

    # Warm up the agent
    agent.warm_up()

    # Run the agent
    messages = [ChatMessage.from_user("What is the capital of France?")]
    result = agent.run(messages=messages)

    print("User:", messages[0].content)
    print("Agent:", result["last_message"].content)
    print()


# Example 2: Agent with tools
def agent_with_tools_example():
    print("=== Agent with Tools Example ===")

    # Create tools
    add_tool = Tool(
        name="add_numbers",
        description="Add two numbers together",
        function=add_numbers,
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    )

    multiply_tool = Tool(
        name="multiply_numbers",
        description="Multiply two numbers together",
        function=multiply_numbers,
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    )

    # Create a chat generator
    llm = OpenAIGenerator(model="gpt-4o-mini")

    # Create an agent with tools
    agent = Agent(
        chat_generator=llm,
        tools=[add_tool, multiply_tool],
        system_prompt="You are a helpful math assistant. Use the tools provided to help with calculations.",
    )

    # Warm up the agent
    agent.warm_up()

    # Run the agent with a calculation request
    messages = [ChatMessage.from_user("What is 15 + 27, and what is the result multiplied by 3?")]
    result = agent.run(messages=messages)

    print("User:", messages[0].content)
    print("Agent:", result["last_message"].content)
    print("\nNumber of messages exchanged:", len(result["messages"]))
    print()


# Example 3: Async Agent
async def async_agent_example():
    print("=== Async Agent Example ===")

    # Create a simple tool
    def get_time() -> str:
        """Get the current time."""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    time_tool = Tool(
        name="get_time",
        description="Get the current time",
        function=get_time,
        parameters={"type": "object", "properties": {}},
    )

    # Create components
    llm = OpenAIGenerator(model="gpt-4o-mini")

    agent = Agent(
        chat_generator=llm,
        tools=[time_tool],
        system_prompt="You are a helpful assistant that can tell the time.",
    )

    # Warm up the agent
    agent.warm_up()

    # Run the agent asynchronously
    messages = [ChatMessage.from_user("What time is it now?")]
    result = await agent.run_async(messages=messages)

    print("User:", messages[0].content)
    print("Agent:", result["last_message"].content)
    print()


# Run examples
if __name__ == "__main__":
    # Run synchronous examples
    simple_agent_example()
    agent_with_tools_example()

    # Run asynchronous example
    asyncio.run(async_agent_example())

    print("=== Tracing Complete ===")
    print("The Agent execution traces have been logged to MLflow.")
    print("You can view the hierarchical trace structure showing:")
    print("- Agent.run as the parent span")
    print("- chat_generator and tool_invoker as child spans")
    print("- All tool invocations and their results")
