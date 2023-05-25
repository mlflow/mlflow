import os
import mlflow

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."
assert "SERPAPI_API_KEY" in os.environ, "Please set the SERPAPI_API_KEY environment variable."

# First, let's load the language model we're going to use to control the agent.
llm = OpenAI(temperature=0)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

with mlflow.start_run():
    logged_model = mlflow.langchain.log_model(agent, "langchain_model")

loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
print(
    loaded_model.predict(
        [
            {
                "input": "What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?"
            }
        ]
    )
)
