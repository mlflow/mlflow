import os

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI

import mlflow

# Note: Ensure that the package 'google-search-results' is installed via pypi to run this example
# and that you have a accounts with SerpAPI and OpenAI to use their APIs.

# Ensuring necessary API keys are set
assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."
assert "SERPAPI_API_KEY" in os.environ, "Please set the SERPAPI_API_KEY environment variable."

# Load the language model for agent control
llm = OpenAI(temperature=0)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Log the agent in an MLflow run
with mlflow.start_run():
    logged_model = mlflow.langchain.log_model(agent, name="langchain_model")

# Load the logged agent model for prediction
loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)

# Generate an inference result using the loaded model
question = "What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?"

answer = loaded_model.predict([{"input": question}])

print(answer)
