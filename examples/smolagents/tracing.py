"""
This is an example for leveraging MLflow's auto tracing capabilities for Smolagents.
For more information about MLflow Tracing, see: https://mlflow.org/docs/latest/llms/tracing/index.html
"""

from smolagents import CodeAgent, LiteLLMModel

import mlflow

# Turn on auto tracing for Smolagents by calling mlflow.smolagents.autolog()
mlflow.smolagents.autolog()

model = LiteLLMModel(model_id="openai/gpt-4o-mini", api_key="API_KEY")
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

result = agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
