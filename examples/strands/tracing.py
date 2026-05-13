import mlflow

mlflow.strands.autolog()
mlflow.set_experiment("Strand Agent")

from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator

model = OpenAIModel(
    client_args={"api_key": "<api-key>"},
    # **model_config
    model_id="gpt-4o",
    params={
        "max_tokens": 1000,
        "temperature": 0.7,
    },
)

agent = Agent(model=model, tools=[calculator])
response = agent("What is 2+2")
print(response)
