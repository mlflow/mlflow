"""
This example demonstrates how to enable automatic tracing for LangChain.

Note: this example requires the `langchain` and `langchain-openai` package to be installed.
"""

import json
import os

from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import OpenAI

import mlflow

exp = mlflow.set_experiment("mlflow-tracing-langchain")
exp_id = exp.experiment_id

# This example uses OpenAI LLM. If you want to use other LLMs, you can
# uncomment the following line and replace `OpenAI` with the desired LLM class.
assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."


# You can enable automatic tracing for LangChain by simply calling `mlflow langchain.autolog()`.
# (Note: By default this only enables tracing and does not log any other artifacts such as
#  models, dataset, etc. To enable auto logging of other artifacts, please refer to the example
#  at examples/langchain/chain_autolog.py)
mlflow.langchain.autolog()

# Build a simple chain
prompt = PromptTemplate(
    input_variables=["question"], template="Please answer this question: {question}"
)
llm = OpenAI(temperature=0.9)
chain = prompt | llm | StrOutputParser()

# Invoke the chain. Each invocation will generate a new trace.
chain.invoke({"question": "What is the capital of Japan?"})
chain.invoke({"question": "How many animals are there in the world?"})
chain.invoke({"question": "Who is the first person to land on the moon?"})

# Retrieve the traces
traces = mlflow.search_traces(experiment_ids=[exp_id], max_results=3, return_type="list")
print(json.dumps([t.to_dict() for t in traces], indent=2))

print(
    "\033[92m"
    + "🤖Now run `mlflow server` and open MLflow UI to see the trace visualization!"
    + "\033[0m"
)
