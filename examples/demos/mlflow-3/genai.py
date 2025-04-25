# MLflow 3 GenAI Example
# In this example, we will create an agent and then evaluate its performance. First, we will define the agent and log it to MLflow.

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import mlflow

# Define the chain
chat_model = ChatOpenAI(name="gpt-4o")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a chatbot that can answer questions about Databricks."),
        ("user", "{messages}"),
    ]
)
chain = prompt | chat_model

# Log the chain with MLflow, specifying its parameters
# As a new feature, the LoggedModel entity is linked to its name and params
logged_model = mlflow.langchain.log_model(
    lc_model=chain,
    name="basic_chain",
    params={"temperature": 0.1, "max_tokens": 2000, "prompt_template": str(prompt)},
    model_type="agent",
    input_example={"messages": "What is MLflow?"},
)

# Inspect the LoggedModel and its properties
print(logged_model.model_id, logged_model.params)
# m-123802d4ba324f4d8baa456eb8b5c061, {'max_tokens': '2000', 'prompt_template': "input_variables=['messages'] messages=[SystemMessagePromptTemplate(...), HumanMessagePromptTemplate(...)]", 'temperature': '0.1'}

# Then, we will interactively query the chain in a notebook to make sure that it's viable enough for further testing. These traces can be viewed in UI, under the Traces tab of the model details page.

# Enable autologging so that interactive traces from the chain are automatically linked to its LoggedModel
mlflow.langchain.autolog()
loaded_chain = mlflow.langchain.load_model(f"models:/{logged_model.model_id}")
chain_inputs = [
    {"messages": "What is MLflow?"},
    {"messages": "What is Unity Catalog?"},
    {"messages": "What are user-defined functions (UDFs)?"},
]

for chain_input in chain_inputs:
    loaded_chain.invoke(chain_input)

# Print out the traces linked to the LoggedModel
print(mlflow.search_traces(model_id=logged_model.model_id))
