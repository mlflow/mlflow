import os
from operator import itemgetter

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

import mlflow

# Uncomment the following to use the full abilities of langchain autologgin
# %pip install `langchain_community>=0.0.16`
# These two libraries enable autologging to log text analysis related artifacts
# %pip install textstat spacy

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."

# Enable mlflow langchain autologging
# Note: We only support auto-logging models that do not contain retrievers
mlflow.langchain.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    registered_model_name="lc_model",
)

prompt_with_history_str = """
Here is a history between you and a human: {chat_history}

Now, please answer this question: {question}
"""
prompt_with_history = PromptTemplate(
    input_variables=["chat_history", "question"], template=prompt_with_history_str
)


def extract_question(input):
    return input[-1]["content"]


def extract_history(input):
    return input[:-1]


llm = OpenAI(temperature=0.9)

# Build a chain with LCEL
chain_with_history = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | prompt_with_history
    | llm
    | StrOutputParser()
)

inputs = {"messages": [{"role": "user", "content": "Who owns MLflow?"}]}

print(chain_with_history.invoke(inputs))
# sample output:
# "1. Databricks\n2. Microsoft\n3. Google\n4. Amazon\n\nEnter your answer: 1\n\n
# Correct! MLflow is an open source project developed by Databricks. ...

# We automatically log the model and trace related artifacts
# A model with name `lc_model` is registered, we can load it back as a PyFunc model
model_name = "lc_model"
model_version = 1
loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
print(loaded_model.predict(inputs))
