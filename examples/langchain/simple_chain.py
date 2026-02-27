import os

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

import mlflow

# Ensure the OpenAI API key is set in the environment
assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."

# Initialize the OpenAI model and the prompt template
llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# Create the LLMChain with the specified model and prompt
chain = LLMChain(llm=llm, prompt=prompt)

# Log the LangChain LLMChain in an MLflow run
with mlflow.start_run():
    logged_model = mlflow.langchain.log_model(chain, name="langchain_model")

# Load the logged model using MLflow's Python function flavor
loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)

# Predict using the loaded model
print(loaded_model.predict([{"product": "colorful socks"}]))
