import os
import mlflow

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
chain = LLMChain(llm=llm, prompt=prompt)

with mlflow.start_run():
    logged_model = mlflow.langchain.log_model(chain, "langchain_model")

loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
print(loaded_model.predict([{"product": "colorful socks"}]))
