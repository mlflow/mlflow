from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import mlflow

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is {product}?",
)
llm = ChatOpenAI(temperature=0.1, stream_usage=True)
chain = prompt | llm | StrOutputParser()

mlflow.models.set_model(chain)
