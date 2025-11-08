from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import mlflow
from mlflow.models import set_model

prompt = ChatPromptTemplate.from_template(
    mlflow.load_prompt("prompts:/qa_prompt@production").to_single_brace_format()
)
chain = prompt | ChatOpenAI(temperature=0) | StrOutputParser()

set_model(chain)
