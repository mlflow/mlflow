from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

from mlflow.models import set_model

model = ChatOpenAI(temperature=0).configurable_fields(
    temperature=ConfigurableField(
        id="temperature",
        name="LLM temperature",
        description="The temperature of the LLM",
    )
)

prompt = PromptTemplate.from_template("Pick a random number above {x}")
chain = prompt | model

set_model(chain)
