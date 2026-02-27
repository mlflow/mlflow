from operator import itemgetter
from typing import Any

from langchain.agents import AgentExecutor, tool
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import SimpleChatModel
from langchain.prompts import PromptTemplate
from langchain.schema.messages import BaseMessage
from langchain.schema.runnable import RunnableLambda

from mlflow.models import ModelConfig, set_model

base_config = ModelConfig(development_config="tests/langchain/agent_executor/config.yml")

prompt_with_history = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=base_config.get("prompt_with_history_str"),
)


def extract_question(input):
    return input[-1]["content"]


def extract_history(input):
    return input[:-1]


@tool
def custom_tool(query: str):
    """
    Mock a tool
    """
    return "Databricks"


class FakeChatModel(SimpleChatModel):
    """Fake Chat Model wrapper for testing purposes."""

    endpoint_name: str = "fake-endpoint"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        return "Databricks"

    @property
    def _llm_type(self) -> str:
        return "fake chat model"


fake_chat_model = FakeChatModel()
llm_with_tools = fake_chat_model.bind(tools=[custom_tool])
agent = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | prompt_with_history
    | llm_with_tools
    | ToolsAgentOutputParser()
)

model = AgentExecutor(agent=agent, tools=[custom_tool])
set_model(model)
