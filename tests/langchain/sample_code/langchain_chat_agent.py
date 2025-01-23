from operator import itemgetter

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

import mlflow
from mlflow.langchain.chat_agent_langchain import LangChainChatAgent
from mlflow.langchain.output_parsers import ChatAgentOutputParser


class FakeOpenAI(ChatOpenAI, extra="allow"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._responses = iter(
            [
                AIMessage(content="1"),
                AIMessage(content="2"),
                AIMessage(content="3"),
            ]
        )

    def _generate(self, *args, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=next(self._responses))])


mlflow.langchain.autolog()


# Helper functions
def extract_user_query_string(messages):
    return messages[-1]["content"]


def extract_chat_history(messages):
    return messages[:-1]


# Define components
prompt = ChatPromptTemplate.from_template(
    """Previous conversation:
{chat_history}

User's question:
{question}"""
)

model = FakeOpenAI()
output_parser = ChatAgentOutputParser()

# Chain definition
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
    }
    | prompt
    | model
    | output_parser
)
chat_agent = LangChainChatAgent(chain)

mlflow.models.set_model(chat_agent)
