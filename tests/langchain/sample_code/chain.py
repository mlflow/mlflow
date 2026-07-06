import dbutils

dbutils.library.restartPython()

import os
from typing import Any

# `databricks-langchain` versions older than ~0.9 eagerly construct a
# `WorkspaceClient` inside `ChatDatabricks.__init__`, which requires
# Databricks credentials. Cross-version test jobs pin those older releases
# (e.g. 0.8.2 with `langchain==0.3.30`), so set fake creds before the fake
# chat model is instantiated below.
os.environ.setdefault("DATABRICKS_HOST", "https://fake-host")
os.environ.setdefault("DATABRICKS_TOKEN", "fake-token")

from databricks_langchain import ChatDatabricks
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters.character import CharacterTextSplitter

import mlflow
from mlflow.models import ModelConfig, set_model, set_retriever_schema

base_config = ModelConfig(development_config="tests/langchain/config.yml")


def get_fake_chat_model(endpoint="fake-endpoint"):
    class FakeChatModel(ChatDatabricks):
        """Fake Chat Model wrapper for testing purposes."""

        endpoint: str = "fake-endpoint"

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            message = AIMessage(content=str(base_config.get("response")))
            return ChatResult(generations=[ChatGeneration(message=message)])

        @property
        def _llm_type(self) -> str:
            return "fake chat model"

    return FakeChatModel(endpoint=endpoint)


# No need to define the model, but simulating common practice in dev notebooks
mlflow.langchain.autolog()

text_path = "tests/langchain/state_of_the_union.txt"
loader = TextLoader(text_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = FakeEmbeddings(size=base_config.get("embedding_size"))
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template(base_config.get("llm_prompt_template"))
retrieval_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | get_fake_chat_model()
    | StrOutputParser()
)

set_model(retrieval_chain)
set_retriever_schema(
    primary_key="primary-key",
    text_column="text-column",
    doc_uri="doc-uri",
    other_columns=["column1", "column2"],
)

retrieval_chain.invoke({"question": "What is the capital of Japan?"})
