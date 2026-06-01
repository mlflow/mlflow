import os
from typing import Any

# See `tests/langchain/sample_code/chain.py` for why fake creds are set.
os.environ.setdefault("DATABRICKS_HOST", "https://fake-host")
os.environ.setdefault("DATABRICKS_TOKEN", "fake-token")

from databricks_langchain import ChatDatabricks
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters.character import CharacterTextSplitter

from mlflow.models import set_model


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
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Databricks"))])

        @property
        def _llm_type(self) -> str:
            return "fake chat model"

    return FakeChatModel(endpoint=endpoint)


text_path = "tests/langchain/state_of_the_union.txt"
loader = TextLoader(text_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = FakeEmbeddings(size=5)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template(
    "Answer the following question based on the context: {context}\nQuestion: {question}"
)
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
