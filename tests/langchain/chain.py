import os
from typing import Any, List, Optional

from langchain.document_loaders import TextLoader
from langchain.embeddings.fake import FakeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

import mlflow
from mlflow.langchain._rag_utils import _set_chain


def get_fake_chat_model(endpoint="fake-endpoint"):
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.chat_models import ChatDatabricks, ChatMlflow
    from langchain.schema.messages import BaseMessage
    from langchain_core.outputs import ChatResult

    class FakeChatModel(ChatDatabricks):
        """Fake Chat Model wrapper for testing purposes."""

        endpoint: str = "fake-endpoint"

        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            response = {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Databricks",
                        },
                        "finish_reason": None,
                    }
                ],
            }
            return ChatMlflow._create_chat_result(response)

        @property
        def _llm_type(self) -> str:
            return "fake chat model"

    return FakeChatModel(endpoint=endpoint)


text_path = mlflow.langchain._rag_utils.__databricks_rag_config_path__
assert (
    os.path.basename(os.path.abspath(mlflow.langchain._rag_utils.__databricks_rag_config_path__))
    == "state_of_the_union.txt"
)
assert os.path.exists(text_path)
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

_set_chain(retrieval_chain)
