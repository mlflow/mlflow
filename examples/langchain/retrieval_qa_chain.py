import mlflow
import os

from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from mlflow.langchain import _LOADER_FN_KEY, _PERSIST_DIR_KEY

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."

# Load the vectorstore from persist_dir
persist_dir = "tests/langchain/faiss_index"
embeddings = OpenAIEmbeddings()
db = FAISS.load_local(persist_dir, embeddings)

# Create the RetrievalQA chain
retrievalQA = RetrievalQA.from_llm(llm=OpenAI(), retriever=db.as_retriever())

# Log the retrievalQA chain
def load_retriever(persist_directory):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(persist_directory, embeddings)
    return vectorstore.as_retriever()


with mlflow.start_run() as run:
    logged_model = mlflow.langchain.log_model(
        retrievalQA,
        artifact_path="retrieval_qa",
        metadata={_LOADER_FN_KEY: load_retriever, _PERSIST_DIR_KEY: persist_dir},
    )

# Load the retrievalQA chain
loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
loaded_model.predict([{"query": "What did the president say about Ketanji Brown Jackson"}])
