import os
import tempfile

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

import mlflow

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."

with tempfile.TemporaryDirectory() as temp_dir:
    persist_dir = os.path.join(temp_dir, "faiss_index")

    # Create the vector db, persist the db to a local fs folder
    loader = TextLoader("tests/langchain/state_of_the_union.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(persist_dir)

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
            name="retrieval_qa",
            loader_fn=load_retriever,
            persist_dir=persist_dir,
        )

# Load the retrievalQA chain
loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
print(loaded_model.predict([{"query": "What did the president say about Ketanji Brown Jackson"}]))
