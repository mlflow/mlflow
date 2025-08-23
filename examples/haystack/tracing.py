import mlflow

mlflow.haystack.autolog()
mlflow.set_experiment("Haystack Tracing")

from haystack import Document, Pipeline
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret

# Write documents to InMemoryDocumentStore
document_store = InMemoryDocumentStore()
document_store.write_documents(
    [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]
)

# Build a RAG pipeline
prompt_template = [
    ChatMessage.from_system("You are a helpful assistant."),
    ChatMessage.from_user(
        "Given these documents, answer the question.\n"
        "Documents:\n{% for doc in documents %}{{ doc.content }}{% endfor %}\n"
        "Question: {{question}}\n"
        "Answer:"
    ),
]

# Define required variables explicitly
prompt_builder = ChatPromptBuilder(
    template=prompt_template, required_variables={"question", "documents"}
)

retriever = InMemoryBM25Retriever(document_store=document_store)
llm = OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"))

rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm.messages")

# Ask a question
question = "Who lives in Paris?"
results = rag_pipeline.run(
    {
        "retriever": {"query": question},
        "prompt_builder": {"question": question},
    }
)

print(results["llm"]["replies"])
