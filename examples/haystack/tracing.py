import os
from getpass import getpass

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.routers import ConditionalRouter
from haystack.components.websearch.serper_dev import SerperDevWebSearch
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

import mlflow

mlflow.set_experiment("Haystack Tracing")
mlflow.haystack.autolog()

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("Enter OpenAI API key:")
if "SERPERDEV_API_KEY" not in os.environ:
    os.environ["SERPERDEV_API_KEY"] = getpass("Enter SerperDev API key:")


document_store = InMemoryDocumentStore()

documents = [
    Document(
        content="""Munich, the vibrant capital of Bavaria in southern Germany, exudes a perfect blend of rich cultural
                                heritage and modern urban sophistication. Nestled along the banks of the Isar River, Munich is renowned
                                for its splendid architecture, including the iconic Neues Rathaus (New Town Hall) at Marienplatz and
                                the grandeur of Nymphenburg Palace. The city is a haven for art enthusiasts, with world-class museums like the
                                Alte Pinakothek housing masterpieces by renowned artists. Munich is also famous for its lively beer gardens, where
                                locals and tourists gather to enjoy the city's famed beers and traditional Bavarian cuisine. The city's annual
                                Oktoberfest celebration, the world's largest beer festival, attracts millions of visitors from around the globe.
                                Beyond its cultural and culinary delights, Munich offers picturesque parks like the English Garden, providing a
                                serene escape within the heart of the bustling metropolis. Visitors are charmed by Munich's warm hospitality,
                                making it a must-visit destination for travelers seeking a taste of both old-world charm and contemporary allure."""
    )
]

document_store.write_documents(documents)

retriever = InMemoryBM25Retriever(document_store)

prompt_template = [
    ChatMessage.from_user(
        """
Answer the following query given the documents.
If the answer is not contained within the documents reply with 'no_answer'

Documents:
{% for document in documents %}
  {{document.content}}
{% endfor %}
Query: {{query}}
"""
    )
]

prompt_builder = ChatPromptBuilder(template=prompt_template, required_variables="*")
llm = OpenAIChatGenerator(model="gpt-4o-mini")

prompt_for_websearch = [
    ChatMessage.from_user(
        """
Answer the following query given the documents retrieved from the web.
Your answer should indicate that your answer was generated from websearch.

Documents:
{% for document in documents %}
  {{document.content}}
{% endfor %}

Query: {{query}}
"""
    )
]

websearch = SerperDevWebSearch()
prompt_builder_for_websearch = ChatPromptBuilder(
    template=prompt_for_websearch, required_variables="*"
)
llm_for_websearch = OpenAIChatGenerator(model="gpt-4o-mini")


routes = [
    {
        "condition": "{{'no_answer' in replies[0].text}}",
        "output": "{{query}}",
        "output_name": "go_to_websearch",
        "output_type": str,
    },
    {
        "condition": "{{'no_answer' not in replies[0].text}}",
        "output": "{{replies[0].text}}",
        "output_name": "answer",
        "output_type": str,
    },
]

router = ConditionalRouter(routes)

agentic_rag_pipe = Pipeline()
agentic_rag_pipe.add_component("retriever", retriever)
agentic_rag_pipe.add_component("prompt_builder", prompt_builder)
agentic_rag_pipe.add_component("llm", llm)
agentic_rag_pipe.add_component("router", router)
agentic_rag_pipe.add_component("websearch", websearch)
agentic_rag_pipe.add_component("prompt_builder_for_websearch", prompt_builder_for_websearch)
agentic_rag_pipe.add_component("llm_for_websearch", llm_for_websearch)

agentic_rag_pipe.connect("retriever", "prompt_builder.documents")
agentic_rag_pipe.connect("prompt_builder.prompt", "llm.messages")
agentic_rag_pipe.connect("llm.replies", "router.replies")
agentic_rag_pipe.connect("router.go_to_websearch", "websearch.query")
agentic_rag_pipe.connect("router.go_to_websearch", "prompt_builder_for_websearch.query")
agentic_rag_pipe.connect("websearch.documents", "prompt_builder_for_websearch.documents")
agentic_rag_pipe.connect("prompt_builder_for_websearch", "llm_for_websearch")


query = "How many people live in Munich?"

result = agentic_rag_pipe.run(
    {"retriever": {"query": query}, "prompt_builder": {"query": query}, "router": {"query": query}}
)

# Print the `replies` generated using the web searched Documents
print(result["llm_for_websearch"]["replies"][0].text)

last_trace_id = mlflow.get_last_active_trace_id()
trace = mlflow.get_trace(trace_id=last_trace_id)

# Print the token usage
total_usage = trace.info.token_usage
print("== Total token usage: ==")
print(f"  Input tokens: {total_usage['input_tokens']}")
print(f"  Output tokens: {total_usage['output_tokens']}")
print(f"  Total tokens: {total_usage['total_tokens']}")

# Print the token usage for each LLM call
print("\n== Detailed usage for each LLM call: ==")
for span in trace.data.spans:
    if usage := span.get_attribute("mlflow.chat.tokenUsage"):
        print(f"{span.name}:")
        print(f"  Input tokens: {usage['input_tokens']}")
        print(f"  Output tokens: {usage['output_tokens']}")
        print(f"  Total tokens: {usage['total_tokens']}")
