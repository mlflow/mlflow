from typing import Literal

from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import Event


class VectorSearchRetrieveEvent(Event):
    """Event for triggering VectorStore index retrieval step."""

    query: str


class BM25RetrieveEvent(Event):
    """Event for triggering BM25 retrieval step."""

    query: str


class TransformQueryEvent(Event):
    """Event for transforming user query into a search query."""

    query: str


class WebsearchEvent(Event):
    """Event for triggering web search tool step."""

    search_query: str


class RetrievalResultEvent(Event):
    """Event to send retrieval result from each retriever to the gather step."""

    nodes: list[NodeWithScore]
    retriever: Literal["vector_search", "bm25", "web_search"]


class RerankEvent(Event):
    """Event to send retrieval result to reranking step."""

    nodes: list[NodeWithScore]


class QueryEvent(Event):
    """Event for triggering the final query step"""

    context: str
