import os

import qdrant_client
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.vector_stores.qdrant import QdrantVectorStore
from workflow.events import *
from workflow.prompts import *

_BM25_PERSIST_DIR = ".bm25_retriever"

_QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
_QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
_QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "mlflow_doc")


class HybridRAGWorkflow(Workflow):
    VALID_RETRIEVERS = {"vector_search", "bm25", "web_search"}

    def __init__(self, retrievers=None, **kwargs):
        super().__init__(**kwargs)
        self.llm = Settings.llm
        self.retrievers = retrievers or []

        if invalid_retrievers := set(self.retrievers) - self.VALID_RETRIEVERS:
            raise ValueError(f"Invalid retrievers specified: {invalid_retrievers}")

        self._use_vs_retriever = "vector_search" in self.retrievers
        self._use_bm25_retriever = "bm25" in self.retrievers
        self._use_web_search = "web_search" in self.retrievers

        if self._use_vs_retriever:
            qd_client = qdrant_client.QdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT)
            vector_store = QdrantVectorStore(
                client=qd_client, collection_name=_QDRANT_COLLECTION_NAME
            )
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            self.vs_retriever = index.as_retriever()

        if self._use_bm25_retriever:
            self.bm25_retriever = BM25Retriever.from_persist_dir(_BM25_PERSIST_DIR)

        if self._use_web_search:
            self.tavily_tool = TavilyToolSpec(api_key=os.environ.get("TAVILY_AI_API_KEY"))

    @step
    async def route_retrieval(
        self, ctx: Context, ev: StartEvent
    ) -> VectorSearchRetrieveEvent | BM25RetrieveEvent | TransformQueryEvent | QueryEvent | None:
        """Route query to the retrieval steps based on the model config."""
        query = ev.get("query")

        if query is None:
            return None

        # Setting the query in the Context object to access it globally
        await ctx.set("query", query)

        # If not retriever is specified, direct to the final query step with an empty context
        if len(self.retrievers) == 0:
            return QueryEvent(context="")

        # Trigger the retrieval steps based on the model config
        if self._use_vs_retriever:
            ctx.send_event(VectorSearchRetrieveEvent(query=query))
        if self._use_bm25_retriever:
            ctx.send_event(BM25RetrieveEvent(query=query))
        if self._use_web_search:
            ctx.send_event(TransformQueryEvent(query=query))

    @step
    async def query_vector_store(self, ev: VectorSearchRetrieveEvent) -> RetrievalResultEvent:
        """Perform retrieval using the vector store."""
        nodes = self.vs_retriever.retrieve(ev.query)
        return RetrievalResultEvent(nodes=nodes, retriever="vector_search")

    @step
    async def query_bm25(self, ev: BM25RetrieveEvent) -> RetrievalResultEvent:
        """Perform retrieval using the BM25 retriever."""
        nodes = self.bm25_retriever.retrieve(ev.query)
        return RetrievalResultEvent(nodes=nodes, retriever="bm25")

    @step
    async def transform_query(self, ev: TransformQueryEvent) -> WebsearchEvent:
        """Transform the user query into a search query."""
        prompt = TRANSFORM_QUERY_TEMPLATE.format(query=ev.query)
        transformed_query = self.llm.complete(prompt).text
        return WebsearchEvent(search_query=transformed_query)

    @step
    async def query_web_search(self, ev: WebsearchEvent) -> RetrievalResultEvent:
        """Perform web search with the transformed query string"""
        search_results = self.tavily_tool.search(ev.search_query, max_results=5)
        nodes = [NodeWithScore(node=document, score=None) for document in search_results]
        return RetrievalResultEvent(nodes=nodes, retriever="web_search")

    @step
    async def gather_retrieval_results(
        self, ctx: Context, ev: RetrievalResultEvent
    ) -> RerankEvent | QueryEvent | None:
        """Gather the retrieved texts and send them to the reranking step."""
        # Wait for results from all retrievers
        results = ctx.collect_events(ev, [RetrievalResultEvent] * len(self.retrievers))

        # Llama Index workflow polls for results until all retrievers have responded.
        # If any retriever has not responded, collect_events will return None and we
        # should return None to wait for the next poll.
        if results is None:
            return None

        # If only one retriever is used, we can skip reranking
        if len(results) == 1:
            context = "\n".join(node.text for node in results[0].nodes)
            return QueryEvent(context=context)

        # Combine the nodes from all retrievers for reranking
        all_nodes = []
        for result in results:
            # Record the source of the retrieved nodes
            for node in result.nodes:
                node.node.metadata["retriever"] = result.retriever
            all_nodes.extend(result.nodes)

        return RerankEvent(nodes=all_nodes)

    @step
    async def rerank(self, ctx: Context, ev: RerankEvent) -> QueryEvent:
        """Evaluate relevancy of retrieved documents with the query."""
        query = await ctx.get("query")

        # Rerank the nodes using LLM (RankGPT based)
        reranker = RankGPTRerank(llm=self.llm, top_n=5)
        reranked_nodes = reranker.postprocess_nodes(ev.nodes, query_str=query)
        reranked_context = "\n".join(node.text for node in reranked_nodes)
        return QueryEvent(context=reranked_context)

    @step
    async def query_result(self, ctx: Context, ev: QueryEvent) -> StopEvent:
        """Get result with relevant text."""
        query = await ctx.get("query")

        prompt = FINAL_QUERY_TEMPLATE.format(context=ev.context, query=query)
        response = self.llm.complete(prompt).text
        return StopEvent(result=response)
