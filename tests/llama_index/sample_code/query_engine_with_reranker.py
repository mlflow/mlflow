"""
Sample code to define a query engine with post processors and save it with model-from-code logging.

Ref: https://qdrant.tech/documentation/quickstart/
"""

from typing import List, Optional

from llama_index.core import Document, QueryBundle, VectorStoreIndex
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore

import mlflow

index = VectorStoreIndex.from_documents(documents=[Document.example()])

# Postprocessor 1: Reranker
reranker = LLMRerank(
    choice_batch_size=5,
    top_n=2,
)


# Postprocessor 2: custom postprocessor
class CustomNodePostprocessor(BaseNodePostprocessor):
    call_count: int = 0

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        # subtracts 1 from the score
        self.call_count += 1
        return nodes


query_engine = index.as_query_engine(
    similarity_top_k=5,
    node_postprocessors=[reranker, CustomNodePostprocessor()],
)


mlflow.models.set_model(query_engine)
