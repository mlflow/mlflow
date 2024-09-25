from mlflow.entities import Document
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore


def _convert_node_with_score_to_document(node: NodeWithScore):
  metadata = {}
  metadata["score"] = node.get_score()
  metadata.update(node.metadata)

  return Document(
    page_content=node.get_content(),
    metadata=metadata,
    id=node.node_id,
  )