from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass
class Document:
    """
    An entity used in MLflow Tracing to represent retrieved documents in a RETRIEVER span.

    Args:
        page_content: The content of the document.
        metadata: A dictionary of metadata associated with the document.
        id: The ID of the document.
    """

    page_content: str
    metadata: Dict[str, any]
    id: Optional[str]

    @classmethod
    def from_langchain_document(cls, document):
        return cls(
            page_content=document.page_content,
            metadata=deepcopy(document.metadata),
            id=document.id,
        )

    @classmethod
    def from_llama_index_node_with_score(cls, node_with_score):
        metadata = {}
        metadata["score"] = node_with_score.get_score()
        # update after setting score so that it can be
        # overridden if the user wishes to do so
        metadata.update(deepcopy(node_with_score.metadata))

        return Document(
            page_content=node_with_score.get_content(),
            metadata=metadata,
            id=node_with_score.node_id,
        )

    def to_dict(self):
        return asdict(self)
