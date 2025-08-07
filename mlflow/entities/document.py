from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any


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
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None

    @classmethod
    def from_langchain_document(cls, document):
        # older versions of langchain do not have the id attribute
        id = getattr(document, "id", None)

        return cls(
            page_content=document.page_content,
            metadata=deepcopy(document.metadata),
            id=id,
        )

    @classmethod
    def from_llama_index_node_with_score(cls, node_with_score):
        metadata = {
            "score": node_with_score.get_score(),
            # update after setting score so that it can be
            # overridden if the user wishes to do so
            **deepcopy(node_with_score.metadata),
        }

        return cls(
            page_content=node_with_score.get_content(),
            metadata=metadata,
            id=node_with_score.node_id,
        )

    def to_dict(self):
        return asdict(self)
