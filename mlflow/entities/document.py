from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass
class Document:
    """
    An entity used in MLflow Tracing to represent retrieved documents in a RETRIEVER span.

    Attributes:
        page_content: The content of the document.
        metadata: A dictionary of metadata associated with the document.
        id: The ID of the document.
    """

    page_content: str
    metadata: Dict[str, any]
    id: Optional[str]

    def to_dict(self):
        return asdict(self)
