from typing import List, Dict, Any

from pydantic import BaseModel, Extra, Field


class Request(BaseModel, extra=Extra.allow):
    text: str


class Response(BaseModel, extra=Extra.allow):
    embeddings: List[float]
    metadata: Optional[Metadata]
