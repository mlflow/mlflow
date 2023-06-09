from typing import List, Optional

from pydantic import BaseModel, Extra, Field

from .common import Metadata


class Message(BaseModel):
    role: str
    content: str


class Request(BaseModel, extra=Extra.allow):
    messages: List[Message] = Field(..., min_items=1)


class Response(BaseModel, extra=Extra.allow):
    candidates: List[Message]
    metadata: Optional[Metadata]
