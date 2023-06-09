from pydantic import BaseModel, Field, Extra
from typing import List, Optional, Union, Dict, Any


class Message(BaseModel):
    role: str = Field(...)
    content: str = Field(...)
    ...


class Request(BaseModel, extra=Extra.allow):
    messages: List[Message] = Field(...)
    ...


class Response(BaseModel, extra=Extra.allow):
    candidates: List[Message] = Field(...)
    metadata: Dict[str, Any] = Field(...)
