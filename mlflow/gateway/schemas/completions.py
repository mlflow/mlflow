from pydantic import BaseModel, Extra

from .chat import Response  # pylint: disable=unused-import


class Request(BaseModel, extra=Extra.allow):
    prompt: str
