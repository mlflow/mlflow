#
from pydantic import BaseModel, Extra


class Request(BaseModel, extra=Extra.allow):
    ...


class Response(BaseModel, extra=Extra.allow):
    ...
