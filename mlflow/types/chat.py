from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel as _BaseModel
from pydantic import Field
from typing_extensions import Annotated

from mlflow.utils import IS_PYDANTIC_V2_OR_NEWER


class BaseModel(_BaseModel):
    @classmethod
    def validate(cls, obj: any):
        if IS_PYDANTIC_V2_OR_NEWER:
            return cls.model_validate(obj)
        else:
            return cls.parse_obj(obj)


class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrl(BaseModel):
    url: str  # either URL of an image, or bas64 encoded data
    detail: Literal["auto", "low", "high"]


class ImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


class InputAudio(BaseModel):
    data: str  # base64 encoded data
    format: Literal["wav", "mp3"]


class AudioContentPart(BaseModel):
    type: Literal["input_audio"]
    input_audio: InputAudio


ContentPartsList = Annotated[
    list[
        Annotated[
            Union[TextContentPart, ImageContentPart, AudioContentPart], Field(discriminator="type")
        ]
    ],
    Field(min_items=1),
]


ContentType = Annotated[Union[str, ContentPartsList], Field(union_mode="left_to_right")]


class Function(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: Function


class RequestMessage(BaseModel):
    """
    A chat request. ``content`` can be a string, or an array of content parts.

    A content part is one of the following:

    - :py:class:`TextContentPart <mlflow.types.chat.TextContentPart>`
    - :py:class:`ImageContentPart <mlflow.types.chat.ImageContentPart>`
    - :py:class:`AudioContentPart <mlflow.types.chat.AudioContentPart>`
    """

    role: str
    content: Optional[ContentType] = None
    tool_calls: Optional[list[ToolCall]] = Field(None, min_items=1)
    tool_call_id: Optional[str] = None
    refusal: Optional[str] = None


class ParamType(BaseModel):
    type: Literal["string", "number", "integer", "object", "array", "boolean", "null"]


class ParamProperty(ParamType):
    description: Optional[str] = None
    enum: Optional[list[str]] = None
    items: Optional[ParamType] = None


class FunctionParams(BaseModel):
    properties: dict[str, ParamProperty]
    type: Literal["object"] = "object"
    required: Optional[list[str]] = None
    additionalProperties: Optional[bool] = None


class FunctionToolDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[FunctionParams] = None
    strict: bool = False


class FunctionTool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionToolDefinition


class UnityCatalogFunctionToolDefinition(BaseModel):
    name: str


class UnityCatalogFunctionTool(BaseModel):
    type: Literal["uc_function"] = "uc_function"
    uc_function: UnityCatalogFunctionToolDefinition


ChatTool = Annotated[Union[FunctionTool, UnityCatalogFunctionTool], Field(discriminator="type")]


class ChatTools(BaseModel):
    tools: Optional[list[ChatTool]] = Field(None, min_items=1)
