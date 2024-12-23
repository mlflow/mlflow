from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel as _BaseModel
from pydantic import Field, ValidationError

from mlflow.exceptions import MlflowException
from mlflow.utils import IS_PYDANTIC_V2_OR_NEWER


class BaseModel(_BaseModel):
    @classmethod
    def validate_compat(cls, obj: Any):
        try:
            if IS_PYDANTIC_V2_OR_NEWER:
                return cls.model_validate(obj)
            else:
                return cls.parse_obj(obj)
        except ValidationError as e:
            raise MlflowException.invalid_parameter_value(e) from e


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


ContentPartsList = list[
    Annotated[
        Union[TextContentPart, ImageContentPart, AudioContentPart], Field(discriminator="type")
    ]
]


ContentType = Annotated[Union[str, ContentPartsList], Field(union_mode="left_to_right")]


class Function(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: Function


class ChatMessage(BaseModel):
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
    strict: Optional[bool] = None


class UnityCatalogFunctionToolDefinition(BaseModel):
    name: str


class ChatTool(BaseModel):
    type: Literal["function", "uc_function"]
    function: Optional[FunctionToolDefinition] = None
    uc_function: Optional[UnityCatalogFunctionToolDefinition] = None


class ChatTools(BaseModel):
    tools: Optional[list[ChatTool]] = Field(None, min_items=1)


class ChatCompletionRequest(ChatTool, BaseModel):
    messages: list[ChatMessage] = Field(..., min_items=1)
    temperature: float = Field(0.0, ge=0, le=2)
    n: int = Field(1, ge=1)
    stop: Optional[list[str]] = Field(None, min_items=1)
    max_tokens: Optional[int] = Field(None, ge=1)
    stream: Optional[bool] = None
    model: Optional[str] = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ChatCompletionResponse(BaseModel):
    id: Optional[str] = None
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage


class ChatChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatChunkChoice(BaseModel):
    index: int
    finish_reason: Optional[str] = None
    delta: ChatChoiceDelta


class ChatCompletionChunk(BaseModel):
    id: Optional[str] = None
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatChunkChoice]
