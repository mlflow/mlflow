from typing import Literal, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from mlflow.gateway.base_models import RequestModel, ResponseModel
from mlflow.utils import IS_PYDANTIC_V2

if IS_PYDANTIC_V2:
    from pydantic import RootModel


class TextContentPart(RequestModel):
    type: Literal["text"]
    text: str


class ImageUrl(RequestModel):
    url: str  # either URL of an image, or bas64 encoded data
    detail: Literal["auto", "low", "high"]


class ImageContentPart(RequestModel):
    type: Literal["image_url"]
    image_url: ImageUrl


class InputAudio(RequestModel):
    data: str  # base64 encoded data
    format: Literal["wav", "mp3"]


class AudioContentPart(RequestModel):
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
"""
An array of content parts, conforming to the OpenAI spec. A content part is one of:

  - :py:class:`TextContentPart <mlflow.gateway.schemas.chat.TextContentPart>`
  - :py:class:`ImageContentPart <mlflow.gateway.schemas.chat.ImageContentPart>`
  - :py:class:`AudioContentPart <mlflow.gateway.schemas.chat.AudioContentPart>`
"""


ContentType = Annotated[Union[str, ContentPartsList], Field(union_mode="left_to_right")]
"""
The type of the `content` field in system/user/tool/assistant messages. One of:

  - str
  - :py:class:`ContentPartsList <mlflow.gateway.schemas.chat.ContentPartsList>`
"""


class SystemMessage(RequestModel):
    role: Literal["system"]
    content: ContentType
    """
    The content of the message. Can be a string, or a
    :py:class:`ContentPartsList <mlflow.gateway.schemas.chat.ContentPartsList>`
    """


class UserMessage(RequestModel):
    role: Literal["user"]
    content: ContentType
    """
    The content of the message. Can be a string, or a
    :py:class:`ContentPartsList <mlflow.gateway.schemas.chat.ContentPartsList>`
    """


class FunctionCallArguments(RequestModel):
    name: str
    arguments: str


class FunctionCall(RequestModel):
    id: str
    function: FunctionCallArguments
    type: Literal["function"]


class ToolMessage(RequestModel):
    role: Literal["tool"]
    content: ContentType
    """
    The content of the message. Can be a string, or a
    :py:class:`ContentPartsList <mlflow.gateway.schemas.chat.ContentPartsList>`
    """
    tool_call_id: str


class AssistantMessage(RequestModel):
    role: Literal["assistant"]
    content: Optional[ContentType] = None
    """
    The content of the message. Can be a string, or a
    :py:class:`ContentPartsList <mlflow.gateway.schemas.chat.ContentPartsList>`
    """
    tool_calls: Optional[list[FunctionCall]] = Field(None, min_items=1)
    refusal: Optional[str] = None


if IS_PYDANTIC_V2:
    RequestMessage = RootModel[
        Annotated[
            Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage],
            Field(discriminator="role"),
        ]
    ]
    """One of the following types of messages:

    - :py:class:`SystemMessage <mlflow.gateway.schemas.chat.SystemMessage>`
    - :py:class:`UserMessage <mlflow.gateway.schemas.chat.UserMessage>`
    - :py:class:`AssistantMessage <mlflow.gateway.schemas.chat.AssistantMessage>`
    - :py:class:`ToolMessage <mlflow.gateway.schemas.chat.ToolMessage>`
    """
else:
    class RequestMessage(BaseModel):
        __root__: Annotated[
            Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage],
            Field(discriminator="role"),
        ]


class ParamType(RequestModel):
    type: Literal["string", "number", "integer", "object", "array", "boolean", "null"]


class ParamProperty(ParamType):
    description: Optional[str] = None
    enum: Optional[list[str]] = None
    items: Optional[ParamType] = None


class FunctionParams(RequestModel):
    properties: dict[str, ParamProperty]
    type: Literal["object"] = "object"
    required: Optional[list[str]] = None
    additionalProperties: Optional[bool] = None


class FunctionToolDefinition(RequestModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[FunctionParams] = None
    strict: bool = False


class FunctionTool(RequestModel):
    type: Literal["function"] = "function"
    function: FunctionToolDefinition


class BaseRequestPayload(RequestModel):
    tools: Optional[list[FunctionTool]] = None
    temperature: float = Field(0.0, ge=0, le=2)
    n: int = Field(1, ge=1)
    stop: Optional[list[str]] = Field(None, min_items=1)
    max_tokens: Optional[int] = Field(None, ge=1)
    stream: Optional[bool] = None
    model: Optional[str] = None


_REQUEST_PAYLOAD_EXTRA_SCHEMA = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    "temperature": 0.0,
    "max_tokens": 64,
    "stop": ["END"],
    "n": 1,
    "stream": False,
}


class RequestPayload(BaseRequestPayload):
    messages: list[RequestMessage] = Field(..., min_items=1)

    class Config:
        if IS_PYDANTIC_V2:
            json_schema_extra = _REQUEST_PAYLOAD_EXTRA_SCHEMA
        else:
            schema_extra = _REQUEST_PAYLOAD_EXTRA_SCHEMA


class Function(ResponseModel):
    name: str
    arguments: str


class ToolCall(ResponseModel):
    id: str
    type: Literal["function"]
    function: Function


class Choice(ResponseModel):
    index: int
    message: AssistantMessage
    finish_reason: Optional[str] = None


class ChatUsage(ResponseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


_RESPONSE_PAYLOAD_EXTRA_SCHEMA = {
    "example": {
        "id": "3cdb958c-e4cc-4834-b52b-1d1a7f324714",
        "object": "chat.completion",
        "created": 1700173217,
        "model": "llama-2-70b-chat-hf",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! I am an AI assistant"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }
}


class ResponsePayload(ResponseModel):
    id: Optional[str] = None
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: ChatUsage

    class Config:
        if IS_PYDANTIC_V2:
            json_schema_extra = _RESPONSE_PAYLOAD_EXTRA_SCHEMA
        else:
            schema_extra = _RESPONSE_PAYLOAD_EXTRA_SCHEMA


class StreamDelta(ResponseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(ResponseModel):
    index: int
    finish_reason: Optional[str] = None
    delta: StreamDelta


_STREAM_RESPONSE_PAYLOAD_EXTRA_SCHEMA = {
    "example": {
        "id": "3cdb958c-e4cc-4834-b52b-1d1a7f324714",
        "object": "chat.completion",
        "created": 1700173217,
        "model": "llama-2-70b-chat-hf",
        "choices": [
            {
                "index": 6,
                "finish_reason": "stop",
                "delta": {"role": "assistant", "content": "you?"},
            }
        ],
    }
}


class StreamResponsePayload(ResponseModel):
    id: Optional[str] = None
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]

    class Config:
        if IS_PYDANTIC_V2:
            json_schema_extra = _STREAM_RESPONSE_PAYLOAD_EXTRA_SCHEMA
        else:
            schema_extra = _STREAM_RESPONSE_PAYLOAD_EXTRA_SCHEMA
