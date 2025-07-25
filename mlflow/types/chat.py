from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel as _BaseModel
from pydantic import Field

from mlflow.utils import IS_PYDANTIC_V2_OR_NEWER


class BaseModel(_BaseModel):
    @classmethod
    def validate_compat(cls, obj: Any):
        if IS_PYDANTIC_V2_OR_NEWER:
            return cls.model_validate(obj)
        else:
            return cls.parse_obj(obj)

    def model_dump_compat(self, **kwargs):
        if IS_PYDANTIC_V2_OR_NEWER:
            return self.model_dump(**kwargs)
        else:
            return self.dict(**kwargs)


class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrl(BaseModel):
    """
    Represents an image URL.

    Attributes:
        url: Either a URL of an image or base64 encoded data.
            https://platform.openai.com/docs/guides/vision?lang=curl#uploading-base64-encoded-images
        detail: The level of resolution for the image when the model receives it.
            For example, when set to "low", the model will see a image resized to
            512x512 pixels, which consumes fewer tokens. In OpenAI, this is optional
            and defaults to "auto".
            https://platform.openai.com/docs/guides/vision?lang=curl#low-or-high-fidelity-image-understanding
    """

    url: str
    detail: Optional[Literal["auto", "low", "high"]] = None


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

    def to_tool_call(self, id=None) -> ToolCall:
        if id is None:
            id = str(uuid4())
        return ToolCall(id=id, type="function", function=self)


class ToolCall(BaseModel):
    id: str
    type: str = Field(default="function")
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
    # NB: In the actual OpenAI chat completion API spec, these fields only
    #   present in either the request or response message (tool_call_id is only in
    #   the request, while the other two are only in the response).
    #   Strictly speaking, we should separate the request and response message types
    #   to match OpenAI's API spec. However, we don't want to do that because we the
    #   request and response message types are not distinguished in many parts of the
    #   codebase, and also we don't want to ask users to use two different classes.
    #   Therefore, we include all fields in this class, while marking them as optional.
    # TODO: Define a sub classes for different type of messages (request/response, and
    #   system/user/assistant/tool, etc), and create a factory function to allow users
    #   to create them without worrying about the details.
    tool_calls: Optional[list[ToolCall]] = None
    refusal: Optional[str] = None
    tool_call_id: Optional[str] = None


class ParamType(BaseModel):
    type: Optional[Literal["string", "number", "integer", "object", "array", "boolean", "null"]] = (
        None
    )


class ParamProperty(ParamType):
    """
    OpenAI uses JSON Schema (https://json-schema.org/) for function parameters.
    See OpenAI function calling reference:
    https://platform.openai.com/docs/guides/function-calling?&api-mode=responses#defining-functions

    JSON Schema enum supports any JSON type (str, int, float, bool, null, arrays, objects),
    but we restrict to basic scalar types for practical use cases and API safety.
    """

    description: Optional[str] = None
    enum: Optional[list[Union[str, int, float, bool]]] = None
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


class ChatTool(BaseModel):
    """
    A tool definition passed to the chat completion API.

    Ref: https://platform.openai.com/docs/guides/function-calling
    """

    type: Literal["function"]
    function: Optional[FunctionToolDefinition] = None


class BaseRequestPayload(BaseModel):
    """Common parameters used for chat completions and completion endpoints."""

    temperature: float = Field(0.0, ge=0, le=2)
    n: int = Field(1, ge=1)
    stop: Optional[list[str]] = (
        Field(None, min_length=1) if IS_PYDANTIC_V2_OR_NEWER else Field(None, min_items=1)
    )
    max_tokens: Optional[int] = Field(None, ge=1)
    stream: Optional[bool] = None
    stream_options: Optional[dict[str, Any]] = None
    model: Optional[str] = None


# NB: For interface constructs that rely on other BaseModel implementations, in
# pydantic 1 the **order** in which classes are defined in this module is absolutely
# critical to prevent ForwardRef errors. Pydantic 2 does not have this limitation.
# To maintain compatibility with Pydantic 1, ensure that all classes that are defined in
# this file have dependencies defined higher than the line of usage.


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ChatChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatChunkChoice(BaseModel):
    index: int
    finish_reason: Optional[str] = None
    delta: ChatChoiceDelta


class ChatCompletionChunk(BaseModel):
    """A chunk of a chat completion stream response."""

    id: Optional[str] = None
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatChunkChoice]


class ChatCompletionRequest(BaseRequestPayload):
    """
    A request to the chat completion API.

    Must be compatible with OpenAI's Chat Completion API.
    https://platform.openai.com/docs/api-reference/chat
    """

    messages: list[ChatMessage] = (
        Field(..., min_length=1) if IS_PYDANTIC_V2_OR_NEWER else Field(..., min_items=1)
    )
    tools: Optional[list[ChatTool]] = (
        Field(None, min_length=1) if IS_PYDANTIC_V2_OR_NEWER else Field(None, min_items=1)
    )


class ChatCompletionResponse(BaseModel):
    """
    A response from the chat completion API.

    Must be compatible with OpenAI's Chat Completion API.
    https://platform.openai.com/docs/api-reference/chat
    """

    id: Optional[str] = None
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage
