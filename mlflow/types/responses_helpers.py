import warnings
from typing import Any, Optional, Union

from pydantic import ConfigDict

from mlflow.types.chat import BaseModel
from mlflow.utils.pydantic_utils import model_validator


#########################
# Response helper classes
#########################
class Status(BaseModel):
    status: Optional[str] = None
    """The status of the item."""

    @model_validator(mode="after")
    def check_status(self) -> "Status":
        if self.status is not None and self.status not in {
            "in_progress",
            "completed",
            "incomplete",
        }:
            raise ValueError(
                f"Invalid status: {self.status}. "
                "Must be 'in_progress', 'completed', or 'incomplete'."
            )
        return self


class ResponseError(BaseModel):
    code: str
    """The error code for the response."""

    message: str
    """A human-readable description of the error."""

    @model_validator(mode="after")
    def check_code(self) -> "ResponseError":
        if self.code not in {
            "server_error",
            "rate_limit_exceeded",
            "invalid_prompt",
            "vector_store_timeout",
            "invalid_image",
            "invalid_image_format",
            "invalid_base64_image",
            "invalid_image_url",
            "image_too_large",
            "image_too_small",
            "image_parse_error",
            "image_content_policy_violation",
            "invalid_image_mode",
            "image_file_too_large",
            "unsupported_image_media_type",
            "empty_image_file",
            "failed_to_download_image",
            "image_file_not_found",
        }:
            raise ValueError(f"Invalid error code: {self.code}")
        return self


class AnnotationFileCitation(BaseModel):
    file_id: str
    index: int
    type: str = "file_citation"


class AnnotationURLCitation(BaseModel):
    end_index: int
    start_index: int
    title: str
    type: str = "url_citation"
    url: str


class AnnotationFilePath(BaseModel):
    file_id: str
    index: int
    type: str = "file_path"


class Annotation(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str

    @model_validator(mode="after")
    def check_type(self) -> "Annotation":
        if self.type == "file_citation":
            AnnotationFileCitation(**self.model_dump_compat())
        elif self.type == "url_citation":
            AnnotationURLCitation(**self.model_dump_compat())
        elif self.type == "file_path":
            AnnotationFilePath(**self.model_dump_compat())
        else:
            raise ValueError(f"Invalid annotation type: {self.type}")
        return self


class ResponseOutputText(BaseModel):
    annotations: Optional[list[Annotation]] = None
    text: str
    type: str = "output_text"


class ResponseOutputRefusal(BaseModel):
    refusal: str
    type: str = "refusal"


class Content(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str

    @model_validator(mode="after")
    def check_type(self) -> "Content":
        if self.type == "output_text":
            ResponseOutputText(**self.model_dump_compat())
        elif self.type == "refusal":
            ResponseOutputRefusal(**self.model_dump_compat())
        else:
            raise ValueError(f"Invalid content type: {self.type}")
        return self


class ResponseOutputMessage(Status):
    id: str
    content: list[Content]
    role: str = "assistant"
    type: str = "message"


class ResponseFunctionToolCall(Status):
    arguments: str
    call_id: str
    name: str
    type: str = "function_call"
    id: Optional[str] = None


class Summary(BaseModel):
    text: str
    type: str = "summary_text"


class ResponseReasoningItem(Status):
    id: str
    summary: list[Summary]
    type: str = "reasoning"


class OutputItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str

    @model_validator(mode="after")
    def check_type(self) -> "OutputItem":
        if self.type == "message":
            ResponseOutputMessage(**self.model_dump_compat())
        elif self.type == "function_call":
            ResponseFunctionToolCall(**self.model_dump_compat())
        elif self.type == "reasoning":
            ResponseReasoningItem(**self.model_dump_compat())
        elif self.type not in {
            "file_search_call",
            "computer_call",
            "web_search_call",
        }:
            warnings.warn(f"Invalid type: {self.type}.")
        return self


class IncompleteDetails(BaseModel):
    reason: Optional[str] = None
    """The reason why the response is incomplete."""

    @model_validator(mode="after")
    def check_reason(self) -> "IncompleteDetails":
        if self.reason not in {"max_output_tokens", "content_filter"}:
            raise ValueError(f"Invalid reason: {self.reason}")
        return self


class ToolChoiceFunction(BaseModel):
    name: str
    """The name of the function to call."""

    type: str = "function"


class FunctionTool(BaseModel):
    name: str
    """The name of the function to call."""

    # TODO bbqiu revisit if we should use FunctionParams from chat.py
    parameters: dict[str, Any]
    """A JSON schema object describing the parameters of the function."""

    strict: bool
    """Whether to enforce strict parameter validation. Default `true`."""

    type: str = "function"

    description: Optional[str] = None
    """A description of the function.

    Used by the model to determine whether or not to call the function.
    """


class Tool(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str

    @model_validator(mode="after")
    def check_type(self) -> "Tool":
        if self.type == "function":
            FunctionTool(**self.model_dump_compat())
        elif self.type not in {"file_search", "computer_use", "web_search"}:
            warnings.warn(f"Invalid tool type: {self.type}")
        return self


class Tools(BaseModel):
    tools: Optional[list[Tool]] = None


class ToolChoice(BaseModel):
    tool_choice: Optional[Union[str, ToolChoiceFunction]] = None
    """
    How the model should select which tool (or tools) to use when generating a
    response. See the `tools` parameter to see how to specify which tools the model
    can call.
    """

    @model_validator(mode="after")
    def check_tool_choice(self) -> "ToolChoice":
        if isinstance(self.tool_choice, str):
            warnings.warn(f"Not validating tool choice: {self.tool_choice}")
        return self


class Reasoning(BaseModel):
    effort: Optional[str] = None

    generate_summary: Optional[str] = None

    @model_validator(mode="after")
    def check_generate_summary(self) -> "Reasoning":
        if self.generate_summary not in {"concise", "detailed"}:
            raise ValueError(f"Invalid generate_summary: {self.generate_summary}")
        return self

    @model_validator(mode="after")
    def check_effort(self) -> "Reasoning":
        if self.effort not in {"low", "medium", "high"}:
            raise ValueError(f"Invalid effort: {self.effort}")
        return self


class InputTokensDetails(BaseModel):
    cached_tokens: int


class OutputTokensDetails(BaseModel):
    reasoning_tokens: int
    """The number of reasoning tokens."""


class ResponseUsage(BaseModel):
    input_tokens: int
    """The number of input tokens."""

    input_tokens_details: InputTokensDetails
    """A detailed breakdown of the input tokens."""

    output_tokens: int
    """The number of output tokens."""

    output_tokens_details: OutputTokensDetails
    """A detailed breakdown of the output tokens."""

    total_tokens: int
    """The total number of tokens used."""


class Truncation(BaseModel):
    truncation: Optional[str] = None

    @model_validator(mode="after")
    def check_truncation(self) -> "Truncation":
        if self.truncation is not None and self.truncation not in {"auto", "disabled"}:
            raise ValueError(
                f"Invalid truncation: {self.truncation}. Must be 'auto' or 'disabled'."
            )
        return self


class Response(Tools, Truncation, ToolChoice):
    id: Optional[str] = None
    created_at: Optional[float] = None
    error: Optional[ResponseError] = None
    incomplete_details: Optional[IncompleteDetails] = None
    instructions: Optional[str] = None
    metadata: Optional[dict[str, str]] = None
    model: Optional[str] = None
    object: str = "response"
    output: list[OutputItem]
    parallel_tool_calls: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    previous_response_id: Optional[str] = None
    reasoning: Optional[Reasoning] = None
    status: Optional[str] = None
    # TODO bbqiu revisit this ResponseFormatTextConfig: TypeAlias
    text: Optional[Any] = None
    usage: Optional[ResponseUsage] = None
    user: Optional[str] = None

    @property
    def output_text(self) -> str:
        """Convenience property that aggregates all `output_text` items from the `output`
        list.

        If no `output_text` content blocks exist, then an empty string is returned.
        """
        texts: list[str] = []
        for output in self.output:
            if output.type == "message":
                for content in output.content:
                    if content.type == "output_text":
                        texts.append(content.text)

        return "".join(texts)

    @model_validator(mode="after")
    def check_output(self) -> "Response":
        if not self.output:
            raise ValueError("output must be a non-empty list")
        for output in self.output:
            if isinstance(
                output, (ResponseOutputMessage, ResponseFunctionToolCall, ResponseReasoningItem)
            ):
                continue
            elif isinstance(output, dict):
                if "type" not in output:
                    raise ValueError("dict must have a type key")
                if output["type"] not in {
                    "file_search_call",
                    "computer_call",
                    "web_search_call",
                }:
                    raise ValueError(f"Invalid type: {output['type']}.")
        return self

    @model_validator(mode="after")
    def check_status(self) -> "Response":
        if self.status is not None and self.status not in {
            "completed",
            "failed",
            "in_progress",
            "incomplete",
        }:
            raise ValueError(
                f"Invalid status: {self.status}. Must be 'completed', 'failed', "
                "'in_progress', or 'incomplete'."
            )
        return self


#################################
# ResponsesRequest helper classes
#################################
class ResponseInputTextParam(BaseModel):
    text: str
    type: str = "input_text"


class Message(Status):
    # TODO bbqiu revisit to see if we shouldn't combine w/ EasyInputMessageParam
    content: Union[str, list[Union[ResponseInputTextParam, dict[str, Any]]]]
    role: str
    status: Optional[str] = None
    type: str = "message"

    @model_validator(mode="after")
    def check_content(self) -> "Message":
        if isinstance(self.content, list):
            for item in self.content:
                if isinstance(item, dict):
                    if "type" not in item:
                        raise ValueError("dict must have a type key")
                    if item["type"] not in {"input_image", "input_file"}:
                        raise ValueError(f"Invalid type: {item['type']}.")
        return self

    @model_validator(mode="after")
    def check_role(self) -> "Message":
        if self.role not in {"user", "assistant", "system", "developer"}:
            raise ValueError(
                f"Invalid role: {self.role}. Must be 'user', 'assistant', 'system', or 'developer'."
            )
        return self


class FunctionCallOutput(Status):
    call_id: str
    output: str
    type: str = "function_call_output"
    id: str


class BaseRequestPayload(Truncation, ToolChoice):
    model: Optional[str] = None
    max_output_tokens: Optional[int] = None
    metadata: Optional[dict[str, str]] = None
    parallel_tool_calls: Optional[bool] = None
    reasoning: Optional[Reasoning] = None
    store: Optional[bool] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    # TODO bbqiu revisit this ResponseFormatTextConfig: TypeAlias
    text: Optional[Any] = None
    top_p: Optional[float] = None
    user: Optional[str] = None


#####################################
# ResponsesStreamEvent helper classes
#####################################


class ResponseContentPartAddedEvent(BaseModel):
    content_index: int
    item_id: str
    output_index: int
    part: Content
    type: str = "response.content_part.added"


class ResponseContentPartDoneEvent(BaseModel):
    content_index: int
    item_id: str
    output_index: int
    part: Content
    type: str = "response.content_part.done"


class ResponseTextDeltaEvent(BaseModel):
    content_index: int
    delta: str
    item_id: str
    output_index: int
    type: str = "response.output_text.delta"


class ResponseTextDoneEvent(BaseModel):
    content_index: int
    item_id: str
    output_index: int
    text: str
    type: str = "response.output_text.done"


class ResponseTextAnnotationDeltaEvent(BaseModel):
    annotation: Union[AnnotationFileCitation, AnnotationURLCitation, AnnotationFilePath]
    annotation_index: int
    content_index: int
    item_id: str
    output_index: int
    type: str = "response.output_text.annotation.added"


class ResponseFunctionCallArgumentsDeltaEvent(BaseModel):
    delta: str
    item_id: str
    output_index: int
    type: str = "response.function_call_arguments.delta"


class ResponseFunctionCallArgumentsDoneEvent(BaseModel):
    arguments: str
    item_id: str
    output_index: int
    type: str = "response.function_call_arguments.done"


class ResponseOutputItemAddedEvent(BaseModel):
    item: OutputItem
    output_index: int
    type: str = "response.output_item.added"


class ResponseOutputItemDoneEvent(BaseModel):
    item: OutputItem
    output_index: int
    type: str = "response.output_item.done"


class ResponseErrorEvent(BaseModel):
    code: Optional[str] = None
    message: str
    param: Optional[str] = None
    type: str = "error"


class ResponseCompletedEvent(BaseModel):
    response: Response
    type: str = "response.completed"
