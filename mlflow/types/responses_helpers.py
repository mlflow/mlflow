from mlflow.utils.pydantic_utils import IS_PYDANTIC_V2_OR_NEWER, model_validator

if not IS_PYDANTIC_V2_OR_NEWER:
    raise ImportError(
        "mlflow.types.responses is not supported in Pydantic v1. "
        "Please upgrade to Pydantic v2 or newer."
    )

import warnings
from typing import Any, Optional, Union

from pydantic import ConfigDict

from mlflow.types.chat import BaseModel

"""
Classes are inspired by classes for Response and ResponseStreamEvent in openai-python

https://github.com/openai/openai-python/blob/ed53107e10e6c86754866b48f8bd862659134ca8/src/openai/types/responses/response.py#L31
https://github.com/openai/openai-python/blob/ed53107e10e6c86754866b48f8bd862659134ca8/src/openai/types/responses/response_stream_event.py#L42
"""


#########################
# Response helper classes
#########################
class Status(BaseModel):
    status: Optional[str] = None

    @model_validator(mode="after")
    def check_status(self) -> "Status":
        if self.status is not None and self.status not in {
            "in_progress",
            "completed",
            "incomplete",
        }:
            raise ValueError(
                f"Invalid status: {self.status} for {self.__class__.__name__}. "
                "Must be 'in_progress', 'completed', or 'incomplete'."
            )
        return self


class ResponseError(BaseModel):
    code: Optional[str] = None
    message: str


class AnnotationFileCitation(BaseModel):
    file_id: str
    index: int
    type: str = "file_citation"


class AnnotationURLCitation(BaseModel):
    end_index: Optional[int] = None
    start_index: Optional[int] = None
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
            raise ValueError(f"Invalid content type: {self.type} for {self.__class__.__name__}")
        return self


class ResponseOutputMessage(Status):
    id: str
    content: list[Content]
    role: str = "assistant"
    type: str = "message"

    @model_validator(mode="after")
    def check_role(self) -> "ResponseOutputMessage":
        if self.role != "assistant":
            raise ValueError(f"Invalid role: {self.role}. Must be 'assistant'.")
        return self

    @model_validator(mode="after")
    def check_content(self) -> "ResponseOutputMessage":
        if not self.content:
            raise ValueError(f"content must not be an empty list for {self.__class__.__name__}")
        return self


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
        elif self.type == "function_call_output":
            FunctionCallOutput(**self.model_dump_compat())
        elif self.type not in {
            "file_search_call",
            "computer_call",
            "web_search_call",
        }:
            raise ValueError(f"Invalid type: {self.type} for {self.__class__.__name__}")
        return self


class IncompleteDetails(BaseModel):
    reason: Optional[str] = None

    @model_validator(mode="after")
    def check_reason(self) -> "IncompleteDetails":
        if self.reason and self.reason not in {"max_output_tokens", "content_filter"}:
            warnings.warn(f"Invalid reason: {self.reason}")
        return self


class ToolChoiceFunction(BaseModel):
    name: str
    type: str = "function"


class FunctionTool(BaseModel):
    name: str
    parameters: dict[str, Any]
    strict: Optional[bool] = None
    type: str = "function"
    description: Optional[str] = None


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


class ToolChoice(BaseModel):
    tool_choice: Optional[Union[str, ToolChoiceFunction]] = None

    @model_validator(mode="after")
    def check_tool_choice(self) -> "ToolChoice":
        if (
            self.tool_choice
            and isinstance(self.tool_choice, str)
            and self.tool_choice not in {"none", "auto", "required"}
        ):
            warnings.warn(f"Invalid tool choice: {self.tool_choice}")
        return self


class ReasoningParams(BaseModel):
    effort: Optional[str] = None
    generate_summary: Optional[str] = None

    @model_validator(mode="after")
    def check_generate_summary(self) -> "ReasoningParams":
        if self.generate_summary and self.generate_summary not in {"concise", "detailed"}:
            warnings.warn(f"Invalid generate_summary: {self.generate_summary}")
        return self

    @model_validator(mode="after")
    def check_effort(self) -> "ReasoningParams":
        if self.effort and self.effort not in {"low", "medium", "high"}:
            warnings.warn(f"Invalid effort: {self.effort}")
        return self


class InputTokensDetails(BaseModel):
    cached_tokens: int


class OutputTokensDetails(BaseModel):
    reasoning_tokens: int


class ResponseUsage(BaseModel):
    input_tokens: int
    input_tokens_details: InputTokensDetails
    output_tokens: int
    output_tokens_details: OutputTokensDetails
    total_tokens: int


class Truncation(BaseModel):
    truncation: Optional[str] = None

    @model_validator(mode="after")
    def check_truncation(self) -> "Truncation":
        if self.truncation and self.truncation not in {"auto", "disabled"}:
            warnings.warn(f"Invalid truncation: {self.truncation}. Must be 'auto' or 'disabled'.")
        return self


class Response(Truncation, ToolChoice):
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
    tools: Optional[list[Tool]] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    previous_response_id: Optional[str] = None
    reasoning: Optional[ReasoningParams] = None
    status: Optional[str] = None
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
    def check_status(self) -> "Response":
        if self.status and self.status not in {
            "completed",
            "failed",
            "in_progress",
            "incomplete",
        }:
            warnings.warn(
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
    content: Union[str, list[Union[ResponseInputTextParam, dict[str, Any]]]]
    role: str
    status: Optional[str] = None
    type: str = "message"

    @model_validator(mode="after")
    def check_content(self) -> "Message":
        if not self.content:
            raise ValueError("content must not be empty")
        if isinstance(self.content, list):
            for item in self.content:
                if isinstance(item, dict):
                    if "type" not in item:
                        raise ValueError(
                            "dictionary type content field values must have key 'type'"
                        )
                    if item["type"] == "input_text":
                        ResponseInputTextParam(**item)
                    elif item["type"] not in {"input_image", "input_file"}:
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


class BaseRequestPayload(Truncation, ToolChoice):
    max_output_tokens: Optional[int] = None
    metadata: Optional[dict[str, str]] = None
    parallel_tool_calls: Optional[bool] = None
    tools: Optional[list[Tool]] = None
    reasoning: Optional[ReasoningParams] = None
    store: Optional[bool] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    text: Optional[Any] = None
    top_p: Optional[float] = None
    user: Optional[str] = None


#####################################
# ResponsesStreamEvent helper classes
#####################################


class ResponseTextDeltaEvent(BaseModel):
    content_index: Optional[int] = None
    delta: str
    item_id: str
    output_index: Optional[int] = None
    type: str = "response.output_text.delta"


class ResponseTextAnnotationDeltaEvent(BaseModel):
    annotation: Annotation
    annotation_index: int
    content_index: Optional[int] = None
    item_id: str
    output_index: Optional[int] = None
    type: str = "response.output_text.annotation.added"


class ResponseOutputItemDoneEvent(BaseModel):
    item: OutputItem
    output_index: Optional[int] = None
    type: str = "response.output_item.done"


class ResponseErrorEvent(BaseModel):
    code: Optional[str] = None
    message: str
    param: Optional[str] = None
    type: str = "error"


class ResponseCompletedEvent(BaseModel):
    response: Response
    type: str = "response.completed"
