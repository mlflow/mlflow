import warnings
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

"""
Classes are inspired by classes for Response and ResponseStreamEvent in openai-python

https://github.com/openai/openai-python/blob/ed53107e10e6c86754866b48f8bd862659134ca8/src/openai/types/responses/response.py#L31
https://github.com/openai/openai-python/blob/ed53107e10e6c86754866b48f8bd862659134ca8/src/openai/types/responses/response_stream_event.py#L42
"""


#########################
# Response helper classes
#########################
class Status(BaseModel):
    status: str | None = None

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
    code: str | None = None
    message: str


class AnnotationFileCitation(BaseModel):
    file_id: str
    index: int
    type: str = "file_citation"


class AnnotationURLCitation(BaseModel):
    end_index: int | None = None
    start_index: int | None = None
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
            AnnotationFileCitation(**self.model_dump())
        elif self.type == "url_citation":
            AnnotationURLCitation(**self.model_dump())
        elif self.type == "file_path":
            AnnotationFilePath(**self.model_dump())
        else:
            raise ValueError(f"Invalid annotation type: {self.type}")
        return self


class ResponseOutputText(BaseModel):
    annotations: list[Annotation] | None = None
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
            ResponseOutputText(**self.model_dump())
        elif self.type == "refusal":
            ResponseOutputRefusal(**self.model_dump())
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
        if self.content is None:
            raise ValueError(f"content must not be None for {self.__class__.__name__}")
        if isinstance(self.content, list) and len(self.content) == 0:
            raise ValueError("content must not be an empty list")
        return self


class ResponseFunctionToolCall(Status):
    arguments: str
    call_id: str
    name: str
    type: str = "function_call"
    id: str | None = None


class Summary(BaseModel):
    text: str
    type: str = "summary_text"


class ResponseReasoningItem(Status):
    id: str
    summary: list[Summary]
    type: str = "reasoning"


class McpApprovalRequest(Status):
    id: str
    arguments: str
    name: str
    server_label: str
    type: str = "mcp_approval_request"


class McpApprovalResponse(Status):
    approval_request_id: str
    approve: bool
    type: str = "mcp_approval_response"
    id: str | None = None
    reason: str | None = None


class OutputItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str

    @model_validator(mode="after")
    def check_type(self) -> "OutputItem":
        if self.type == "message":
            ResponseOutputMessage(**self.model_dump())
        elif self.type == "function_call":
            ResponseFunctionToolCall(**self.model_dump())
        elif self.type == "reasoning":
            ResponseReasoningItem(**self.model_dump())
        elif self.type == "function_call_output":
            FunctionCallOutput(**self.model_dump())
        elif self.type == "mcp_approval_request":
            McpApprovalRequest(**self.model_dump())
        elif self.type == "mcp_approval_response":
            McpApprovalResponse(**self.model_dump())
        elif self.type not in {
            "file_search_call",
            "computer_call",
            "web_search_call",
        }:
            raise ValueError(f"Invalid type: {self.type} for {self.__class__.__name__}")
        return self


class IncompleteDetails(BaseModel):
    reason: str | None = None

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
    strict: bool | None = None
    type: str = "function"
    description: str | None = None


class Tool(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str

    @model_validator(mode="after")
    def check_type(self) -> "Tool":
        if self.type == "function":
            FunctionTool(**self.model_dump())
        elif self.type not in {"file_search", "computer_use", "web_search"}:
            warnings.warn(f"Invalid tool type: {self.type}")
        return self


class ToolChoice(BaseModel):
    tool_choice: str | ToolChoiceFunction | None = None

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
    effort: str | None = None
    generate_summary: str | None = None

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
    truncation: str | None = None

    @model_validator(mode="after")
    def check_truncation(self) -> "Truncation":
        if self.truncation and self.truncation not in {"auto", "disabled"}:
            warnings.warn(f"Invalid truncation: {self.truncation}. Must be 'auto' or 'disabled'.")
        return self


class Response(Truncation, ToolChoice):
    id: str | None = None
    created_at: float | None = None
    error: ResponseError | None = None
    incomplete_details: IncompleteDetails | None = None
    instructions: str | None = None
    metadata: dict[str, str] | None = None
    model: str | None = None
    object: str = "response"
    output: list[OutputItem]
    parallel_tool_calls: bool | None = None
    temperature: float | None = None
    tools: list[Tool] | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    previous_response_id: str | None = None
    reasoning: ReasoningParams | None = None
    status: str | None = None
    text: Any | None = None
    usage: ResponseUsage | None = None
    user: str | None = None

    @property
    def output_text(self) -> str:
        """Convenience property that aggregates all `output_text` items from the `output`
        list.

        If no `output_text` content blocks exist, then an empty string is returned.
        """
        texts: list[str] = []
        for output in self.output:
            if output.type == "message":
                texts.extend(
                    content.text for content in output.content if content.type == "output_text"
                )

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
    content: str | list[ResponseInputTextParam | dict[str, Any]]
    role: str
    status: str | None = None
    type: str = "message"

    @model_validator(mode="after")
    def check_content(self) -> "Message":
        if self.content is None:
            raise ValueError("content must not be None")
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
    max_output_tokens: int | None = None
    metadata: dict[str, str] | None = None
    parallel_tool_calls: bool | None = None
    tools: list[Tool] | None = None
    reasoning: ReasoningParams | None = None
    store: bool | None = None
    stream: bool | None = None
    temperature: float | None = None
    text: Any | None = None
    top_p: float | None = None
    user: str | None = None


#####################################
# ResponsesStreamEvent helper classes
#####################################


class ResponseTextDeltaEvent(BaseModel):
    content_index: int | None = None
    delta: str
    item_id: str
    output_index: int | None = None
    type: str = "response.output_text.delta"


class ResponseTextAnnotationDeltaEvent(BaseModel):
    annotation: Annotation
    annotation_index: int
    content_index: int | None = None
    item_id: str
    output_index: int | None = None
    type: str = "response.output_text.annotation.added"


class ResponseOutputItemDoneEvent(BaseModel):
    item: OutputItem
    output_index: int | None = None
    type: str = "response.output_item.done"


class ResponseErrorEvent(BaseModel):
    code: str | None = None
    message: str
    param: str | None = None
    type: str = "error"


class ResponseCompletedEvent(BaseModel):
    response: Response
    type: str = "response.completed"
