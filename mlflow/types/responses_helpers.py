import warnings
from typing import Any, Optional, Union

from pydantic import ConfigDict

from mlflow.types.chat import BaseModel
from mlflow.utils.pydantic_utils import IS_PYDANTIC_V2_OR_NEWER, model_validator


#########################
# Response helper classes
#########################
class Status(BaseModel):
    status: Optional[str] = None

    @model_validator(mode="after")
    def check_status(cls, values) -> "Status":
        status = values.status if IS_PYDANTIC_V2_OR_NEWER else values.get("status")
        if status is not None and status not in {
            "in_progress",
            "completed",
            "incomplete",
        }:
            raise ValueError(
                f"Invalid status: {status} for {cls.__name__}. "
                "Must be 'in_progress', 'completed', or 'incomplete'."
            )
        return values


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
    def check_type(cls, values) -> "Annotation":
        type = values.type if IS_PYDANTIC_V2_OR_NEWER else values.get("type")
        if type == "file_citation":
            AnnotationFileCitation(**values.model_dump_compat())
        elif type == "url_citation":
            AnnotationURLCitation(**values.model_dump_compat())
        elif type == "file_path":
            AnnotationFilePath(**values.model_dump_compat())
        else:
            raise ValueError(f"Invalid annotation type: {type}")
        return values


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
    def check_type(cls, values) -> "Content":
        type = values.type if IS_PYDANTIC_V2_OR_NEWER else values.get("type")
        if type == "output_text":
            ResponseOutputText(**values.model_dump_compat())
        elif type == "refusal":
            ResponseOutputRefusal(**values.model_dump_compat())
        else:
            raise ValueError(f"Invalid content type: {type} for {cls.__class__.__name__}")
        return values


class ResponseOutputMessage(Status):
    id: str
    content: list[Content]
    role: str = "assistant"
    type: str = "message"

    @model_validator(mode="after")
    def check_role(cls, values) -> "ResponseOutputMessage":
        role = values.role if IS_PYDANTIC_V2_OR_NEWER else values.get("role")
        if role != "assistant":
            raise ValueError(f"Invalid role: {role}. Must be 'assistant'.")
        return values

    @model_validator(mode="after")
    def check_content(cls, values) -> "ResponseOutputMessage":
        content = values.content if IS_PYDANTIC_V2_OR_NEWER else values.get("content")
        if not content:
            raise ValueError(f"content must not be an empty list for {cls.__class__.__name__}")
        return values


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
    def check_type(cls, values) -> "OutputItem":
        type = values.type if IS_PYDANTIC_V2_OR_NEWER else values.get("type")
        if type == "message":
            ResponseOutputMessage(**values.model_dump_compat())
        elif type == "function_call":
            ResponseFunctionToolCall(**values.model_dump_compat())
        elif type == "reasoning":
            ResponseReasoningItem(**values.model_dump_compat())
        elif type == "function_call_output":
            FunctionCallOutput(**values.model_dump_compat())
        elif type not in {
            "file_search_call",
            "computer_call",
            "web_search_call",
        }:
            raise ValueError(f"Invalid type: {type} for {cls.__class__.__name__}")
        return values


class IncompleteDetails(BaseModel):
    reason: Optional[str] = None

    @model_validator(mode="after")
    def check_reason(cls, values) -> "IncompleteDetails":
        reason = values.reason if IS_PYDANTIC_V2_OR_NEWER else values.get("reason")
        if reason and reason not in {"max_output_tokens", "content_filter"}:
            warnings.warn(f"Invalid reason: {reason}")
        return values


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
    def check_type(cls, values) -> "Tool":
        type = values.type if IS_PYDANTIC_V2_OR_NEWER else values.get("type")
        if type == "function":
            FunctionTool(**values.model_dump_compat())
        elif type not in {"file_search", "computer_use", "web_search"}:
            warnings.warn(f"Invalid tool type: {type}")
        return values


class ToolChoice(BaseModel):
    tool_choice: Optional[Union[str, ToolChoiceFunction]] = None

    @model_validator(mode="after")
    def check_tool_choice(cls, values) -> "ToolChoice":
        tool_choice = values.tool_choice if IS_PYDANTIC_V2_OR_NEWER else values.get("tool_choice")
        if (
            tool_choice
            and isinstance(tool_choice, str)
            and tool_choice not in {"none", "auto", "required"}
        ):
            warnings.warn(f"Invalid tool choice: {tool_choice}")
        return values


class ReasoningParams(BaseModel):
    effort: Optional[str] = None
    generate_summary: Optional[str] = None

    @model_validator(mode="after")
    def check_generate_summary(cls, values) -> "ReasoningParams":
        generate_summary = (
            values.generate_summary if IS_PYDANTIC_V2_OR_NEWER else values.get("generate_summary")
        )
        if generate_summary and generate_summary not in {"concise", "detailed"}:
            warnings.warn(f"Invalid generate_summary: {generate_summary}")
        return values

    @model_validator(mode="after")
    def check_effort(cls, values) -> "ReasoningParams":
        effort = values.effort if IS_PYDANTIC_V2_OR_NEWER else values.get("effort")
        if effort and effort not in {"low", "medium", "high"}:
            warnings.warn(f"Invalid effort: {effort}")
        return values


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
    def check_truncation(cls, values) -> "Truncation":
        truncation = values.truncation if IS_PYDANTIC_V2_OR_NEWER else values.get("truncation")
        if truncation and truncation not in {"auto", "disabled"}:
            warnings.warn(f"Invalid truncation: {truncation}. Must be 'auto' or 'disabled'.")
        return values


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
    def check_status(cls, values) -> "Response":
        status = values.status if IS_PYDANTIC_V2_OR_NEWER else values.get("status")
        if status and status not in {
            "completed",
            "failed",
            "in_progress",
            "incomplete",
        }:
            warnings.warn(
                f"Invalid status: {status}. Must be 'completed', 'failed', "
                "'in_progress', or 'incomplete'."
            )
        return values


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
    def check_content(cls, values) -> "Message":
        content = values.content if IS_PYDANTIC_V2_OR_NEWER else values.get("content")
        if not content:
            raise ValueError("content must not be empty")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if "type" not in item:
                        raise ValueError(
                            "dictionary type content field values must have key 'type'"
                        )
                    if item["type"] == "input_text":
                        ResponseInputTextParam(**item)
                    elif item["type"] not in {"input_image", "input_file"}:
                        raise ValueError(f"Invalid type: {item['type']}.")
        return values

    @model_validator(mode="after")
    def check_role(cls, values) -> "Message":
        role = values.role if IS_PYDANTIC_V2_OR_NEWER else values.get("role")
        if role not in {"user", "assistant", "system", "developer"}:
            raise ValueError(
                f"Invalid role: {role}. Must be 'user', 'assistant', 'system', or 'developer'."
            )
        return values


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
