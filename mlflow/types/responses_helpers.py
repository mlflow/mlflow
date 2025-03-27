from typing import Any, Optional, Union

from mlflow.types.chat import BaseModel
from mlflow.utils.pydantic_utils import model_validator

################################
# Response helper classes
################################


class ResponseError(BaseModel):
    code: str
    """The error code for the response."""

    message: str
    """A human-readable description of the error."""

    @model_validator(mode="after")
    def check_message(self) -> "ResponseError":
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
    """The ID of the file."""

    index: int
    """The index of the file in the list of files."""

    type: str = "file_citation"
    """The type of the file citation. Always `file_citation`."""

    @model_validator(mode="after")
    def check_type(self) -> "AnnotationFileCitation":
        if self.type != "file_citation":
            raise ValueError(f"Invalid type: {self.type}")
        return self


class AnnotationURLCitation(BaseModel):
    end_index: int
    """The index of the last character of the URL citation in the message."""

    start_index: int
    """The index of the first character of the URL citation in the message."""

    title: str
    """The title of the web resource."""

    type: str = "url_citation"
    """The type of the URL citation. Always `url_citation`."""

    @model_validator(mode="after")
    def check_type(self) -> "AnnotationURLCitation":
        if self.type != "url_citation":
            raise ValueError(f"Invalid type: {self.type}")
        return self

    url: str
    """The URL of the web resource."""


class AnnotationFilePath(BaseModel):
    file_id: str
    """The ID of the file."""

    index: int
    """The index of the file in the list of files."""

    type: str = "file_path"
    """The type of the file path. Always `file_path`."""

    @model_validator(mode="after")
    def check_type(self) -> "AnnotationFilePath":
        if self.type != "file_path":
            raise ValueError(f"Invalid type: {self.type}")
        return self


class ResponseOutputText(BaseModel):
    annotations: list[Union[AnnotationFileCitation, AnnotationURLCitation, AnnotationFilePath]]
    """The annotations of the text output."""

    text: str
    """The text output from the model."""

    type: str = "output_text"
    """The type of the output text. Always `output_text`."""

    @model_validator(mode="after")
    def check_type(self) -> "ResponseOutputText":
        if self.type != "output_text":
            raise ValueError(f"Invalid type: {self.type}")
        return self


class ResponseOutputRefusal(BaseModel):
    refusal: str
    """The refusal explanationfrom the model."""

    type: str = "refusal"
    """The type of the refusal. Always `refusal`."""

    @model_validator(mode="after")
    def check_type(self) -> "ResponseOutputRefusal":
        if self.type != "refusal":
            raise ValueError(f"Invalid type: {self.type}")
        return self


class ResponseOutputMessage(BaseModel):
    id: str
    """The unique ID of the output message."""

    content: list[Union[ResponseOutputText, ResponseOutputRefusal]]
    """The content of the output message."""

    role: str = "assistant"
    """The role of the output message. Always `assistant`."""

    status: str
    """The status of the message input.

    One of `in_progress`, `completed`, or `incomplete`. Populated when input items
    are returned via API.
    """

    type: str = "message"
    """The type of the output message. Always `message`."""

    @model_validator(mode="after")
    def check_type(self) -> "ResponseOutputMessage":
        if self.type != "message":
            raise ValueError(f"Invalid type: {self.type}")
        return self

    @model_validator(mode="after")
    def check_role(self) -> "ResponseOutputMessage":
        if self.role != "assistant":
            raise ValueError(f"Invalid role: {self.role}")
        return self

    @model_validator(mode="after")
    def check_status(self) -> "ResponseOutputMessage":
        if self.status not in {"in_progress", "completed", "incomplete"}:
            raise ValueError(f"Invalid status: {self.status}")
        return self


class Result(BaseModel):
    attributes: Optional[dict[str, Union[str, float, bool]]] = None
    """Set of 16 key-value pairs that can be attached to an object.

    This can be useful for storing additional information about the object in a
    structured format, and querying for objects via API or the dashboard. Keys are
    strings with a maximum length of 64 characters. Values are strings with a
    maximum length of 512 characters, booleans, or numbers.
    """

    file_id: Optional[str] = None
    """The unique ID of the file."""

    filename: Optional[str] = None
    """The name of the file."""

    score: Optional[float] = None
    """The relevance score of the file - a value between 0 and 1."""

    text: Optional[str] = None
    """The text that was retrieved from the file."""


class ResponseFileSearchToolCall(BaseModel):
    id: str
    """The unique ID of the file search tool call."""

    queries: list[str]
    """The queries used to search for files."""

    status: str
    """The status of the file search tool call.

    One of `in_progress`, `searching`, `incomplete` or `failed`,
    """

    type: str = "file_search_call"
    """The type of the file search tool call. Always `file_search_call`."""

    results: Optional[list[Result]] = None
    """The results of the file search tool call."""

    @model_validator(mode="after")
    def check_type(self) -> "ResponseFileSearchToolCall":
        if self.type != "file_search_call":
            raise ValueError(f"Invalid type: {self.type}")
        return self

    @model_validator(mode="after")
    def check_status(self) -> "ResponseFileSearchToolCall":
        if self.status not in {"in_progress", "searching", "incomplete", "failed"}:
            raise ValueError(f"Invalid status: {self.status}")
        return self


class ResponseFunctionToolCall(BaseModel):
    arguments: str
    """A JSON string of the arguments to pass to the function."""

    call_id: str
    """The unique ID of the function tool call generated by the model."""

    name: str
    """The name of the function to run."""

    type: str = "function_call"
    """The type of the function tool call. Always `function_call`."""

    id: Optional[str] = None
    """The unique ID of the function tool call."""

    status: Optional[str] = None
    """The status of the item.

    One of `in_progress`, `completed`, or `incomplete`. Populated when items are
    returned via API.
    """

    @model_validator(mode="after")
    def check_type(self) -> "ResponseFunctionToolCall":
        if self.type != "function_call":
            raise ValueError(f"Invalid type: {self.type}")
        return self

    @model_validator(mode="after")
    def check_status(self) -> "ResponseFunctionToolCall":
        if self.status not in {"in_progress", "completed", "incomplete"}:
            raise ValueError(f"Invalid status: {self.status}")
        return self


class ResponseFunctionWebSearch(BaseModel):
    id: str
    """The unique ID of the web search tool call."""

    status: str
    """The status of the web search tool call."""

    type: str = "web_search_call"
    """The type of the web search tool call. Always `web_search_call`."""

    @model_validator(mode="after")
    def check_type(self) -> "ResponseFunctionWebSearch":
        if self.type != "web_search_call":
            raise ValueError(f"Invalid type: {self.type}")
        return self

    @model_validator(mode="after")
    def check_status(self) -> "ResponseFunctionWebSearch":
        if self.status not in {"in_progress", "searching", "completed", "failed"}:
            raise ValueError(f"Invalid status: {self.status}")
        return self


class PendingSafetyCheck(BaseModel):
    id: str
    """The ID of the pending safety check."""

    code: str
    """The type of the pending safety check."""

    message: str
    """Details about the pending safety check."""


class ResponseComputerToolCall(BaseModel):
    id: str
    """The unique ID of the computer call."""

    # TODO bbqiu revisit this to see if we need to validate Action: TypeAlias
    action: Any
    """A click action."""

    call_id: str
    """An identifier used when responding to the tool call with output."""

    pending_safety_checks: list[PendingSafetyCheck]
    """The pending safety checks for the computer call."""

    status: str
    """The status of the item.

    One of `in_progress`, `completed`, or `incomplete`. Populated when items are
    returned via API.
    """

    type: str = "computer_call"
    """The type of the computer call. Always `computer_call`."""

    @model_validator(mode="after")
    def check_type(self) -> "ResponseComputerToolCall":
        if self.type != "computer_call":
            raise ValueError(f"Invalid type: {self.type}")
        return self

    @model_validator(mode="after")
    def check_status(self) -> "ResponseComputerToolCall":
        if self.status not in {"in_progress", "completed", "incomplete"}:
            raise ValueError(f"Invalid status: {self.status}")
        return self


class Summary(BaseModel):
    text: str
    """
    A short summary of the reasoning used by the model when generating the response.
    """

    type: str = "summary_text"
    """The type of the object. Always `summary_text`."""

    @model_validator(mode="after")
    def check_type(self) -> "Summary":
        if self.type != "summary_text":
            raise ValueError(f"Invalid type: {self.type}")
        return self


class ResponseReasoningItem(BaseModel):
    id: str
    """The unique identifier of the reasoning content."""

    summary: list[Summary]
    """Reasoning text contents."""

    type: str = "reasoning"
    """The type of the object. Always `reasoning`."""

    status: Optional[str] = None
    """The status of the item.

    One of `in_progress`, `completed`, or `incomplete`. Populated when items are
    returned via API.
    """

    @model_validator(mode="after")
    def check_type(self) -> "ResponseReasoningItem":
        if self.type != "reasoning":
            raise ValueError(f"Invalid type: {self.type}")
        return self

    @model_validator(mode="after")
    def check_status(self) -> "ResponseReasoningItem":
        if self.status not in {"in_progress", "completed", "incomplete"}:
            raise ValueError(f"Invalid status: {self.status}")
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
    """For function calling, the type is always `function`."""


class RankingOptions(BaseModel):
    ranker: Optional[str] = None
    """The ranker to use for the file search."""

    score_threshold: Optional[float] = None
    """
    The score threshold for the file search, a number between 0 and 1. Numbers
    closer to 1 will attempt to return only the most relevant results, but may
    return fewer results.
    """

    @model_validator(mode="after")
    def ranker(self) -> "RankingOptions":
        if self.ranker not in {"auto", "default-2024-11-15"}:
            raise ValueError(f"Invalid ranker: {self.ranker}")
        return self


class FileSearchTool(BaseModel):
    type: str = "file_search"
    """The type of the file search tool. Always `file_search`."""

    vector_store_ids: list[str]
    """The IDs of the vector stores to search."""

    # TODO bbqiu revisit this to see if we need to validate the filters
    filters: Optional[Any] = None
    """A filter to apply based on file attributes."""

    max_num_results: Optional[int] = None
    """The maximum number of results to return.

    This number should be between 1 and 50 inclusive.
    """

    ranking_options: Optional[RankingOptions] = None
    """Ranking options for search."""


class FunctionTool(BaseModel):
    name: str
    """The name of the function to call."""

    # TODO bbqiu revisit if we should use FunctionParams from chat.py
    parameters: dict[str, Any]
    """A JSON schema object describing the parameters of the function."""

    strict: bool
    """Whether to enforce strict parameter validation. Default `true`."""

    type: str = "function"
    """The type of the function tool. Always `function`."""

    description: Optional[str] = None
    """A description of the function.

    Used by the model to determine whether or not to call the function.
    """


class ComputerTool(BaseModel):
    display_height: float
    """The height of the computer display."""

    display_width: float
    """The width of the computer display."""

    environment: str
    """The type of computer environment to control."""

    type: str = "computer_use_preview"
    """The type of the computer use tool. Always `computer_use_preview`."""

    @model_validator(mode="after")
    def environment(self) -> "ComputerTool":
        if self.environment not in {"mac", "windows", "ubuntu", "browser"}:
            raise ValueError(f"Invalid environment: {self.environment}")
        return self

    @model_validator(mode="after")
    def type(self) -> "ComputerTool":
        if self.type != "computer_use_preview":
            raise ValueError(f"Invalid type: {self.type}")
        return self


class UserLocation(BaseModel):
    type: str = "approximate"
    """The type of location approximation. Always `approximate`."""

    city: Optional[str] = None
    """Free text input for the city of the user, e.g. `San Francisco`."""

    country: Optional[str] = None
    """
    The two-letter [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1) of
    the user, e.g. `US`.
    """

    region: Optional[str] = None
    """Free text input for the region of the user, e.g. `California`."""

    timezone: Optional[str] = None
    """
    The [IANA timezone](https://timeapi.io/documentation/iana-timezones) of the
    user, e.g. `America/Los_Angeles`.
    """


class WebSearchTool(BaseModel):
    # TODO bbqiu not setting this validator bc it's very subject to change
    type: str
    """The type of the web search tool. One of:

    - `web_search_preview`
    - `web_search_preview_2025_03_11`
    """

    search_context_size: Optional[str] = None
    """
    High level guidance for the amount of context window space to use for the
    search. One of `low`, `medium`, or `high`. `medium` is the default.
    """

    user_location: Optional[UserLocation] = None

    @model_validator(mode="after")
    def search_context_size(self) -> "WebSearchTool":
        if self.search_context_size not in {"low", "medium", "high"}:
            raise ValueError(f"Invalid search_context_size: {self.search_context_size}")
        return self


class Reasoning(BaseModel):
    effort: Optional[str] = None
    """**o-series models only**

    Constrains effort on reasoning for
    [reasoning models](https://platform.openai.com/docs/guides/reasoning). Currently
    supported values are `low`, `medium`, and `high`. Reducing reasoning effort can
    result in faster responses and fewer tokens used on reasoning in a response.
    """

    generate_summary: Optional[str] = None
    """**computer_use_preview only**

    A summary of the reasoning performed by the model. This can be useful for
    debugging and understanding the model's reasoning process. One of `concise` or
    `detailed`.
    """

    @model_validator(mode="after")
    def generate_summary(self) -> "Reasoning":
        if self.generate_summary not in {"concise", "detailed"}:
            raise ValueError(f"Invalid generate_summary: {self.generate_summary}")
        return self

    @model_validator(mode="after")
    def effort(self) -> "Reasoning":
        if self.effort not in {"low", "medium", "high"}:
            raise ValueError(f"Invalid effort: {self.effort}")
        return self


class InputTokensDetails(BaseModel):
    cached_tokens: int
    """The number of tokens that were retrieved from the cache.

    [More on prompt caching](https://platform.openai.com/docs/guides/prompt-caching).
    """


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


class Response(BaseModel):
    id: str
    """Unique identifier for this Response."""

    created_at: float
    """Unix timestamp (in seconds) of when this Response was created."""

    error: Optional[ResponseError] = None
    """An error object returned when the model fails to generate a Response."""

    incomplete_details: Optional[IncompleteDetails] = None
    """Details about why the response is incomplete."""

    instructions: Optional[str] = None
    """
    Inserts a system (or developer) message as the first item in the model's
    context.

    When using along with `previous_response_id`, the instructions from a previous
    response will be not be carried over to the next response. This makes it simple
    to swap out system (or developer) messages in new responses.
    """

    metadata: Optional[dict[str, str]] = None
    """Set of 16 key-value pairs that can be attached to an object.

    This can be useful for storing additional information about the object in a
    structured format, and querying for objects via API or the dashboard.

    Keys are strings with a maximum length of 64 characters. Values are strings with
    a maximum length of 512 characters.
    """

    model: str  # TODO bbqiu revisit this to see if we need to validate the model names
    """Model ID used to generate the response, like `gpt-4o` or `o1`.

    OpenAI offers a wide range of models with different capabilities, performance
    characteristics, and price points. Refer to the
    [model guide](https://platform.openai.com/docs/models) to browse and compare
    available models.
    """

    object: str = "response"
    """The object type of this resource - always set to `response`."""

    output: list[
        Union[
            ResponseOutputMessage,
            ResponseFileSearchToolCall,
            ResponseFunctionToolCall,
            ResponseFunctionWebSearch,
            ResponseComputerToolCall,
            ResponseReasoningItem,
        ]
    ]
    """An array of content items generated by the model.

    - The length and order of items in the `output` array is dependent on the
      model's response.
    - Rather than accessing the first item in the `output` array and assuming it's
      an `assistant` message with the content generated by the model, you might
      consider using the `output_text` property where supported in SDKs.
    """

    parallel_tool_calls: bool
    """Whether to allow the model to run tool calls in parallel."""

    temperature: Optional[float] = None
    """What sampling temperature to use, between 0 and 2.

    Higher values like 0.8 will make the output more random, while lower values like
    0.2 will make it more focused and deterministic. We generally recommend altering
    this or `top_p` but not both.
    """

    tool_choice: Union[str, ToolChoiceFunction]
    """
    How the model should select which tool (or tools) to use when generating a
    response. See the `tools` parameter to see how to specify which tools the model
    can call.
    """

    tools: list[Union[FileSearchTool, FunctionTool, ComputerTool, WebSearchTool]]
    """An array of tools the model may call while generating a response.

    You can specify which tool to use by setting the `tool_choice` parameter.

    The two categories of tools you can provide the model are:

    - **Built-in tools**: Tools that are provided by OpenAI that extend the model's
      capabilities, like
      [web search](https://platform.openai.com/docs/guides/tools-web-search) or
      [file search](https://platform.openai.com/docs/guides/tools-file-search).
      Learn more about
      [built-in tools](https://platform.openai.com/docs/guides/tools).
    - **Function calls (custom tools)**: Functions that are defined by you, enabling
      the model to call your own code. Learn more about
      [function calling](https://platform.openai.com/docs/guides/function-calling).
    """

    top_p: Optional[float] = None
    """
    An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with top_p probability mass. So 0.1
    means only the tokens comprising the top 10% probability mass are considered.

    We generally recommend altering this or `temperature` but not both.
    """

    max_output_tokens: Optional[int] = None
    """
    An upper bound for the number of tokens that can be generated for a response,
    including visible output tokens and
    [reasoning tokens](https://platform.openai.com/docs/guides/reasoning).
    """

    previous_response_id: Optional[str] = None
    """The unique ID of the previous response to the model.

    Use this to create multi-turn conversations. Learn more about
    [conversation state](https://platform.openai.com/docs/guides/conversation-state).
    """

    reasoning: Optional[Reasoning] = None
    """**o-series models only**

    Configuration options for
    [reasoning models](https://platform.openai.com/docs/guides/reasoning).
    """

    status: Optional[str] = None
    """The status of the response generation.

    One of `completed`, `failed`, `in_progress`, or `incomplete`.
    """

    # TODO bbqiu revisit this ResponseFormatTextConfig: TypeAlias
    text: Optional[Any] = None
    """Configuration options for a text response from the model.

    Can be plain text or structured JSON data. Learn more:

    - [Text inputs and outputs](https://platform.openai.com/docs/guides/text)
    - [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
    """

    truncation: Optional[str] = None
    """The truncation strategy to use for the model response.

    - `auto`: If the context of this response and previous ones exceeds the model's
      context window size, the model will truncate the response to fit the context
      window by dropping input items in the middle of the conversation.
    - `disabled` (default): If a model response will exceed the context window size
      for a model, the request will fail with a 400 error.
    """

    usage: Optional[ResponseUsage] = None
    """
    Represents token usage details including input tokens, output tokens, a
    breakdown of output tokens, and the total tokens used.
    """

    user: Optional[str] = None
    """
    A unique identifier representing your end-user, which can help OpenAI to monitor
    and detect abuse.
    [Learn more](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids).
    """

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
    def check_tool_choice(self) -> "Response":
        if isinstance(self.tool_choice, str) and self.tool_choice not in {
            "none",
            "auto",
            "required",
            "file_search",
            "web_search_preview",
            "computer_use_preview",
            "web_search_preview_2025_03_11",
        }:
            raise ValueError(f"Invalid tool_choice: {self.tool_choice}")
        return self

    @model_validator(mode="after")
    def status(self) -> "Response":
        if self.status not in {"completed", "failed", "in_progress", "incomplete"}:
            raise ValueError(f"Invalid status: {self.status}")
        return self

    @model_validator(mode="after")
    def truncation(self) -> "Response":
        if self.truncation not in {"auto", "disabled"}:
            raise ValueError(f"Invalid truncation: {self.truncation}")
        return self
