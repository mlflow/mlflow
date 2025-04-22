import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Optional

from mlflow.types.schema import AnyType, Array, ColSpec, DataType, Map, Object, Property, Schema

# TODO: Switch to pydantic in a future version of MLflow.
#       For now, to prevent adding pydantic as a core dependency,
#       we use dataclasses instead.
#
#       Unfortunately, validation for generic types is not that
#       straightforward. For example, `isinstance(thing, List[T])``
#       is not supported, so the code here is a little ugly.


JSON_SCHEMA_TYPES = ["string", "number", "integer", "object", "array", "boolean", "null"]


class _BaseDataclass:
    def _validate_field(self, key, val_type, required):
        value = getattr(self, key, None)
        if required and value is None:
            raise ValueError(f"`{key}` is required")
        if value is not None and not isinstance(value, val_type):
            raise ValueError(
                f"`{key}` must be of type {val_type.__name__}, got {type(value).__name__}"
            )

    def _validate_literal(self, key, allowed_values, required):
        value = getattr(self, key, None)
        if required and value is None:
            raise ValueError(f"`{key}` is required")
        if value is not None and value not in allowed_values:
            raise ValueError(f"`{key}` must be one of {allowed_values}, got {value}")

    def _validate_list(self, key, val_type, required):
        values = getattr(self, key, None)
        if required and values is None:
            raise ValueError(f"`{key}` is required")

        if values is not None:
            if isinstance(values, list) and not all(isinstance(v, val_type) for v in values):
                raise ValueError(f"All items in `{key}` must be of type {val_type.__name__}")
            elif not isinstance(values, list):
                raise ValueError(f"`{key}` must be a list, got {type(values).__name__}")

    def _convert_dataclass(self, key: str, cls: "_BaseDataclass", required=True):
        value = getattr(self, key)
        if value is None:
            if required:
                raise ValueError(f"`{key}` is required")
            return

        if isinstance(value, cls):
            return

        if not isinstance(value, dict):
            raise ValueError(
                f"Expected `{key}` to be either an instance of `{cls.__name__}` or "
                f"a dict matching the schema. Received `{type(value).__name__}`"
            )

        try:
            setattr(self, key, cls.from_dict(value))
        except TypeError as e:
            raise ValueError(f"Error when coercing {value} to {cls.__name__}: {e}")

    def _convert_dataclass_list(self, key: str, cls: "_BaseDataclass", required=True):
        values = getattr(self, key)
        if values is None:
            if required:
                raise ValueError(f"`{key}` is required")
            return
        if not isinstance(values, list):
            raise ValueError(f"`{key}` must be a list")

        if len(values) > 0:
            # if the items are all dicts, try to convert them to the desired class
            if all(isinstance(v, dict) for v in values):
                try:
                    setattr(self, key, [cls.from_dict(v) for v in values])
                except TypeError as e:
                    raise ValueError(f"Error when coercing {values} to {cls.__name__}: {e}")
            elif any(not isinstance(v, cls) for v in values):
                raise ValueError(
                    f"Items in `{key}` must all have the same type: {cls.__name__} or dict"
                )

    def _convert_dataclass_map(self, key, cls, required=True):
        mapping = getattr(self, key)
        if mapping is None:
            if required:
                raise ValueError(f"`{key}` is required")
            return

        if not isinstance(mapping, dict):
            raise ValueError(f"`{key}` must be a dict")

        # create a new map to avoid mutating the original
        new_mapping = {}
        for k, v in mapping.items():
            if isinstance(v, cls):
                new_mapping[k] = v
            elif isinstance(v, dict):
                try:
                    new_mapping[k] = cls.from_dict(v)
                except TypeError as e:
                    raise ValueError(f"Error when coercing {v} to {cls.__name__}: {e}")
            else:
                raise ValueError(
                    f"Items in `{key}` must be either an instance of `{cls.__name__}` "
                    f"or a dict matching the schema. Received `{type(v).__name__}`"
                )
        setattr(self, key, new_mapping)

    def to_dict(self):
        return asdict(self, dict_factory=lambda obj: {k: v for (k, v) in obj if v is not None})

    @classmethod
    def from_dict(cls, data):
        """
        Create an instance of the class from a dict, ignoring any undefined fields.
        This is useful when the dict contains extra fields, causing cls(**data) to fail.
        """
        field_names = [field.name for field in fields(cls)]
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)


@dataclass
class FunctionToolCallArguments(_BaseDataclass):
    """
    The arguments of a function tool call made by the model.

    Args:
        arguments (str): A JSON string of arguments that should be passed to the tool.
        name (str): The name of the tool that is being called.
    """

    name: str
    arguments: str

    def __post_init__(self):
        self._validate_field("name", str, True)
        self._validate_field("arguments", str, True)

    def to_tool_call(self, id=None):
        if id is None:
            id = str(uuid.uuid4())
        return ToolCall(id=id, function=self)


@dataclass
class ToolCall(_BaseDataclass):
    """
    A tool call made by the model.

    Args:
        function (:py:class:`FunctionToolCallArguments`): The arguments of the function tool call.
        id (str): The ID of the tool call. Defaults to a random UUID.
        type (str): The type of the object. Defaults to "function".
    """

    function: FunctionToolCallArguments
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "function"

    def __post_init__(self):
        self._validate_field("id", str, True)
        self._convert_dataclass("function", FunctionToolCallArguments, True)
        self._validate_field("type", str, True)


@dataclass
class ChatMessage(_BaseDataclass):
    """
    A message in a chat request or response.

    Args:
        role (str): The role of the entity that sent the message (e.g. ``"user"``,
            ``"system"``, ``"assistant"``, ``"tool"``).
        content (str): The content of the message.
            **Optional** Can be ``None`` if refusal or tool_calls are provided.
        refusal (str): The refusal message content.
            **Optional** Supplied if a refusal response is provided.
        name (str): The name of the entity that sent the message. **Optional**.
        tool_calls (List[:py:class:`ToolCall`]): A list of tool calls made by the model.
            **Optional** defaults to ``None``
        tool_call_id (str): The ID of the tool call that this message is a response to.
            **Optional** defaults to ``None``
    """

    role: str
    content: Optional[str] = None
    refusal: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def __post_init__(self):
        self._validate_field("role", str, True)

        if self.refusal:
            self._validate_field("refusal", str, True)
            if self.content:
                raise ValueError("Both `content` and `refusal` cannot be set")
        elif self.tool_calls:
            self._validate_field("content", str, False)
        else:
            self._validate_field("content", str, True)

        self._validate_field("name", str, False)
        self._convert_dataclass_list("tool_calls", ToolCall, False)
        self._validate_field("tool_call_id", str, False)


@dataclass
class ChatChoiceDelta(_BaseDataclass):
    """
    A streaming message delta in a chat response.

    Args:
        role (str): The role of the entity that sent the message (e.g. ``"user"``,
            ``"system"``, ``"assistant"``, ``"tool"``).
            **Optional** defaults to ``"assistant"``
            This is optional because OpenAI clients can explicitly return None for
            the role
        content (str): The content of the new token being streamed
            **Optional** Can be ``None`` on the last delta chunk or if refusal or
            tool_calls are provided
        refusal (str): The refusal message content.
            **Optional** Supplied if a refusal response is provided.
        name (str): The name of the entity that sent the message. **Optional**.
        tool_calls (List[:py:class:`ToolCall`]): A list of tool calls made by the model.
            **Optional** defaults to ``None``
    """

    role: Optional[str] = "assistant"
    content: Optional[str] = None
    refusal: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None

    def __post_init__(self):
        self._validate_field("role", str, False)

        if self.refusal:
            self._validate_field("refusal", str, True)
            if self.content:
                raise ValueError("Both `content` and `refusal` cannot be set")
        self._validate_field("content", str, False)
        self._validate_field("name", str, False)
        self._convert_dataclass_list("tool_calls", ToolCall, False)


@dataclass
class ParamType(_BaseDataclass):
    type: Literal["string", "number", "integer", "object", "array", "boolean", "null"]

    def __post_init__(self):
        self._validate_literal("type", JSON_SCHEMA_TYPES, True)


@dataclass
class ParamProperty(ParamType):
    """
    A single parameter within a function definition.

    Args:
        type (str): The type of the parameter. Possible values are "string", "number", "integer",
            "object", "array", "boolean", or "null", conforming to the JSON Schema specification.
        description (str): A description of the parameter.
            **Optional**, defaults to ``None``
        enum (List[str]): Used to constrain the possible values for the parameter.
            **Optional**, defaults to ``None``
        items (:py:class:`ParamProperty`): If the param is of ``array`` type, this field can be
            used to specify the type of its items. **Optional**, defaults to ``None``
    """

    description: Optional[str] = None
    enum: Optional[list[str]] = None
    items: Optional[ParamType] = None

    def __post_init__(self):
        self._validate_field("description", str, False)
        self._validate_list("enum", str, False)
        self._convert_dataclass("items", ParamType, False)
        super().__post_init__()


@dataclass
class ToolParamsSchema(_BaseDataclass):
    """
    A tool parameter definition.

    Args:
        properties (Dict[str, :py:class:`ParamProperty`]): A mapping of parameter names to
            their definitions.
        type (str): The type of the parameter. Currently only "object" is supported.
        required (List[str]): A list of required parameter names. **Optional**, defaults to ``None``
        additionalProperties (bool): Whether additional properties are allowed in the object.
            **Optional**, defaults to ``None``
    """

    properties: dict[str, ParamProperty]
    type: Literal["object"] = "object"
    required: Optional[list[str]] = None
    additionalProperties: Optional[bool] = None

    def __post_init__(self):
        self._convert_dataclass_map("properties", ParamProperty, True)
        self._validate_literal("type", ["object"], True)
        self._validate_list("required", str, False)
        self._validate_field("additionalProperties", bool, False)


@dataclass
class FunctionToolDefinition(_BaseDataclass):
    """
    Definition for function tools (currently the only supported type of tool).

    Args:
        name (str): The name of the tool.
        description (str): A description of what the tool does, and how it should be used.
            **Optional**, defaults to ``None``
        parameters: A mapping of parameter names to their
            definitions. If not provided, this defines a function without parameters.
            **Optional**, defaults to ``None``
        strict (bool): A flag that represents whether or not the model should
            strictly follow the schema provided.
    """

    name: str
    description: Optional[str] = None
    parameters: Optional[ToolParamsSchema] = None
    strict: bool = False

    def __post_init__(self):
        self._validate_field("name", str, True)
        self._validate_field("description", str, False)
        self._convert_dataclass("parameters", ToolParamsSchema, False)
        self._validate_field("strict", bool, True)

    def to_tool_definition(self):
        """
        Convenience function for wrapping this in a ToolDefinition
        """
        return ToolDefinition(type="function", function=self)


@dataclass
class ToolDefinition(_BaseDataclass):
    """
    Definition for tools that can be called by the model.

    Args:
        function (:py:class:`FunctionToolDefinition`): The definition of a function tool.
        type (str): The type of the tool. Currently only "function" is supported.
    """

    function: FunctionToolDefinition
    type: Literal["function"] = "function"

    def __post_init__(self):
        self._validate_literal("type", ["function"], True)
        self._convert_dataclass("function", FunctionToolDefinition, True)


@dataclass
class ChatParams(_BaseDataclass):
    """
    Common parameters used for chat inference

    Args:
        temperature (float): A param used to control randomness and creativity during inference.
            **Optional**, defaults to ``1.0``
        max_tokens (int): The maximum number of new tokens to generate.
            **Optional**, defaults to ``None`` (unlimited)
        stop (List[str]): A list of tokens at which to stop generation.
            **Optional**, defaults to ``None``
        n (int): The number of responses to generate.
            **Optional**, defaults to ``1``
        stream (bool): Whether to stream back responses as they are generated.
            **Optional**, defaults to ``False``
        top_p (float): An optional param to control sampling with temperature, the model considers
            the results of the tokens with top_p probability mass. E.g., 0.1 means only the tokens
            comprising the top 10% probability mass are considered.
        top_k (int): An optional param for reducing the vocabulary size to top k tokens
            (sorted in descending order by their probabilities).
        frequency_penalty: (float): An optional param of positive or negative value,
            positive values penalize new tokens based on
            their existing frequency in the text so far, decreasing the model's likelihood to repeat
            the same line verbatim.
        presence_penalty: (float): An optional param of positive or negative value,
            positive values penalize new tokens based on whether they appear in the text so far,
            increasing the model's likelihood to talk about new topics.
        custom_inputs (Dict[str, Any]): An optional param to provide arbitrary additional context
            to the model. The dictionary values must be JSON-serializable.
        tools (List[:py:class:`ToolDefinition`]): An optional list of tools that can be called by
            the model.

    .. warning::

        In an upcoming MLflow release, default values for `temperature`, `n` and `stream` will be
        removed. Please provide these values explicitly in your code if needed.
    """

    temperature: float = 1.0
    max_tokens: Optional[int] = None
    stop: Optional[list[str]] = None
    n: int = 1
    stream: bool = False

    top_p: Optional[float] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    custom_inputs: Optional[dict[str, Any]] = None
    tools: Optional[list[ToolDefinition]] = None

    def __post_init__(self):
        self._validate_field("temperature", float, True)
        self._validate_field("max_tokens", int, False)
        self._validate_list("stop", str, False)
        self._validate_field("n", int, True)
        self._validate_field("stream", bool, True)

        self._validate_field("top_p", float, False)
        self._validate_field("top_k", int, False)
        self._validate_field("frequency_penalty", float, False)
        self._validate_field("presence_penalty", float, False)
        self._convert_dataclass_list("tools", ToolDefinition, False)

        # validate that the custom_inputs field is a map from string to string
        if self.custom_inputs is not None:
            if not isinstance(self.custom_inputs, dict):
                raise ValueError(
                    "Expected `custom_inputs` to be a dictionary, "
                    f"received `{type(self.custom_inputs).__name__}`"
                )
            for key, value in self.custom_inputs.items():
                if not isinstance(key, str):
                    raise ValueError(
                        "Expected `custom_inputs` to be of type `Dict[str, Any]`, "
                        f"received key of type `{type(key).__name__}` (key: {key})"
                    )

    @classmethod
    def keys(cls) -> set[str]:
        """
        Return the keys of the dataclass
        """
        return {field.name for field in fields(cls)}


@dataclass()
class ChatCompletionRequest(ChatParams):
    """
    Format of the request object expected by the chat endpoint.

    Args:
        messages (List[:py:class:`ChatMessage`]): A list of :py:class:`ChatMessage`
            that will be passed to the model. **Optional**, defaults to empty list (``[]``)
        temperature (float): A param used to control randomness and creativity during inference.
            **Optional**, defaults to ``1.0``
        max_tokens (int): The maximum number of new tokens to generate.
            **Optional**, defaults to ``None`` (unlimited)
        stop (List[str]): A list of tokens at which to stop generation.
            **Optional**, defaults to ``None``
        n (int): The number of responses to generate.
            **Optional**, defaults to ``1``
        stream (bool): Whether to stream back responses as they are generated.
            **Optional**, defaults to ``False``
        top_p (float): An optional param to control sampling with temperature, the model considers
            the results of the tokens with top_p probability mass. E.g., 0.1 means only the tokens
            comprising the top 10% probability mass are considered.
        top_k (int): An optional param for reducing the vocabulary size to top k tokens
            (sorted in descending order by their probabilities).
        frequency_penalty: (float): An optional param of positive or negative value,
            positive values penalize new tokens based on
            their existing frequency in the text so far, decreasing the model's likelihood to repeat
            the same line verbatim.
        presence_penalty: (float): An optional param of positive or negative value,
            positive values penalize new tokens based on whether they appear in the text so far,
            increasing the model's likelihood to talk about new topics.
        custom_inputs (Dict[str, Any]): An optional param to provide arbitrary additional context
            to the model. The dictionary values must be JSON-serializable.
        tools (List[:py:class:`ToolDefinition`]): An optional list of tools that can be called by
            the model.

    .. warning::

        In an upcoming MLflow release, default values for `temperature`, `n` and `stream` will be
        removed. Please provide these values explicitly in your code if needed.
    """

    messages: list[ChatMessage] = field(default_factory=list)

    def __post_init__(self):
        self._convert_dataclass_list("messages", ChatMessage)
        super().__post_init__()


@dataclass
class TopTokenLogProb(_BaseDataclass):
    """
    Token and its log probability.

    Args:
        token: The token.
        logprob: The log probability of this token, if it is within the top
            20 most likely tokens. Otherwise, the value -9999.0 is used to
            signify that the token is very unlikely.
        bytes: A list of integers representing the UTF-8 bytes representation
            of the token. Useful in instances where characters are represented
            by multiple tokens and their byte representations must be combined
            to generate the correct text representation. Can be null if there
            is no bytes representation for the token.
    """

    token: str
    logprob: float
    bytes: Optional[list[int]] = None

    def __post_init__(self):
        self._validate_field("token", str, True)
        self._validate_field("logprob", float, True)
        self._validate_list("bytes", int, False)


@dataclass
class TokenLogProb(_BaseDataclass):
    """
    Message content token with log probability information.

    Args:
        token: The token.
        logprob: The log probability of this token, if it is within the top
            20 most likely tokens. Otherwise, the value -9999.0 is used to
            signify that the token is very unlikely.
        bytes: A list of integers representing the UTF-8 bytes representation
            of the token. Useful in instances where characters are represented
            by multiple tokens and their byte representations must be combined
            to generate the correct text representation. Can be null if there
            is no bytes representation for the token.
        top_logprobs: List of the most likely tokens and their log probability,
            at this token position. In rare cases, there may be fewer than the
            number of requested top_logprobs returned.
    """

    token: str
    logprob: float
    top_logprobs: list[TopTokenLogProb]
    bytes: Optional[list[int]] = None

    def __post_init__(self):
        self._validate_field("token", str, True)
        self._validate_field("logprob", float, True)
        self._convert_dataclass_list("top_logprobs", TopTokenLogProb)
        self._validate_list("bytes", int, False)


@dataclass
class ChatChoiceLogProbs(_BaseDataclass):
    """
    Log probability information for the choice.

    Args:
        content: A list of message content tokens with log probability information.
    """

    content: Optional[list[TokenLogProb]] = None

    def __post_init__(self):
        self._convert_dataclass_list("content", TokenLogProb, False)


@dataclass
class ChatChoice(_BaseDataclass):
    """
    A single chat response generated by the model.
    ref: https://platform.openai.com/docs/api-reference/chat/object

    Args:
        message (:py:class:`ChatMessage`): The message that was generated.
        index (int): The index of the response in the list of responses.
            Defaults to ``0``
        finish_reason (str): The reason why generation stopped.
            **Optional**, defaults to ``"stop"``
        logprobs (:py:class:`ChatChoiceLogProbs`): Log probability information for the choice.
            **Optional**, defaults to ``None``
    """

    message: ChatMessage
    index: int = 0
    finish_reason: str = "stop"
    logprobs: Optional[ChatChoiceLogProbs] = None

    def __post_init__(self):
        self._validate_field("index", int, True)
        self._validate_field("finish_reason", str, True)
        self._convert_dataclass("message", ChatMessage, True)
        self._convert_dataclass("logprobs", ChatChoiceLogProbs, False)


@dataclass
class ChatChunkChoice(_BaseDataclass):
    """
    A single chat response chunk generated by the model.
    ref: https://platform.openai.com/docs/api-reference/chat/streaming

    Args:
        index (int): The index of the response in the list of responses.
            defaults to ``0``
        delta (:py:class:`ChatChoiceDelta`): The streaming chunk message that was generated.
        finish_reason (str): The reason why generation stopped.
            **Optional**, defaults to ``None``
        logprobs (:py:class:`ChatChoiceLogProbs`): Log probability information for the choice.
            **Optional**, defaults to ``None``
    """

    delta: ChatChoiceDelta
    index: int = 0
    finish_reason: Optional[str] = None
    logprobs: Optional[ChatChoiceLogProbs] = None

    def __post_init__(self):
        self._validate_field("index", int, True)
        self._validate_field("finish_reason", str, False)
        self._convert_dataclass("delta", ChatChoiceDelta, True)
        self._convert_dataclass("logprobs", ChatChoiceLogProbs, False)


@dataclass
class TokenUsageStats(_BaseDataclass):
    """
    Stats about the number of tokens used during inference.

    Args:
        prompt_tokens (int): The number of tokens in the prompt.
            **Optional**, defaults to ``None``
        completion_tokens (int): The number of tokens in the generated completion.
            **Optional**, defaults to ``None``
        total_tokens (int): The total number of tokens used.
            **Optional**, defaults to ``None``
    """

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    def __post_init__(self):
        self._validate_field("prompt_tokens", int, False)
        self._validate_field("completion_tokens", int, False)
        self._validate_field("total_tokens", int, False)


@dataclass
class ChatCompletionResponse(_BaseDataclass):
    """
    The full response object returned by the chat endpoint.

    Args:
        choices (List[:py:class:`ChatChoice`]): A list of :py:class:`ChatChoice` objects
            containing the generated responses
        usage (:py:class:`TokenUsageStats`): An object describing the tokens used by the request.
            **Optional**, defaults to ``None``.
        id (str): The ID of the response. **Optional**, defaults to ``None``
        model (str): The name of the model used. **Optional**, defaults to ``None``
        object (str): The object type. Defaults to 'chat.completion'
        created (int): The time the response was created.
            **Optional**, defaults to the current time.
        custom_outputs (Dict[str, Any]): An field that can contain arbitrary additional context.
            The dictionary values must be JSON-serializable.
            **Optional**, defaults to ``None``
    """

    choices: list[ChatChoice]
    usage: Optional[TokenUsageStats] = None
    id: Optional[str] = None
    model: Optional[str] = None
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    custom_outputs: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self._validate_field("id", str, False)
        self._validate_field("object", str, True)
        self._validate_field("created", int, True)
        self._validate_field("model", str, False)
        self._convert_dataclass_list("choices", ChatChoice)
        self._convert_dataclass("usage", TokenUsageStats, False)


@dataclass
class ChatCompletionChunk(_BaseDataclass):
    """
    The streaming chunk returned by the chat endpoint.
    ref: https://platform.openai.com/docs/api-reference/chat/streaming

    Args:
        choices (List[:py:class:`ChatChunkChoice`]): A list of :py:class:`ChatChunkChoice` objects
            containing the generated chunk of a streaming response
        usage (:py:class:`TokenUsageStats`): An object describing the tokens used by the request.
            **Optional**, defaults to ``None``.
        id (str): The ID of the response. **Optional**, defaults to ``None``
        model (str): The name of the model used. **Optional**, defaults to ``None``
        object (str): The object type. Defaults to 'chat.completion.chunk'
        created (int): The time the response was created.
            **Optional**, defaults to the current time.
        custom_outputs (Dict[str, Any]): An field that can contain arbitrary additional context.
            The dictionary values must be JSON-serializable.
            **Optional**, defaults to ``None``
    """

    choices: list[ChatChunkChoice]
    usage: Optional[TokenUsageStats] = None
    id: Optional[str] = None
    model: Optional[str] = None
    object: str = "chat.completion.chunk"
    created: int = field(default_factory=lambda: int(time.time()))
    custom_outputs: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self._validate_field("id", str, False)
        self._validate_field("object", str, True)
        self._validate_field("created", int, True)
        self._validate_field("model", str, False)
        self._convert_dataclass_list("choices", ChatChunkChoice)
        self._convert_dataclass("usage", TokenUsageStats, False)


# turn off formatting for the model signatures to preserve readability
# fmt: off

_token_usage_stats_col_spec = ColSpec(
    name="usage",
    type=Object(
        [
            Property("prompt_tokens", DataType.long),
            Property("completion_tokens", DataType.long),
            Property("total_tokens", DataType.long),
        ]
    ),
    required=False,
)
_custom_inputs_col_spec = ColSpec(name="custom_inputs", type=Map(AnyType()), required=False)
_custom_outputs_col_spec = ColSpec(name="custom_outputs", type=Map(AnyType()), required=False)

CHAT_MODEL_INPUT_SCHEMA = Schema(
    [
        ColSpec(
            name="messages",
            type=Array(
                Object(
                    [
                        Property("role", DataType.string),
                        Property("content", DataType.string, False),
                        Property("name", DataType.string, False),
                        Property("refusal", DataType.string, False),
                        Property("tool_calls", Array(Object([
                            Property("id", DataType.string),
                            Property("function", Object([
                                Property("name", DataType.string),
                                Property("arguments", DataType.string),
                            ])),
                            Property("type", DataType.string),
                        ])), False),
                        Property("tool_call_id", DataType.string, False),
                    ]
                )
            ),
        ),
        ColSpec(name="temperature", type=DataType.double, required=False),
        ColSpec(name="max_tokens", type=DataType.long, required=False),
        ColSpec(name="stop", type=Array(DataType.string), required=False),
        ColSpec(name="n", type=DataType.long, required=False),
        ColSpec(name="stream", type=DataType.boolean, required=False),
        ColSpec(name="top_p", type=DataType.double, required=False),
        ColSpec(name="top_k", type=DataType.long, required=False),
        ColSpec(name="frequency_penalty", type=DataType.double, required=False),
        ColSpec(name="presence_penalty", type=DataType.double, required=False),
        ColSpec(
            name="tools",
            type=Array(
                Object([
                    Property("type", DataType.string),
                    Property("function", Object([
                        Property("name", DataType.string),
                        Property("description", DataType.string, False),
                        Property("parameters", Object([
                            Property("properties", Map(Object([
                                Property("type", DataType.string),
                                Property("description", DataType.string, False),
                                Property("enum", Array(DataType.string), False),
                                Property("items", Object([Property("type", DataType.string)]), False), # noqa
                            ]))),
                            Property("type", DataType.string, False),
                            Property("required", Array(DataType.string), False),
                            Property("additionalProperties", DataType.boolean, False),
                        ])),
                        Property("strict", DataType.boolean, False),
                    ]), False),
                ]),
            ),
            required=False,
        ),
        _custom_inputs_col_spec,
    ]
)

CHAT_MODEL_OUTPUT_SCHEMA = Schema(
    [
        ColSpec(name="id", type=DataType.string),
        ColSpec(name="object", type=DataType.string),
        ColSpec(name="created", type=DataType.long),
        ColSpec(name="model", type=DataType.string),
        ColSpec(
            name="choices",
            type=Array(Object([
                Property("index", DataType.long),
                Property("message", Object([
                    Property("role", DataType.string),
                    Property("content", DataType.string, False),
                    Property("name", DataType.string, False),
                    Property("refusal", DataType.string, False),
                    Property("tool_calls",Array(Object([
                        Property("id", DataType.string),
                        Property("function", Object([
                            Property("name", DataType.string),
                            Property("arguments", DataType.string),
                        ])),
                        Property("type", DataType.string),
                    ])), False),
                    Property("tool_call_id", DataType.string, False),
                ])),
                Property("finish_reason", DataType.string),
            ])),
        ),
        _token_usage_stats_col_spec,
        _custom_outputs_col_spec
    ]
)

CHAT_MODEL_INPUT_EXAMPLE = {
    "messages": [
        {"role": "user", "content": "Hello!"},
    ],
    "temperature": 1.0,
    "max_tokens": 10,
    "stop": ["\n"],
    "n": 1,
    "stream": False,
}

COMPLETIONS_MODEL_INPUT_SCHEMA = Schema(
    [
        ColSpec(name="prompt", type=DataType.string),
        ColSpec(name="temperature", type=DataType.double, required=False),
        ColSpec(name="max_tokens", type=DataType.long, required=False),
        ColSpec(name="stop", type=Array(DataType.string), required=False),
        ColSpec(name="n", type=DataType.long, required=False),
        ColSpec(name="stream", type=DataType.boolean, required=False),
    ]
)

COMPLETIONS_MODEL_OUTPUT_SCHEMA = Schema(
    [
        ColSpec(name="id", type=DataType.string),
        ColSpec(name="object", type=DataType.string),
        ColSpec(name="created", type=DataType.long),
        ColSpec(name="model", type=DataType.string),
        ColSpec(
            name="choices",
            type=Array(
                Object(
                    [
                        Property("index", DataType.long),
                        Property(
                            "text",
                            DataType.string,
                        ),
                        Property("finish_reason", DataType.string),
                    ]
                )
            ),
        ),
        ColSpec(
            name="usage",
            type=Object(
                [
                    Property("prompt_tokens", DataType.long),
                    Property("completion_tokens", DataType.long),
                    Property("total_tokens", DataType.long),
                ]
            ),
        ),
    ]
)

EMBEDDING_MODEL_INPUT_SCHEMA = Schema(
    [
        ColSpec(name="input", type=DataType.string),
    ]
)

EMBEDDING_MODEL_OUTPUT_SCHEMA = Schema(
    [
        ColSpec(name="object", type=DataType.string),
        ColSpec(
            name="data",
            type=Array(
                Object(
                    [
                        Property("index", DataType.long),
                        Property("object", DataType.string),
                        Property("embedding", Array(DataType.double)),
                    ]
                )
            ),
        ),
        ColSpec(
            name="usage",
            type=Object(
                [
                    Property("prompt_tokens", DataType.long),
                    Property("total_tokens", DataType.long),
                ]
            ),
        ),
    ]
)
# fmt: on
