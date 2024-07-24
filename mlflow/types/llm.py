import time
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional

from mlflow.types.schema import Array, ColSpec, DataType, Object, Property, Schema

# TODO: Switch to pydantic in a future version of MLflow.
#       For now, to prevent adding pydantic as a core dependency,
#       we use dataclasses instead.
#
#       Unfortunately, validation for generic types is not that
#       straightforward. For example, `isinstance(thing, List[T])``
#       is not supported, so the code here is a little ugly.


class _BaseDataclass:
    def _validate_field(self, key, val_type, required):
        value = getattr(self, key, None)
        if required and value is None:
            raise ValueError(f"`{key}` is required")
        if value is not None and not isinstance(value, val_type):
            raise ValueError(
                f"`{key}` must be of type {val_type.__name__}, got {type(value).__name__}"
            )

    def _validate_list(self, key, val_type, required):
        values = getattr(self, key, None)
        if required and values is None:
            raise ValueError(f"`{key}` is required")

        if values is not None:
            if isinstance(values, list) and not all(isinstance(v, val_type) for v in values):
                raise ValueError(f"All items in `{key}` must be of type {val_type.__name__}")
            elif not isinstance(values, list):
                raise ValueError(f"`{key}` must be a list, got {type(values).__name__}")

    def _convert_dataclass_list(self, key, cls, required=True):
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
                    setattr(self, key, [cls(**v) for v in values])
                except TypeError as e:
                    raise ValueError(f"Error when coercing {values} to {cls.__name__}: {e}")
            elif any(not isinstance(v, cls) for v in values):
                raise ValueError(
                    f"Items in `{key}` must all have the same type: {cls.__name__} or dict"
                )

    def to_dict(self):
        return asdict(self, dict_factory=lambda obj: {k: v for (k, v) in obj if v is not None})


@dataclass
class ChatMessage(_BaseDataclass):
    """
    A message in a chat request or response.

    Args:
        role (str): The role of the entity that sent the message (e.g. ``"user"``, ``"system"``).
        content (str): The content of the message.
        name (str): The name of the entity that sent the message. **Optional**.
    """

    role: str
    content: str
    name: Optional[str] = None

    def __post_init__(self):
        self._validate_field("role", str, True)
        self._validate_field("content", str, True)
        self._validate_field("name", str, False)


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
            (sorted in descending order by their probabilites).
        frequency_penalty: (float): An optional param of positive or negative value,
            positive values penalize new tokens based on
            their existing frequency in the text so far, decreasing the model's likelihood to repeat
            the same line verbatim.
        presence_penalty: (float): An optional param of positive or negative value,
            positive values penalize new tokens based on whether they appear in the text so far,
            increasing the model's likelihood to talk about new topics.
    """

    temperature: float = 1.0
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    n: int = 1
    stream: bool = False

    top_p: Optional[float] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

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


@dataclass()
class ChatRequest(ChatParams):
    """
    Format of the request object expected by the chat endpoint.

    Args:
        messages (List[:py:class:`ChatMessage`]): A list of :py:class:`ChatMessage`
            that will be passed to the model. **Optional**, defaults to empty list (``[]``)
        temperature (float): A param used to control randomness and creativity during inference.
            **Optional**, defaults to ``1.0``
        max_tokens (int): The maximum number of new tokens to generate.
            **Optional**, defaults to ``None`` (unlimited)
        stop (List[str]): A list of tokens at which to stop generation. **Optional**,
            defaults to ``None``
        n (int): The number of responses to generate. **Optional**,
            defaults to ``1``
        stream (bool): Whether to stream back responses as they are generated. **Optional**,
            defaults to ``False``
    """

    messages: List[ChatMessage] = field(default_factory=list)

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
    bytes: Optional[List[int]] = None

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
    top_logprobs: List[TopTokenLogProb]
    bytes: Optional[List[int]] = None

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

    content: Optional[List[TokenLogProb]] = None

    def __post_init__(self):
        self._convert_dataclass_list("content", TokenLogProb, False)


@dataclass
class ChatChoice(_BaseDataclass):
    """
    A single chat response generated by the model.
    ref: https://platform.openai.com/docs/api-reference/chat/object

    Args:
        index (int): The index of the response in the list of responses.
        message (:py:class:`ChatMessage`): The message that was generated.
        finish_reason (str): The reason why generation stopped.
            **Optional**, defaults to ``"stop"``
        logprobs (:py:class:`ChatChoiceLogProbs`): Log probability information for the choice.
    """

    index: int
    message: ChatMessage
    finish_reason: str = "stop"
    logprobs: Optional[ChatChoiceLogProbs] = None

    def __post_init__(self):
        self._validate_field("index", int, True)
        self._validate_field("finish_reason", str, True)
        if isinstance(self.message, dict):
            self.message = ChatMessage(**self.message)
        if not isinstance(self.message, ChatMessage):
            raise ValueError(
                f"Expected `message` to be of type ChatMessage or dict, got {type(self.message)}"
            )
        if isinstance(self.logprobs, dict):
            self.logprobs = ChatChoiceLogProbs(**self.logprobs)
        if self.logprobs and not isinstance(self.logprobs, ChatChoiceLogProbs):
            raise ValueError(
                f"Expected `logprobs` to be of type LogProbs or dict, got {type(self.logprobs)}"
            )


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
class ChatResponse(_BaseDataclass):
    """
    The full response object returned by the chat endpoint.

    Args:
        choices (List[:py:class:`ChatChoice`]): A list of :py:class:`ChatChoice` objects
            containing the generated responses
        usage (:py:class:`TokenUsageStats`): An object describing the tokens used by the request.
        id (str): The ID of the response. **Optional**, defaults to ``None``
        model (str): The name of the model used. **Optional**, defaults to ``None``
        object (str): The object type. The value should always be 'chat.completion'
        created (int): The time the response was created.
            **Optional**, defaults to the current time.
    """

    choices: List[ChatChoice]
    usage: TokenUsageStats
    id: Optional[str] = None
    model: Optional[str] = None
    object: Literal["chat.completion"] = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))

    def __post_init__(self):
        self._validate_field("id", str, False)
        self._validate_field("object", str, True)
        self._validate_field("created", int, True)
        self._validate_field("model", str, False)
        self._convert_dataclass_list("choices", ChatChoice)
        if isinstance(self.usage, dict):
            self.usage = TokenUsageStats(**self.usage)
        if not isinstance(self.usage, TokenUsageStats):
            raise ValueError(
                f"Expected `usage` to be of type TokenUsageStats or dict, got {type(self.usage)}"
            )

    def __setattr__(self, name, value):
        # A hack to ensure users cannot overwrite 'object' field
        if name == "object" and value != "chat.completion":
            raise ValueError("`object` field must be 'chat.completion'")
        return super().__setattr__(name, value)


CHAT_MODEL_INPUT_SCHEMA = Schema(
    [
        ColSpec(
            name="messages",
            type=Array(
                Object(
                    [
                        Property("role", DataType.string),
                        Property("content", DataType.string),
                        Property("name", DataType.string, False),
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
            type=Array(
                Object(
                    [
                        Property("index", DataType.long),
                        Property(
                            "message",
                            Object(
                                [
                                    Property("role", DataType.string),
                                    Property("content", DataType.string),
                                    Property("name", DataType.string, False),
                                ]
                            ),
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

CHAT_MODEL_INPUT_EXAMPLE = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
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
