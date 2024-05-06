

from dataclasses import dataclass
from typing import List, Optional, Union

from mlflow.types.schema import ColSpec, Schema


@dataclass
class Message:
  role: str  = "user" # "system", "user", or "assistant"
  content: str = "What is mlflow?"

@dataclass
class ChatCompletionRequest:
  messages: List[Message] = [Message(role="user", content="What is mlflow?")]

class ChatCompletionRequestSchema(Schema):
  # create a mlflow.types.Schema object for ChatCompletionRequest
  # using a ColSpec object for each field
    def __init__(self):
        super().__init__(inputs=[ColSpec("messages", ChatCompletionRequest)])

class MultiturnChatRequest:
  query: str
  history: Optional[List[Message]]


class ChainCompletionChoice:
  index: int
  message: Message
  finish_reason: str

class ChainCompletionChunk:
  index: int
  delta: Message
  finish_reason: str

class ChatCompletionResponse:
  choices: Union[List[ChainCompletionChoice],
                 List[ChainCompletionChunk]]

class StringResponse:
  content: str

