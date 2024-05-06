

from typing import List, Optional, Union

from mlflow.types.schema import ColSpec, Schema


class Message:
  role: str # "system", "user", or "assistant"
  content: str

class ChatCompletionRequest:
  messages: List[Message]

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

