

from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class Message:
  role: str  = "user" # "system", "user", or "assistant"
  content: str = "What is mlflow?"

@dataclass
class ChatCompletionRequest:
  messages: List[Message] = [Message(role="user", content="What is mlflow?")]

@dataclass
class MultiturnChatRequest:
  query: str = "What is mlflow?"
  history: Optional[List[Message]] = None


@dataclass
class ChainCompletionChoice:
  index: int = 0
  message: Message = Message(role="assistant", content="MLflow is an open source platform for the complete machine learning lifecycle.")
  finish_reason: str = "stop"

@dataclass
class ChainCompletionChunk:
  index: int = 0
  delta: Message = Message(role="assistant", content="MLflow is an open source platform for the complete machine learning lifecycle.")
  finish_reason: str = "stop"

@dataclass
class ChatCompletionResponse:
  choices: Union[List[ChainCompletionChoice],
                 List[ChainCompletionChunk]] = [ChainCompletionChoice(index=0, message=Message(role="assistant", content="MLflow is an open source platform for the complete machine learning lifecycle."))]

@dataclass
class StringResponse:
  content: str = "MLflow is an open source platform for the complete machine learning lifecycle."

