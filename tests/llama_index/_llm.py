from typing import Any, List, Sequence

import numpy as np
from llama_index.embeddings import BaseEmbedding
from llama_index.llms import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.llms.base import llm_chat_callback, llm_completion_callback
from llama_index.llms.generic_utils import completion_response_to_chat_response

# Context window size
_CONTEXT_WINDOW = 2048

# Number of output tokens
_NUM_OUTPUT = 256

DUMMY_CHAT_RESPONSE = "chat response"
DUMMY_RETRIEVE_RESPONSE = "retrieve response"
DUMMY_COMPLETE_RESPONSE = "complete response"
DUMMY_EMBEDDING_RESPONSE = np.random.uniform(-1, _NUM_OUTPUT)


class DatabricksLLM(CustomLLM, extra="allow"):
    def __init__(self, **data: Any):
        super().__init__(**data)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=_CONTEXT_WINDOW,
            num_output=_NUM_OUTPUT,
            model_name=self.endpoint,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=DUMMY_COMPLETE_RESPONSE, raw={})

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError()


class DatabricksEmbedding(BaseEmbedding, extra="allow"):
    def __init__(self, **data: Any):
        super().__init__(**data)

    @classmethod
    def class_name(cls) -> str:
        return "TestingEmbedding"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return DUMMY_EMBEDDING_RESPONSE

    def _get_text_embedding(self, text: str) -> List[float]:
        return DUMMY_EMBEDDING_RESPONSE

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [DUMMY_EMBEDDING_RESPONSE for _ in texts]
