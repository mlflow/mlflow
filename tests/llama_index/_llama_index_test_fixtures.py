import json
import os
import re
import shutil
import sys
from contextlib import contextmanager
from functools import lru_cache
from typing import Dict, Iterator, List, Tuple
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import (
    Document,
    KnowledgeGraphIndex,
    PromptTemplate,
    Settings,
    VectorStoreIndex,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.node_parser import SentenceSplitter
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.create_embedding_response import CreateEmbeddingResponse
from packaging.version import Version
from pyspark.sql import SparkSession


def version_between_inclusive(
    version: str, min_version: str = "0.0.0", max_version: str = "999.999.999"
) -> bool:
    return Version(min_version) <= Version(version) <= Version(max_version)


def mock_lookup(mock_map: Dict, version: str) -> Tuple[str, MagicMock]:
    for version_range, return_value in mock_map.items():
        if version_between_inclusive(version, *version_range):
            return return_value

    raise KeyError(f"No mock found for version {version} in {mock_map.keys()}")


@lru_cache
def openai_sync_client_mocks():
    # Signature
    # - version (str)
    # - openai object path (str)
    # - method name (str)
    # - new object

    sync_mocks = {}
    sync_mocks[()] = {}

    # Syncronous Chat Completions
    # NOTE: key structure is ("min_version", "max_version")
    # TODO: convert to rangedict
    sync_mocks[()]["openai.resources.chat.completions.Completions.create"] = ChatCompletion(
        **{
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo-0125",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "\n\nPATCHED!! Hello there, how may I assist you today?",
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
    )

    sync_mocks[()]["openai.resources.embeddings.Embeddings.create"] = CreateEmbeddingResponse(
        **{
            "data": [{"index": 0, "object": "embedding", "embedding": [0.1] * 1536}],
            "model": "text-embedding-3-small",
            "object": "list",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
    )

    return sync_mocks


def _append_mocks_to_patches(patches: List, mock_map: Dict, version: str) -> List:
    mocks = mock_lookup(mock_map, version)
    for qualified_path, return_value in mocks.items():
        patches.append(patch(qualified_path, return_value=return_value))

    return patches


@contextmanager
def llama_index_patches(
    patch_openai_sync: bool = True, set_openai_token: bool = True
) -> Iterator[None]:
    if set_openai_token:
        old_openai_value = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "PATCHED_KEY"

    patches = []

    if patch_openai_sync:
        import openai

        patches = _append_mocks_to_patches(patches, openai_sync_client_mocks(), openai.__version__)

    try:
        for p in patches:
            p.start()
        yield
    finally:
        for p in patches:
            p.stop()

    if set_openai_token:
        if old_openai_value:
            os.environ["OPENAI_API_KEY"] = old_openai_value
        else:
            os.environ.pop("OPENAI_API_KEY")

    # AChat Completions


#### General ####
@pytest.fixture
def model_path(tmp_path):
    model_path = tmp_path.joinpath("model")
    yield model_path

    if os.getenv("GITHUB_ACTIONS") == "true":
        shutil.rmtree(model_path, ignore_errors=True)


@pytest.fixture(scope="module")
def spark():
    # NB: ensure that the driver and workers have the same python version
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    with SparkSession.builder.master("local[*]").getOrCreate() as s:
        yield s


#### Settings ####
class MockChatLLM(MagicMock):
    def chat(self, prompt: str) -> ChatResponse:
        test_object = {"hello": "chat"}
        text = json.dumps(test_object)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text))

    @property
    def metadata(self) -> LLMMetadata:
        metadata = LLMMetadata()
        metadata.is_chat_model = True
        return metadata


@pytest.fixture
def embed_model():
    return MockEmbedding(embed_dim=1)


@pytest.fixture
def _callback_manager():
    return CallbackManager([TokenCountingHandler()])


def _mock_tokenizer(text: str) -> List[str]:
    """Mock tokenizer."""
    tokens = re.split(r"[ \n]", text)
    result = []
    for token in tokens:
        if token.strip() == "":
            continue
        result.append(token.strip())
    return result


@pytest.fixture
def settings(embed_model):
    # Settings.llm = OpenAI(api_key="mock key")
    # Settings.embed_model = OpenAI(api_key="mock key")
    Settings._tokenizer = _mock_tokenizer  # must bypass setter
    Settings.context_window = 4096  # this enters the _prompt_helper field
    Settings.node_parser = SentenceSplitter(chunk_size=1024)
    Settings.transformations = [SentenceSplitter(chunk_size=1024)]

    assert all(Settings.__dict__.values())  # ensure the full object is populated
    return Settings


#### Indexes ####
@pytest.fixture
def document():
    return Document.example()


@pytest.fixture
def single_index(settings, document):
    return VectorStoreIndex(nodes=[document], embed_model=MockEmbedding(embed_dim=1))


@pytest.fixture
def multi_index(settings, document):
    return VectorStoreIndex(nodes=[document] * 5, embed_model=MockEmbedding(embed_dim=1))


@pytest.fixture
def single_graph(settings, document):
    return KnowledgeGraphIndex.from_documents([document])


#### Prompt Templates ####
@pytest.fixture
def qa_prompt_template():
    return PromptTemplate(
        template="""
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Please write the answer in the style of {tone_name}
    Query: {query_str}
    Answer:
    """
    )
