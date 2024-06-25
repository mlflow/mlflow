import json
import os
import re
import shutil
import sys
from typing import List
from unittest.mock import MagicMock

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
from pyspark.sql import SparkSession


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
    Settings.llm = MockChatLLM()
    Settings.embed_model = embed_model
    Settings.callback_manager = CallbackManager([TokenCountingHandler()])
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
