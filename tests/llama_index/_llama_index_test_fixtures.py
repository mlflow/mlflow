import os
import re
import shutil
from typing import List

import pytest
from llama_index.core import (
    Document,
    KnowledgeGraphIndex,
    PromptTemplate,
    Settings,
    VectorStoreIndex,
)
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms.mock import MockLLM
from llama_index.core.node_parser import SentenceSplitter


#### General ####
@pytest.fixture
def model_path(tmp_path):
    model_path = tmp_path.joinpath("model")
    yield model_path

    if os.getenv("GITHUB_ACTIONS") == "true":
        shutil.rmtree(model_path, ignore_errors=True)


#### Settings ####
@pytest.fixture
def llm():
    return MockLLM()


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
def settings(llm, embed_model):
    Settings.llm = llm
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
def single_index(settings):
    return VectorStoreIndex(nodes=[Document.example()], embed_model=MockEmbedding(embed_dim=1))


@pytest.fixture
def multi_index(settings):
    return VectorStoreIndex(nodes=[Document.example()] * 5, embed_model=MockEmbedding(embed_dim=1))


@pytest.fixture
def single_graph(settings):
    return KnowledgeGraphIndex.from_documents([Document.example()])


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
