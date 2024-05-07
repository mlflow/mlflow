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

    # Pytest keeps the temporary directory created by `tmp_path` fixture for 3 recent test sessions
    # by default. This is useful for debugging during local testing, but in CI it just wastes the
    # disk space.
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
def callback_manager():
    return CallbackManager([TokenCountingHandler()])


@pytest.fixture
def mock_tokenizer():
    def _mock_tokenizer(text: str) -> List[str]:
        """Mock tokenizer."""
        tokens = re.split(r"[ \n]", text)
        result = []
        for token in tokens:
            if token.strip() == "":
                continue
            result.append(token.strip())
        return result

    return _mock_tokenizer


@pytest.fixture
def sentence_splitter():
    return SentenceSplitter(chunk_size=1024)


@pytest.fixture
def settings(llm, embed_model, callback_manager, mock_tokenizer, sentence_splitter):
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.callback_manager = callback_manager
    Settings._tokenizer = mock_tokenizer  # must bypass setter
    Settings.context_window = 4096  # this enters the _prompt_helper field
    Settings.node_parser = sentence_splitter
    Settings.transformations = [sentence_splitter]

    assert all(Settings.__dict__.values())  # ensure the full object is populated
    return Settings


#### Indexes ####
@pytest.fixture
def single_index():
    return VectorStoreIndex(nodes=[Document.example()], embed_model=MockEmbedding(embed_dim=1))


@pytest.fixture
def multi_index():
    return VectorStoreIndex(nodes=[Document.example()] * 5, embed_model=MockEmbedding(embed_dim=1))


@pytest.fixture
def single_graph():
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
