import os
import re
import shutil
import sys

import pytest
from llama_index.core import (
    Document,
    KnowledgeGraphIndex,
    PromptTemplate,
    Settings,
    VectorStoreIndex,
)
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from mlflow.tracing.provider import trace_disabled

from tests.helper_functions import start_mock_openai_server


#### General ####
@pytest.fixture
def model_path(tmp_path):
    model_path = tmp_path.joinpath("model")
    yield model_path

    if os.getenv("GITHUB_ACTIONS") == "true":
        shutil.rmtree(model_path, ignore_errors=True)


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    # NB: ensure that the driver and workers have the same python version
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    with SparkSession.builder.master("local[*]").getOrCreate() as s:
        yield s


@pytest.fixture(scope="module", autouse=True)
def mock_openai():
    with start_mock_openai_server() as base_url:
        yield base_url


#### Settings ####
def _mock_tokenizer(text: str) -> list[str]:
    """Mock tokenizer."""
    tokens = re.split(r"[ \n]", text)
    result = []
    for token in tokens:
        if token.strip() == "":
            continue
        result.append(token.strip())
    return result


@pytest.fixture(autouse=True)
def settings(monkeypatch, mock_openai):
    """Set the LLM and Embedding model to the mock OpenAI server."""
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_BASE", mock_openai)
    monkeypatch.setattr(Settings, "llm", OpenAI())
    monkeypatch.setattr(Settings, "embed_model", OpenAIEmbedding())
    monkeypatch.setattr(Settings, "callback_manager", CallbackManager([LlamaDebugHandler()]))
    monkeypatch.setattr(Settings, "_tokenizer", _mock_tokenizer)  # must bypass setter
    monkeypatch.setattr(Settings, "context_window", 4096)  # this enters the _prompt_helper field
    monkeypatch.setattr(Settings, "node_parser", SentenceSplitter(chunk_size=1024))
    monkeypatch.setattr(Settings, "transformations", [SentenceSplitter(chunk_size=1024)])

    assert all(Settings.__dict__.values())  # ensure the full object is populated

    return Settings


#### Indexes ####
@pytest.fixture
def document():
    return Document.example()


@pytest.fixture
@trace_disabled
def single_index(document):
    return VectorStoreIndex(nodes=[document])


@pytest.fixture
@trace_disabled
def multi_index(document):
    return VectorStoreIndex(nodes=[document] * 5)


@pytest.fixture
def single_graph(document):
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
