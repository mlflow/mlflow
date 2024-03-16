import importlib
import json
import os
import re
import shutil
import sqlite3
from contextlib import contextmanager
from operator import itemgetter
from typing import Any, DefaultDict, Dict, List, Mapping, Optional
from unittest import mock

import langchain
import numpy as np
import openai
import pytest
import transformers
from langchain import SQLDatabase
from langchain.agents import AgentType, initialize_agent
from langchain.chains import (
    APIChain,
    ConversationChain,
    HypotheticalDocumentEmbedder,
    LLMChain,
    RetrievalQA,
)
from langchain.chains.api import open_meteo_docs
from langchain.chains.base import Chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings.base import Embeddings
from langchain.embeddings.fake import FakeEmbeddings
from langchain.evaluation.qa import QAEvalChain
from langchain.llms import HuggingFacePipeline, OpenAI
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.requests import TextRequestsWrapper
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import Tool
from langchain.vectorstores import FAISS
from langchain_experimental.sql import SQLDatabaseChain
from packaging import version
from packaging.version import Version
from pydantic import BaseModel
from pyspark.sql import SparkSession

import mlflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow.deployments import PredictionsResponse
from mlflow.exceptions import MlflowException
from mlflow.langchain.api_request_parallel_processor import APIRequest
from mlflow.models.signature import ModelSignature, Schema, infer_signature
from mlflow.types.schema import Array, ColSpec, DataType, Object, Property
from mlflow.utils.openai_utils import (
    TEST_CONTENT,
    TEST_INTERMEDIATE_STEPS,
    TEST_SOURCE_DOCUMENTS,
    _mock_chat_completion_response,
    _mock_request,
    _MockResponse,
)

from tests.helper_functions import pyfunc_serve_and_score_model

try:
    import langchain_community

    # this kwarg was added in langchain_community 0.0.27, and
    # prevents the use of pickled objects if not provided.
    VECTORSTORE_KWARGS = (
        {"allow_dangerous_deserialization": True}
        if Version(langchain_community.__version__) >= Version("0.0.27")
        else {}
    )
except ImportError:
    VECTORSTORE_KWARGS = {}


@contextmanager
def _mock_async_request(content=TEST_CONTENT):
    with _mock_request(return_value=_mock_chat_completion_response(content)) as m:
        yield m


@pytest.fixture
def model_path(tmp_path):
    return tmp_path / "model"


@pytest.fixture(scope="module")
def spark():
    with SparkSession.builder.master("local[*]").getOrCreate() as s:
        yield s


@pytest.fixture(autouse=True)
def set_envs(monkeypatch):
    monkeypatch.setenvs(
        {
            "MLFLOW_TESTING": "true",
            "OPENAI_API_KEY": "test",
            "SERPAPI_API_KEY": "test",
        }
    )
    importlib.reload(openai)


def create_huggingface_model(model_path):
    architecture = "lordtt13/emo-mobilebert"
    mlflow.transformers.save_model(
        transformers_model={
            "model": transformers.TFMobileBertForSequenceClassification.from_pretrained(
                architecture
            ),
            "tokenizer": transformers.AutoTokenizer.from_pretrained(architecture),
        },
        path=model_path,
    )
    llm = mlflow.transformers.load_model(model_path)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    hf_pipe = HuggingFacePipeline(pipeline=llm)
    return LLMChain(llm=hf_pipe, prompt=prompt)


def create_openai_llmchain():
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    return LLMChain(llm=llm, prompt=prompt)


def create_qa_eval_chain():
    llm = OpenAI(temperature=0)
    return QAEvalChain.from_llm(llm)


def create_qa_with_sources_chain():
    # StuffDocumentsChain
    return load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")


def create_openai_llmagent(return_intermediate_steps=False):
    from langchain.agents import AgentType, initialize_agent, load_tools

    # First, let's load the language model we're going to use to control the agent.
    llm = OpenAI(temperature=0)

    # Next, let's load some tools to use.
    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # Finally, let's initialize an agent with the tools.
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=return_intermediate_steps,
    )


class FakeLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    queries: Optional[Mapping] = None
    endpoint_name: str = "fake-llm-endpoint"

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager=None) -> str:
        """First try to lookup in queries, else return 'foo' or 'bar'."""
        if self.queries is not None:
            return self.queries[prompt]
        if stop is None:
            return "foo"
        else:
            return "bar"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}


class FakeChain(Chain):
    """Fake chain class for testing purposes."""

    be_correct: bool = True
    the_input_keys: List[str] = ["foo"]
    the_output_keys: List[str] = ["bar"]

    @property
    def input_keys(self) -> List[str]:
        """Input keys."""
        return self.the_input_keys

    @property
    def output_keys(self) -> List[str]:
        """Output key of bar."""
        return self.the_output_keys

    def _call(self, inputs: Dict[str, str], run_manager=None) -> Dict[str, str]:
        if self.be_correct:
            return {"bar": "baz"}
        else:
            return {"baz": "bar"}


def get_fake_chat_model(endpoint_name="fake-endpoint"):
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.chat_models.base import SimpleChatModel
    from langchain.schema.messages import BaseMessage

    class FakeChatModel(SimpleChatModel):
        """Fake Chat Model wrapper for testing purposes."""

        endpoint_name: str = "fake-endpoint"

        def _call(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            return "Databricks"

        @property
        def _llm_type(self) -> str:
            return "fake chat model"

    return FakeChatModel(endpoint_name=endpoint_name)


@pytest.fixture
def fake_chat_model():
    return get_fake_chat_model()


@pytest.fixture
def fake_classifier_chat_model():
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.chat_models.base import SimpleChatModel
    from langchain.schema.messages import BaseMessage

    class FakeMLflowClassifier(SimpleChatModel):
        """Fake Chat Model wrapper for testing purposes."""

        def _call(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            if "MLflow" in messages[0].content.split(":")[1]:
                return "yes"
            if "cat" in messages[0].content.split(":")[1]:
                return "no"
            return "unknown"

        @property
        def _llm_type(self) -> str:
            return "fake mlflow classifier"

    return FakeMLflowClassifier()


def test_langchain_native_save_and_load_model(model_path):
    model = create_openai_llmchain()
    mlflow.langchain.save_model(model, model_path)

    loaded_model = mlflow.langchain.load_model(model_path)
    assert type(loaded_model) == langchain.chains.llm.LLMChain
    assert type(loaded_model.llm) == langchain.llms.openai.OpenAI
    assert type(loaded_model.prompt) == langchain.prompts.PromptTemplate
    assert loaded_model.prompt.template == "What is a good name for a company that makes {product}?"


def test_langchain_native_log_and_load_model():
    model = create_openai_llmchain()
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, "langchain_model")

    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)

    assert "langchain" in logged_model.flavors
    assert str(logged_model.signature.inputs) == "['product': string (required)]"
    assert str(logged_model.signature.outputs) == "['text': string (required)]"

    assert type(loaded_model) == langchain.chains.llm.LLMChain
    assert type(loaded_model.llm) == langchain.llms.openai.OpenAI
    assert type(loaded_model.prompt) == langchain.prompts.PromptTemplate
    assert loaded_model.prompt.template == "What is a good name for a company that makes {product}?"


def test_pyfunc_load_openai_model():
    model = create_openai_llmchain()
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, "langchain_model")

    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)

    assert "langchain" in logged_model.flavors
    assert type(loaded_model) == mlflow.pyfunc.PyFuncModel


def test_langchain_model_predict():
    with _mock_request(return_value=_mock_chat_completion_response()):
        model = create_openai_llmchain()
        with mlflow.start_run():
            logged_model = mlflow.langchain.log_model(model, "langchain_model")
        loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
        result = loaded_model.predict([{"product": "MLflow"}])
        assert result == [TEST_CONTENT]


def test_pyfunc_spark_udf_with_langchain_model(spark):
    model = create_openai_llmchain()
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, "langchain_model")
    loaded_model = mlflow.pyfunc.spark_udf(spark, logged_model.model_uri, result_type="string")
    df = spark.createDataFrame([("MLflow",), ("Spark",)], ["product"])
    df = df.withColumn("answer", loaded_model())
    pdf = df.toPandas()
    assert pdf["answer"].tolist() == [TEST_CONTENT, TEST_CONTENT]


def test_langchain_log_huggingface_hub_model_metadata(model_path):
    model = create_huggingface_model(model_path)
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, "langchain_model")

    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)

    assert "langchain" in logged_model.flavors
    assert str(logged_model.signature.inputs) == "['product': string (required)]"
    assert str(logged_model.signature.outputs) == "['text': string (required)]"

    assert type(loaded_model) == langchain.chains.llm.LLMChain
    assert type(loaded_model.llm) == HuggingFacePipeline
    assert type(loaded_model.prompt) == langchain.prompts.PromptTemplate
    assert loaded_model.prompt.template == "What is a good name for a company that makes {product}?"


@pytest.mark.parametrize("return_intermediate_steps", [False, True])
def test_langchain_agent_model_predict(return_intermediate_steps):
    langchain_agent_output = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "text": f"Final Answer: {TEST_CONTENT}",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }
    model = create_openai_llmagent(return_intermediate_steps=return_intermediate_steps)
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, "langchain_model")
    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    langchain_input = {
        "input": "What was the high temperature in SF yesterday in Fahrenheit?"
        "What is that number raised to the .023 power?"
    }

    if return_intermediate_steps:
        langchain_output = [{"output": TEST_CONTENT, "intermediate_steps": TEST_INTERMEDIATE_STEPS}]
        # hardcoded output key because that is the default for an agent
        # but it is not an attribute of the agent or anything that we log
    else:
        langchain_output = [TEST_CONTENT]

    with _mock_request(return_value=_MockResponse(200, langchain_agent_output)):
        result = loaded_model.predict([langchain_input])
        assert result == langchain_output

    inference_payload = json.dumps({"inputs": langchain_input})
    langchain_agent_output_serving = {"predictions": langchain_agent_output}
    with _mock_request(return_value=_MockResponse(200, langchain_agent_output_serving)):
        response = pyfunc_serve_and_score_model(
            logged_model.model_uri,
            data=inference_payload,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
            extra_args=["--env-manager", "local"],
        )

        assert (
            PredictionsResponse.from_json(response.content.decode("utf-8"))
            == langchain_agent_output_serving
        )


def test_langchain_native_log_and_load_qaevalchain():
    # QAEvalChain is a subclass of LLMChain
    model = create_qa_eval_chain()
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, "langchain_model")

    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)
    assert model == loaded_model


def test_langchain_native_log_and_load_qa_with_sources_chain():
    # StuffDocumentsChain is a subclass of Chain
    model = create_qa_with_sources_chain()
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, "langchain_model")

    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)
    assert model == loaded_model


@pytest.mark.skipif(
    version.parse(langchain.__version__) < version.parse("0.0.194"),
    reason="Saving RetrievalQA chains requires langchain>=0.0.194",
)
def test_log_and_load_retrieval_qa_chain(tmp_path):
    # Create the vector db, persist the db to a local fs folder
    loader = TextLoader("tests/langchain/state_of_the_union.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = FakeEmbeddings(size=5)
    db = FAISS.from_documents(docs, embeddings)
    persist_dir = str(tmp_path / "faiss_index")
    db.save_local(persist_dir)

    # Create the RetrievalQA chain
    retrievalQA = RetrievalQA.from_llm(llm=OpenAI(), retriever=db.as_retriever())

    # Log the RetrievalQA chain
    def load_retriever(persist_directory):
        embeddings = FakeEmbeddings(size=5)
        vectorstore = FAISS.load_local(
            persist_directory,
            embeddings,
            **VECTORSTORE_KWARGS,
        )
        return vectorstore.as_retriever()

    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            retrievalQA,
            "retrieval_qa_chain",
            loader_fn=load_retriever,
            persist_dir=persist_dir,
        )

    # Remove the persist_dir
    shutil.rmtree(persist_dir)

    # Load the chain
    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)
    assert loaded_model == retrievalQA

    loaded_pyfunc_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    langchain_input = {"query": "What did the president say about Ketanji Brown Jackson"}
    langchain_output = [TEST_CONTENT]
    result = loaded_pyfunc_model.predict([langchain_input])
    assert result == langchain_output

    # Serve the chain
    inference_payload = json.dumps({"inputs": langchain_input})
    langchain_output_serving = {"predictions": langchain_output}

    response = pyfunc_serve_and_score_model(
        logged_model.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    assert (
        PredictionsResponse.from_json(response.content.decode("utf-8")) == langchain_output_serving
    )


@pytest.mark.skipif(
    version.parse(langchain.__version__) < version.parse("0.0.194"),
    reason="Saving RetrievalQA chains requires langchain>=0.0.194",
)
def test_log_and_load_retrieval_qa_chain_multiple_output(tmp_path):
    # Create the vector db, persist the db to a local fs folder
    loader = TextLoader("tests/langchain/state_of_the_union.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = FakeEmbeddings(size=5)
    db = FAISS.from_documents(docs, embeddings)
    persist_dir = str(tmp_path / "faiss_index")
    db.save_local(persist_dir)

    # Create the RetrievalQA chain
    retrievalQA = RetrievalQA.from_llm(
        llm=OpenAI(), retriever=db.as_retriever(), return_source_documents=True
    )

    # Log the RetrievalQA chain
    def load_retriever(persist_directory):
        embeddings = FakeEmbeddings(size=5)
        vectorstore = FAISS.load_local(
            persist_directory,
            embeddings,
            **VECTORSTORE_KWARGS,
        )
        return vectorstore.as_retriever()

    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            retrievalQA,
            "retrieval_qa_chain",
            loader_fn=load_retriever,
            persist_dir=persist_dir,
        )

    # Remove the persist_dir
    shutil.rmtree(persist_dir)

    # Load the chain
    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)
    assert loaded_model == retrievalQA

    loaded_pyfunc_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    langchain_input = {"query": "What did the president say about Ketanji Brown Jackson"}
    langchain_output = [
        {
            loaded_model.output_key: TEST_CONTENT,
            "source_documents": TEST_SOURCE_DOCUMENTS,
        }
    ]
    result = loaded_pyfunc_model.predict([langchain_input])

    assert result == langchain_output

    # Serve the chain
    inference_payload = json.dumps({"inputs": langchain_input})
    langchain_output_serving = {"predictions": langchain_output}

    response = pyfunc_serve_and_score_model(
        logged_model.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    assert (
        PredictionsResponse.from_json(response.content.decode("utf-8")) == langchain_output_serving
    )


# Define a special embedding for testing
class DeterministicDummyEmbeddings(Embeddings, BaseModel):
    size: int

    def _get_embedding(self, text: str) -> List[float]:
        seed = abs(hash(text)) % (10**8)
        np.random.seed(seed)
        return list(np.random.normal(size=self.size))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)


def assert_equal_retrievers(retriever, expected_retreiver):
    from langchain.schema.retriever import BaseRetriever

    assert isinstance(retriever, BaseRetriever)
    assert isinstance(retriever, type(expected_retreiver))
    assert isinstance(retriever.vectorstore, type(expected_retreiver.vectorstore))
    assert retriever.tags == expected_retreiver.tags
    assert retriever.metadata == expected_retreiver.metadata
    assert retriever.search_type == expected_retreiver.search_type
    assert retriever.search_kwargs == expected_retreiver.search_kwargs


def test_log_and_load_retriever_chain(tmp_path):
    # Create the vector db, persist the db to a local fs folder
    loader = TextLoader("tests/langchain/state_of_the_union.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = DeterministicDummyEmbeddings(size=5)
    db = FAISS.from_documents(docs, embeddings)
    persist_dir = str(tmp_path / "faiss_index")
    db.save_local(persist_dir)

    # Define the loader_fn
    def load_retriever(persist_directory):
        from typing import List  # clint: disable=lazy-builtin-import

        import numpy as np
        from langchain.embeddings.base import Embeddings
        from pydantic import BaseModel

        class DeterministicDummyEmbeddings(Embeddings, BaseModel):
            size: int

            def _get_embedding(self, text: str) -> List[float]:
                if isinstance(text, np.ndarray):
                    text = text.item()
                seed = abs(hash(text)) % (10**8)
                np.random.seed(seed)
                return list(np.random.normal(size=self.size))

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return [self._get_embedding(t) for t in texts]

            def embed_query(self, text: str) -> List[float]:
                return self._get_embedding(text)

        embeddings = DeterministicDummyEmbeddings(size=5)
        vectorstore = FAISS.load_local(
            persist_directory,
            embeddings,
            **VECTORSTORE_KWARGS,
        )
        return vectorstore.as_retriever()

    # Log the retriever
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            db.as_retriever(),
            "retriever",
            loader_fn=load_retriever,
            persist_dir=persist_dir,
        )

    # Remove the persist_dir
    shutil.rmtree(persist_dir)

    # Load the retriever
    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)
    assert_equal_retrievers(loaded_model, db.as_retriever())

    loaded_pyfunc_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    query = "What did the president say about Ketanji Brown Jackson"
    langchain_input = {"query": query}
    result = loaded_pyfunc_model.predict([langchain_input])
    expected_result = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in db.as_retriever().get_relevant_documents(query)
    ]
    assert result == [expected_result]

    # Serve the retriever
    inference_payload = json.dumps({"inputs": langchain_input})
    response = pyfunc_serve_and_score_model(
        logged_model.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    pred = PredictionsResponse.from_json(response.content.decode("utf-8"))["predictions"]
    assert type(pred) == list
    assert len(pred) == 1
    docs_list = pred[0]
    assert type(docs_list) == list
    assert len(docs_list) == 4
    # The returned docs are non-deterministic when used with dummy embeddings,
    # so we cannot assert pred == {"predictions": [expected_result]}


def load_requests_wrapper(_):
    return TextRequestsWrapper(headers=None, aiosession=None)


def test_log_and_load_api_chain():
    llm = OpenAI(temperature=0)
    apichain = APIChain.from_llm_and_api_docs(
        llm,
        open_meteo_docs.OPEN_METEO_DOCS,
        verbose=True,
        limit_to_domains=["test.com"],
    )

    # Log the APIChain
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            apichain,
            "api_chain",
            loader_fn=load_requests_wrapper,
        )

    # Load the chain
    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)
    assert loaded_model == apichain


def test_log_and_load_subclass_of_specialized_chain():
    class APIChainSubclass(APIChain):
        pass

    llm = OpenAI(temperature=0)
    apichain_subclass = APIChainSubclass.from_llm_and_api_docs(
        llm,
        open_meteo_docs.OPEN_METEO_DOCS,
        verbose=True,
        limit_to_domains=["test.com"],
    )

    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            apichain_subclass,
            "apichain_subclass",
            loader_fn=load_requests_wrapper,
        )

    # Load the chain
    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)
    assert loaded_model == apichain_subclass


def load_base_embeddings(_):
    return FakeEmbeddings(size=32)


@pytest.mark.skip(reason="This fails due to https://github.com/hwchase17/langchain/issues/5131")
def test_log_and_load_hyde_chain():
    # Create the HypotheticalDocumentEmbedder chain
    base_embeddings = FakeEmbeddings(size=32)
    llm = OpenAI()
    # Load with `web_search` prompt
    embeddings = HypotheticalDocumentEmbedder.from_llm(llm, base_embeddings, "web_search")

    # Log the hyde chain
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            embeddings,
            "hyde_chain",
            loader_fn=load_base_embeddings,
        )

    # Load the chain
    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)
    assert loaded_model == embeddings


def create_sqlite_db_file(db_dir):
    # Connect to SQLite database (or create it if it doesn't exist)
    with sqlite3.connect(db_dir) as conn:
        # Create a cursor
        c = conn.cursor()

        # Create a dummy table
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS employees(
                id INTEGER PRIMARY KEY,
                name TEXT,
                salary REAL,
                department TEXT,
                position TEXT,
                hireDate TEXT);
            """
        )

        # Insert dummy data into the table
        c.execute(
            """
            INSERT INTO employees (name, salary, department, position, hireDate)
            VALUES ('John Doe', 80000, 'IT', 'Engineer', '2023-06-26');
            """
        )


def load_db(persist_dir):
    db_file_path = os.path.join(persist_dir, "my_database.db")
    sqlite_uri = f"sqlite:///{db_file_path}"
    return SQLDatabase.from_uri(sqlite_uri)


@pytest.mark.skipif(
    version.parse(langchain.__version__) < version.parse("0.0.297"),
    reason="Saving SQLDatabaseChain chains requires langchain>=0.0.297",
)
def test_log_and_load_sql_database_chain(tmp_path):
    # Create the SQLDatabaseChain
    db_file_path = tmp_path / "my_database.db"
    sqlite_uri = f"sqlite:///{db_file_path}"
    llm = OpenAI(temperature=0)
    create_sqlite_db_file(db_file_path)
    db = SQLDatabase.from_uri(sqlite_uri)
    db_chain = SQLDatabaseChain.from_llm(llm, db)

    # Log the SQLDatabaseChain
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            db_chain,
            "sql_database_chain",
            loader_fn=load_db,
            persist_dir=tmp_path,
        )

    # Load the chain
    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)
    assert loaded_model == db_chain


def test_saving_not_implemented_for_memory():
    conversation = ConversationChain(llm=OpenAI(temperature=0), memory=ConversationBufferMemory())
    with pytest.raises(
        ValueError,
        match="Saving of memory is not yet supported.",
    ):
        with mlflow.start_run():
            mlflow.langchain.log_model(conversation, "conversation_model")


def test_saving_not_implemented_chain_type():
    chain = FakeChain()
    if version.parse(langchain.__version__) < version.parse("0.0.309"):
        error_message = "Saving not supported for this chain type"
    else:
        error_message = f"Chain {chain} does not support saving."
    with pytest.raises(
        NotImplementedError,
        match=error_message,
    ):
        with mlflow.start_run():
            mlflow.langchain.log_model(chain, "fake_chain")


def test_unsupported_class():
    llm = FakeLLM()
    with pytest.raises(
        MlflowException,
        match="MLflow langchain flavor only supports subclasses of "
        + "langchain.chains.base.Chain",
    ):
        with mlflow.start_run():
            mlflow.langchain.log_model(llm, "fake_llm")


def test_agent_with_unpicklable_tools(tmp_path):
    tmp_file = tmp_path / "temp_file.txt"
    with open(tmp_file, mode="w") as temp_file:
        # files that aren't opened for reading cannot be pickled
        tools = [
            Tool.from_function(
                func=lambda: temp_file,
                name="Write 0",
                description="If you need to write 0 to a file",
            )
        ]
        agent = initialize_agent(
            llm=OpenAI(temperature=0),
            tools=tools,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

        with pytest.raises(
            MlflowException,
            match=(
                "Error when attempting to pickle the AgentExecutor tools. "
                "This model likely does not support serialization."
            ),
        ):
            with mlflow.start_run():
                mlflow.langchain.log_model(agent, "unpicklable_tools")


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"),
    reason="feature not existing",
)
def test_save_load_runnable_passthrough():
    from langchain.schema.runnable import RunnablePassthrough

    runnable = RunnablePassthrough()
    assert runnable.invoke("hello") == "hello"

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(runnable, "model_path")

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke("hello") == "hello"
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict(["hello"]) == ["hello"]

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": ["hello"]}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": ["hello"]
    }


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"),
    reason="feature not existing",
)
def test_save_load_runnable_lambda(spark):
    from langchain.schema.runnable import RunnableLambda

    def add_one(x: int) -> int:
        return x + 1

    runnable = RunnableLambda(add_one)

    assert runnable.invoke(1) == 2
    assert runnable.batch([1, 2, 3]) == [2, 3, 4]

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            runnable, "runnable_lambda", input_example=[1, 2, 3]
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke(1) == 2
    assert loaded_model.batch([1, 2, 3]) == [2, 3, 4]

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert loaded_model.predict(1) == [2]
    assert loaded_model.predict([1, 2, 3]) == [2, 3, 4]

    udf = mlflow.pyfunc.spark_udf(spark, model_info.model_uri, result_type="long")
    df = spark.createDataFrame([(1,), (2,), (3,)], ["data"])
    df = df.withColumn("answer", udf("data"))
    pdf = df.toPandas()
    assert pdf["answer"].tolist() == [2, 3, 4]

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": [1, 2, 3]}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [2, 3, 4]
    }


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"),
    reason="feature not existing",
)
def test_save_load_runnable_lambda_in_sequence():
    from langchain.schema.runnable import RunnableLambda

    def add_one(x):
        return x + 1

    def mul_two(x):
        return x * 2

    runnable_1 = RunnableLambda(add_one)
    runnable_2 = RunnableLambda(mul_two)
    sequence = runnable_1 | runnable_2
    assert sequence.invoke(1) == 4

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(sequence, "model_path", input_example=[1, 2, 3])

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke(1) == 4
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict(1) == [4]
    assert pyfunc_loaded_model.predict([1, 2, 3]) == [4, 6, 8]

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": [1, 2, 3]}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [4, 6, 8]
    }


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_predict_with_callbacks(fake_chat_model):
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser

    class TestCallbackHandler(BaseCallbackHandler):
        def __init__(self):
            super().__init__()
            self.num_llm_start_calls = 0

        def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: List[str],
            **kwargs: Any,
        ) -> Any:
            self.num_llm_start_calls += 1

    prompt = ChatPromptTemplate.from_template("What's your favorite {industry} company?")
    chain = prompt | fake_chat_model | StrOutputParser()
    # Test the basic functionality of the chain
    assert chain.invoke({"industry": "tech"}) == "Databricks"

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain, "model_path", input_example={"industry": "tech"}
        )

    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    callback_handler1 = TestCallbackHandler()
    callback_handler2 = TestCallbackHandler()

    # Ensure handlers have not been called yet
    assert callback_handler1.num_llm_start_calls == 0
    assert callback_handler2.num_llm_start_calls == 0

    assert (
        pyfunc_loaded_model._model_impl._predict_with_callbacks(
            {"industry": "tech"},
            callback_handlers=[callback_handler1, callback_handler2],
        )
        == "Databricks"
    )

    # Test that the callback handlers were called
    assert callback_handler1.num_llm_start_calls == 1
    assert callback_handler2.num_llm_start_calls == 1

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": {"industry": "tech"}}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": ["Databricks"]
    }


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_predict_with_callbacks_supports_chat_response_conversion(fake_chat_model):
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser

    prompt = ChatPromptTemplate.from_template("What's your favorite {industry} company?")
    chain = prompt | fake_chat_model | StrOutputParser()
    # Test the basic functionality of the chain
    assert chain.invoke({"industry": "tech"}) == "Databricks"

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain, "model_path", input_example={"industry": "tech"}
        )

    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    expected_chat_response = {
        "id": None,
        "object": "chat.completion",
        "created": 1677858242,
        "model": None,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Databricks"},
                "finish_reason": None,
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    }
    with mock.patch("time.time", return_value=1677858242):
        assert (
            pyfunc_loaded_model._model_impl._predict_with_callbacks(
                {"industry": "tech"},
                convert_chat_responses=True,
            )
            == expected_chat_response
        )

        assert (
            pyfunc_loaded_model._model_impl._predict_with_callbacks(
                {"industry": "tech"},
                convert_chat_responses=False,
            )
            == "Databricks"
        )


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"),
    reason="feature not existing",
)
def test_save_load_runnable_parallel():
    from langchain.schema.runnable import RunnableParallel

    def fake_llm(prompt: str) -> str:
        return "completion"

    runnable = RunnableParallel({"llm": fake_llm})
    assert runnable.invoke("hello") == {"llm": "completion"}
    assert runnable.batch(["hello", "world"]) == [
        {"llm": "completion"},
        {"llm": "completion"},
    ]
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            runnable, "model_path", input_example=["hello", "world"]
        )
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke("hello") == {"llm": "completion"}
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict("hello") == [{"llm": "completion"}]
    assert pyfunc_loaded_model.predict(["hello", "world"]) == [
        {"llm": "completion"},
        {"llm": "completion"},
    ]

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": ["hello", "world"]}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [{"llm": "completion"}, {"llm": "completion"}]
    }


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"),
    reason="feature not existing",
)
def tests_save_load_complex_runnable_parallel():
    from langchain.schema.runnable import RunnableParallel

    with _mock_request(return_value=_mock_chat_completion_response()):
        chain = create_openai_llmchain()
        runnable = RunnableParallel({"llm": chain})
        expected_result = {"llm": {"product": "MLflow", "text": TEST_CONTENT}}
        assert runnable.invoke({"product": "MLflow"}) == expected_result
        with mlflow.start_run():
            model_info = mlflow.langchain.log_model(
                runnable, "model_path", input_example=[{"product": "MLflow"}]
            )
        loaded_model = mlflow.langchain.load_model(model_info.model_uri)
        assert loaded_model.invoke("MLflow") == expected_result
        pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        assert pyfunc_loaded_model.predict([{"product": "MLflow"}]) == [expected_result]

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": [{"product": "MLflow"}, {"product": "MLflow"}]}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [expected_result, expected_result]
    }


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"),
    reason="feature not existing",
)
def test_save_load_runnable_parallel_and_assign_in_sequence():
    from langchain.schema.runnable import RunnablePassthrough

    def fake_llm(prompt: str) -> str:
        return "completion"

    runnable = {
        "llm1": fake_llm,
        "llm2": fake_llm,
    } | RunnablePassthrough.assign(total_chars=lambda inputs: len(inputs["llm1"] + inputs["llm2"]))
    expected_result = {
        "llm1": "completion",
        "llm2": "completion",
        "total_chars": 20,
    }
    assert runnable.invoke("hello") == expected_result

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            runnable, "model_path", input_example=["hello", "world"]
        )
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke("hello") == expected_result
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict(["hello"]) == [expected_result]

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": ["hello", "world"]}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [expected_result, expected_result]
    }


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"),
    reason="feature not existing",
)
def test_save_load_complex_runnable_assign(fake_chat_model):
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnableParallel
    from langchain.schema.runnable.passthrough import RunnableAssign

    prompt = ChatPromptTemplate.from_template(
        "What is a good name for a company that makes {product}?"
    )
    chain = prompt | fake_chat_model | StrOutputParser()

    def fake_llm(prompt: str) -> str:
        return "completion"

    runnable_assign = RunnableAssign(mapper=RunnableParallel({"product": chain, "test": fake_llm}))
    expected_result = {
        "product": "Databricks",
        "test": "completion",
    }
    input_example = {"product": "MLflow", "test": "test"}
    assert runnable_assign.invoke(input_example) == expected_result

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            runnable_assign, "model_path", input_example=input_example
        )
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke(input_example) == expected_result
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict([input_example]) == [expected_result]

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": input_example}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [expected_result]
    }


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_save_load_runnable_sequence():
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnableSequence

    prompt1 = PromptTemplate.from_template("what is the city {person} is from?")
    llm = OpenAI(temperature=0.9)
    model = prompt1 | llm | StrOutputParser()

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(model, "model_path")

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert type(loaded_model) == RunnableSequence
    assert type(loaded_model.steps[0]) == PromptTemplate
    assert type(loaded_model.steps[1]) == OpenAI
    assert type(loaded_model.steps[2]) == StrOutputParser


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_save_load_long_runnable_sequence(model_path):
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnablePassthrough, RunnableSequence

    prompt1 = PromptTemplate.from_template("what is the city {person} is from?")
    llm = OpenAI(temperature=0.9)
    model = prompt1 | llm | StrOutputParser()
    for _ in range(10):
        model = model | RunnablePassthrough()

    with mlflow.start_run():
        mlflow.langchain.save_model(model, model_path)

    loaded_model = mlflow.langchain.load_model(model_path)
    assert type(loaded_model) == RunnableSequence
    assert type(loaded_model.steps[0]) == PromptTemplate
    assert type(loaded_model.steps[1]) == OpenAI
    assert type(loaded_model.steps[2]) == StrOutputParser
    for i in range(3, 13):
        assert type(loaded_model.steps[i]) == RunnablePassthrough


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"),
    reason="feature not existing",
)
def test_save_load_complex_runnable_sequence():
    from langchain.schema.runnable import RunnablePassthrough

    with _mock_request(return_value=_mock_chat_completion_response()):
        llm_chain = create_openai_llmchain()
        chain = llm_chain | RunnablePassthrough()
        expected_result = {"product": "MLflow", "text": TEST_CONTENT}
        assert chain.invoke({"product": "MLflow"}) == expected_result

        with mlflow.start_run():
            model_info = mlflow.langchain.log_model(
                chain, "model_path", input_example=[{"product": "MLflow"}]
            )

        loaded_model = mlflow.langchain.load_model(model_info.model_uri)
        result = loaded_model.invoke({"product": "MLflow"})
        assert result == expected_result
        pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        assert pyfunc_loaded_model.predict([{"product": "MLflow"}]) == [expected_result]

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": [{"product": "MLflow"}]}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [expected_result]
    }


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_save_load_simple_chat_model(spark, fake_chat_model):
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser

    prompt = ChatPromptTemplate.from_template(
        "What is a good name for a company that makes {product}?"
    )
    chain = prompt | fake_chat_model | StrOutputParser()
    assert chain.invoke({"product": "MLflow"}) == "Databricks"
    # signature is required for spark_udf
    signature = infer_signature({"product": "MLflow"}, "Databricks")
    assert signature == ModelSignature(
        Schema([ColSpec("string", "product")]), Schema([ColSpec("string")])
    )
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(chain, "model_path", signature=signature)
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke({"product": "MLflow"}) == "Databricks"
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict([{"product": "MLflow"}]) == ["Databricks"]

    udf = mlflow.pyfunc.spark_udf(spark, model_info.model_uri, result_type="string")
    df = spark.createDataFrame([("MLflow",), ("Spark",)], ["product"])
    df = df.withColumn("answer", udf("product"))
    pdf = df.toPandas()
    assert pdf["answer"].tolist() == ["Databricks", "Databricks"]

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": {"product": "MLflow"}}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    # Because of the schema enforcement converts input to pandas dataframe
    # the prediction result is wrapped in a list in api_request_parallel_processor
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": ["Databricks"]
    }


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_save_load_rag(tmp_path, spark, fake_chat_model):
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnablePassthrough

    # Create the vector db, persist the db to a local fs folder
    loader = TextLoader("tests/langchain/state_of_the_union.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = DeterministicDummyEmbeddings(size=5)
    db = FAISS.from_documents(docs, embeddings)
    persist_dir = str(tmp_path / "faiss_index")
    db.save_local(persist_dir)
    retriever = db.as_retriever()

    def load_retriever(persist_directory):
        embeddings = FakeEmbeddings(size=5)
        vectorstore = FAISS.load_local(
            persist_directory,
            embeddings,
            **VECTORSTORE_KWARGS,
        )
        return vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        "Answer the following question based on the context: {context}\nQuestion: {question}"
    )
    retrieval_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | fake_chat_model
        | StrOutputParser()
    )
    question = "What is a good name for a company that makes MLflow?"
    answer = "Databricks"
    assert retrieval_chain.invoke(question) == answer
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            retrieval_chain,
            "model_path",
            loader_fn=load_retriever,
            persist_dir=persist_dir,
            input_example=question,
        )

    # Remove the persist_dir
    shutil.rmtree(persist_dir)

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke(question) == answer
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict(question) == [answer]

    udf = mlflow.pyfunc.spark_udf(spark, model_info.model_uri, result_type="string")
    df = spark.createDataFrame([(question,), (question,)], ["question"])
    df = df.withColumn("answer", udf("question"))
    pdf = df.toPandas()
    assert pdf["answer"].tolist() == [answer, answer]

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": question}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [answer]
    }


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_runnable_branch_save_load():
    from langchain.schema.runnable import RunnableBranch

    branch = RunnableBranch(
        (lambda x: isinstance(x, str), lambda x: x.upper()),
        (lambda x: isinstance(x, int), lambda x: x + 1),
        (lambda x: isinstance(x, float), lambda x: x * 2),
        lambda x: "goodbye",
    )

    assert branch.invoke("hello") == "HELLO"
    assert branch.invoke({}) == "goodbye"

    with mlflow.start_run():
        # We only support single input format for now, so we should
        # not save signature for runnable branch which accepts multiple
        # input types
        model_info = mlflow.langchain.log_model(branch, "model_path")

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke("hello") == "HELLO"
    assert loaded_model.invoke({}) == "goodbye"
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict("hello") == "HELLO"
    assert pyfunc_loaded_model.predict({}) == "goodbye"

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": "hello"}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": "HELLO"
    }


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_complex_runnable_branch_save_load(fake_chat_model, fake_classifier_chat_model):
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnableBranch, RunnableLambda

    prompt = ChatPromptTemplate.from_template("{question_is_relevant}\n{query}")
    # Need to add prompt here as the chat model doesn't accept dict input
    answer_model = prompt | fake_chat_model

    decline_to_answer = RunnableLambda(
        lambda x: "I cannot answer questions that are not about MLflow."
    )
    something_went_wrong = RunnableLambda(lambda x: "Something went wrong.")

    is_question_about_mlflow_prompt = ChatPromptTemplate.from_template(
        "You are classifying documents to know if this question "
        "is related with MLflow. Only answer with yes or no. The question is: {query}"
    )

    branch_node = RunnableBranch(
        (lambda x: x["question_is_relevant"].lower() == "yes", answer_model),
        (lambda x: x["question_is_relevant"].lower() == "no", decline_to_answer),
        something_went_wrong,
    )

    chain = (
        {
            "question_is_relevant": is_question_about_mlflow_prompt
            | fake_classifier_chat_model
            | StrOutputParser(),
            "query": itemgetter("query"),
        }
        | branch_node
        | StrOutputParser()
    )

    assert chain.invoke({"query": "Who owns MLflow?"}) == "Databricks"
    assert (
        chain.invoke({"query": "Do you like cat?"})
        == "I cannot answer questions that are not about MLflow."
    )
    assert chain.invoke({"query": "Are you happy today?"}) == "Something went wrong."

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain, "model_path", input_example={"query": "Who owns MLflow?"}
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke({"query": "Who owns MLflow?"}) == "Databricks"
    assert (
        loaded_model.invoke({"query": "Do you like cat?"})
        == "I cannot answer questions that are not about MLflow."
    )
    assert loaded_model.invoke({"query": "Are you happy today?"}) == "Something went wrong."
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict({"query": "Who owns MLflow?"}) == ["Databricks"]
    assert pyfunc_loaded_model.predict({"query": "Do you like cat?"}) == [
        "I cannot answer questions that are not about MLflow."
    ]
    assert pyfunc_loaded_model.predict({"query": "Are you happy today?"}) == [
        "Something went wrong."
    ]

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": {"query": "Who owns MLflow?"}}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": ["Databricks"]
    }


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_chat_with_history(spark, fake_chat_model):
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnableLambda

    prompt_with_history_str = """
    Here is a history between you and a human: {chat_history}

    Now, please answer this question: {question}
    """

    prompt_with_history = PromptTemplate(
        input_variables=["chat_history", "question"], template=prompt_with_history_str
    )

    def extract_question(input):
        return input[-1]["content"]

    def extract_history(input):
        return input[:-1]

    chain_with_history = (
        {
            "question": itemgetter("messages") | RunnableLambda(extract_question),
            "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
        }
        | prompt_with_history
        | fake_chat_model
        | StrOutputParser()
    )

    input_example = {"messages": [{"role": "user", "content": "Who owns MLflow?"}]}
    assert chain_with_history.invoke(input_example) == "Databricks"

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain_with_history, "model_path", input_example=input_example
        )
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke(input_example) == "Databricks"
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    input_schema = pyfunc_loaded_model.metadata.get_input_schema()
    assert input_schema == Schema(
        [
            ColSpec(
                Array(
                    Object(
                        [
                            Property("role", DataType.string),
                            Property("content", DataType.string),
                        ]
                    )
                ),
                "messages",
            )
        ]
    )
    assert pyfunc_loaded_model.predict(input_example) == ["Databricks"]

    udf = mlflow.pyfunc.spark_udf(spark, model_info.model_uri, result_type="string")
    df = spark.createDataFrame([(input_example["messages"],)], ["messages"])
    df = df.withColumn("answer", udf("messages"))
    pdf = df.toPandas()
    assert pdf["answer"].tolist() == ["Databricks"]

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": input_example}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": ["Databricks"]
    }


def _extract_endpoint_name_from_lc_model(lc_model, dependency_dict: DefaultDict[str, List[Any]]):
    if type(lc_model).__name__ == type(get_fake_chat_model()).__name__:
        dependency_dict["fake_chat_model_endpoint_name"].append(lc_model.endpoint_name)


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
@mock.patch(
    "mlflow.langchain.databricks_dependencies._extract_dependency_dict_from_lc_model",
    _extract_endpoint_name_from_lc_model,
)
def test_databricks_dependency_extraction_from_lcel_chain():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt_1 = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
    prompt_2 = ChatPromptTemplate.from_template(
        "compare which joke is better {joke1} or {joke2}. Output the better joke."
    )
    model_1 = get_fake_chat_model(endpoint_name="fake-endpoint-1")
    model_2 = get_fake_chat_model(endpoint_name="fake-endpoint-2")
    model_3 = get_fake_chat_model(endpoint_name="fake-endpoint-3")
    output_parser = StrOutputParser()

    chain = prompt_1 | {"joke1": model_1, "joke2": model_2} | prompt_2 | model_3 | output_parser

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(chain, "basic_chain")

    langchain_flavor = model_info.flavors["langchain"]
    assert langchain_flavor["databricks_dependency"] == {
        "fake_chat_model_endpoint_name": [
            "fake-endpoint-1",
            "fake-endpoint-2",
            "fake-endpoint-3",
        ]
    }


def _extract_databricks_dependencies_from_retriever(
    retriever, dependency_dict: DefaultDict[str, List[Any]]
):
    import langchain_community

    vectorstore = getattr(retriever, "vectorstore", None)
    if vectorstore and (embeddings := getattr(vectorstore, "embeddings", None)):
        if isinstance(vectorstore, langchain_community.vectorstores.faiss.FAISS):
            dependency_dict["fake_index"].append("faiss-index")

        if isinstance(embeddings, FakeEmbeddings):
            dependency_dict["fake_embeddings_size"].append(embeddings.size)


def _extract_databricks_dependencies_from_llm(llm, dependency_dict: DefaultDict[str, List[Any]]):
    if isinstance(llm, FakeLLM):
        dependency_dict["fake_llm_endpoint_name"].append(llm.endpoint_name)


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
@mock.patch(
    "mlflow.langchain.databricks_dependencies._extract_databricks_dependencies_from_llm",
    _extract_databricks_dependencies_from_llm,
)
@mock.patch(
    "mlflow.langchain.databricks_dependencies._extract_databricks_dependencies_from_retriever",
    _extract_databricks_dependencies_from_retriever,
)
def test_databricks_dependency_extraction_from_retrieval_qa_chain(tmp_path):
    # Create the vector db, persist the db to a local fs folder
    loader = TextLoader("tests/langchain/state_of_the_union.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = FakeEmbeddings(size=5)
    db = FAISS.from_documents(docs, embeddings)
    persist_dir = str(tmp_path / "faiss_index")
    db.save_local(persist_dir)

    # Create the RetrievalQA chain
    retrievalQA = RetrievalQA.from_llm(llm=FakeLLM(), retriever=db.as_retriever())

    # Log the RetrievalQA chain
    def load_retriever(persist_directory):
        embeddings = FakeEmbeddings(size=5)
        vectorstore = FAISS.load_local(persist_directory, embeddings)
        return vectorstore.as_retriever()

    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            retrievalQA,
            "retrieval_qa_chain",
            loader_fn=load_retriever,
            persist_dir=persist_dir,
        )
    langchain_flavor = logged_model.flavors["langchain"]
    assert langchain_flavor["databricks_dependency"] == {
        "fake_llm_endpoint_name": ["fake-llm-endpoint"],
        "fake_index": ["faiss-index"],
        "fake_embeddings_size": [5],
    }


def _error_func(*args, **kwargs):
    raise ValueError("error")


@mock.patch(
    "mlflow.langchain.databricks_dependencies._traverse_runnable",
    _error_func,
)
@mock.patch("mlflow.langchain.databricks_dependencies._logger.warning")
def test_databricks_dependency_extraction_log_errors_as_warnings(mock_warning):
    from mlflow.langchain.databricks_dependencies import (
        _DATABRICKS_DEPENDENCY_KEY,
        _detect_databricks_dependencies,
    )

    model = create_openai_llmchain()

    _detect_databricks_dependencies(model, log_errors_as_warnings=True)
    mock_warning.assert_called_once_with(
        "Unable to detect Databricks dependencies. "
        "Set logging level to DEBUG to see the full traceback."
    )

    with pytest.raises(ValueError, match="error"):
        _detect_databricks_dependencies(model, log_errors_as_warnings=False)

    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, "langchain_model")
    assert logged_model.flavors["langchain"].get(_DATABRICKS_DEPENDENCY_KEY) is None


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_predict_with_builtin_pyfunc_chat_conversion(spark):
    from langchain.chat_models.base import SimpleChatModel
    from langchain.schema.output_parser import StrOutputParser

    class ChatModel(SimpleChatModel):
        def _call(self, messages, stop, run_manager, **kwargs):
            return "\n".join([f"{message.type}: {message.content}" for message in messages])

        @property
        def _llm_type(self) -> str:
            return "chat model"

    input_example = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "What would you like to ask?"},
            {"role": "user", "content": "Who owns MLflow?"},
        ]
    }
    content = (
        "system: You are a helpful assistant.\n"
        "ai: What would you like to ask?\n"
        "human: Who owns MLflow?"
    )
    example_output = {
        "id": "some_id",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "some_model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20,
        },
    }
    signature = infer_signature(model_input=input_example, model_output=example_output)

    chain = ChatModel() | StrOutputParser()
    assert chain.invoke([HumanMessage(content="Who owns MLflow?")]) == "human: Who owns MLflow?"
    with pytest.raises(ValueError, match="Invalid input type"):
        chain.invoke(input_example)

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain, "model_path", signature=signature, input_example=input_example
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert (
        loaded_model.invoke([HumanMessage(content="Who owns MLflow?")]) == "human: Who owns MLflow?"
    )

    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    expected_chat_response = {
        "id": None,
        "object": "chat.completion",
        "created": 1677858242,
        "model": None,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": None,
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    }

    with mock.patch("time.time", return_value=1677858242):
        assert pyfunc_loaded_model.predict(input_example) == [expected_chat_response]
        assert pyfunc_loaded_model.predict([input_example, input_example]) == [
            expected_chat_response,
            expected_chat_response,
        ]

    with pytest.raises(MlflowException, match="Unrecognized chat message role"):
        pyfunc_loaded_model.predict({"messages": [{"role": "foobar", "content": "test content"}]})

    udf = mlflow.pyfunc.spark_udf(spark, model_info.model_uri)
    df = spark.createDataFrame([(input_example["messages"],)], ["messages"])
    with mock.patch("time.time", return_value=1677858242):
        df = df.withColumn("answer", udf("messages"))
        assert (
            df.collect()[0]["answer"].asDict(recursive=True)["choices"][0]["message"]["content"]
            == content
        )

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps(input_example),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert (
        json.loads(response.content.decode("utf-8"))[0]["choices"][0]["message"]["content"]
        == content
    )


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_predict_with_builtin_pyfunc_chat_conversion_for_aimessage_response():
    from langchain.chat_models.base import SimpleChatModel

    class ChatModel(SimpleChatModel):
        def _call(self, messages, stop, run_manager, **kwargs):
            return "You own MLflow"

        @property
        def _llm_type(self) -> str:
            return "chat model"

    input_example = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "What would you like to ask?"},
            {"role": "user", "content": "Who owns MLflow?"},
        ]
    }
    signature = infer_signature(model_input=input_example)

    chain = ChatModel()
    assert chain.invoke([HumanMessage(content="Who owns MLflow?")]) == AIMessage(
        content="You own MLflow"
    )

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain, "model_path", signature=signature, input_example=input_example
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke([HumanMessage(content="Who owns MLflow?")]) == AIMessage(
        content="You own MLflow"
    )

    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    with mock.patch("time.time", return_value=1677858242):
        assert pyfunc_loaded_model.predict(input_example) == [
            {
                "id": None,
                "object": "chat.completion",
                "created": 1677858242,
                "model": None,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "You own MLflow",
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
            }
        ]


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_pyfunc_builtin_chat_request_conversion_fails_gracefully():
    from langchain.schema.runnable import RunnablePassthrough

    chain = RunnablePassthrough() | itemgetter("messages")
    # Ensure we're going to test that "messages" remains intact & unchanged even if it
    # doesn't appear explicitly in the chain's input schema
    assert "messages" not in chain.input_schema().__fields__

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(chain, "model_path")
        pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    assert pyfunc_loaded_model.predict({"messages": "not an array"}) == "not an array"

    # Verify that messages aren't converted to LangChain format if extra keys are present,
    # under the assumption that additional keys can't be specified when calling LangChain invoke()
    # / batch() with chat messages
    assert pyfunc_loaded_model.predict(
        {
            "messages": [{"role": "user", "content": "blah"}],
            "extrakey": "extra",
        }
    ) == [
        {"role": "user", "content": "blah"},
    ]

    # Verify that messages aren't converted to LangChain format if role / content are missing
    # or extra keys are present in the message
    assert pyfunc_loaded_model.predict(
        {
            "messages": [{"content": "blah"}],
        }
    ) == [
        {"content": "blah"},
    ]
    assert pyfunc_loaded_model.predict(
        {
            "messages": [{"role": "user", "content": "blah"}, {"role": "blah"}],
        }
    ) == [
        {"role": "user", "content": "blah"},
        {"role": "blah"},
    ]
    assert pyfunc_loaded_model.predict(
        {
            "messages": [{"role": "role", "content": "content", "extra": "extra"}],
        }
    ) == [
        {"role": "role", "content": "content", "extra": "extra"},
    ]

    # Verify behavior for batches of message histories
    assert pyfunc_loaded_model.predict(
        [
            {
                "messages": "not an array",
            },
            {
                "messages": [{"role": "user", "content": "content"}],
            },
        ]
    ) == [
        "not an array",
        [{"role": "user", "content": "content"}],
    ]
    assert pyfunc_loaded_model.predict(
        [
            {
                "messages": [{"role": "user", "content": "content"}],
            },
            {"messages": [{"role": "user", "content": "content"}], "extrakey": "extra"},
        ]
    ) == [
        [{"role": "user", "content": "content"}],
        [{"role": "user", "content": "content"}],
    ]
    assert pyfunc_loaded_model.predict(
        [
            {
                "messages": [{"role": "user", "content": "content"}],
            },
            {
                "messages": [{"role": "user", "content": "content"}, {"role": "user"}],
            },
        ]
    ) == [
        [{"role": "user", "content": "content"}],
        [{"role": "user", "content": "content"}, {"role": "user"}],
    ]


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_pyfunc_builtin_chat_response_conversion_fails_gracefully():
    from langchain.schema.runnable import RunnablePassthrough

    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["messages"],
        template="What is a good name for a company that makes {messages}?",
    )
    chain = RunnablePassthrough() | LLMChain(llm=llm, prompt=prompt) | RunnablePassthrough()

    input_example = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "What would you like to ask?"},
            {"role": "user", "content": "Who owns MLflow?"},
        ]
    }
    signature = infer_signature(model_input=input_example)

    with _mock_request(return_value=_mock_chat_completion_response()):
        with mlflow.start_run():
            logged_model = mlflow.langchain.log_model(
                chain,
                "langchain_model",
                signature=signature,
                input_example=input_example,
            )
        loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
        result = loaded_model.predict(input_example)
        # Verify that the chat request format was converted into LangChain messages correctly, but
        # the response was not converted to the chat response format because it does not have the
        # expected structure (a nonstandard dict with 'messages' and 'text' fields is returned)
        assert result == [
            {
                "messages": [
                    SystemMessage(content="You are a helpful assistant."),
                    AIMessage(content="What would you like to ask?"),
                    HumanMessage(content="Who owns MLflow?"),
                ],
                "text": TEST_CONTENT,
            }
        ]


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_save_load_chain_as_code():
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is a good name for a company that makes MLflow?",
            }
        ]
    }
    with mlflow.start_run():
        signature = ModelSignature(
            inputs=Schema(
                [
                    ColSpec(
                        type=Array(
                            Object(
                                [
                                    Property("role", DataType.string),
                                    Property("content", DataType.string),
                                ]
                            ),
                        ),
                        name="messages",
                    ),
                    ColSpec(
                        type=Object(
                            [
                                Property("return_trace", DataType.string, required=False),
                            ]
                        ),
                        name="databricks_options",
                        required=False,
                    ),
                ]
            ),
            outputs=Schema(
                [
                    ColSpec(name="id", type=DataType.string),
                    ColSpec(name="object", type=DataType.string),
                    ColSpec(name="created", type=DataType.long),
                    ColSpec(name="model", type=DataType.string),
                    ColSpec(name="choices", type=DataType.string),
                    ColSpec(name="usage", type=DataType.string),
                ]
            ),
        )

        model_info = mlflow.langchain.log_model(
            lc_model="tests/langchain/chain.py",
            artifact_path="model_path",
            signature=signature,
            input_example=input_example,
            code_paths=["tests/langchain/state_of_the_union.txt"],
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    answer = "Databricks"
    assert loaded_model.invoke(input_example) == answer
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert (
        pyfunc_loaded_model.predict(input_example)[0]
        .get("choices")[0]
        .get("message")
        .get("content")
        == answer
    )

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": input_example}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [APIRequest._try_transform_response_to_chat_format(answer)]
    }

    langchain_flavor = model_info.flavors["langchain"]
    assert langchain_flavor["databricks_dependency"] == {
        "databricks_chat_endpoint_name": ["fake-endpoint"]
    }


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_save_load_chain_errors():
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is a good name for a company that makes MLflow?",
            }
        ]
    }
    with mlflow.start_run():
        signature = ModelSignature(
            inputs=Schema(
                [
                    ColSpec(
                        type=Array(
                            Object(
                                [
                                    Property("role", DataType.string),
                                    Property("content", DataType.string),
                                ]
                            ),
                        ),
                        name="messages",
                    ),
                    ColSpec(
                        type=Object(
                            [
                                Property("return_trace", DataType.string, required=False),
                            ]
                        ),
                        name="databricks_options",
                        required=False,
                    ),
                ]
            ),
            outputs=Schema(
                [
                    ColSpec(name="id", type=DataType.string),
                    ColSpec(name="object", type=DataType.string),
                    ColSpec(name="created", type=DataType.long),
                    ColSpec(name="model", type=DataType.string),
                    ColSpec(name="choices", type=DataType.string),
                    ColSpec(name="usage", type=DataType.string),
                ]
            ),
        )

        incorrect_path = "tests/langchain/chain1.py"
        with pytest.raises(
            MlflowException,
            match=f"If {incorrect_path} is a string, it must be the path to a "
            "file named `chain.py` on the local filesystem.",
        ):
            mlflow.langchain.log_model(
                lc_model=incorrect_path,
                artifact_path="model_path",
                signature=signature,
                input_example=input_example,
                code_paths=["tests/langchain/state_of_the_union.txt"],
            )

        incorrect_path = "tests/langchain1/chain.py"
        with pytest.raises(
            MlflowException,
            match=f"If the {incorrect_path} is a string, it must be a valid "
            "python file path containing the code for defining the chain instance.",
        ):
            mlflow.langchain.log_model(
                lc_model=incorrect_path,
                artifact_path="model_path",
                signature=signature,
                input_example=input_example,
                code_paths=["tests/langchain/state_of_the_union.txt"],
            )

        incorrect_path = "tests/langchain/chain.py"
        code_paths = [
            "tests/langchain/state_of_the_union.txt",
            "tests/langchain/chain.py",
        ]

        with pytest.raises(
            MlflowException,
            match=re.escape(
                "When the model is a string, and if the code_paths are specified, "
                "it should contain only one path."
                "This config path is used to set config.yml file path "
                "for the model. This path should be passed in via the code_paths. "
                f"Current code paths: {code_paths}"
            ),
        ):
            mlflow.langchain.log_model(
                lc_model=incorrect_path,
                artifact_path="model_path",
                signature=signature,
                input_example=input_example,
                code_paths=code_paths,
            )


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_save_load_chain_as_code_optional_code_path():
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is a good name for a company that makes MLflow?",
            }
        ]
    }
    with mlflow.start_run():
        signature = ModelSignature(
            inputs=Schema(
                [
                    ColSpec(
                        type=Array(
                            Object(
                                [
                                    Property("role", DataType.string),
                                    Property("content", DataType.string),
                                ]
                            ),
                        ),
                        name="messages",
                    ),
                    ColSpec(
                        type=Object(
                            [
                                Property("return_trace", DataType.string, required=False),
                            ]
                        ),
                        name="databricks_options",
                        required=False,
                    ),
                ]
            ),
            outputs=Schema(
                [
                    ColSpec(name="id", type=DataType.string),
                    ColSpec(name="object", type=DataType.string),
                    ColSpec(name="created", type=DataType.long),
                    ColSpec(name="model", type=DataType.string),
                    ColSpec(name="choices", type=DataType.string),
                    ColSpec(name="usage", type=DataType.string),
                ]
            ),
        )

        model_info = mlflow.langchain.log_model(
            lc_model="tests/langchain/no_config/chain.py",
            artifact_path="model_path",
            signature=signature,
            input_example=input_example,
            code_paths=[],
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    answer = "Databricks"
    assert loaded_model.invoke(input_example) == answer
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert (
        pyfunc_loaded_model.predict(input_example)[0]
        .get("choices")[0]
        .get("message")
        .get("content")
        == answer
    )

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": input_example}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [APIRequest._try_transform_response_to_chat_format(answer)]
    }

    langchain_flavor = model_info.flavors["langchain"]
    assert langchain_flavor["databricks_dependency"] == {
        "databricks_chat_endpoint_name": ["fake-endpoint"]
    }
