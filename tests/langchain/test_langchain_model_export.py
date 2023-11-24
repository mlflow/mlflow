import importlib
import json
import os
import shutil
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Mapping, Optional

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
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import Tool
from langchain.vectorstores import FAISS
from langchain_experimental.sql import SQLDatabaseChain
from packaging import version
from pydantic import BaseModel
from pyspark.sql import SparkSession

import mlflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow.deployments import PredictionsResponse
from mlflow.exceptions import MlflowException
from mlflow.openai.utils import (
    TEST_CONTENT,
    TEST_INTERMEDIATE_STEPS,
    TEST_SOURCE_DOCUMENTS,
    _mock_chat_completion_response,
    _mock_request,
    _MockResponse,
)

from tests.helper_functions import pyfunc_serve_and_score_model


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

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"

    # pylint: disable=arguments-differ
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

    # pylint: disable=arguments-differ
    def _call(self, inputs: Dict[str, str], run_manager=None) -> Dict[str, str]:
        if self.be_correct:
            return {"bar": "baz"}
        else:
            return {"baz": "bar"}


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
    assert str(logged_model.signature.inputs) == "['product': string]"
    assert str(logged_model.signature.outputs) == "['text': string]"

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
    assert str(logged_model.signature.inputs) == "['product': string]"
    assert str(logged_model.signature.outputs) == "['text': string]"

    assert type(loaded_model) == langchain.chains.llm.LLMChain
    assert type(loaded_model.llm) == langchain.llms.huggingface_pipeline.HuggingFacePipeline
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
        vectorstore = FAISS.load_local(persist_directory, embeddings)
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
        vectorstore = FAISS.load_local(persist_directory, embeddings)
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
        {loaded_model.output_key: TEST_CONTENT, "source_documents": TEST_SOURCE_DOCUMENTS}
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
        from typing import List  # pylint: disable=lazy-builtin-import

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
        vectorstore = FAISS.load_local(persist_directory, embeddings)
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
        llm, open_meteo_docs.OPEN_METEO_DOCS, verbose=True, limit_to_domains=["test.com"]
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
        llm, open_meteo_docs.OPEN_METEO_DOCS, verbose=True, limit_to_domains=["test.com"]
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
            llm=OpenAI(temperature=0), tools=tools, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
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
