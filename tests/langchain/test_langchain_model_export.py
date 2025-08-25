import inspect
import json
import os
import shutil
import sqlite3
import sys
import warnings
from importlib.metadata import version
from operator import itemgetter
from typing import Any, Iterator, Mapping
from unittest import mock

import langchain
import pytest
import yaml
from langchain.agents import AgentType, initialize_agent
from langchain.chains import (
    APIChain,
    ConversationChain,
    LLMChain,
    RetrievalQA,
)
from langchain.chains.api import open_meteo_docs
from langchain.chains.base import Chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.evaluation.qa import QAEvalChain

from mlflow.environment_variables import (
    MLFLOW_CONVERT_MESSAGES_DICT_FOR_LANGCHAIN,
)
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.export.inference_table import pop_trace
from mlflow.types.schema import Object, Property

from tests.tracing.helper import get_traces

try:
    from langchain_huggingface import HuggingFacePipeline
except ImportError:
    from langchain_community.llms import HuggingFacePipeline
from unittest.mock import ANY

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models.base import SimpleChatModel
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableBinding,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
)
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import Tool

# TODO: We should use langchain_openai instead of the community models
# once the partner package loading issue is resolved
from langchain_community.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.utilities import SQLDatabase, TextRequestsWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks.base import BaseCallbackHandler
from packaging import version
from packaging.version import Version
from pyspark.sql import SparkSession

import mlflow
import mlflow.models.model
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow.deployments import PredictionsResponse
from mlflow.exceptions import MlflowException
from mlflow.langchain.langchain_tracer import MlflowLangchainTracer
from mlflow.langchain.utils.chat import (
    transform_request_json_for_chat_if_necessary,
    try_transform_response_to_chat_format,
)
from mlflow.langchain.utils.logging import (
    IS_PICKLE_SERIALIZATION_RESTRICTED,
    lc_runnables_types,
)
from mlflow.models import Model
from mlflow.models.dependencies_schemas import DependenciesSchemasType
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksVectorSearchIndex,
)
from mlflow.models.signature import ModelSignature, Schema, infer_signature
from mlflow.models.utils import load_serving_example
from mlflow.pyfunc.context import Context
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.schema import AnyType, Array, ColSpec, DataType, Object, Property

from tests.helper_functions import (
    _compare_logged_code_paths,
    pyfunc_serve_and_score_model,
)
from tests.langchain.conftest import DeterministicDummyEmbeddings

# this kwarg was added in langchain_community 0.0.27, and
# prevents the use of pickled objects if not provided.
VECTORSTORE_KWARGS = (
    {"allow_dangerous_deserialization": True} if IS_PICKLE_SERIALIZATION_RESTRICTED else {}
)

IS_LANGCHAIN_03 = version.parse(langchain.__version__) >= version.parse("0.3.0")

# The mock OAI completion endpoint returns payload as it is
TEST_CONTENT = '[{"role": "user", "content": "What is MLflow?"}]'


@pytest.fixture
def model_path(tmp_path):
    return tmp_path / "model"


@pytest.fixture(scope="module")
def spark():
    with SparkSession.builder.master("local[*]").getOrCreate() as s:
        yield s


def create_openai_llmchain():
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is {product}?",
    )
    return LLMChain(llm=llm, prompt=prompt)


def create_openai_runnable():
    from langchain_core.output_parsers import StrOutputParser

    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is {product}?",
    )
    return prompt | ChatOpenAI(temperature=0.9) | StrOutputParser()


def create_qa_eval_chain():
    llm = OpenAI(temperature=0)
    return QAEvalChain.from_llm(llm)


def create_qa_with_sources_chain():
    # StuffDocumentsChain
    return load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")


def create_uc_tools(
    monkeypatch, warehouse_id, expected_catalog_name, expected_schema_name, functions
):
    try:
        from langchain_community.tools.databricks import UCFunctionToolkit
    except Exception:
        return []

    from databricks.sdk.service.catalog import FunctionInfo

    # Return 2 functions from the function lis
    def mock_function_list(self, catalog_name, schema_name):
        assert catalog_name == expected_catalog_name
        assert schema_name == expected_schema_name
        return [FunctionInfo(full_name=function) for function in functions]

    # For each function ensure that it returns a tool which takes one input
    def mock_function_get(self, function_name):
        components = function_name.split(".")
        param_dict = {
            "parameters": [
                {
                    "name": "param",
                    "parameter_type": "PARAM",
                    "position": 0,
                    "type_json": '{"name":"param","type":"string","nullable":true,"metadata":{}}',
                    "type_name": "STRING",
                    "type_precision": 0,
                    "type_scale": 0,
                    "type_text": "string",
                }
            ]
        }
        return FunctionInfo.from_dict(
            {
                "catalog_name": components[0],
                "schema_name": components[1],
                "name": components[2],
                "input_params": param_dict,
            }
        )

    monkeypatch.setenv("DATABRICKS_HOST", "my-default-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "my-default-token")
    monkeypatch.setattr("databricks.sdk.service.catalog.FunctionsAPI.list", mock_function_list)
    monkeypatch.setattr("databricks.sdk.service.catalog.FunctionsAPI.get", mock_function_get)

    # Create an toolkit with the '*' syntax
    return (
        UCFunctionToolkit(warehouse_id=warehouse_id)
        .include(f"{expected_catalog_name}.{expected_schema_name}.*")
        .get_tools()
    )


def create_retriever_tool(monkeypatch):
    from langchain.tools.retriever import create_retriever_tool
    from langchain_community.vectorstores import DatabricksVectorSearch

    vsc = MockVectorSearchClient()
    vs_index = vsc.get_index(
        endpoint_name="dbdemos_vs_endpoint",
        index_name="mlflow.rag.vs_index",
        has_embedding_endpoint=True,
    )

    mock_module = mock.MagicMock()
    mock_module.VectorSearchIndex = MockVectorSearchIndex
    monkeypatch.setitem(sys.modules, "databricks.vector_search.client", mock_module)

    vectorstore = DatabricksVectorSearch(vs_index, text_column="content")
    retriever = vectorstore.as_retriever()
    return create_retriever_tool(retriever, "vs_index_name", "vs_index_desc")


class FakeLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    queries: Mapping | None = None
    endpoint_name: str = "fake-llm-endpoint"

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"

    def _call(self, prompt: str, stop: list[str] | None = None, run_manager=None) -> str:
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
    the_input_keys: list[str] = ["foo"]
    the_output_keys: list[str] = ["bar"]

    @property
    def input_keys(self) -> list[str]:
        """Input keys."""
        return self.the_input_keys

    @property
    def output_keys(self) -> list[str]:
        """Output key of bar."""
        return self.the_output_keys

    def _call(self, inputs: dict[str, str], run_manager=None) -> dict[str, str]:
        if self.be_correct:
            return {"bar": "baz"}
        else:
            return {"baz": "bar"}


class MockVectorSearchIndex:
    def __init__(self, endpoint_name, index_name, has_embedding_endpoint=False) -> None:
        self.endpoint_name = endpoint_name
        self.name = index_name
        self.has_embedding_endpoint = has_embedding_endpoint

    def describe(self):
        if self.has_embedding_endpoint:
            return {
                "name": self.name,
                "endpoint_name": self.endpoint_name,
                "primary_key": "id",
                "index_type": "DELTA_SYNC",
                "delta_sync_index_spec": {
                    "source_table": "ml.schema.databricks_documentation",
                    "embedding_source_columns": [
                        {"name": "content", "embedding_model_endpoint_name": "embedding-model"}
                    ],
                    "pipeline_type": "TRIGGERED",
                    "pipeline_id": "79a76fcc-67ad-4ac6-8d8e-20f7d485ffa6",
                },
                "status": {
                    "detailed_state": "OFFLINE_FAILED",
                    "message": "Index creation failed.",
                    "indexed_row_count": 0,
                    "failed_status": {"error_message": ""},
                    "ready": False,
                    "index_url": "e2-dogfood.staging.cloud.databricks.com/rest_of_url",
                },
                "creator": "first.last@databricks.com",
            }
        else:
            return {
                "name": self.name,
                "endpoint_name": self.endpoint_name,
                "primary_key": "id",
                "index_type": "DELTA_SYNC",
                "delta_sync_index_spec": {
                    "source_table": "ml.schema.databricks_documentation",
                    "embedding_vector_columns": [],
                    "pipeline_type": "TRIGGERED",
                    "pipeline_id": "fbbd5bf1-2b9b-4a7e-8c8d-c0f6cc1030de",
                },
                "status": {
                    "detailed_state": "ONLINE",
                    "message": "Index is currently online",
                    "indexed_row_count": 17183,
                    "ready": True,
                    "index_url": "e2-dogfood.staging.cloud.databricks.com/rest_of_url",
                },
                "creator": "first.last@databricks.com",
            }


class MockVectorSearchClient:
    def get_index(self, endpoint_name, index_name, has_embedding_endpoint=False):
        return MockVectorSearchIndex(endpoint_name, index_name, has_embedding_endpoint)


def get_fake_chat_model(endpoint_name="fake-endpoint"):
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.chat_models.base import SimpleChatModel
    from langchain.schema.messages import BaseMessage

    class FakeChatModel(SimpleChatModel):
        """Fake Chat Model wrapper for testing purposes."""

        endpoint_name: str = "fake-endpoint"

        def _call(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
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

    class FakeMlflowClassifier(SimpleChatModel):
        """Fake Chat Model wrapper for testing purposes."""

        def _call(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
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

    return FakeMlflowClassifier()


@pytest.mark.skipif(IS_LANGCHAIN_03, reason="LLMChain is deprecated")
def test_langchain_llm_chain():
    model = create_openai_llmchain()
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, name="langchain_model")

    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)

    assert "langchain" in logged_model.flavors
    assert str(logged_model.signature.inputs) == "['product': string (required)]"
    assert str(logged_model.signature.outputs) == "['text': string (required)]"

    assert type(loaded_model) == LLMChain
    assert type(loaded_model.llm) == OpenAI
    assert type(loaded_model.prompt) == PromptTemplate
    assert loaded_model.prompt.template == "What is {product}?"


def test_langchain_native_log_and_load_model():
    model = create_openai_runnable()

    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            model, name="langchain_model", input_example={"product": "MLflow"}
        )

    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)

    assert "langchain" in logged_model.flavors
    assert str(logged_model.signature.inputs) == "['product': string (required)]"
    assert str(logged_model.signature.outputs) == "[string (required)]"

    assert type(loaded_model) == RunnableSequence
    assert loaded_model.steps[0].template == "What is {product}?"
    assert type(loaded_model.steps[1]) == ChatOpenAI

    # Predict
    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    result = loaded_model.predict([{"product": "MLflow"}])
    assert result == [TEST_CONTENT]

    # Predict stream
    result = loaded_model.predict_stream([{"product": "MLflow"}])
    assert inspect.isgenerator(result)
    assert list(result) == ["Hello", " world"]


def test_pyfunc_spark_udf_with_langchain_model(spark):
    model = create_openai_runnable()
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            model, name="langchain_model", input_example={"product": "MLflow"}
        )
    loaded_model = mlflow.pyfunc.spark_udf(spark, logged_model.model_uri, result_type="string")
    df = spark.createDataFrame([("MLflow",), ("Spark",)], ["product"])
    df = df.withColumn("answer", loaded_model())
    pdf = df.toPandas()
    assert pdf["answer"].tolist() == [
        '[{"role": "user", "content": "What is MLflow?"}]',
        '[{"role": "user", "content": "What is Spark?"}]',
    ]


def test_save_and_load_azure_chat_openai(model_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_VERSION", "2023-05-15")
    monkeypatch.setenv("OPENAI_API_BASE", "https://mlflowtest.foo.bar/")

    llm = AzureChatOpenAI(temperature=0.9)
    prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
    parser = StrOutputParser()
    chain = prompt | llm | parser
    mlflow.langchain.save_model(chain, model_path)

    loaded_model = mlflow.langchain.load_model(model_path)
    assert isinstance(loaded_model, RunnableSequence)
    assert loaded_model.steps[0] == prompt
    assert loaded_model.steps[1]._identifying_params == llm._identifying_params
    assert loaded_model.steps[2] == parser


def test_save_model_with_partner_package(tmp_path):
    from langchain_community.chat_models import ChatOpenAI as ChatOpenAICommunity
    from langchain_openai import ChatOpenAI as ChatOpenAIPartner

    # 1. Saving a model with LLM from a community package
    #    -> no warning should be raised
    chain = ChatOpenAICommunity() | StrOutputParser()

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*LangChain partner package.*")
        mlflow.langchain.save_model(chain, tmp_path / "community-model")

    # 2. Saving a model with LLM from a partner package
    #    -> a warning should be raised and incorrect class is loaded
    chain = ChatOpenAIPartner() | StrOutputParser()

    with pytest.warns(match=r".*LangChain partner package.*"):
        mlflow.langchain.save_model(chain, tmp_path / "partner-model")

    loaded_model = mlflow.langchain.load_model(tmp_path / "partner-model")
    loaded_llm = loaded_model.steps[0]
    assert type(loaded_llm) == ChatOpenAICommunity

    # 3. Saving a model using model-from-code
    #    -> no warning should be raised and the correct class is loaded
    with open(tmp_path / "model.py", "w") as f:
        f.write(
            """
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
import mlflow

chain = ChatOpenAI() | StrOutputParser()
mlflow.models.set_model(chain)
"""
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*LangChain partner package.*")
        mlflow.langchain.save_model(
            lc_model=str(tmp_path / "model.py"),
            path=tmp_path / "model-from-code",
        )

    loaded_model = mlflow.langchain.load_model(tmp_path / "model-from-code")
    loaded_llm = loaded_model.steps[0]
    assert type(loaded_llm) == ChatOpenAIPartner


def test_langchain_log_huggingface_hub_model_metadata(model_path):
    import transformers

    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    pipeline = transformers.pipeline("text-generation", model="distilgpt2")
    hf_pipe = HuggingFacePipeline(pipeline=pipeline)
    model = prompt | hf_pipe | StrOutputParser()

    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            model, name="langchain_model", input_example={"product": "MLflow"}
        )

    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)

    assert "langchain" in logged_model.flavors
    assert str(logged_model.signature.inputs) == "['product': string (required)]"
    assert str(logged_model.signature.outputs) == "[string (required)]"

    assert isinstance(loaded_model, RunnableSequence)
    assert loaded_model.steps[0] == prompt
    # TODO: Check the type once https://github.com/langchain-ai/langchain/issues/22520 is resolved
    assert type(loaded_model.steps[1]).__name__ == "HuggingFacePipeline"


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.2.0"),
    reason="Agent behavior is not stable across minor versions",
)
@pytest.mark.parametrize("return_intermediate_steps", [False, True])
def test_langchain_agent_model_predict(return_intermediate_steps, monkeypatch):
    input_example = {"input": "What is 2 * 3?"}

    # Use env var to control the return_intermediate_steps without modifying the code
    monkeypatch.setenv("RETURN_INTERMEDIATE_STEPS", str(return_intermediate_steps))

    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            # OpenAI Client since 1.0 contains thread lock object that cannot be
            # pickled. Therefore, AgentExecutor cannot be saved with the legacy
            # object-based logging and we need to use Model-from-Code logging.
            "tests/langchain/sample_code/openai_agent.py",
            name="langchain_model",
            input_example=input_example,
        )

    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    response = loaded_model.predict([input_example])

    if return_intermediate_steps:
        expected_output = [
            {
                "output": "The result of 2 * 3 is 6.",
                "intermediate_steps": [
                    # tuple of (action, observation)
                    (
                        {
                            "log": mock.ANY,
                            "message_log": [mock.ANY],
                            "tool": "multiply",
                            "tool_call_id": "123",
                            "tool_input": {"a": 2, "b": 3},
                            "type": "AgentActionMessageLog",
                        },
                        6,
                    )
                ],
            }
        ]
        # hardcoded output key because that is the default for an agent
        # but it is not an attribute of the agent or anything that we log
    else:
        expected_output = ["The result of 2 * 3 is 6."]

    assert response == expected_output

    inference_payload = load_serving_example(logged_model.model_uri)
    response = pyfunc_serve_and_score_model(
        logged_model.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    # TODO: The response is not wrapped by the "predictions" key. This is a bug in
    # output handling. Often the user input contains a key "input" because it is
    # used in popular agent prompts in the hub. However, this confuses the scoring
    # server to treat it as a llm/v1/completion request.
    response = json.loads(response.content.decode("utf-8"))
    if return_intermediate_steps:
        # Tuples are converted to lists during JSON serialization
        response[0]["intermediate_steps"] = [tuple(r) for r in response[0]["intermediate_steps"]]
    assert response == expected_output


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.2.0"),
    reason="Agent behavior is not stable across minor versions",
)
def test_langchain_agent_model_predict_stream():
    input_example = {"input": "What is 2 * 3?"}
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            # OpenAI Client since 1.0 contains thread lock object that cannot be
            # pickled. Therefore, AgentExecutor cannot be saved with the legacy
            # object-based logging and we need to use Model-from-Code logging.
            "tests/langchain/sample_code/openai_agent.py",
            name="langchain_model",
            input_example=input_example,
        )

    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    response = loaded_model.predict_stream([input_example])
    assert inspect.isgenerator(response)
    assert list(response) == [
        {
            "actions": [mock.ANY],
            "messages": [mock.ANY],
        },
        {
            "steps": [
                {
                    "action": mock.ANY,
                    "observation": 6,
                }
            ],
            "messages": [mock.ANY],
        },
        {
            "output": "The result of 2 * 3 is 6.",
            "messages": [mock.ANY],
        },
    ]


@pytest.mark.skipif(IS_LANGCHAIN_03, reason="Saving QAEvalChain does not work with LangChain 0.3.0")
def test_langchain_native_log_and_load_qaevalchain():
    # QAEvalChain is a subclass of LLMChain
    model = create_qa_eval_chain()
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, name="langchain_model")

    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)
    assert model == loaded_model


@pytest.mark.skipif(IS_LANGCHAIN_03, reason="Saving QAEvalChain does not work with LangChain 0.3.0")
def test_langchain_native_log_and_load_qa_with_sources_chain():
    # StuffDocumentsChain is a subclass of Chain
    model = create_qa_with_sources_chain()
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, name="langchain_model")

    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)
    assert model == loaded_model


@pytest.mark.skipif(IS_LANGCHAIN_03, reason="RetrievalQA is deprecated")
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

    langchain_input = {"query": "What did the president say about Ketanji Brown Jackson"}
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            retrievalQA,
            name="retrieval_qa_chain",
            loader_fn=load_retriever,
            persist_dir=persist_dir,
            input_example=langchain_input,
        )

    # Remove the persist_dir
    shutil.rmtree(persist_dir)

    # Load the chain
    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)
    assert loaded_model == retrievalQA

    loaded_pyfunc_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    result = loaded_pyfunc_model.predict([langchain_input])
    # The mock OpenAI endpoint simply echos the input
    assert result[0].startswith("Use the following pieces of context")

    # Serve the chain
    inference_payload = load_serving_example(logged_model.model_uri)

    response = pyfunc_serve_and_score_model(
        logged_model.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    response = PredictionsResponse.from_json(response.content.decode("utf-8"))
    response["predictions"][0].startswith("Use the following pieces of context")


@pytest.mark.skipif(IS_LANGCHAIN_03, reason="RetrievalQA is deprecated")
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

    langchain_input = {"query": "What did the president say about Ketanji Brown Jackson"}
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            retrievalQA,
            name="retrieval_qa_chain",
            loader_fn=load_retriever,
            persist_dir=persist_dir,
            input_example=langchain_input,
        )

    # Remove the persist_dir
    shutil.rmtree(persist_dir)

    # Load the chain
    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)
    assert loaded_model == retrievalQA

    loaded_pyfunc_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    result = loaded_pyfunc_model.predict([langchain_input])
    assert result[0][loaded_model.output_key].startswith("Use the following")

    # Serve the chain
    inference_payload = load_serving_example(logged_model.model_uri)

    response = pyfunc_serve_and_score_model(
        logged_model.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    response = PredictionsResponse.from_json(response.content.decode("utf-8"))
    assert response["predictions"][0][loaded_model.output_key].startswith("Use the following")


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
    text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = DeterministicDummyEmbeddings(size=5)
    db = FAISS.from_documents(docs, embeddings)
    persist_dir = str(tmp_path / "faiss_index")
    db.save_local(persist_dir)

    # Define the loader_fn
    def load_retriever(persist_directory):
        import numpy as np
        from langchain.embeddings.base import Embeddings
        from pydantic import BaseModel

        class DeterministicDummyEmbeddings(Embeddings, BaseModel):
            size: int

            def _get_embedding(self, text: str) -> list[float]:
                if isinstance(text, np.ndarray):
                    text = text.item()
                seed = abs(hash(text)) % (10**8)
                np.random.seed(seed)
                return list(np.random.normal(size=self.size))

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [self._get_embedding(t) for t in texts]

            def embed_query(self, text: str) -> list[float]:
                return self._get_embedding(text)

        embeddings = DeterministicDummyEmbeddings(size=5)
        vectorstore = FAISS.load_local(
            persist_directory,
            embeddings,
            **VECTORSTORE_KWARGS,
        )
        return vectorstore.as_retriever()

    query = "What did the president say about Ketanji Brown Jackson"
    langchain_input = {"query": query}
    # Log the retriever
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            db.as_retriever(),
            name="retriever",
            loader_fn=load_retriever,
            persist_dir=persist_dir,
            input_example=langchain_input,
        )

    # Remove the persist_dir
    shutil.rmtree(persist_dir)

    # Load the retriever
    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)
    assert_equal_retrievers(loaded_model, db.as_retriever())

    loaded_pyfunc_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    result = loaded_pyfunc_model.predict([langchain_input])
    expected_result = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "type": "Document",
        }
        for doc in db.as_retriever().get_relevant_documents(query)
    ]
    # "id" field was added to Document model in langchain 0.2.7
    if Version(langchain.__version__) >= Version("0.2.7"):
        expected_result = [{**d, "id": ANY} for d in expected_result]
    assert result == [expected_result]

    # Serve the retriever
    inference_payload = load_serving_example(logged_model.model_uri)
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


@pytest.mark.skipif(IS_LANGCHAIN_03, reason="APIChain is deprecated")
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
            name="api_chain",
            loader_fn=load_requests_wrapper,
        )

    # Load the chain
    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)
    assert loaded_model == apichain


@pytest.mark.skipif(IS_LANGCHAIN_03, reason="LLMChain is deprecated")
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
            name="apichain_subclass",
            loader_fn=load_requests_wrapper,
        )

    # Load the chain
    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)
    assert loaded_model == apichain_subclass


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
    version.parse(langchain.__version__) in (version.parse("0.1.14"), version.parse("0.1.15")),
    reason="LangChain 0.1.14 and 0.1.15 has a bug in loading SQLDatabaseChain",
)
@pytest.mark.skipif(
    IS_LANGCHAIN_03, reason="Saving SQLDatabaseChain does not work with LangChain 0.3.0"
)
def test_log_and_load_sql_database_chain(tmp_path):
    from langchain_experimental.sql import SQLDatabaseChain

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
            name="sql_database_chain",
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
            mlflow.langchain.log_model(conversation, name="conversation_model")


def test_saving_not_implemented_chain_type():
    chain = FakeChain()
    error_message = f"Chain {chain} does not support saving."
    with pytest.raises(
        NotImplementedError,
        match=error_message,
    ):
        with mlflow.start_run():
            mlflow.langchain.log_model(chain, name="fake_chain")


def test_unsupported_class():
    llm = FakeLLM()
    with pytest.raises(
        MlflowException,
        match="MLflow langchain flavor only supports subclasses of "
        + "\\(<class 'langchain.chains.base.Chain'>",
    ):
        with mlflow.start_run():
            mlflow.langchain.log_model(llm, name="fake_llm")


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
                mlflow.langchain.log_model(agent, name="unpicklable_tools")


def test_save_load_runnable_passthrough():
    runnable = RunnablePassthrough()
    assert runnable.invoke("hello") == "hello"

    input_example = "hello"
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            runnable, name="model_path", input_example=input_example
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke(input_example) == "hello"
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict(["hello"]) == ["hello"]

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": ["hello"]
    }


def test_save_load_runnable_lambda(spark):
    def add_one(x: int) -> int:
        return x + 1

    runnable = RunnableLambda(add_one)

    assert runnable.invoke(1) == 2
    assert runnable.batch([1, 2, 3]) == [2, 3, 4]

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            runnable, name="runnable_lambda", input_example=[1, 2, 3]
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

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [2, 3, 4]
    }


def test_save_load_runnable_lambda_in_sequence():
    def add_one(x):
        return x + 1

    def mul_two(x):
        return x * 2

    runnable_1 = RunnableLambda(add_one)
    runnable_2 = RunnableLambda(mul_two)
    sequence = runnable_1 | runnable_2
    assert sequence.invoke(1) == 4

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            sequence, name="model_path", input_example=[1, 2, 3]
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke(1) == 4
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict(1) == [4]
    assert pyfunc_loaded_model.predict([1, 2, 3]) == [4, 6, 8]

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [4, 6, 8]
    }


def test_predict_with_callbacks(fake_chat_model):
    class TestCallbackHandler(BaseCallbackHandler):
        def __init__(self):
            super().__init__()
            self.num_llm_start_calls = 0

        def on_llm_start(
            self,
            serialized: dict[str, Any],
            prompts: list[str],
            **kwargs: Any,
        ) -> Any:
            self.num_llm_start_calls += 1

    prompt = ChatPromptTemplate.from_template("What's your favorite {industry} company?")
    chain = prompt | fake_chat_model | StrOutputParser()
    # Test the basic functionality of the chain
    assert chain.invoke({"industry": "tech"}) == "Databricks"

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain, name="model_path", input_example={"industry": "tech"}
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

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": ["Databricks"]
    }


def test_predict_with_callbacks_supports_chat_response_conversion(fake_chat_model):
    prompt = ChatPromptTemplate.from_template("What's your favorite {industry} company?")
    chain = prompt | fake_chat_model | StrOutputParser()
    # Test the basic functionality of the chain
    assert chain.invoke({"industry": "tech"}) == "Databricks"

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain, name="model_path", input_example={"industry": "tech"}
        )

    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    expected_chat_response = {
        "id": None,
        "object": "chat.completion",
        "created": 1677858242,
        "model": "",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Databricks",
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


def test_save_load_runnable_parallel():
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
            runnable, name="model_path", input_example=["hello", "world"]
        )
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke("hello") == {"llm": "completion"}
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict("hello") == [{"llm": "completion"}]
    assert pyfunc_loaded_model.predict(["hello", "world"]) == [
        {"llm": "completion"},
        {"llm": "completion"},
    ]

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [{"llm": "completion"}, {"llm": "completion"}]
    }


def test_simple_chat_model_inference():
    class ChatModel(SimpleChatModel):
        def _call(self, messages, stop, run_manager, **kwargs):
            return "\n".join([f"{message.type}: {message.content}" for message in messages])

        @property
        def _llm_type(self) -> str:
            return "chat model"

    model = ChatModel()

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(model, name="model")

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    input_example = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "What would you like to ask?"},
            {"role": "user", "content": "Who owns MLflow?"},
        ]
    }
    expected_resp_content = {
        "role": "assistant",
        "content": (
            "system: You are a helpful assistant.\n"
            "ai: What would you like to ask?\n"
            "human: Who owns MLflow?"
        ),
    }
    response1 = loaded_model.predict([input_example])
    assert len(response1) == 1
    assert response1[0]["choices"][0]["message"] == expected_resp_content
    response2 = loaded_model.predict(input_example)
    assert response2["choices"][0]["message"] == expected_resp_content
    response3 = loaded_model.predict([input_example, input_example])
    assert len(response3) == 2
    for i in range(2):
        assert response3[i]["choices"][0]["message"] == expected_resp_content


def test_save_load_complex_runnable_parallel():
    runnable = RunnableParallel({"llm": create_openai_runnable()})
    expected_result = {"llm": TEST_CONTENT}
    assert runnable.invoke({"product": "MLflow"}) == expected_result
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            runnable, name="model_path", input_example=[{"product": "MLflow"}]
        )
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke({"product": "MLflow"}) == expected_result
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict([{"product": "MLflow"}]) == [expected_result]

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [expected_result]
    }


@pytest.mark.skipif(
    IS_LANGCHAIN_03,
    reason="RunnableAssign has a bug in LangChain 0.3.x. "
    "https://github.com/langchain-ai/langchain/issues/26862",
)
def test_save_load_runnable_parallel_and_assign_in_sequence():
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
            runnable, name="model_path", input_example=["hello", "world"]
        )
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke("hello") == expected_result
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict(["hello"]) == [expected_result]

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [expected_result, expected_result]
    }


@pytest.mark.skipif(
    IS_LANGCHAIN_03,
    reason="RunnableAssign has a bug in LangChain 0.3.x. "
    "https://github.com/langchain-ai/langchain/issues/26862",
)
def test_save_load_complex_runnable_assign(fake_chat_model):
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
            runnable_assign, name="model_path", input_example=input_example
        )
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke(input_example) == expected_result
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict([input_example]) == [expected_result]

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [expected_result]
    }


def test_save_load_runnable_sequence():
    prompt1 = PromptTemplate.from_template("what is the city {person} is from?")
    llm = OpenAI(temperature=0.9)
    model = prompt1 | llm | StrOutputParser()

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(model, name="model_path")

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert type(loaded_model) == RunnableSequence
    assert type(loaded_model.steps[0]) == PromptTemplate
    assert type(loaded_model.steps[1]) == OpenAI
    assert type(loaded_model.steps[2]) == StrOutputParser


def test_save_load_long_runnable_sequence(model_path):
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


def test_save_load_runnable_sequence_with_chat_openai():
    prompt1 = PromptTemplate.from_template("what is the city {person} is from?")
    llm = ChatOpenAI(temperature=0.9)
    model = prompt1 | llm | StrOutputParser()

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(model, name="model_path")

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert type(loaded_model) == RunnableSequence
    assert type(loaded_model.steps[0]) == PromptTemplate
    assert type(loaded_model.steps[1]) == ChatOpenAI
    assert type(loaded_model.steps[2]) == StrOutputParser


def test_save_load_chain_with_model_paths():
    prompt1 = PromptTemplate.from_template("what is the city {person} is from?")
    llm = ChatOpenAI(temperature=0.9)
    model = prompt1 | llm | StrOutputParser()

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(model, name="model_path")
    artifact_path = "model_path"
    with (
        mlflow.start_run(),
        mock.patch("mlflow.langchain.model._add_code_from_conf_to_system_path") as add_mock,
    ):
        model_info = mlflow.langchain.log_model(model, name=artifact_path, code_paths=[__file__])
        mlflow.langchain.load_model(model_info.model_uri)
        model_uri = model_info.model_uri
        _compare_logged_code_paths(__file__, model_uri, mlflow.langchain.FLAVOR_NAME)
        add_mock.assert_called()


def test_save_load_simple_chat_model(spark, fake_chat_model):
    prompt = ChatPromptTemplate.from_template(
        "What is a good name for a company that makes {product}?"
    )
    chain = prompt | fake_chat_model | StrOutputParser()
    input_example = {"product": "MLflow"}
    assert chain.invoke(input_example) == "Databricks"
    # signature is required for spark_udf
    signature = infer_signature({"product": "MLflow"}, "Databricks")
    assert signature == ModelSignature(
        Schema([ColSpec("string", "product")]), Schema([ColSpec("string")])
    )
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain, name="model_path", input_example=input_example
        )
    assert model_info.signature == signature
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke({"product": "MLflow"}) == "Databricks"
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict([{"product": "MLflow"}]) == ["Databricks"]

    udf = mlflow.pyfunc.spark_udf(spark, model_info.model_uri, result_type="string")
    df = spark.createDataFrame([("MLflow",), ("Spark",)], ["product"])
    df = df.withColumn("answer", udf("product"))
    pdf = df.toPandas()
    assert pdf["answer"].tolist() == ["Databricks", "Databricks"]

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    # Because of the schema enforcement converts input to pandas dataframe
    # the prediction result is wrapped in a list in api_request_parallel_processor
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": ["Databricks"]
    }


def test_save_load_rag(tmp_path, spark, fake_chat_model):
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
            name="model_path",
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

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [answer]
    }


def test_runnable_branch_save_load():
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
        model_info = mlflow.langchain.log_model(branch, name="model_path")

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


def test_complex_runnable_branch_save_load(fake_chat_model, fake_classifier_chat_model):
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
            chain, name="model_path", input_example={"query": "Who owns MLflow?"}
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

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": ["Databricks"]
    }


def test_chat_with_history(spark, fake_chat_model):
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
            chain_with_history, name="model_path", input_example=input_example
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

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert json.loads(response.content.decode("utf-8")) == ["Databricks"]


def _extract_endpoint_name_from_lc_model(lc_model):
    if type(lc_model).__name__ == type(get_fake_chat_model()).__name__:
        yield DatabricksServingEndpoint(endpoint_name=lc_model.endpoint_name)


@mock.patch(
    "mlflow.langchain.databricks_dependencies._extract_dependency_list_from_lc_model",
    _extract_endpoint_name_from_lc_model,
)
def test_databricks_dependency_extraction_from_lcel_chain():
    prompt_1 = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
    prompt_2 = ChatPromptTemplate.from_template(
        "compare which joke is better {joke1} or {joke2}. Output the better joke."
    )
    model_1 = get_fake_chat_model(endpoint_name="fake-endpoint-1")
    model_2 = get_fake_chat_model(endpoint_name="fake-endpoint-2")
    model_3 = get_fake_chat_model(endpoint_name="fake-endpoint-3")
    output_parser = StrOutputParser()

    chain = prompt_1 | {"joke1": model_1, "joke2": model_2} | prompt_2 | model_3 | output_parser

    pyfunc_artifact_path = "basic_chain"
    with mlflow.start_run(), mock.patch("mlflow.langchain.model.logger.info") as mock_log_info:
        model_info = mlflow.langchain.log_model(chain, name=pyfunc_artifact_path)
        mock_log_info.assert_called_once_with(
            "Attempting to auto-detect Databricks resource dependencies for the current "
            "langchain model. Dependency auto-detection is best-effort and may not capture "
            "all dependencies of your langchain model, resulting in authorization errors when "
            "serving or querying your model. We recommend that you explicitly pass `resources` "
            "to mlflow.langchain.log_model() to ensure authorization to dependent resources "
            "succeeds when the model is deployed."
        )
    pyfunc_model_path = _download_artifact_from_uri(model_info.model_uri)
    reloaded_model = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    assert reloaded_model.resources["databricks"] == {
        "serving_endpoint": [
            {"name": "fake-endpoint-1"},
            {"name": "fake-endpoint-2"},
            {"name": "fake-endpoint-3"},
        ]
    }


def _extract_databricks_dependencies_from_retriever(retriever):
    import langchain_community

    vectorstore = getattr(retriever, "vectorstore", None)
    if vectorstore:
        if isinstance(vectorstore, langchain_community.vectorstores.faiss.FAISS):
            yield DatabricksVectorSearchIndex(index_name="faiss-index")

        embeddings = getattr(vectorstore, "embeddings", None)
        if isinstance(embeddings, FakeEmbeddings):
            yield DatabricksServingEndpoint(endpoint_name="fake-embeddings")


def _extract_databricks_dependencies_from_llm(llm):
    if isinstance(llm, FakeLLM):
        yield DatabricksServingEndpoint(endpoint_name=llm.endpoint_name)


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

    pyfunc_artifact_path = "retrieval_qa_chain"
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            retrievalQA,
            name=pyfunc_artifact_path,
            loader_fn=load_retriever,
            persist_dir=persist_dir,
        )
    pyfunc_model_path = _download_artifact_from_uri(model_info.model_uri)
    reloaded_model = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    actual = reloaded_model.resources["databricks"]
    expected = {
        "serving_endpoint": [
            {"name": "fake-llm-endpoint"},
            {"name": "fake-embeddings"},
        ],
        "vector_search_index": [{"name": "faiss-index"}],
    }
    assert all(item in actual["serving_endpoint"] for item in expected["serving_endpoint"])
    assert all(item in expected["serving_endpoint"] for item in actual["serving_endpoint"])
    assert actual["vector_search_index"] == expected["vector_search_index"]


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.2.0"),
    reason="Langgraph are not supported the way we want in earlier versions",
)
def test_databricks_dependency_extraction_from_langgraph_agent(monkeypatch):
    from langchain_community.chat_models import ChatDatabricks
    from langchain_core.runnables import RunnableLambda
    from langgraph.prebuilt import create_react_agent

    # Mocking Cloudpickle because serialization in this setup is failing
    monkeypatch.setattr("cloudpickle.dump", mock.MagicMock())

    uc_functions = ["rag.studio.test_function_a", "rag.studio.test_function_b"]
    uc_function_tools = create_uc_tools(
        monkeypatch,
        warehouse_id="test_id_1",
        expected_catalog_name="rag",
        expected_schema_name="studio",
        functions=uc_functions,
    )
    retriever_tool = create_retriever_tool(monkeypatch)
    chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens=500)

    agent = create_react_agent(chat_model, uc_function_tools + [retriever_tool])

    def wrap_agent(input):
        return agent.invoke(input)

    pyfunc_artifact_path = "retrieval_qa_chain"
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            RunnableLambda(wrap_agent),
            name=pyfunc_artifact_path,
        )
    pyfunc_model_path = _download_artifact_from_uri(model_info.model_uri)
    reloaded_model = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    actual = reloaded_model.resources["databricks"]

    # Ensure both functions are outputted
    expected = {
        "serving_endpoint": [{"name": "databricks-llama-2-70b-chat"}, {"name": "embedding-model"}],
        "vector_search_index": [{"name": "mlflow.rag.vs_index"}],
        "sql_warehouse": [{"name": "test_id_1"}],
        "function": [{"name": function} for function in uc_functions],
    }

    assert all(item in actual["serving_endpoint"] for item in expected["serving_endpoint"])
    assert all(item in expected["serving_endpoint"] for item in actual["serving_endpoint"])
    assert actual["vector_search_index"] == expected["vector_search_index"]
    if uc_function_tools:
        assert actual["sql_warehouse"] == expected["sql_warehouse"]
        assert all(item in actual["function"] for item in expected["function"])
        assert all(item in expected["function"] for item in actual["function"])


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.1.0"),
    reason="Tools are not supported the way we want in earlier versions",
)
def test_databricks_dependency_extraction_from_agent_chain(monkeypatch):
    from langchain_community.chat_models import ChatDatabricks

    # Mocking Cloudpickle because serialization in this setup is failing
    monkeypatch.setattr("cloudpickle.dump", mock.MagicMock())

    uc_functions = ["rag.studio.test_function_a", "rag.studio.test_function_b"]
    uc_function_tools = create_uc_tools(
        monkeypatch,
        warehouse_id="test_id_1",
        expected_catalog_name="rag",
        expected_schema_name="studio",
        functions=uc_functions,
    )
    retriever_tool = create_retriever_tool(monkeypatch)
    chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens=500)

    agent = initialize_agent(
        uc_function_tools + [retriever_tool],
        chat_model,
        verbose=True,
    )

    pyfunc_artifact_path = "retrieval_qa_chain"
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            agent,
            name=pyfunc_artifact_path,
        )
    pyfunc_model_uri = model_info.model_uri
    pyfunc_model_path = _download_artifact_from_uri(pyfunc_model_uri)
    reloaded_model = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    actual = reloaded_model.resources["databricks"]
    # Ensure both functions are outputted
    expected = {
        "serving_endpoint": [{"name": "databricks-llama-2-70b-chat"}, {"name": "embedding-model"}],
        "vector_search_index": [{"name": "mlflow.rag.vs_index"}],
    }

    if len(uc_function_tools) > 0:
        uc_expected = {
            "sql_warehouse": [{"name": "test_id_1"}],
            "function": [{"name": function} for function in uc_functions],
        }
        expected.update(uc_expected)

    assert all(item in actual["serving_endpoint"] for item in expected["serving_endpoint"])
    assert all(item in expected["serving_endpoint"] for item in actual["serving_endpoint"])
    assert actual["vector_search_index"] == expected["vector_search_index"]
    if uc_function_tools:
        assert actual["sql_warehouse"] == expected["sql_warehouse"]
        assert all(item in actual["function"] for item in expected["function"])
        assert all(item in expected["function"] for item in actual["function"])


def _error_func(*args, **kwargs):
    raise ValueError("error")


@mock.patch(
    "mlflow.langchain.databricks_dependencies._traverse_runnable",
    _error_func,
)
@mock.patch("mlflow.langchain.databricks_dependencies._logger.warning")
def test_databricks_dependency_extraction_log_errors_as_warnings(mock_warning):
    from mlflow.langchain.databricks_dependencies import _detect_databricks_dependencies

    model = create_openai_llmchain()

    _detect_databricks_dependencies(model, log_errors_as_warnings=True)
    mock_warning.assert_called_once_with(
        "Unable to detect Databricks dependencies. "
        "Set logging level to DEBUG to see the full traceback."
    )

    with pytest.raises(ValueError, match="error"):
        _detect_databricks_dependencies(model, log_errors_as_warnings=False)

    pyfunc_artifact_path = "langchain_model"
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(model, name=pyfunc_artifact_path)
    pyfunc_model_path = _download_artifact_from_uri(model_info.model_uri)
    reloaded_model = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    assert reloaded_model.resources is None


class ChatModel(SimpleChatModel):
    def _call(self, messages, stop, run_manager, **kwargs):
        return "\n".join([f"{message.type}: {message.content}" for message in messages])

    @property
    def _llm_type(self) -> str:
        return "chat model"


def test_predict_with_builtin_pyfunc_chat_conversion(spark):
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

    chain = ChatModel() | StrOutputParser()
    assert chain.invoke([HumanMessage(content="Who owns MLflow?")]) == "human: Who owns MLflow?"
    with pytest.raises(ValueError, match="Invalid input type"):
        chain.invoke(input_example)

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain, name="model_path", input_example=input_example
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
        "model": "",
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
        result1 = pyfunc_loaded_model.predict(input_example)
        result1[0]["id"] = None
        assert result1 == [expected_chat_response]
        result2 = pyfunc_loaded_model.predict([input_example, input_example])
        result2[0]["id"] = None
        result2[1]["id"] = None
        assert result2 == [
            expected_chat_response,
            expected_chat_response,
        ]

    with pytest.raises(MlflowException, match="Unrecognized chat message role"):
        pyfunc_loaded_model.predict({"messages": [{"role": "foobar", "content": "test content"}]})


def test_predict_with_builtin_pyfunc_chat_conversion_for_aimessage_response():
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

    chain = ChatModel()
    result = chain.invoke([HumanMessage(content="Who owns MLflow?")])
    assert isinstance(result, AIMessage)
    assert result.content == "You own MLflow"

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain, name="model_path", input_example=input_example
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    result = loaded_model.invoke([HumanMessage(content="Who owns MLflow?")])
    assert isinstance(result, AIMessage)
    assert result.content == "You own MLflow"

    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    with mock.patch("time.time", return_value=1677858242):
        result = pyfunc_loaded_model.predict(input_example)
        assert "id" in result[0], "Response message id is lost."
        result[0]["id"] = None
        assert result == [
            {
                "id": None,
                "object": "chat.completion",
                "created": 1677858242,
                "model": "",
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


def test_pyfunc_builtin_chat_request_conversion_fails_gracefully():
    chain = RunnablePassthrough() | itemgetter("messages")
    # Ensure we're going to test that "messages" remains intact & unchanged even if it
    # doesn't appear explicitly in the chain's input schema
    assert "messages" not in chain.input_schema().__fields__

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(chain, name="model_path")
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
            "messages": [{"role": "user", "content": "blah"}, {}],
        }
    ) == [
        {"role": "user", "content": "blah"},
        {},
    ]
    assert pyfunc_loaded_model.predict(
        {
            "messages": [{"role": "user", "content": 123}],
        }
    ) == [
        {"role": "user", "content": 123},
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
                "messages": [
                    {"role": "user", "content": "content"},
                    {"role": "user", "content": 123},
                ],
            },
        ]
    ) == [
        [{"role": "user", "content": "content"}],
        [{"role": "user", "content": "content"}, {"role": "user", "content": 123}],
    ]


@pytest.mark.skipif(IS_LANGCHAIN_03, reason="LLMChain is deprecated")
def test_pyfunc_builtin_chat_response_conversion_fails_gracefully():
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["messages"],
        template="What is {messages}?",
    )
    chain = RunnablePassthrough() | LLMChain(llm=llm, prompt=prompt) | RunnablePassthrough()

    input_example = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "What would you like to ask?"},
            {"role": "user", "content": "Who owns MLflow?"},
        ]
    }

    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            chain,
            name="langchain_model",
            input_example=input_example,
        )
    assert logged_model.signature is not None
    assert logged_model.signature.outputs is not None
    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    result = loaded_model.predict(input_example)
    # Verify that the chat request format was converted into LangChain messages correctly, but
    # the response was not converted to the chat response format because it does not have the
    # expected structure (a nonstandard dict with 'messages' and 'text' fields is returned)
    assert result[0]["messages"] == [
        SystemMessage(content="You are a helpful assistant."),
        AIMessage(content="What would you like to ask?"),
        HumanMessage(content="Who owns MLflow?"),
    ]
    assert result[0]["text"].startswith("What is ")


def test_save_load_chain_that_relies_on_pickle_serialization(monkeypatch, model_path):
    from langchain_community.llms.databricks import Databricks

    monkeypatch.setattr(
        "langchain_community.llms.databricks._DatabricksServingEndpointClient",
        mock.MagicMock(),
    )
    monkeypatch.setenv("DATABRICKS_HOST", "test-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")

    llm_kwargs = {"endpoint_name": "test-endpoint", "temperature": 0.9}
    if IS_PICKLE_SERIALIZATION_RESTRICTED:
        llm_kwargs["allow_dangerous_deserialization"] = True

    llm = Databricks(**llm_kwargs)
    prompt = PromptTemplate(input_variables=["question"], template="I have a question: {question}")
    chain = prompt | llm | StrOutputParser()

    # Not passing an input_example to avoid triggering prediction
    mlflow.langchain.save_model(chain, model_path)

    if IS_PICKLE_SERIALIZATION_RESTRICTED and Version(langchain.__version__) < Version("0.1.14"):
        # For LangChain between 0.1.12 and 0.1.14, MLflow cannot load a model that relies on pickle
        # serialization, instead, raises an MlflowException with a message that explains the issue.
        with pytest.raises(MlflowException, match=r"Since langchain-community 0.0.27, loading a"):
            mlflow.langchain.load_model(model_path)
        return
    loaded_model = mlflow.langchain.load_model(model_path)

    # Check if the deserialized model has the same endpoint and temperature
    loaded_databricks_llm = loaded_model.middle[0]
    assert loaded_databricks_llm.endpoint_name == "test-endpoint"
    assert loaded_databricks_llm.temperature == 0.9


@pytest.fixture
def chain_model_signature():
    return ModelSignature(
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


def _get_message_content(predictions):
    return predictions[0]["choices"][0]["message"]["content"]


@pytest.mark.parametrize(
    ("chain_path", "model_config"),
    [
        (
            os.path.abspath("tests/langchain/sample_code/chain.py"),
            os.path.abspath("tests/langchain/sample_code/config.yml"),
        ),
        (
            "tests/langchain/../langchain/sample_code/chain.py",
            "tests/langchain/../langchain/sample_code/config.yml",
        ),
    ],
)
def test_save_load_chain_as_code(chain_model_signature, chain_path, model_config, monkeypatch):
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is a good name for a company that makes MLflow?",
            }
        ]
    }
    artifact_path = "model_path"
    with mlflow.start_run() as run:
        model_info = mlflow.langchain.log_model(
            chain_path,
            name=artifact_path,
            signature=chain_model_signature,
            input_example=input_example,
            model_config=model_config,
        )

    client = mlflow.tracking.MlflowClient()
    run_id = run.info.run_id
    assert client.get_run(run_id).data.params == {
        "llm_prompt_template": "Answer the following question based on "
        "the context: {context}\nQuestion: {question}",
        "embedding_size": "5",
        "not_used_array": "[1, 2, 3]",
        "response": "Databricks",
    }

    assert mlflow.models.model_config.__mlflow_model_config__ is None
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)

    # During the loading process, MLflow executes the chain.py file to
    # load the model class. It should not generate any traces even if
    # the code enables autologging and invoke chain.
    assert len(get_traces()) == 0

    assert mlflow.models.model_config.__mlflow_model_config__ is None
    answer = "Databricks"
    assert loaded_model.invoke(input_example) == answer
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert answer == _get_message_content(pyfunc_loaded_model.predict(input_example))

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    predictions = json.loads(response.content.decode("utf-8"))
    # Mock out the `created` timestamp as it is not deterministic
    expected = [{**try_transform_response_to_chat_format(answer), "created": mock.ANY}]
    assert expected == predictions

    pyfunc_model_path = _download_artifact_from_uri(model_info.model_uri)
    reloaded_model = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    assert reloaded_model.resources["databricks"] == {
        "serving_endpoint": [{"name": "fake-endpoint"}]
    }
    assert reloaded_model.metadata["dependencies_schemas"] == {
        DependenciesSchemasType.RETRIEVERS.value: [
            {
                "doc_uri": "doc-uri",
                "name": "retriever",
                "other_columns": ["column1", "column2"],
                "primary_key": "primary-key",
                "text_column": "text-column",
            }
        ]
    }

    # Emulate the model serving environment
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")
    monkeypatch.setenv("ENABLE_MLFLOW_TRACING", "true")
    mlflow.tracing.reset()

    request_id = "mock_request_id"
    tracer = MlflowLangchainTracer(prediction_context=Context(request_id))
    input_example = {"messages": [{"role": "user", "content": TEST_CONTENT}]}
    response = pyfunc_loaded_model._model_impl._predict_with_callbacks(
        data=input_example, callback_handlers=[tracer]
    )
    assert response["choices"][0]["message"]["content"] == "Databricks"
    trace = pop_trace(request_id)
    assert trace["info"]["tags"][DependenciesSchemasType.RETRIEVERS.value] == json.dumps(
        [
            {
                "doc_uri": "doc-uri",
                "name": "retriever",
                "other_columns": ["column1", "column2"],
                "primary_key": "primary-key",
                "text_column": "text-column",
            }
        ]
    )


@pytest.mark.parametrize(
    "chain_path",
    [
        os.path.abspath("tests/langchain/sample_code/chain.py"),
        "tests/langchain/../langchain/sample_code/chain.py",
    ],
)
def test_save_load_chain_as_code_model_config_dict(chain_model_signature, chain_path):
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is a good name for a company that makes MLflow?",
            }
        ]
    }
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain_path,
            name="model_path",
            signature=chain_model_signature,
            input_example=input_example,
            model_config={
                "response": "modified response",
                "embedding_size": 5,
                "llm_prompt_template": "answer the question",
            },
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    answer = "modified response"
    assert loaded_model.invoke(input_example) == answer
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert answer == _get_message_content(pyfunc_loaded_model.predict(input_example))


@pytest.mark.parametrize(
    "model_config",
    [
        os.path.abspath("tests/langchain/sample_code/config.yml"),
        "tests/langchain/../langchain/sample_code/config.yml",
    ],
)
def test_save_load_chain_as_code_with_different_names(
    tmp_path, chain_model_signature, model_config
):
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is a good name for a company that makes MLflow?",
            }
        ]
    }

    # Read the contents of the original chain file
    with open("tests/langchain/sample_code/chain.py") as chain_file:
        chain_file_content = chain_file.read()

    temp_file = tmp_path / "model.py"
    temp_file.write_text(chain_file_content)

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            str(temp_file),
            name="model_path",
            signature=chain_model_signature,
            input_example=input_example,
            model_config=model_config,
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    answer = "Databricks"
    assert loaded_model.invoke(input_example) == answer
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert answer == _get_message_content(pyfunc_loaded_model.predict(input_example))


@pytest.mark.parametrize(
    "chain_path",
    [
        os.path.abspath("tests/langchain/sample_code/chain.py"),
        "tests/langchain/../langchain/sample_code/chain.py",
    ],
)
@pytest.mark.parametrize(
    "model_config",
    [
        os.path.abspath("tests/langchain/sample_code/config.yml"),
        "tests/langchain/../langchain/sample_code/config.yml",
    ],
)
def test_save_load_chain_as_code_multiple_times(
    tmp_path, chain_model_signature, chain_path, model_config
):
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is a good name for a company that makes MLflow?",
            }
        ]
    }
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain_path,
            name="model_path",
            signature=chain_model_signature,
            input_example=input_example,
            model_config=model_config,
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    with open(model_config) as f:
        base_config = yaml.safe_load(f)

    assert loaded_model.middle[0].messages[0].prompt.template == base_config["llm_prompt_template"]

    file_name = "config_updated.yml"
    new_config_file = str(tmp_path.joinpath(file_name))

    new_config = base_config.copy()
    new_config["llm_prompt_template"] = "new_template"
    with open(new_config_file, "w") as f:
        yaml.dump(new_config, f)

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain_path,
            name="model_path",
            signature=chain_model_signature,
            input_example=input_example,
            model_config=new_config_file,
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.middle[0].messages[0].prompt.template == new_config["llm_prompt_template"]


@pytest.mark.parametrize(
    "chain_path",
    [
        os.path.abspath("tests/langchain/sample_code/chain.py"),
        "tests/langchain/../langchain/sample_code/chain.py",
    ],
)
def test_save_load_chain_as_code_with_model_paths(chain_model_signature, chain_path):
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is a good name for a company that makes MLflow?",
            }
        ]
    }
    artifact_path = "model_path"
    with (
        mlflow.start_run(),
        mock.patch("mlflow.langchain.model._add_code_from_conf_to_system_path") as add_mock,
    ):
        model_info = mlflow.langchain.log_model(
            chain_path,
            name=artifact_path,
            signature=chain_model_signature,
            input_example=input_example,
            code_paths=[__file__],
            model_config={
                "response": "modified response",
                "embedding_size": 5,
                "llm_prompt_template": "answer the question",
            },
        )
        loaded_model = mlflow.langchain.load_model(model_info.model_uri)
        answer = "modified response"
        _compare_logged_code_paths(__file__, model_info.model_uri, mlflow.langchain.FLAVOR_NAME)
        assert loaded_model.invoke(input_example) == answer
        add_mock.assert_called()


@pytest.mark.parametrize("chain_path", [os.path.abspath("tests/langchain1/sample_code/chain.py")])
def test_save_load_chain_errors(chain_model_signature, chain_path):
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is a good name for a company that makes MLflow?",
            }
        ]
    }
    with mlflow.start_run():
        with pytest.raises(
            MlflowException,
            match=f"The provided model path '{chain_path}' does not exist. "
            "Ensure the file path is valid and try again.",
        ):
            mlflow.langchain.log_model(
                chain_path,
                name="model_path",
                signature=chain_model_signature,
                input_example=input_example,
                model_config="tests/langchain/state_of_the_union.txt",
            )


@pytest.mark.parametrize(
    "chain_path",
    [
        os.path.abspath("tests/langchain/sample_code/no_config/chain.py"),
        "tests/langchain/../langchain/sample_code/no_config/chain.py",
    ],
)
def test_save_load_chain_as_code_optional_code_path(chain_model_signature, chain_path):
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is a good name for a company that makes MLflow?",
            }
        ]
    }
    artifact_path = "new_model_path"
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain_path,
            name=artifact_path,
            signature=chain_model_signature,
            input_example=input_example,
        )

    assert mlflow.models.model_config.__mlflow_model_config__ is None
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert mlflow.models.model_config.__mlflow_model_config__ is None
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

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    # avoid minor diff of created time in the response
    prediction_result = json.loads(response.content.decode("utf-8"))
    prediction_result[0]["created"] = 123
    expected_prediction = try_transform_response_to_chat_format(answer)
    expected_prediction["created"] = 123
    assert prediction_result == [expected_prediction]

    pyfunc_model_path = _download_artifact_from_uri(model_info.model_uri)
    reloaded_model = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    assert reloaded_model.resources["databricks"] == {
        "serving_endpoint": [{"name": "fake-endpoint"}]
    }
    assert reloaded_model.metadata is None


def get_fake_chat_stream_model(endpoint_name="fake-stream-endpoint"):
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.chat_models.base import SimpleChatModel
    from langchain.schema.messages import AIMessageChunk, BaseMessage
    from langchain_core.outputs import ChatGenerationChunk

    class FakeChatStreamModel(SimpleChatModel):
        """Fake Chat Stream Model wrapper for testing purposes."""

        endpoint_name: str = "fake-stream-endpoint"

        def _call(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> str:
            return "Databricks"

        def _stream(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> Iterator[ChatGenerationChunk]:
            for chunk_content, finish_reason in [
                ("Da", None),
                ("tab", None),
                ("ricks", "stop"),
            ]:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=chunk_content),
                    generation_info={"finish_reason": finish_reason},
                )
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text, chunk=chunk)

                yield chunk

        @property
        def _llm_type(self) -> str:
            return "fake chat model"

    return FakeChatStreamModel(endpoint_name=endpoint_name)


@pytest.fixture
def fake_chat_stream_model():
    return get_fake_chat_stream_model()


@pytest.mark.parametrize("provide_signature", [True, False])
def test_simple_chat_model_stream_inference(fake_chat_stream_model, provide_signature):
    input_example = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "What would you like to ask?"},
            {"role": "user", "content": "Who owns MLflow?"},
        ]
    }
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            fake_chat_stream_model,
            name="model",
        )

    if provide_signature:
        signature = infer_signature(model_input=input_example)
        with mlflow.start_run():
            model_with_siginature_info = mlflow.langchain.log_model(
                fake_chat_stream_model, name="model", signature=signature
            )
    else:
        with mlflow.start_run():
            model_with_siginature_info = mlflow.langchain.log_model(
                fake_chat_stream_model, name="model", input_example=input_example
            )

    for model_uri in [model_info.model_uri, model_with_siginature_info.model_uri]:
        loaded_model = mlflow.pyfunc.load_model(model_uri)

        chunk_iter = loaded_model.predict_stream(input_example)

        finish_reason = None if Version(langchain.__version__) < Version("0.1.8") else "stop"

        with mock.patch("time.time", return_value=1677858242):
            chunks = list(chunk_iter)

            for chunk in chunks:
                assert "id" in chunk, "chunk id is lost."
                chunk["id"] = None

            assert chunks == [
                {
                    "id": None,
                    "object": "chat.completion.chunk",
                    "created": 1677858242,
                    "model": "",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": None,
                            "delta": {"role": "assistant", "content": "Da"},
                        }
                    ],
                },
                {
                    "id": None,
                    "object": "chat.completion.chunk",
                    "created": 1677858242,
                    "model": "",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": None,
                            "delta": {"role": "assistant", "content": "tab"},
                        }
                    ],
                },
                {
                    "id": None,
                    "object": "chat.completion.chunk",
                    "created": 1677858242,
                    "model": "",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": finish_reason,
                            "delta": {"role": "assistant", "content": "ricks"},
                        }
                    ],
                },
            ]


def test_simple_chat_model_stream_with_callbacks(fake_chat_stream_model):
    class TestCallbackHandler(BaseCallbackHandler):
        def __init__(self):
            super().__init__()
            self.num_llm_start_calls = 0

        def on_llm_start(
            self,
            serialized: dict[str, Any],
            prompts: list[str],
            **kwargs: Any,
        ) -> Any:
            self.num_llm_start_calls += 1

    prompt = ChatPromptTemplate.from_template("What's your favorite {industry} company?")
    chain = prompt | fake_chat_stream_model | StrOutputParser()
    # Test the basic functionality of the chain
    assert chain.invoke({"industry": "tech"}) == "Databricks"

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain, name="model_path", input_example={"industry": "tech"}
        )

    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    callback_handler1 = TestCallbackHandler()
    callback_handler2 = TestCallbackHandler()

    # Ensure handlers have not been called yet
    assert callback_handler1.num_llm_start_calls == 0
    assert callback_handler2.num_llm_start_calls == 0

    stream = pyfunc_loaded_model._model_impl._predict_stream_with_callbacks(
        {"industry": "tech"},
        callback_handlers=[callback_handler1, callback_handler2],
    )
    assert list(stream) == ["Da", "tab", "ricks"]

    # Test that the callback handlers were called
    assert callback_handler1.num_llm_start_calls == 1
    assert callback_handler2.num_llm_start_calls == 1


def test_langchain_model_save_exception(fake_chat_model):
    prompt = PromptTemplate.from_template(
        "What's your favorite {industry} company in {country}?", partial_variables={"country": "US"}
    )
    chain = prompt | fake_chat_model | StrOutputParser()
    assert chain.invoke({"industry": "tech"}) == "Databricks"

    with pytest.raises(
        MlflowException, match=r"Failed to save runnable sequence: {'0': 'PromptTemplate -- "
    ):
        with mlflow.start_run():
            mlflow.langchain.log_model(chain, name="model_path", input_example={"industry": "tech"})


def test_langchain_model_save_load_with_listeners(fake_chat_model):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    def retrieve_history(input):
        return {"history": [], "question": input["question"], "name": input["name"]}

    chain = (
        {"question": itemgetter("question"), "name": itemgetter("name")}
        | (RunnableLambda(retrieve_history) | prompt | fake_chat_model).with_listeners()
        | StrOutputParser()
        | RunnablePassthrough()
    )
    input_example = {"question": "Who owns MLflow?", "name": ""}
    assert chain.invoke(input_example) == "Databricks"

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain, name="model_path", input_example=input_example
        )
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke(input_example) == "Databricks"
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict(input_example) == ["Databricks"]

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": ["Databricks"]
    }


@pytest.mark.parametrize("env_var", ["MLFLOW_ENABLE_TRACE_IN_SERVING", "ENABLE_MLFLOW_TRACING"])
def test_langchain_model_not_inject_callback_when_disabled(monkeypatch, model_path, env_var):
    # Emulate the model serving environment
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")

    # Disable tracing
    monkeypatch.setenv(env_var, "false")

    model = create_openai_runnable()
    mlflow.langchain.save_model(model, model_path)

    loaded_model = mlflow.pyfunc.load_model(model_path)
    loaded_model.predict({"product": "shoe"})

    # Trace should be logged to the inference table
    from mlflow.tracing.export.inference_table import _TRACE_BUFFER

    assert _TRACE_BUFFER == {}


@pytest.mark.parametrize(
    "chain_path",
    [
        os.path.abspath("tests/langchain/sample_code/no_config/chain.py"),
        "tests/langchain/../langchain/sample_code/no_config/chain.py",
    ],
)
def test_save_model_as_code_correct_streamable(chain_model_signature, chain_path):
    input_example = {"messages": [{"role": "user", "content": "Who owns MLflow?"}]}
    answer = "Databricks"
    artifact_path = "model_path"
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain_path,
            name=artifact_path,
            signature=chain_model_signature,
            input_example=input_example,
        )

    assert model_info.flavors["langchain"]["streamable"] is True
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    with mock.patch("time.time", return_value=1677858242):
        assert pyfunc_loaded_model._model_impl._predict_with_callbacks(input_example) == {
            "id": None,
            "object": "chat.completion",
            "created": 1677858242,
            "model": "",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Databricks",
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

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    # avoid minor diff of created time in the response
    prediction_result = json.loads(response.content.decode("utf-8"))
    prediction_result[0]["created"] = 123
    expected_prediction = try_transform_response_to_chat_format(answer)
    expected_prediction["created"] = 123
    assert prediction_result == [expected_prediction]

    pyfunc_model_path = _download_artifact_from_uri(model_info.model_uri)
    reloaded_model = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    assert reloaded_model.resources["databricks"] == {
        "serving_endpoint": [{"name": "fake-endpoint"}]
    }


def test_save_load_langchain_binding(fake_chat_model):
    runnable_binding = RunnableBinding(bound=fake_chat_model, kwargs={"stop": ["-"]})
    model = runnable_binding | StrOutputParser()
    assert model.invoke("Say something") == "Databricks"

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            model, name="model_path", input_example="Say something"
        )
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.first.kwargs == {"stop": ["-"]}
    assert loaded_model.invoke("hello") == "Databricks"
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict("hello") == ["Databricks"]

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": ["Databricks"]
    }


def test_save_load_langchain_binding_llm_with_tool():
    from langchain_core.tools import tool

    # We need to use ChatOpenAI from langchain_openai as community one does not support bind_tools
    from langchain_openai import ChatOpenAI

    @tool
    def add(a: int, b: int) -> int:
        """Adds a and b.

        Args:
            a: first int
            b: second int
        """
        return a + b

    runnable_binding = ChatOpenAI(temperature=0.9).bind_tools([add])
    model = runnable_binding | StrOutputParser()
    expected_output = '[{"role": "user", "content": "hello"}]'
    assert model.invoke("hello") == expected_output

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(model, name="model_path", input_example="hello")

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke("hello") == expected_output
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict("hello") == [expected_output]


def test_langchain_bindings_save_load_with_config_and_types(fake_chat_model):
    class CustomCallbackHandler(BaseCallbackHandler):
        def __init__(self):
            self.count = 0

        def on_chain_start(
            self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
        ) -> None:
            self.count += 1

        def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
            self.count += 1

    model = fake_chat_model | StrOutputParser()
    callback = CustomCallbackHandler()
    model = model.with_config(run_name="test_run", callbacks=[callback]).with_types(
        input_type=str, output_type=str
    )
    assert model.invoke("Say something") == "Databricks"
    assert callback.count == 4

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(model, name="model_path", input_example="hello")
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.config["run_name"] == "test_run"
    assert loaded_model.custom_input_type == str
    assert loaded_model.custom_output_type == str
    callback = loaded_model.config["callbacks"][0]
    assert loaded_model.invoke("hello") == "Databricks"
    assert callback.count > 8  # accumulated count (inside model logging we also call the callbacks)
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict("hello") == ["Databricks"]

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": ["Databricks"]
    }


def test_langchain_2_12_model_loads():
    TEST_DIR = "tests"
    TEST_MLFLOW_12_2_LANGCHAIN_MODEL = os.path.join(
        TEST_DIR, "resources", "example_mlflow_2.12_langchain_model"
    )

    model = mlflow.langchain.load_model(TEST_MLFLOW_12_2_LANGCHAIN_MODEL)
    pyfunc_model = mlflow.pyfunc.load_model(TEST_MLFLOW_12_2_LANGCHAIN_MODEL)
    assert (
        model.invoke({"messages": [{"role": "user", "content": "Who owns MLflow?"}]})
        == "Databricks"
    )
    output = pyfunc_model.predict({"messages": [{"role": "user", "content": "Who owns MLflow?"}]})
    assert output[0]["choices"][0]["message"]["content"] == "Databricks"


@pytest.mark.parametrize(
    "chain_path",
    [
        os.path.abspath("tests/langchain/sample_code/chain.py"),
        "tests/langchain/../langchain/sample_code/chain.py",
    ],
)
@pytest.mark.parametrize(
    "model_config",
    [
        os.path.abspath("tests/langchain/sample_code/config.yml"),
        "tests/langchain/../langchain/sample_code/config.yml",
    ],
)
def test_load_chain_with_model_config_overrides_saved_config(
    chain_model_signature, chain_path, model_config
):
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is a good name for a company that makes MLflow?",
            }
        ]
    }
    artifact_path = "model_path"
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain_path,
            name=artifact_path,
            signature=chain_model_signature,
            input_example=input_example,
            model_config=model_config,
        )

    with mock.patch("mlflow.langchain.model._load_model_code_path") as load_model_code_path_mock:
        mlflow.pyfunc.load_model(model_info.model_uri, model_config={"embedding_size": 2})
        args, kwargs = load_model_code_path_mock.call_args
        assert args[1] == {
            "embedding_size": 2,
            "llm_prompt_template": "Answer the following question based on the "
            "context: {context}\nQuestion: {question}",
            "not_used_array": [
                1,
                2,
                3,
            ],
            "response": "Databricks",
        }


@pytest.mark.parametrize("streamable", [True, False, None])
def test_langchain_model_streamable_param_in_log_model(streamable, fake_chat_model):
    prompt = ChatPromptTemplate.from_template("What's your favorite {industry} company?")
    chain = prompt | fake_chat_model | StrOutputParser()

    runnable = RunnableParallel({"llm": lambda _: "completion"})

    llm = ChatOpenAI(temperature=0.9)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    for model in [chain, runnable, llm_chain]:
        with mock.patch("mlflow.langchain.model._save_model"), mlflow.start_run():
            model_info = mlflow.langchain.log_model(
                model,
                name="model",
                streamable=streamable,
                pip_requirements=[],
            )

            expected = (streamable is None) or streamable
            assert model_info.flavors["langchain"]["streamable"] is expected


@pytest.fixture
def model_type(request):
    return lc_runnables_types()[request.param]


@pytest.mark.parametrize("streamable", [True, False, None])
@pytest.mark.parametrize("model_type", range(len(lc_runnables_types())), indirect=True)
def test_langchain_model_streamable_param_in_log_model_for_lc_runnable_types(
    streamable, model_type
):
    with mock.patch("mlflow.langchain.model._save_model"), mlflow.start_run():
        model = mock.MagicMock(spec=model_type)
        assert hasattr(model, "stream") is True
        model_info = mlflow.langchain.log_model(
            model,
            name="model",
            streamable=streamable,
            pip_requirements=[],
        )

        expected = (streamable is None) or streamable
        assert model_info.flavors["langchain"]["streamable"] is expected

        del model.stream
        assert hasattr(model, "stream") is False
        model_info = mlflow.langchain.log_model(
            model,
            name="model",
            streamable=streamable,
            pip_requirements=[],
        )
        assert model_info.flavors["langchain"]["streamable"] is bool(streamable)


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.1.20"), reason="feature not existing"
)
def test_agent_executor_model_with_messages_input():
    question = {"messages": [{"role": "user", "content": "Who owns MLflow?"}]}

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            os.path.abspath("tests/langchain/agent_executor/chain.py"),
            name="model_path",
            input_example=question,
            model_config=os.path.abspath("tests/langchain/agent_executor/config.yml"),
        )
    native_model = mlflow.langchain.load_model(model_info.model_uri)
    assert native_model.invoke(question)["output"] == "Databricks"
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    # TODO: in the future we should fix this and output shouldn't be wrapped
    # The result is wrapped in a list because during signature enforcement we convert
    # input data to pandas dataframe, then inside _convert_llm_input_data
    # we convert pandas dataframe back to records, and a single row will be
    # wrapped inside a list.
    assert pyfunc_model.predict(question) == ["Databricks"]

    # Test stream output
    response = pyfunc_model.predict_stream(question)
    assert inspect.isgenerator(response)

    expected_response = [
        {
            "output": "Databricks",
            "messages": [
                {
                    "additional_kwargs": {},
                    "content": "Databricks",
                    "example": False,
                    "id": None,
                    "invalid_tool_calls": [],
                    "name": None,
                    "response_metadata": {},
                    "tool_calls": [],
                    "type": "ai",
                }
            ],
        }
    ]
    if Version(langchain.__version__) >= Version("0.2.0"):
        expected_response[0]["messages"][0]["usage_metadata"] = None
    assert list(response) == expected_response


def test_signature_inference_succeeds_with_any_type(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLFLOW_TESTING", "false")

    model = RunnableLambda(lambda x: x)

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            model,
            name="model",
            input_example={"chat": []},
        )

    schema = Schema([ColSpec(AnyType(), name="chat")])
    assert model_info.signature.inputs == schema
    assert model_info.signature.outputs == schema


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.2.0"),
    reason="Configurable fields are not supported correctly in old versions",
)
def test_invoking_model_with_params():
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            os.path.abspath("tests/langchain/sample_code/model_with_config.py"),
            name="model",
        )
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    data = {"x": 0}
    pyfunc_model.predict(data)
    params = {"config": {"temperature": 3.0}}
    with mock.patch("mlflow.pyfunc._validate_prediction_input", return_value=(data, params)):
        # This proves the temperature is passed to the model
        with pytest.raises(MlflowException, match=r"Input should be less than or equal to 2"):
            pyfunc_model.predict(data=data, params=params)


def test_custom_resources(chain_model_signature, tmp_path):
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is a good name for a company that makes MLflow?",
            }
        ]
    }
    expected_resources = {
        "api_version": "1",
        "databricks": {
            "serving_endpoint": [
                {"name": "databricks-mixtral-8x7b-instruct"},
                {"name": "databricks-bge-large-en"},
                {"name": "azure-eastus-model-serving-2_vs_endpoint"},
            ],
            "vector_search_index": [{"name": "rag.studio_bugbash.databricks_docs_index"}],
            "sql_warehouse": [{"name": "testid"}],
            "function": [
                {"name": "rag.studio.test_function_a"},
                {"name": "rag.studio.test_function_b"},
            ],
        },
    }
    artifact_path = "model_path"
    chain_path = "tests/langchain/sample_code/chain.py"
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain_path,
            name=artifact_path,
            signature=chain_model_signature,
            input_example=input_example,
            model_config="tests/langchain/sample_code/config.yml",
            resources=[
                DatabricksServingEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct"),
                DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
                DatabricksServingEndpoint(endpoint_name="azure-eastus-model-serving-2_vs_endpoint"),
                DatabricksVectorSearchIndex(index_name="rag.studio_bugbash.databricks_docs_index"),
                DatabricksSQLWarehouse(warehouse_id="testid"),
                DatabricksFunction(function_name="rag.studio.test_function_a"),
                DatabricksFunction(function_name="rag.studio.test_function_b"),
            ],
        )

        model_path = _download_artifact_from_uri(model_info.model_uri)
        reloaded_model = Model.load(os.path.join(model_path, "MLmodel"))
        assert reloaded_model.resources == expected_resources

    yaml_file = tmp_path.joinpath("resources.yaml")
    with open(yaml_file, "w") as f:
        f.write(
            """
            api_version: "1"
            databricks:
                vector_search_index:
                - name: rag.studio_bugbash.databricks_docs_index
                serving_endpoint:
                - name: databricks-mixtral-8x7b-instruct
                - name: databricks-bge-large-en
                - name: azure-eastus-model-serving-2_vs_endpoint
                sql_warehouse:
                - name: testid
                function:
                - name: rag.studio.test_function_a
                - name: rag.studio.test_function_b
            """
        )

    artifact_path_2 = "model_path_2"
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain_path,
            name=artifact_path_2,
            signature=chain_model_signature,
            input_example=input_example,
            model_config="tests/langchain/sample_code/config.yml",
            resources=yaml_file,
        )

        model_path = _download_artifact_from_uri(model_info.model_uri)
        reloaded_model = Model.load(os.path.join(model_path, "MLmodel"))
        assert reloaded_model.resources == expected_resources


def chain_accepts_list_messages():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a chatbot that can answer questions about Databricks."),
            ("user", "{question}"),
        ]
    )
    fake_chat_model = get_fake_chat_model()
    return prompt | fake_chat_model | StrOutputParser()


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.1.20"), reason="feature not existing"
)
@pytest.mark.parametrize(
    ("model", "should_convert", "input_example", "needs_env_var"),
    [
        (
            chain_accepts_list_messages(),
            True,
            {"messages": [{"role": "user", "content": "Hello"}]},
            False,
        ),
        (
            # This model is an example when the model expects a chat request
            # format input, but the input should not be converted to List[BaseMessage]
            RunnablePassthrough.assign(problem=lambda x: x["messages"][-1]["content"])
            | itemgetter("problem"),
            False,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Databricks",
                    }
                ]
            },
            True,
        ),
    ],
)
def test_pyfunc_converts_chat_request_correctly(
    model, should_convert, input_example, needs_env_var, monkeypatch
):
    request = (
        transform_request_json_for_chat_if_necessary(model, input_example)
        if should_convert
        else input_example
    )
    assert model.invoke(request) == "Databricks"

    if needs_env_var:
        monkeypatch.setenv(
            MLFLOW_CONVERT_MESSAGES_DICT_FOR_LANGCHAIN.name,
            str(should_convert),
        )
    # pyfunc model can accepts chat request format even the chain
    # itself does not accept it, but we need to use the correct
    # input example to infer model signature
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            model,
            name="model",
            input_example=input_example,
        )
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    result = pyfunc_model.predict(input_example)
    if should_convert:
        # output are converted to chatResponse format if input is converted
        assert result[0]["choices"][0]["message"]["content"] == "Databricks"
    else:
        assert result == ["Databricks"]

    # Test stream output
    response = pyfunc_model.predict_stream(input_example)
    assert inspect.isgenerator(response)

    if should_convert:
        assert list(response)[0]["choices"][0]["delta"]["content"] == "Databricks"
    else:
        assert list(response) == ["Databricks"], list(response)


def test_log_langchain_model_with_prompt():
    mlflow.register_prompt(
        name="qa_prompt",
        template="What is a good name for a company that makes {{product}}?",
        commit_message="Prompt for generating company names",
    )
    mlflow.set_prompt_alias("qa_prompt", alias="production", version=1)

    mlflow.register_prompt(name="another_prompt", template="Hi")

    # If the model code involves `mlflow.load_prompt()` call, the prompt version
    # should be automatically logged to the Run
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            os.path.abspath("tests/langchain/sample_code/chain_with_mlflow_prompt.py"),
            name="model",
            # Manually associate another prompt
            prompts=["prompts:/another_prompt/1"],
        )

    # Check that prompts were linked to the run via the linkedPrompts tag
    from mlflow.prompt.constants import LINKED_PROMPTS_TAG_KEY

    run = mlflow.MlflowClient().get_run(model_info.run_id)
    linked_prompts_tag = run.data.tags.get(LINKED_PROMPTS_TAG_KEY)
    assert linked_prompts_tag is not None

    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 2
    assert {p["name"] for p in linked_prompts} == {"qa_prompt", "another_prompt"}

    prompt = mlflow.load_prompt("qa_prompt", 1)
    assert prompt.aliases == ["production"]

    prompt = mlflow.load_prompt("another_prompt", 1)

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    response = pyfunc_model.predict({"product": "shoe"})
    # Fake OpenAI server echo the input
    assert (
        response
        == '[{"role": "user", "content": "What is a good name for a company that makes shoe?"}]'
    )


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.2.0"),
    reason="Feature not existing",
)
def test_predict_with_callbacks_with_tracing(monkeypatch):
    # Simulate the model serving environment
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")
    monkeypatch.setenv("ENABLE_MLFLOW_TRACING", "true")
    mlflow.tracing.reset()

    model_info = mlflow.langchain.log_model(
        os.path.abspath("tests/langchain/sample_code/workflow.py"),
        name="model_path",
        input_example={"messages": [{"role": "user", "content": "What is MLflow?"}]},
    )
    # serving environment only reads from this environment variable
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", mlflow.last_logged_model().experiment_id)

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)

    request_id = "mock_request_id"
    tracer = MlflowLangchainTracer(prediction_context=Context(request_id))
    input_example = {"messages": [{"role": "user", "content": TEST_CONTENT}]}

    with mock.patch("mlflow.tracing.client.TracingClient.start_trace") as mock_start_trace:
        pyfunc_model._model_impl._predict_with_callbacks(
            data=input_example, callback_handlers=[tracer]
        )
        mlflow.flush_trace_async_logging()
        mock_start_trace.assert_called_once()
        trace_info = mock_start_trace.call_args[0][0]
        assert trace_info.client_request_id == request_id
        assert trace_info.request_metadata[TraceMetadataKey.MODEL_ID] == model_info.model_id
