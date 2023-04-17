import os
import langchain
import mlflow
import pytest
import transformers

from contextlib import contextmanager
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.llms import HuggingFacePipeline
from langchain.llms.base import LLM
from langchain.chains.base import Chain
from pyspark.sql import SparkSession
from typing import Any, List, Mapping, Optional, Dict

from mlflow.openai.utils import _mock_chat_completion_response, _mock_request, TEST_CONTENT


@contextmanager
def _mock_async_request():
    with _mock_request(return_value=_mock_chat_completion_response()) as m:
        yield m


@pytest.fixture
def model_path(tmp_path):
    return tmp_path.joinpath("model")


@pytest.fixture(scope="module")
def spark():
    with SparkSession.builder.master("local[*]").getOrCreate() as s:
        yield s


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
    os.environ["OPENAI_API_KEY"] = "foo"
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    return LLMChain(llm=llm, prompt=prompt)


def create_model(llm_type, model_path=None):
    if llm_type == "openai":
        return create_openai_llmchain()
    if llm_type == "huggingfacehub":
        return create_huggingface_model(model_path)
    if llm_type == "fake":
        return FakeLLM()
    raise NotImplementedError("This model is not supported yet.")


class FakeLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    queries: Optional[Mapping] = None

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
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

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        if self.be_correct:
            return {"bar": "baz"}
        else:
            return {"baz": "bar"}


def test_langchain_native_save_and_load_model(model_path):
    model = create_model("openai")
    mlflow.langchain.save_model(model, model_path)

    loaded_model = mlflow.langchain.load_model(model_path)
    assert type(loaded_model) == langchain.chains.llm.LLMChain
    assert type(loaded_model.llm) == langchain.llms.openai.OpenAI
    assert type(loaded_model.prompt) == langchain.prompts.PromptTemplate
    assert loaded_model.prompt.template == "What is a good name for a company that makes {product}?"


def test_langchain_native_log_and_load_model():
    model = create_model("openai")
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, "langchain_model")

    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)

    assert "langchain" in logged_model.flavors
    assert "['product': string]" == str(logged_model.signature.inputs)
    assert "['text': string]" == str(logged_model.signature.outputs)

    assert type(loaded_model) == langchain.chains.llm.LLMChain
    assert type(loaded_model.llm) == langchain.llms.openai.OpenAI
    assert type(loaded_model.prompt) == langchain.prompts.PromptTemplate
    assert loaded_model.prompt.template == "What is a good name for a company that makes {product}?"


def test_pyfunc_load_uni_var_model():
    model = create_model("openai")
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, "langchain_model")

    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)

    assert "langchain" in logged_model.flavors
    assert type(loaded_model) == mlflow.pyfunc.PyFuncModel


def test_langchain_model_predict():
    with _mock_request(return_value=_mock_chat_completion_response()):
        model = create_model("openai")
        with mlflow.start_run():
            logged_model = mlflow.langchain.log_model(model, "langchain_model")
        loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
        result = loaded_model.predict([{"product": "MLflow"}])
        assert result == [TEST_CONTENT]


def test_pyfunc_spark_udf_with_langchain_model(spark):
    model = create_model("openai")
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, "langchain_model")
    loaded_model = mlflow.pyfunc.spark_udf(spark, logged_model.model_uri, result_type="string")
    df = spark.createDataFrame([("MLflow",), ("Spark",)], ["product"])
    df = df.withColumn("answer", loaded_model())
    pdf = df.toPandas()
    assert pdf["answer"].tolist() == [TEST_CONTENT, TEST_CONTENT]


def test_langchain_log_huggingface_hub_model_metadata(model_path):
    model = create_model("huggingfacehub", model_path)
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, "langchain_model")

    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)

    assert "langchain" in logged_model.flavors
    assert "['product': string]" == str(logged_model.signature.inputs)
    assert "['text': string]" == str(logged_model.signature.outputs)

    assert type(loaded_model) == langchain.chains.llm.LLMChain
    assert type(loaded_model.llm) == langchain.llms.huggingface_pipeline.HuggingFacePipeline
    assert type(loaded_model.prompt) == langchain.prompts.PromptTemplate
    assert loaded_model.prompt.template == "What is a good name for a company that makes {product}?"


def test_unsupported_chain_types():
    chain = FakeChain()
    with pytest.raises(
        TypeError,
        match="MLflow langchain flavor only supports logging langchain.chains.llm.LLMChain",
    ):
        with mlflow.start_run():
            mlflow.langchain.log_model(chain, "fake_chain_model")
