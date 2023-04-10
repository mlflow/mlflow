import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import logging
from typing import Any, List, Mapping, Optional, Dict

from langchain.llms.base import LLM
from langchain.chains.base import Chain

from pyspark.sql import SparkSession

# from pyspark.sql.types import (
#     ArrayType,
#     StringType,
#     StructType,
#     StructField,
# )
import langchain
import pytest

import mlflow

# mock openai
# model hugging face hub model?


@pytest.fixture
def model_path(tmp_path):
    return tmp_path.joinpath("model")


def create_openai_llmchain():
    os.environ[
        "OPENAI_API_KEY"
    ] = "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"  # no credit
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    return LLMChain(llm=llm, prompt=prompt)


def create_model(llm_type):
    if llm_type == "openai":
        return create_openai_llmchain()
    if llm_type == "huggingfacehub":
        return 2
    if llm_type == "qachain":
        return 3
    if llm_type == "agent":
        return 4
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
    assert "['product': string]" == logged_model.signature.inputs
    assert "['text': string]" in logged_model.signature.outputs

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

    assert type(loaded_model) == langchain.chains.llm.LLMChain
    assert type(loaded_model.llm) == langchain.llms.openai.OpenAI
    assert type(loaded_model.prompt) == langchain.prompts.PromptTemplate
    assert loaded_model.prompt.template == "What is a good name for a company that makes {product}?"


def test_pyfunc_spark_udf_with_langchain_model(spark):
    model = create_model("openai")
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, "langchain_model")
        # todo
        loaded_model = mlflow.pyfunc.spark_udf(spark, logged_model.model_uri)

    assert "langchain" in logged_model.flavors

    assert type(loaded_model) == langchain.chains.llm.LLMChain
    assert type(loaded_model.llm) == langchain.llms.openai.OpenAI
    assert type(loaded_model.prompt) == langchain.prompts.PromptTemplate
    assert loaded_model.prompt.template == "What is a good name for a company that makes {product}?"


def test_langchain_log_openai_model_metadata():
    pass


def test_langchain_log_huggingface_hub_model_metadata():
    pass


def test_pyfunc_load_inference_multi_var_model():
    # qachain
    pass


def test_unsupported_llm_types(caplog):
    # TODO: fix this: caplog cannot capture the warning
    prompt = PromptTemplate(input_variables=["input"], template="{input}?")
    chain = LLMChain(llm=FakeLLM(), prompt=prompt)
    with caplog.at_level(logging.WARNING):
        with mlflow.start_run():
            mlflow.langchain.log_model(chain, "fake_llm_model")
        assert (
            "MLflow does not guarantee support for LLMChains outside of HuggingFaceHub"
            in caplog.text
        )
        assert "OpenAI" in caplog.text


def test_unsupported_chain_types():
    chain = FakeChain()
    with pytest.raises(
        TypeError,
        match="MLflow langchain flavor only supports logging langchain.chains.llm.LLMChain",
    ):
        with mlflow.start_run():
            mlflow.langchain.log_model(chain, "fake_chain_model")


def test_credential_available_in_pyfunc_spark_udf():
    # create a model with fake_llm that checks for api_key before responding
    pass


def test_langchain_native_log_agent_model():
    # langchain serde does not support this well
    pass


@pytest.fixture(scope="module")
def spark():
    with SparkSession.builder.master("local[*]").getOrCreate() as s:
        yield s


# @pytest.mark.parametrize(
#     "task",
#     [
#         openai.ChatCompletion,
#         openai.ChatCompletion.OBJECT_NAME,
#     ],
# )
# def test_log_model(task):
#     with mlflow.start_run():
#         model_info = mlflow.openai.log_model(
#             model="gpt-3.5-turbo",
#             task=task,
#             artifact_path="model",
#             temperature=0.9,
#             messages=[{"role": "user", "content": "What is MLflow?"}],
#         )

#     loaded_model = mlflow.openai.load_model(model_info.model_uri)
#     assert loaded_model["model"] == "gpt-3.5-turbo"
#     assert loaded_model["task"] == "chat.completions"
#     assert loaded_model["temperature"] == 0.9
#     assert loaded_model["messages"] == [{"role": "user", "content": "What is MLflow?"}]

#     with _mock_request_chat_completion() as mock:
#         completion = openai.ChatCompletion.create(
#             model=loaded_model["model"],
#             messages=loaded_model["messages"],
#         )
#         assert completion.choices[0].message.content == _TEST_CONTENT
#         mock.assert_called_once()


# def test_save_model_with_secret_scope(tmp_path, monkeypatch):
#     scope = "test"
#     monkeypatch.setenv("MLFLOW_OPENAI_SECRET_SCOPE", scope)
#     mlflow.openai.save_model(
#         model="gpt-3.5-turbo",
#         task="chat.completions",
#         path=tmp_path,
#         messages=[{"role": "user", "content": "What is MLflow?"}],
#     )
#     with tmp_path.joinpath("openai.json").open() as f:
#         creds = json.load(f)
#         assert creds == {
#             "OPENAI_API_TYPE": f"{scope}:openai_api_type",
#             "OPENAI_API_KEY": f"{scope}:openai_api_key",
#             "OPENAI_API_BASE": f"{scope}:openai_api_base",
#             "OPENAI_API_VERSION": f"{scope}:openai_api_version",
#         }


# def test_spark_udf(spark):
#     with mlflow.start_run():
#         model_info = mlflow.openai.log_model(
#             model="gpt-3.5-turbo",
#             task="chat.completions",
#             artifact_path="model",
#             messages=[{"role": "user", "content": "What is MLflow?"}],
#         )

#     loaded_model = mlflow.openai.load_model(model_info.model_uri)
#     assert loaded_model["model"] == "gpt-3.5-turbo"
#     assert loaded_model["task"] == "chat.completions"
#     assert loaded_model["messages"] == [{"role": "user", "content": "What is MLflow?"}]

#     udf = mlflow.pyfunc.spark_udf(spark=spark, model_uri=model_info.model_uri,
# result_type="string")
#     schema = StructType(
#         [
#             StructField(
#                 "messages",
#                 ArrayType(
#                     StructType(
#                         [
#                             StructField("role", StringType()),
#                             StructField(
#                                 "content",
#                                 StringType(),
#                             ),
#                         ]
#                     )
#                 ),
#             )
#         ]
#     )
#     df = spark.createDataFrame(
#         [
#             ([("user", "What is MLflow?")],),
#             ([("user", "What is Spark?")],),
#         ],
#         schema,
#     )
#     df = df.withColumn("answer", udf("messages"))
#     pdf = df.toPandas()
#     assert pdf["answer"].tolist() == [_TEST_CONTENT, _TEST_CONTENT]
