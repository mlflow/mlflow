import os
import re
from unittest import mock

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

import mlflow
from mlflow import MlflowClient
from mlflow.models import Model
from mlflow.models.signature import infer_signature
from mlflow.models.utils import _read_example
from mlflow.utils.openai_utils import (
    TEST_CONTENT,
    _mock_chat_completion_response,
    _mock_request,
)

MODEL_DIR = "model"
TEST_CONTENT = "test"
try:
    from langchain_community.callbacks.mlflow_callback import (
        get_text_complexity_metrics,
        mlflow_callback_metrics,
    )

    MLFLOW_CALLBACK_METRICS = mlflow_callback_metrics()
    TEXT_COMPLEXITY_METRICS = get_text_complexity_metrics()
# TODO: remove this when langchain_community change is merged
except ImportError:
    MLFLOW_CALLBACK_METRICS = [
        "step",
        "starts",
        "ends",
        "errors",
        "text_ctr",
        "chain_starts",
        "chain_ends",
        "llm_starts",
        "llm_ends",
        "llm_streams",
        "tool_starts",
        "tool_ends",
        "agent_ends",
    ]
    TEXT_COMPLEXITY_METRICS = [
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "smog_index",
        "coleman_liau_index",
        "automated_readability_index",
        "dale_chall_readability_score",
        "difficult_words",
        "linsear_write_formula",
        "gunning_fog",
        "fernandez_huerta",
        "szigriszt_pazos",
        "gutierrez_polini",
        "crawford",
        "gulpease_index",
        "osman",
    ]


def get_mlflow_callback_artifacts(
    contains_llm=False,
    llm_new_token=False,
    contains_chain=False,
    contains_tool=False,
    contains_agent=False,
):
    artifacts = [
        "chat_html.html",
        "table_action_records.html",
        "table_session_analysis.html",
        re.compile(r"on_text_\d+\.json"),
    ]
    if contains_llm:
        artifacts += [
            re.compile(r"llm_start_\d+_prompt_\d+\.json"),
            re.compile(r"llm_end_\d+_generation_\d+\.json"),
        ]
    if llm_new_token:
        artifacts += [re.compile(r"llm_new_tokens_\d+\.json")]
    if contains_chain:
        artifacts += [
            re.compile(r"chain_start_\d+\.json"),
            re.compile(r"chain_end_\d+\.json"),
        ]
    if contains_tool:
        artifacts += [
            re.compile(r"tool_start_\d+\.json"),
            re.compile(r"tool_end_\d+\.json"),
        ]
    if contains_agent:
        artifacts += [
            re.compile(r"agent_start_\d+\.json"),
            re.compile(r"agent_finish_\d+\.json"),
        ]
    return artifacts


def get_mlflow_model(artifact_uri, model_subpath=MODEL_DIR):
    model_conf_path = os.path.join(artifact_uri, model_subpath, "MLmodel")
    return Model.load(model_conf_path)


def create_openai_llmchain():
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    return LLMChain(llm=llm, prompt=prompt)


def test_llmchain_autolog():
    mlflow.langchain.autolog(log_models=True)
    question = "MLflow"
    answer = {"product": "MLflow", "text": TEST_CONTENT}
    with _mock_request(return_value=_mock_chat_completion_response()):
        model = create_openai_llmchain()
        with mock.patch("mlflow.langchain.log_model") as log_model_mock:
            assert model.invoke(question) == answer
            # Call twice to test that the model is only logged once
            assert model.invoke(question) == answer
            log_model_mock.assert_called_once()


def test_llmchain_autolog_with_registered_model_name():
    registered_model_name = "llmchain"
    mlflow.langchain.autolog(log_models=True, registered_model_name=registered_model_name)
    with _mock_request(return_value=_mock_chat_completion_response()):
        model = create_openai_llmchain()
        model.invoke("MLflow")
        registered_model = MlflowClient().get_registered_model(registered_model_name)
        assert registered_model.name == registered_model_name


def test_llmchain_autolog_metrics():
    mlflow.langchain.autolog()
    with _mock_request(return_value=_mock_chat_completion_response()):
        model = create_openai_llmchain()
        with mlflow.start_run() as run:
            model.invoke("MLflow")
            run_id = run.info.run_id
        client = MlflowClient()
        metrics = client.get_run(run_id).data.metrics
        for metric_key in MLFLOW_CALLBACK_METRICS + TEXT_COMPLEXITY_METRICS:
            assert metric_key in metrics
    assert mlflow.active_run() is None


def test_llmchain_autolog_artifacts():
    mlflow.langchain.autolog()
    with _mock_request(return_value=_mock_chat_completion_response()):
        model = create_openai_llmchain()
        with mlflow.start_run() as run:
            model.invoke("MLflow")
        client = MlflowClient()
        artifacts = client.list_artifacts(run.info.run_id)
        artifacts = [x.path.split(os.sep)[-1] for x in artifacts]
        for artifact_name in get_mlflow_callback_artifacts(contains_llm=True):
            if isinstance(artifact_name, str):
                assert artifact_name in artifacts
            else:
                assert any(artifact_name.match(artifact) for artifact in artifacts)


def test_loaded_llmchain_autolog():
    mlflow.langchain.autolog(log_models=True, log_input_examples=True)

    model = create_openai_llmchain()
    question = {"product": "MLflow"}
    answer = {"product": "MLflow", "text": TEST_CONTENT}
    with _mock_request(return_value=_mock_chat_completion_response()):
        with mlflow.start_run() as run:
            assert model.invoke(question) == answer
        with mock.patch("mlflow.langchain.log_model") as log_model_mock:
            loaded_model = mlflow.langchain.load_model(f"runs:/{run.info.run_id}/model")
            assert loaded_model.invoke(question) == answer
            log_model_mock.assert_not_called()

            mlflow_model = get_mlflow_model(run.info.artifact_uri)
            model_path = os.path.join(run.info.artifact_uri, MODEL_DIR)
            input_example = _read_example(mlflow_model, model_path)
            assert input_example.to_dict("records") == [question]

            pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
            # _TestLangChainWrapper mocks this result
            assert pyfunc_model.predict(question) == [TEST_CONTENT]
            log_model_mock.assert_not_called()

            signature = mlflow_model.signature
            assert signature == infer_signature(question, [TEST_CONTENT])


def test_llmchain_autolog_log_inference_history(tmp_path):
    mlflow.langchain.autolog(log_inference_history=True)
    question = {"product": "MLflow"}
    answer = {"product": "MLflow", "text": TEST_CONTENT}
    with _mock_request(return_value=_mock_chat_completion_response()):
        model = create_openai_llmchain()
        with mlflow.start_run() as run:
            model.invoke(question)
        loaded_table = mlflow.load_table("inference_history.json", run_ids=[run.info.run_id])
        loaded_dict = loaded_table.to_dict("records")
        assert loaded_dict == [{"input": question, "output": answer}]

        # with or without a new run wrapper, the inference history is appended
        # to the same table stored in last run
        with mlflow.start_run():
            model.invoke(question)
        model.invoke(question)
        loaded_table = mlflow.load_table("inference_history.json", run_ids=[run.info.run_id])
        loaded_dict = loaded_table.to_dict("records")
        assert loaded_dict == [{"input": question, "output": answer}] * 3


# def test_runnable_sequence_autolog_model():
#     from langchain.schema.output_parser import StrOutputParser
#     from langchain.schema.runnable import RunnableLambda

#     mlflow.langchain.autolog(log_models=True)

#     from mlflow.langchain.utils import _fake_simple_chat_model

#     prompt_with_history_str = """
#     Here is a history between you and a human: {chat_history}

#     Now, please answer this question: {question}
#     """

#     prompt_with_history = PromptTemplate(
#         input_variables=["chat_history", "question"], template=prompt_with_history_str
#     )

#     chat_model = _fake_simple_chat_model()()

#     def extract_question(input):
#         return input[-1]["content"]

#     def extract_history(input):
#         return input[:-1]

#     chain_with_history = (
#         {
#             "question": itemgetter("messages") | RunnableLambda(extract_question),
#             "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
#         }
#         | prompt_with_history
#         | chat_model
#         | StrOutputParser()
#     )
#     input_example = {"messages": [{"role": "user", "content": "Who owns MLflow?"}]}
#     assert chain_with_history.invoke(input_example) == "Databricks"
#     assert chain_with_history.invoke(input_example) == "Databricks"
