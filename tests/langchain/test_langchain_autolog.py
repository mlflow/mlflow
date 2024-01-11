import os
import re
from operator import itemgetter
from typing import Any, List, Optional
from unittest import mock

import pytest
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.signature import infer_signature
from mlflow.models.utils import _read_example
from mlflow.utils.openai_utils import (
    TEST_CONTENT,
    _mock_chat_completion_response,
    _mock_request,
    _MockResponse,
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
    contains_agent_action=False,
    contains_on_text_action=True,
):
    artifacts = [
        "chat_html.html",
        "table_action_records.html",
        "table_session_analysis.html",
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
            re.compile(r"agent_finish_\d+\.json"),
        ]
    if contains_agent_action:
        artifacts += [
            re.compile(r"agent_action_\d+\.json"),
        ]
    if contains_on_text_action:
        artifacts += [
            re.compile(r"on_text_\d+\.json"),
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


def create_openai_llmagent():
    from langchain.agents import AgentType, initialize_agent, load_tools

    llm = OpenAI(temperature=0)
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    model = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=False,
    )
    agent_input = {
        "input": "What was the high temperature in SF yesterday in Fahrenheit?"
        "What is that number raised to the .023 power?"
    }
    mock_response = {
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
    return model, agent_input, mock_response


def create_runnable_sequence():
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.chat_models.base import SimpleChatModel
    from langchain.schema.messages import BaseMessage
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnableLambda

    prompt_with_history_str = """
    Here is a history between you and a human: {chat_history}

    Now, please answer this question: {question}
    """
    prompt_with_history = PromptTemplate(
        input_variables=["chat_history", "question"], template=prompt_with_history_str
    )

    class FakeChatModel(SimpleChatModel):
        """Fake Chat Model wrapper for testing purposes."""

        def _call(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            return TEST_CONTENT

        @property
        def _llm_type(self) -> str:
            return "fake chat model"

    def extract_question(input):
        return input[-1]["content"]

    def extract_history(input):
        return input[:-1]

    chat_model = FakeChatModel()
    chain_with_history = (
        {
            "question": itemgetter("messages") | RunnableLambda(extract_question),
            "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
        }
        | prompt_with_history
        | chat_model
        | StrOutputParser()
    )
    input_example = {"messages": [{"role": "user", "content": "Who owns MLflow?"}]}
    return chain_with_history, input_example


def test_autolog_manage_run():
    mlflow.langchain.autolog(log_models=True)
    run = mlflow.start_run()
    with _mock_request(return_value=_mock_chat_completion_response()):
        model = create_openai_llmchain()
        model.invoke("MLflow")
    assert mlflow.active_run() is not None
    assert MlflowClient().get_run(run.info.run_id).data.metrics != {}
    mlflow.end_run()


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
        client = MlflowClient()
        metrics = client.get_run(run.info.run_id).data.metrics
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


def test_llmchain_autolog_log_inference_history():
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

        # inference history is appended to the same table
        with mlflow.start_run(run.info.run_id):
            model.invoke(question)
        model.invoke(question)
        loaded_table = mlflow.load_table("inference_history.json", run_ids=[run.info.run_id])
        loaded_dict = loaded_table.to_dict("records")
        assert loaded_dict == [{"input": question, "output": answer}] * 3

        with pytest.raises(MlflowException, match="Please end current run when autologging is on "):
            with mlflow.start_run():
                model.invoke(question)


def test_agent_autolog():
    mlflow.langchain.autolog(log_models=True)
    model, input, mock_response = create_openai_llmagent()
    with _mock_request(return_value=_MockResponse(200, mock_response)), mock.patch(
        "mlflow.langchain.log_model"
    ) as log_model_mock:
        # ensure __call__ is patched
        assert model(input, return_only_outputs=True) == {"output": TEST_CONTENT}
        assert model(input, return_only_outputs=True) == {"output": TEST_CONTENT}
        log_model_mock.assert_called_once()

    model, input, mock_response = create_openai_llmagent()
    with _mock_request(return_value=_MockResponse(200, mock_response)), mock.patch(
        "mlflow.langchain.log_model"
    ) as log_model_mock:
        # ensure invoke is patched
        assert model.invoke(input, return_only_outputs=True) == {"output": TEST_CONTENT}
        assert model.invoke(input, return_only_outputs=True) == {"output": TEST_CONTENT}
        log_model_mock.assert_called_once()


def test_agent_autolog_metrics_and_artifacts():
    mlflow.langchain.autolog()
    model, input, mock_response = create_openai_llmagent()
    with _mock_request(return_value=_MockResponse(200, mock_response)):
        with mlflow.start_run() as run:
            model(input)
        client = MlflowClient()
        metrics = client.get_run(run.info.run_id).data.metrics
        for metric_key in MLFLOW_CALLBACK_METRICS + TEXT_COMPLEXITY_METRICS:
            assert metric_key in metrics

        artifacts = client.list_artifacts(run.info.run_id)
        artifacts = [x.path.split(os.sep)[-1] for x in artifacts]
        for artifact_name in get_mlflow_callback_artifacts(
            contains_llm=True, contains_agent=True, contains_chain=True
        ):
            if isinstance(artifact_name, str):
                assert artifact_name in artifacts
            else:
                assert any(artifact_name.match(artifact) for artifact in artifacts)
    assert mlflow.active_run() is None


def test_loaded_agent_autolog():
    mlflow.langchain.autolog(log_models=True, log_input_examples=True)
    model, input, mock_response = create_openai_llmagent()
    with _mock_request(return_value=_MockResponse(200, mock_response)):
        with mlflow.start_run() as run:
            assert model(input, return_only_outputs=True) == {"output": TEST_CONTENT}
        with mock.patch("mlflow.langchain.log_model") as log_model_mock:
            loaded_model = mlflow.langchain.load_model(f"runs:/{run.info.run_id}/model")
            assert loaded_model(input, return_only_outputs=True) == {"output": TEST_CONTENT}
            log_model_mock.assert_not_called()

            mlflow_model = get_mlflow_model(run.info.artifact_uri)
            model_path = os.path.join(run.info.artifact_uri, MODEL_DIR)
            input_example = _read_example(mlflow_model, model_path)
            assert input_example.to_dict("records") == [input]

            pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
            assert pyfunc_model.predict(input) == [TEST_CONTENT]
            log_model_mock.assert_not_called()

            signature = mlflow_model.signature
            assert signature == infer_signature(input, [TEST_CONTENT])


def test_agent_autolog_log_inference_history():
    mlflow.langchain.autolog(log_inference_history=True)
    model, input, mock_response = create_openai_llmagent()
    output = {"output": TEST_CONTENT}
    with _mock_request(return_value=_MockResponse(200, mock_response)):
        with mlflow.start_run() as run:
            assert model(input, return_only_outputs=True) == output
        loaded_table = mlflow.load_table("inference_history.json", run_ids=[run.info.run_id])
        loaded_dict = loaded_table.to_dict("records")
        assert loaded_dict == [{"input": input, "output": output}]

        with mlflow.start_run(run.info.run_id):
            model.invoke(input, return_only_outputs=True)
        loaded_table = mlflow.load_table("inference_history.json", run_ids=[run.info.run_id])
        loaded_dict = loaded_table.to_dict("records")
        assert loaded_dict == [{"input": input, "output": output}] * 2


def test_runnable_sequence_autolog():
    mlflow.langchain.autolog(log_models=True)
    chain, input_example = create_runnable_sequence()
    with mock.patch("mlflow.langchain.log_model") as log_model_mock:
        assert chain.invoke(input_example) == TEST_CONTENT
        assert chain.invoke(input_example) == TEST_CONTENT
        log_model_mock.assert_called_once()


def test_runnable_sequence_autolog_metrics_and_artifacts():
    mlflow.langchain.autolog()
    chain, input_example = create_runnable_sequence()
    with mlflow.start_run() as run:
        chain.invoke(input_example)
    client = MlflowClient()
    metrics = client.get_run(run.info.run_id).data.metrics
    for metric_key in MLFLOW_CALLBACK_METRICS + TEXT_COMPLEXITY_METRICS:
        assert metric_key in metrics

    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = [x.path.split(os.sep)[-1] for x in artifacts]
    for artifact_name in get_mlflow_callback_artifacts(
        contains_llm=True, contains_on_text_action=False
    ):
        if isinstance(artifact_name, str):
            assert artifact_name in artifacts
        else:
            assert any(artifact_name.match(artifact) for artifact in artifacts)
    assert mlflow.active_run() is None


def test_loaded_runnable_sequence_autolog():
    mlflow.langchain.autolog(log_models=True, log_input_examples=True)
    chain, input_example = create_runnable_sequence()
    with mlflow.start_run() as run:
        assert chain.invoke(input_example) == TEST_CONTENT
    with mock.patch("mlflow.langchain.log_model") as log_model_mock:
        loaded_model = mlflow.langchain.load_model(f"runs:/{run.info.run_id}/model")
        assert loaded_model.invoke(input_example) == TEST_CONTENT
        log_model_mock.assert_not_called()

        mlflow_model = get_mlflow_model(run.info.artifact_uri)
        model_path = os.path.join(run.info.artifact_uri, MODEL_DIR)
        saved_example = _read_example(mlflow_model, model_path)
        assert saved_example.to_dict("records") == [input_example]

        pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
        assert pyfunc_model.predict(input_example) == [TEST_CONTENT]
        log_model_mock.assert_not_called()

        signature = mlflow_model.signature
        assert signature == infer_signature(input_example, [TEST_CONTENT])


def test_runnable_sequence_autolog_log_inference_history():
    mlflow.langchain.autolog(log_inference_history=True)
    chain, input_example = create_runnable_sequence()
    output = TEST_CONTENT
    with mlflow.start_run() as run:
        assert chain.invoke(input_example) == output
    loaded_table = mlflow.load_table("inference_history.json", run_ids=[run.info.run_id])
    loaded_dict = loaded_table.to_dict("records")
    assert loaded_dict == [{"input": input_example, "output": output}]

    with mlflow.start_run(run.info.run_id):
        chain.invoke(input_example)
    loaded_table = mlflow.load_table("inference_history.json", run_ids=[run.info.run_id])
    loaded_dict = loaded_table.to_dict("records")
    assert loaded_dict == [{"input": input_example, "output": output}] * 2
