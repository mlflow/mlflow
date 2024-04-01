import os
import re
from operator import itemgetter
from typing import Any, List, Optional
from unittest import mock

import pandas as pd
import pytest
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from test_langchain_model_export import FAISS, DeterministicDummyEmbeddings

import mlflow
from mlflow import MlflowClient
from mlflow.langchain._langchain_autolog import (
    INFERENCE_FILE_NAME,
    UNSUPPORT_LOG_MODEL_MESSAGE,
    _combine_input_and_output,
)
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

from langchain_community.callbacks.mlflow_callback import (
    get_text_complexity_metrics,
    mlflow_callback_metrics,
)

MLFLOW_CALLBACK_METRICS = mlflow_callback_metrics()
TEXT_COMPLEXITY_METRICS = get_text_complexity_metrics()


def get_mlflow_callback_artifacts(
    contains_llm=True,
    contains_on_text_action=True,
    llm_new_token=False,
    contains_chain=False,
    contains_tool=False,
    contains_agent=False,
    contains_agent_action=False,
    contains_retriever=False,
):
    artifacts = [
        "table_action_records.html",
    ]
    if contains_llm:
        artifacts += [
            "chat_html.html",
            "table_session_analysis.html",
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
    if contains_retriever:
        artifacts += [
            re.compile(r"retriever_start_\d+\.json"),
            re.compile(r"retriever_end_\d+\.json"),
        ]
    return artifacts


def get_mlflow_model(artifact_uri, model_subpath=MODEL_DIR):
    model_conf_path = os.path.join(artifact_uri, model_subpath, "MLmodel")
    return Model.load(model_conf_path)


def get_artifacts(run_id):
    client = MlflowClient()
    artifacts = client.list_artifacts(run_id)
    artifacts_folder = [x for x in artifacts if x.path.split("/")[-1].startswith("artifacts-")][0]
    artifacts = client.list_artifacts(run_id, artifacts_folder.path)
    return [x.path.split("/")[-1] for x in artifacts]


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


def create_retriever(tmp_path):
    # Create the vector db, persist the db to a local fs folder
    loader = TextLoader("tests/langchain/state_of_the_union.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = DeterministicDummyEmbeddings(size=5)
    db = FAISS.from_documents(docs, embeddings)
    persist_dir = str(tmp_path / "faiss_index")
    db.save_local(persist_dir)
    query = "What did the president say about Ketanji Brown Jackson"
    return db.as_retriever(), query


def create_fake_chat_model():
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.chat_models.base import SimpleChatModel
    from langchain.schema.messages import BaseMessage

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

    return FakeChatModel()


def create_runnable_sequence():
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

    chat_model = create_fake_chat_model()
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
    mlflow.langchain.autolog(log_models=True, extra_tags={"test_tag": "test"})
    run = mlflow.start_run()
    with _mock_request(return_value=_mock_chat_completion_response()):
        model = create_openai_llmchain()
        model.invoke("MLflow")
    assert mlflow.active_run() is not None
    assert MlflowClient().get_run(run.info.run_id).data.metrics != {}
    assert MlflowClient().get_run(run.info.run_id).data.tags["test_tag"] == "test"
    assert MlflowClient().get_run(run.info.run_id).data.tags["mlflow.autologging"] == "langchain"
    mlflow.end_run()


def test_autolog_manage_run_no_active_run():
    mlflow.langchain.autolog(log_models=True, extra_tags={"test_tag": "test"})
    assert mlflow.active_run() is None
    with _mock_request(return_value=_mock_chat_completion_response()):
        model = create_openai_llmchain()
        model.invoke("MLflow")
    assert mlflow.active_run() is None
    run = MlflowClient().get_run(model.run_id)
    assert run.data.metrics != {}
    assert run.data.tags["test_tag"] == "test"
    assert run.data.tags["mlflow.autologging"] == "langchain"


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
        artifacts = get_artifacts(run.info.run_id)
        for artifact_name in get_mlflow_callback_artifacts():
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


def test_llmchain_autolog_log_inputs_outputs():
    mlflow.langchain.autolog(log_models=True, log_inputs_outputs=True)
    question = {"product": "MLflow"}
    answer = {"product": "MLflow", "text": TEST_CONTENT}
    with _mock_request(return_value=_mock_chat_completion_response()):
        model = create_openai_llmchain()
        with mlflow.start_run() as run:
            model.invoke(question)
        loaded_table = mlflow.load_table(INFERENCE_FILE_NAME, run_ids=[run.info.run_id])
        loaded_dict = loaded_table.to_dict("records")
        assert len(loaded_dict) == 1
        assert loaded_dict[0]["input-product"] == question["product"]
        assert loaded_dict[0]["output-text"] == answer["text"]
        session_id = loaded_dict[0]["session_id"]

        # inference history is appended to the same table
        with mlflow.start_run(run.info.run_id):
            model.invoke(question)
        with mlflow.start_run():
            model.invoke(question)
        model.invoke(question)
        loaded_table = mlflow.load_table(INFERENCE_FILE_NAME, run_ids=[run.info.run_id])
        loaded_dict = loaded_table.to_dict("records")
        assert (
            loaded_dict
            == [
                {
                    "input-product": question["product"],
                    "output-product": answer["product"],
                    "output-text": answer["text"],
                    "session_id": session_id,
                }
            ]
            * 4
        )

        # A different inference session adds a different session_id
        loaded_model = mlflow.langchain.load_model(f"runs:/{run.info.run_id}/model")
        loaded_model.invoke(question)
        loaded_table = mlflow.load_table(INFERENCE_FILE_NAME, run_ids=[run.info.run_id])
        loaded_dict = loaded_table.to_dict("records")
        assert len(loaded_dict) == 5
        new_session_id = loaded_dict[-1]["session_id"]
        assert new_session_id != session_id


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

        artifacts = get_artifacts(run.info.run_id)
        for artifact_name in get_mlflow_callback_artifacts(
            contains_agent=True, contains_chain=True
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


def test_agent_autolog_log_inputs_outputs():
    mlflow.langchain.autolog(log_inputs_outputs=True)
    model, input, mock_response = create_openai_llmagent()
    output = {"output": TEST_CONTENT}
    with _mock_request(return_value=_MockResponse(200, mock_response)):
        with mlflow.start_run() as run:
            assert model(input, return_only_outputs=True) == output
        loaded_table = mlflow.load_table(INFERENCE_FILE_NAME, run_ids=[run.info.run_id])
        loaded_dict = loaded_table.to_dict("records")
        assert len(loaded_dict) == 1
        assert loaded_dict[0]["input-input"] == input["input"]
        assert loaded_dict[0]["output-output"] == output["output"]
        session_id = loaded_dict[0]["session_id"]

        with mlflow.start_run(run.info.run_id):
            model.invoke(input, return_only_outputs=True)
        loaded_table = mlflow.load_table(INFERENCE_FILE_NAME, run_ids=[run.info.run_id])
        loaded_dict = loaded_table.to_dict("records")
        assert (
            loaded_dict
            == [
                {
                    "input-input": input["input"],
                    "output-output": output["output"],
                    "session_id": session_id,
                }
            ]
            * 2
        )


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

    artifacts = get_artifacts(run.info.run_id)
    for artifact_name in get_mlflow_callback_artifacts(contains_on_text_action=False):
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


def test_runnable_sequence_autolog_log_inputs_outputs():
    mlflow.langchain.autolog(log_inputs_outputs=True)
    chain, input_example = create_runnable_sequence()
    output = TEST_CONTENT
    with mlflow.start_run() as run:
        assert chain.invoke(input_example) == output
    loaded_table = mlflow.load_table(INFERENCE_FILE_NAME, run_ids=[run.info.run_id])
    loaded_dict = loaded_table.to_dict("records")
    assert len(loaded_dict) == 1
    assert loaded_dict[0]["input-messages"] == input_example["messages"][0]
    assert loaded_dict[0]["output"] == output
    session_id = loaded_dict[0]["session_id"]

    with mlflow.start_run(run.info.run_id):
        chain.invoke(input_example)
    loaded_table = mlflow.load_table(INFERENCE_FILE_NAME, run_ids=[run.info.run_id])
    loaded_dict = loaded_table.to_dict("records")
    assert (
        loaded_dict
        == [
            {
                "input-messages": input_example["messages"][0],
                "output": output,
                "session_id": session_id,
            }
        ]
        * 2
    )


def test_retriever_autolog(tmp_path):
    mlflow.langchain.autolog(log_models=True)
    model, query = create_retriever(tmp_path)
    with mock.patch("mlflow.langchain.log_model") as log_model_mock, mock.patch(
        "mlflow.langchain._langchain_autolog._logger.info"
    ) as logger_mock:
        model.get_relevant_documents(query)
        log_model_mock.assert_not_called()
        logger_mock.assert_called_once_with(UNSUPPORT_LOG_MODEL_MESSAGE)


def test_retriever_metrics_and_artifacts(tmp_path):
    mlflow.langchain.autolog()
    model, query = create_retriever(tmp_path)
    with mlflow.start_run() as run:
        model.get_relevant_documents(query)
    client = MlflowClient()
    metrics = client.get_run(run.info.run_id).data.metrics
    for metric_key in MLFLOW_CALLBACK_METRICS:
        assert metric_key in metrics

    artifacts = get_artifacts(run.info.run_id)
    for artifact_name in get_mlflow_callback_artifacts(
        contains_llm=False,
        contains_on_text_action=False,
        contains_retriever=True,
    ):
        if isinstance(artifact_name, str):
            assert artifact_name in artifacts
        else:
            assert any(artifact_name.match(artifact) for artifact in artifacts)
    assert mlflow.active_run() is None


def test_retriever_autlog_inputs_outputs(tmp_path):
    mlflow.langchain.autolog(log_inputs_outputs=True)
    model, query = create_retriever(tmp_path)
    with mlflow.start_run() as run:
        documents = model.get_relevant_documents(query)
        documents = [
            {"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents
        ]
    loaded_table = mlflow.load_table(INFERENCE_FILE_NAME, run_ids=[run.info.run_id])
    loaded_dict = loaded_table.to_dict("records")
    assert len(loaded_dict) == 1
    assert loaded_dict[0]["input"] == query
    assert loaded_dict[0]["output"] == documents
    session_id = loaded_dict[0]["session_id"]

    with mlflow.start_run(run.info.run_id):
        model.get_relevant_documents(query)
    loaded_table = mlflow.load_table(INFERENCE_FILE_NAME, run_ids=[run.info.run_id])
    loaded_dict = loaded_table.to_dict("records")
    assert loaded_dict == [{"input": query, "output": documents, "session_id": session_id}] * 2


def test_unsupported_log_model_models_autolog(tmp_path):
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnablePassthrough

    mlflow.langchain.autolog(log_models=True)
    retriever, _ = create_retriever(tmp_path)
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question based on the context: {context}\nQuestion: {question}"
    )
    chat_model = create_fake_chat_model()
    retrieval_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )
    question = "What is a good name for a company that makes MLflow?"
    with _mock_request(return_value=_mock_chat_completion_response()), mock.patch(
        "mlflow.langchain._langchain_autolog._logger.info"
    ) as logger_mock, mock.patch("mlflow.langchain.log_model") as log_model_mock:
        assert retrieval_chain.invoke(question) == TEST_CONTENT
        logger_mock.assert_called_once_with(UNSUPPORT_LOG_MODEL_MESSAGE)
        log_model_mock.assert_not_called()


@pytest.mark.parametrize(
    ("input", "output", "expected"),
    [
        ("data", "result", {"input": ["data"], "output": ["result"], "session_id": ["session_id"]}),
        (
            "data",
            {"result": "some_result"},
            {"input": ["data"], "output-result": "some_result", "session_id": ["session_id"]},
        ),
        (
            "data",
            ["some_result"],
            {"input": ["data"], "output": ["some_result"], "session_id": ["session_id"]},
        ),
        (
            ["data"],
            "some_result",
            {"input": ["data"], "output": ["some_result"], "session_id": ["session_id"]},
        ),
        (
            {"data": "some_data"},
            ["some_result"],
            {"input-data": "some_data", "output": ["some_result"], "session_id": ["session_id"]},
        ),
        (
            {"data": "some_data"},
            {"result": "some_result"},
            {
                "input-data": "some_data",
                "output-result": "some_result",
                "session_id": ["session_id"],
            },
        ),
        (
            [{"data": "some_data"}],
            {"result": "some_result"},
            {
                "input": [{"data": "some_data"}],
                "output-result": "some_result",
                "session_id": ["session_id"],
            },
        ),
    ],
)
def test_combine_input_and_output(input, output, expected):
    assert (
        _combine_input_and_output(input, output, session_id="session_id", func_name="") == expected
    )
    with mlflow.start_run() as run:
        mlflow.log_table(expected, INFERENCE_FILE_NAME, run.info.run_id)
    loaded_table = mlflow.load_table(INFERENCE_FILE_NAME, run_ids=[run.info.run_id])
    pdf = pd.DataFrame(expected)
    pd.testing.assert_frame_equal(loaded_table, pdf)
