import os
from operator import itemgetter
from typing import Any, Dict, List, Optional
from unittest import mock

import pandas as pd
import pytest
from langchain.chains.llm import LLMChain
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable.config import RunnableConfig
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.callbacks.base import (
    AsyncCallbackHandler,
    BaseCallbackHandler,
    BaseCallbackManager,
)
from test_langchain_model_export import FAISS, DeterministicDummyEmbeddings

import mlflow
from mlflow import MlflowClient
from mlflow.entities.trace_status import TraceStatus
from mlflow.langchain._langchain_autolog import (
    INFERENCE_FILE_NAME,
    UNSUPPORT_LOG_MODEL_MESSAGE,
    _combine_input_and_output,
    _resolve_tags,
)
from mlflow.langchain.langchain_tracer import MlflowLangchainTracer
from mlflow.models import Model
from mlflow.models.dependencies_schemas import DependenciesSchemasType, set_retriever_schema
from mlflow.models.signature import infer_signature
from mlflow.models.utils import _read_example
from mlflow.pyfunc.context import Context, set_prediction_context
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey, TraceTagKey
from mlflow.utils.openai_utils import (
    TEST_CONTENT,
    _mock_chat_completion_response,
    _mock_openai_arequest,
    _mock_request,
    _MockResponse,
)

from tests.tracing.helper import get_traces

MODEL_DIR = "model"
TEST_CONTENT = "test"

from langchain_community.callbacks.mlflow_callback import (
    get_text_complexity_metrics,
    mlflow_callback_metrics,
)

MLFLOW_CALLBACK_METRICS = mlflow_callback_metrics()
TEXT_COMPLEXITY_METRICS = get_text_complexity_metrics()


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


def create_retriever(tmp_path):
    # Create the vector db, persist the db to a local fs folder
    loader = TextLoader("tests/langchain/state_of_the_union.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
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
        assert model.run_id == run.info.run_id

        # The run_id should be propagated to the second call via model instance to
        # avoid duplicate logging
        model.invoke("MLflow")
        assert model.run_id == run.info.run_id

    # Active run created by an user should not be terminated
    assert mlflow.active_run() is not None
    assert run.info.status != "FINISHED"

    assert MlflowClient().get_run(run.info.run_id).data.tags["test_tag"] == "test"
    assert MlflowClient().get_run(run.info.run_id).data.tags["mlflow.autologging"] == "langchain"
    mlflow.end_run()

    traces = get_traces()
    assert len(traces) == 2
    for trace in traces:
        attrs = trace.data.spans[0].attributes
        assert attrs[SpanAttributeKey.SPAN_TYPE] == "CHAIN"
        assert attrs[SpanAttributeKey.INPUTS] == {"product": "MLflow"}
        assert attrs[SpanAttributeKey.OUTPUTS] == {"text": TEST_CONTENT}


def test_autolog_manage_run_no_active_run():
    mlflow.langchain.autolog(log_models=True, extra_tags={"test_tag": "test"})
    assert mlflow.active_run() is None

    with _mock_request(return_value=_mock_chat_completion_response()):
        model = create_openai_llmchain()
        model.invoke("MLflow")

        # A new run should be created, and terminated after the inference
        run = MlflowClient().get_run(model.run_id)
        assert run.info.run_name.startswith("langchain-")
        assert run.info.status == "FINISHED"

        # The run_id should be propagated to the second call via model instance to
        # avoid duplicate logging
        model.invoke("MLflow")
        assert model.run_id == run.info.run_id

    assert mlflow.active_run() is None
    assert run.data.tags["test_tag"] == "test"
    assert run.data.tags["mlflow.autologging"] == "langchain"


def test_resolve_tags():
    extra_tags = {"test_tag": "test"}
    # System tags and extra tags should be logged
    actual_tags = set(_resolve_tags(extra_tags).keys())
    assert actual_tags == {
        "mlflow.autologging",
        "mlflow.source.name",
        "mlflow.source.type",
        "mlflow.user",
        "test_tag",
    }

    with mlflow.start_run() as run:
        actual_tags = set(_resolve_tags(extra_tags, run).keys())

    # The immutable tags starts with 'mlflow.' in the run should not be overridden
    assert actual_tags == {
        "mlflow.autologging",
        "test_tag",
    }


def test_autolog_record_exception():
    from langchain.schema.runnable import RunnableLambda

    def always_fail(input):
        raise Exception("Error!")

    model = RunnableLambda(always_fail)

    mlflow.langchain.autolog()

    with pytest.raises(Exception, match="Error!"):
        model.invoke("test")

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.status == "ERROR"
    assert len(trace.data.spans) == 1
    assert trace.data.spans[0].name == "always_fail"


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

    traces = get_traces()
    assert len(traces) == 2
    for trace in traces:
        spans = trace.data.spans
        assert len(spans) == 2  # chain + llm
        assert spans[0].name == "LLMChain"
        attrs = spans[0].attributes
        assert attrs[SpanAttributeKey.SPAN_TYPE] == "CHAIN"
        assert attrs[SpanAttributeKey.INPUTS] == {"product": "MLflow"}
        assert attrs[SpanAttributeKey.OUTPUTS] == {"text": TEST_CONTENT}
        assert spans[1].name == "OpenAI"
        assert spans[1].parent_id == spans[0].span_id
        attrs = spans[1].attributes
        assert attrs[SpanAttributeKey.SPAN_TYPE] == "LLM"
        assert attrs[SpanAttributeKey.INPUTS] == [
            "What is a good name for a company that makes MLflow?"
        ]
        assert attrs[SpanAttributeKey.OUTPUTS]["generations"][0][0]["text"] == "test"
        assert attrs["invocation_params"]["model_name"] == "gpt-3.5-turbo-instruct"
        assert attrs["invocation_params"]["temperature"] == 0.9


def test_llmchain_autolog_should_not_generate_trace_while_saving_models(tmp_path):
    mlflow.langchain.autolog()
    question = "MLflow"

    with _mock_request(return_value=_mock_chat_completion_response()):
        model = create_openai_llmchain()
        # Either save_model or log_model should not generate traces
        mlflow.langchain.save_model(model, path=tmp_path / "model", input_example=question)
        with mlflow.start_run():
            mlflow.langchain.log_model(model, "model", input_example=question)

    traces = get_traces()
    assert len(traces) == 0


def test_llmchain_autolog_no_optional_artifacts_by_default():
    mlflow.langchain.autolog()
    question = "MLflow"
    answer = {"product": "MLflow", "text": TEST_CONTENT}
    with _mock_request(return_value=_mock_chat_completion_response()):
        model = create_openai_llmchain()
        with mock.patch("mlflow.MlflowClient.create_run") as create_run_mock:
            assert model.invoke(question) == answer
            create_run_mock.assert_not_called()

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert len(spans) == 2


def test_llmchain_autolog_with_registered_model_name():
    registered_model_name = "llmchain"
    mlflow.langchain.autolog(log_models=True, registered_model_name=registered_model_name)
    with _mock_request(return_value=_mock_chat_completion_response()):
        model = create_openai_llmchain()
        model.invoke("MLflow")
        registered_model = MlflowClient().get_registered_model(registered_model_name)
        assert registered_model.name == registered_model_name


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
            assert input_example == question

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
    with _mock_request(return_value=_mock_chat_completion_response()), mock.patch(
        "mlflow.langchain._langchain_autolog._logger.warning"
    ) as mock_warning:
        model = create_openai_llmchain()
        with mlflow.start_run() as run:
            model.invoke(question)
        mock_warning.assert_called_once()
        assert (
            "The log_inputs_outputs option is deprecated and will be removed in a future release"
            in mock_warning.call_args[0][0]
        )
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


@mock.patch("mlflow.tracing.export.mlflow.get_display_handler")
def test_loaded_llmchain_within_model_evaluation(mock_get_display, tmp_path):
    # Disable autolog here as it is enabled in other tests.
    mlflow.langchain.autolog(disable=True)

    model = create_openai_llmchain()
    model_path = tmp_path / "model"
    mlflow.langchain.save_model(model, path=model_path)
    loaded_model = mlflow.pyfunc.load_model(model_path)

    request_id = "eval-123"
    with mlflow.start_run(run_name="eval-run") as run:
        run_id = run.info.run_id
        with set_prediction_context(Context(request_id=request_id, is_evaluate=True)):
            with _mock_request(return_value=_mock_chat_completion_response()):
                response = loaded_model.predict({"product": "MLflow"})

    assert response == ["test"]
    trace = mlflow.get_trace(request_id)
    assert trace.info.tags[TraceTagKey.EVAL_REQUEST_ID] == request_id
    assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_id

    # Trace should not be displayed in the notebook cell if it is in evaluation
    mock_display_handler = mock_get_display.return_value
    mock_display_handler.display_traces.assert_not_called()


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

    traces = get_traces()
    assert len(traces) == 4
    for trace in traces:
        spans = [(s.name, s.attributes[SpanAttributeKey.SPAN_TYPE]) for s in trace.data.spans]
        assert spans == [
            ("AgentExecutor", "CHAIN"),
            ("LLMChain", "CHAIN"),
            ("OpenAI", "LLM"),
        ]
        attrs = trace.data.spans[0].attributes
        assert attrs[SpanAttributeKey.INPUTS] == input
        assert attrs[SpanAttributeKey.OUTPUTS] == {"output": TEST_CONTENT}


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
            assert input_example == input

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

    traces = get_traces()
    assert len(traces) == 2
    for trace in traces:
        spans = {(s.name, s.attributes[SpanAttributeKey.SPAN_TYPE]) for s in trace.data.spans}
        # Since the chain includes parallel execution, the order of some
        # spans is not deterministic.
        assert spans == {
            ("RunnableSequence_1", "CHAIN"),
            ("RunnableParallel<question,chat_history>", "CHAIN"),
            ("RunnableSequence_2", "CHAIN"),
            ("RunnableLambda_1", "CHAIN"),
            ("extract_question", "CHAIN"),
            ("RunnableSequence_3", "CHAIN"),
            ("RunnableLambda_2", "CHAIN"),
            ("extract_history", "CHAIN"),
            ("PromptTemplate", "CHAIN"),
            ("FakeChatModel", "CHAT_MODEL"),
            ("StrOutputParser", "CHAIN"),
        }


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
        assert saved_example == input_example

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

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "VectorStoreRetriever"
    attrs = spans[0].attributes
    assert attrs[SpanAttributeKey.SPAN_TYPE] == "RETRIEVER"
    assert attrs[SpanAttributeKey.INPUTS] == query
    assert attrs[SpanAttributeKey.OUTPUTS][0]["metadata"] == {
        "source": "tests/langchain/state_of_the_union.txt"
    }


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


class CustomCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.logs = []

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        self.logs.append("chain_start")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        self.logs.append("chain_end")


class AsyncCustomCallbackHandler(AsyncCallbackHandler):
    def __init__(self):
        self.logs = []

    async def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        self.logs.append("chain_start")

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        self.logs.append("chain_end")


@pytest.mark.parametrize("invoke_arg", ["args", "kwargs", None])
@pytest.mark.parametrize(
    "generate_callbacks",
    [
        lambda: [CustomCallbackHandler()],
        lambda: BaseCallbackManager([CustomCallbackHandler()]),
        lambda: None,
    ],
)
def test_langchain_autolog_callback_injection_in_invoke(invoke_arg, generate_callbacks):
    mlflow.langchain.autolog(log_models=True, extra_tags={"test_tag": "test"})
    callbacks = generate_callbacks()
    with mlflow.start_run() as run, _mock_request(return_value=_mock_chat_completion_response()):
        model = create_openai_llmchain()
        if invoke_arg == "args":
            model.invoke("MLflow", RunnableConfig(callbacks=callbacks))
        elif invoke_arg == "kwargs":
            model.invoke("MLflow", config=RunnableConfig(callbacks=callbacks))
        elif invoke_arg is None:
            model.invoke("MLflow")
        assert mlflow.active_run() is not None
    run_data = MlflowClient().get_run(run.info.run_id).data
    assert run_data.tags["test_tag"] == "test"
    assert run_data.tags["mlflow.autologging"] == "langchain"
    if invoke_arg and callbacks:
        if isinstance(callbacks, BaseCallbackManager):
            assert callbacks.handlers[0].logs == ["chain_start", "chain_end"]
            assert len(callbacks.handlers) == 1
        else:
            assert callbacks[0].logs == ["chain_start", "chain_end"]
            assert len(callbacks) == 1


@pytest.mark.parametrize("invoke_arg", ["args", "kwargs", None])
@pytest.mark.parametrize(
    "generate_callbacks",
    [
        lambda: [CustomCallbackHandler()],
        lambda: BaseCallbackManager([CustomCallbackHandler()]),
        lambda: None,
        lambda: [AsyncCustomCallbackHandler()],
        lambda: BaseCallbackManager([AsyncCustomCallbackHandler()]),
    ],
)
@pytest.mark.asyncio
async def test_langchain_autolog_callback_injection_in_ainvoke(invoke_arg, generate_callbacks):
    mlflow.langchain.autolog(
        log_models=True,
        extra_tags={"test_tag": "test"},
        log_inputs_outputs=True,
    )
    callbacks = generate_callbacks()

    with mlflow.start_run() as run, _mock_openai_arequest():
        model = create_openai_llmchain()
        if invoke_arg == "args":
            await model.ainvoke("MLflow", RunnableConfig(callbacks=callbacks))
        elif invoke_arg == "kwargs":
            await model.ainvoke("MLflow", config=RunnableConfig(callbacks=callbacks))
        elif invoke_arg is None:
            await model.ainvoke("MLflow")
        assert mlflow.active_run() is not None
    run_data = MlflowClient().get_run(run.info.run_id).data
    assert run_data.tags["test_tag"] == "test"
    assert run_data.tags["mlflow.autologging"] == "langchain"
    # original callback still works as expected
    if invoke_arg and callbacks:
        if isinstance(callbacks, BaseCallbackManager):
            assert callbacks.handlers[0].logs == ["chain_start", "chain_end"]
        else:
            assert callbacks[0].logs == ["chain_start", "chain_end"]


@pytest.mark.parametrize("invoke_arg", ["args", "kwargs"])
@pytest.mark.parametrize(
    "generate_config",
    [
        lambda: RunnableConfig(callbacks=[CustomCallbackHandler()]),
        lambda: RunnableConfig(callbacks=BaseCallbackManager([CustomCallbackHandler()])),
        lambda: RunnableConfig(callbacks=None),
        lambda: [
            RunnableConfig(callbacks=[CustomCallbackHandler()]),
            RunnableConfig(callbacks=[CustomCallbackHandler()]),
        ],
        lambda: [
            RunnableConfig(callbacks=BaseCallbackManager([CustomCallbackHandler()])),
            RunnableConfig(callbacks=BaseCallbackManager([CustomCallbackHandler()])),
        ],
        lambda: [
            RunnableConfig(callbacks=None),
            RunnableConfig(callbacks=None),
        ],
    ],
)
def test_langchain_autolog_callback_injection_in_batch(invoke_arg, generate_config):
    mlflow.langchain.autolog(
        log_models=True, extra_tags={"test_tag": "test"}, log_inputs_outputs=True
    )
    config = generate_config()
    with mlflow.start_run() as run, _mock_request(return_value=_mock_chat_completion_response()):
        model = create_openai_llmchain()
        inputs = ["MLflow"] * 2
        if invoke_arg == "args":
            model.batch(inputs, config)
        elif invoke_arg == "kwargs":
            model.batch(inputs, config=config)
        elif invoke_arg is None:
            model.batch(inputs)
        assert mlflow.active_run() is not None
    run_data = MlflowClient().get_run(run.info.run_id).data
    assert run_data.tags["test_tag"] == "test"
    assert run_data.tags["mlflow.autologging"] == "langchain"
    if isinstance(config, list):
        callbacks = config[0]["callbacks"]
        expected_logs = sorted(["chain_start", "chain_end"])
    else:
        callbacks = config["callbacks"]
        expected_logs = sorted(["chain_start", "chain_end"] * 2)
    if invoke_arg and callbacks:
        if isinstance(callbacks, BaseCallbackManager):
            assert sorted(callbacks.handlers[0].logs) == expected_logs
            assert len(callbacks.handlers) == 1
        else:
            assert sorted(callbacks[0].logs) == expected_logs
            assert len(callbacks) == 1


@pytest.mark.parametrize("invoke_arg", ["args", "kwargs", None])
@pytest.mark.parametrize(
    "generate_config",
    [
        lambda: RunnableConfig(callbacks=[CustomCallbackHandler()]),
        lambda: RunnableConfig(callbacks=[AsyncCustomCallbackHandler()]),
        lambda: RunnableConfig(callbacks=None),
        lambda: RunnableConfig(callbacks=BaseCallbackManager([CustomCallbackHandler()])),
        lambda: RunnableConfig(callbacks=BaseCallbackManager([AsyncCustomCallbackHandler()])),
        lambda: [
            RunnableConfig(callbacks=[CustomCallbackHandler()]),
            RunnableConfig(callbacks=[CustomCallbackHandler()]),
        ],
        lambda: [
            RunnableConfig(callbacks=[AsyncCustomCallbackHandler()]),
            RunnableConfig(callbacks=[AsyncCustomCallbackHandler()]),
        ],
        lambda: [
            RunnableConfig(callbacks=BaseCallbackManager([CustomCallbackHandler()])),
            RunnableConfig(callbacks=BaseCallbackManager([CustomCallbackHandler()])),
        ],
        lambda: [
            RunnableConfig(callbacks=BaseCallbackManager([AsyncCustomCallbackHandler()])),
            RunnableConfig(callbacks=BaseCallbackManager([AsyncCustomCallbackHandler()])),
        ],
        lambda: [
            RunnableConfig(callbacks=None),
            RunnableConfig(callbacks=None),
        ],
    ],
)
@pytest.mark.asyncio
async def test_langchain_autolog_callback_injection_in_abatch(invoke_arg, generate_config):
    mlflow.langchain.autolog(
        log_models=True,
        extra_tags={"test_tag": "test"},
        log_inputs_outputs=True,
    )
    config = generate_config()
    with mlflow.start_run() as run, _mock_openai_arequest():
        model = create_openai_llmchain()
        inputs = ["MLflow"] * 2
        if invoke_arg == "args":
            await model.abatch(inputs, config)
        elif invoke_arg == "kwargs":
            await model.abatch(inputs, config=config)
        elif invoke_arg is None:
            await model.abatch(inputs)
        assert mlflow.active_run() is not None
    run_data = MlflowClient().get_run(run.info.run_id).data
    assert run_data.tags["test_tag"] == "test"
    assert run_data.tags["mlflow.autologging"] == "langchain"
    if isinstance(config, list):
        callbacks = config[0]["callbacks"]
        expected_logs = sorted(["chain_start", "chain_end"])
    else:
        callbacks = config["callbacks"]
        expected_logs = sorted(["chain_start", "chain_end"] * 2)
    if invoke_arg and callbacks:
        if isinstance(callbacks, BaseCallbackManager):
            assert sorted(callbacks.handlers[0].logs) == expected_logs
        else:
            assert sorted(callbacks[0].logs) == expected_logs


@pytest.mark.parametrize("invoke_arg", ["args", "kwargs", None])
@pytest.mark.parametrize(
    "generate_config",
    [
        lambda: RunnableConfig(callbacks=[CustomCallbackHandler()]),
        lambda: RunnableConfig(callbacks=BaseCallbackManager([CustomCallbackHandler()])),
        lambda: RunnableConfig(callbacks=None),
    ],
)
def test_langchain_autolog_callback_injection_in_stream(invoke_arg, generate_config):
    mlflow.langchain.autolog(log_models=True, extra_tags={"test_tag": "test"})
    config = generate_config()
    with mlflow.start_run() as run, _mock_request(return_value=_mock_chat_completion_response()):
        model = create_openai_llmchain()
        # convert iterator to list to materialize the stream compute
        if invoke_arg == "args":
            next(model.stream("MLflow", config))
        elif invoke_arg == "kwargs":
            next(model.stream("MLflow", config=config))
        elif invoke_arg is None:
            next(model.stream("MLflow"))
    run_data = MlflowClient().get_run(run.info.run_id).data
    assert run_data.tags["test_tag"] == "test"
    assert run_data.tags["mlflow.autologging"] == "langchain"
    callbacks = config["callbacks"]
    expected_logs = ["chain_start", "chain_end"]
    if invoke_arg and callbacks:
        if isinstance(callbacks, BaseCallbackManager):
            assert callbacks.handlers[0].logs == expected_logs
            # original callbacks should not be modified
            assert (
                sum(isinstance(handler, MlflowLangchainTracer) for handler in callbacks.handlers)
                == 0
            )
            assert len(callbacks.handlers) == 1
        else:
            assert callbacks[0].logs == expected_logs
            assert sum(isinstance(handler, MlflowLangchainTracer) for handler in callbacks) == 0
            assert len(callbacks) == 1


@pytest.mark.parametrize("invoke_arg", ["args", "kwargs", None])
@pytest.mark.parametrize(
    "generate_config",
    [
        lambda: RunnableConfig(callbacks=[CustomCallbackHandler()]),
        lambda: RunnableConfig(callbacks=[AsyncCustomCallbackHandler()]),
        lambda: RunnableConfig(callbacks=BaseCallbackManager([CustomCallbackHandler()])),
        lambda: RunnableConfig(callbacks=BaseCallbackManager([AsyncCustomCallbackHandler()])),
        lambda: RunnableConfig(callbacks=None),
    ],
)
@pytest.mark.asyncio
async def test_langchain_autolog_callback_injection_in_astream(invoke_arg, generate_config):
    mlflow.langchain.autolog(log_models=True, extra_tags={"test_tag": "test"})
    config = generate_config()

    async def _test_astream(model, invoke_arg, config):
        if invoke_arg == "args":
            async for result in model.astream("MLflow", config):
                return result
        elif invoke_arg == "kwargs":
            async for result in model.astream("MLflow", config=config):
                return result
        elif invoke_arg is None:
            async for result in model.astream("MLflow"):
                return result

    with mlflow.start_run() as run, _mock_openai_arequest():
        model = create_openai_llmchain()
        await _test_astream(model, invoke_arg, config)
    run_data = MlflowClient().get_run(run.info.run_id).data
    assert run_data.tags["test_tag"] == "test"
    assert run_data.tags["mlflow.autologging"] == "langchain"
    callbacks = config["callbacks"]
    expected_logs = ["chain_start", "chain_end"]
    if invoke_arg and callbacks:
        if isinstance(callbacks, BaseCallbackManager):
            assert callbacks.handlers[0].logs == expected_logs
            assert (
                sum(isinstance(handler, MlflowLangchainTracer) for handler in callbacks.handlers)
                == 0
            )
        else:
            assert callbacks[0].logs == expected_logs
            assert sum(isinstance(handler, MlflowLangchainTracer) for handler in callbacks) == 0


def test_langchain_autolog_produces_expected_traces_with_streaming(tmp_path):
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnablePassthrough

    mlflow.langchain.autolog()
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
    with _mock_request(return_value=_mock_chat_completion_response()):
        list(retrieval_chain.stream(question))
        retrieval_chain.invoke(question)

    traces = get_traces()
    assert len(traces) == 2
    stream_trace = traces[0]
    invoke_trace = traces[1]

    assert stream_trace.info.status == invoke_trace.info.status == TraceStatus.OK
    assert stream_trace.data.request == invoke_trace.data.request
    assert stream_trace.data.response == invoke_trace.data.response
    assert len(stream_trace.data.spans) == len(invoke_trace.data.spans)


@pytest.mark.parametrize("log_traces", [True, False, None])
def test_langchain_tracer_injection_for_arbitrary_runnables(log_traces):
    from langchain.schema.runnable import RouterRunnable, RunnableLambda

    should_log_traces = log_traces is not False

    if log_traces is not None:
        mlflow.langchain.autolog(log_traces=log_traces)
    else:
        mlflow.langchain.autolog()

    add = RunnableLambda(func=lambda x: x + 1)
    square = RunnableLambda(func=lambda x: x**2)
    model = RouterRunnable(runnables={"add": add, "square": square})

    with mock.patch("mlflow.langchain._langchain_autolog._logger.debug") as mock_debug:
        model.invoke({"key": "square", "input": 3})
        if should_log_traces:
            mock_debug.assert_called_once_with("Injected MLflow callbacks into the model.")
        else:
            mock_debug.assert_not_called()
    traces = get_traces()
    if should_log_traces:
        assert len(traces) == 1
        assert traces[0].data.spans[0].attributes[SpanAttributeKey.SPAN_TYPE] == "CHAIN"
    else:
        assert len(traces) == 0


def test_langchain_autolog_extra_model_classes_no_duplicate_patching():
    from langchain.schema.runnable import Runnable

    class CustomRunnable(Runnable):
        def invoke(self, input, config=None):
            return "test"

        def _type(self):
            return "CHAIN"

    class AnotherRunnable(CustomRunnable):
        def invoke(self, input, config=None):
            return super().invoke(input)

        def _type(self):
            return "CHAT_MODEL"

    mlflow.langchain.autolog(extra_model_classes=[CustomRunnable, AnotherRunnable])
    model = AnotherRunnable()
    with mock.patch("mlflow.langchain._langchain_autolog._logger.debug") as mock_debug:
        assert model.invoke("test") == "test"
        mock_debug.assert_called_once_with("Injected MLflow callbacks into the model.")
        assert mock_debug.call_count == 1


def test_langchain_autolog_extra_model_classes_warning():
    from langchain.schema.runnable import Runnable

    class NotARunnable:
        def __init__(self, x):
            self.x = x

    with mock.patch("mlflow.langchain.logger.warning") as mock_warning:
        mlflow.langchain.autolog(extra_model_classes=[NotARunnable])
        mock_warning.assert_called_once_with(
            "Unsupported classes found in extra_model_classes: ['NotARunnable']. "
            "Only subclasses of Runnable are supported."
        )
        mock_warning.reset_mock()

        mlflow.langchain.autolog(extra_model_classes=[Runnable])
        mock_warning.assert_not_called()


@pytest.mark.skip(reason="This test is not thread safe, please run locally")
def test_set_retriever_schema_work_for_langchain_model():
    set_retriever_schema(
        primary_key="primary-key",
        text_column="text-column",
        doc_uri="doc-uri",
        other_columns=["column1", "column2"],
    )

    model = create_openai_llmchain()
    with _mock_request(return_value=_mock_chat_completion_response()):
        with mlflow.start_run():
            model_info = mlflow.langchain.log_model(model, "model", input_example="MLflow")

        mlflow.langchain.autolog()

        pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
        pyfunc_model.predict("MLflow")

    trace = mlflow.get_last_active_trace()
    assert DependenciesSchemasType.RETRIEVERS.value in trace.info.tags
