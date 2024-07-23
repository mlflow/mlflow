import json

import openai
import pytest

import mlflow
from mlflow import MlflowClient

from tests.openai.conftest import is_v1


@pytest.fixture
def client(mock_openai):
    return openai.OpenAI(api_key="test", base_url=mock_openai)


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_chat_completions_autolog_artifacts(client, monkeypatch):
    mlflow.openai.autolog(log_models=True)
    messages = [{"role": "user", "content": "test"}]
    with mlflow.start_run() as run:
        client.chat.completions._mlflow_session_id = "test_session_id"
        client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo",
            temperature=0,
        )

    artifact_dir = MlflowClient().download_artifacts(run.info.run_id, "artifacts-test_session_id-0")
    with open(f"{artifact_dir}/input.json") as f:
        assert json.load(f)["messages"] == messages

    with open(f"{artifact_dir}/output.json") as f:
        assert json.load(f)["id"] == "chatcmpl-123"


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_chat_completions_autolog_streaming(client, monkeypatch):
    mlflow.openai.autolog(log_models=True)
    messages = [{"role": "user", "content": "test"}]
    with mlflow.start_run() as run:
        client.chat.completions._mlflow_session_id = "test_session_id"
        stream = client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo",
            temperature=0,
            stream=True,
        )
        for _ in stream:
            pass

    artifact_dir = MlflowClient().download_artifacts(run.info.run_id, "artifacts-test_session_id-0")
    with open(f"{artifact_dir}/input.json") as f:
        assert json.load(f)["messages"] == messages

    with open(f"{artifact_dir}/output.json") as f:
        output = json.load(f)
        assert len(output) == 2
        assert output[0]["id"] == "chatcmpl-123"
        assert output[0]["choices"][0]["delta"]["content"] == "Hello"
        assert output[1]["id"] == "chatcmpl-123"
        assert output[1]["choices"][0]["delta"]["content"] == " world"

    trace = mlflow.get_last_active_trace()
    assert trace is not None
    assert isinstance(trace.data.response, str)

    response_data = json.loads(trace.data.response)

    assert response_data["result"] == "Hello world"

    stream_event_data = trace.data.spans[0].attributes["events"]

    assert stream_event_data[0]["id"] == "chatcmpl-123"
    assert stream_event_data[0]["choices"][0]["delta"]["content"] == "Hello"
    assert stream_event_data[1]["id"] == "chatcmpl-123"
    assert stream_event_data[1]["choices"][0]["delta"]["content"] == " world"


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_loaded_chat_completions_autolog(client, monkeypatch):
    mlflow.openai.autolog(log_models=True)
    messages = [{"role": "user", "content": "test"}]
    with mlflow.start_run() as run:
        client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo",
            temperature=0,
        )

    loaded_model = mlflow.openai.load_model(f"runs:/{run.info.run_id}/model")
    assert loaded_model == {
        "model": "gpt-3.5-turbo",
        "task": "chat.completions",
    }

    monkeypatch.setenvs(
        {
            "OPENAI_API_KEY": "test",
            "OPENAI_API_BASE": client.base_url,
        }
    )
    pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
    # expected output from mock_openai
    assert pyfunc_model.predict("test") == [json.dumps(messages)]


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_completions_autolog_artifacts(client, monkeypatch):
    mlflow.openai.autolog(log_models=True)
    with mlflow.start_run() as run:
        client.completions._mlflow_session_id = "test_session_id"
        client.completions.create(
            prompt="test",
            model="gpt-3.5-turbo",
            temperature=0,
        )

    artifact_dir = MlflowClient().download_artifacts(run.info.run_id, "artifacts-test_session_id-0")
    with open(f"{artifact_dir}/input.json") as f:
        assert json.load(f)["prompt"] == "test"

    with open(f"{artifact_dir}/output.json") as f:
        assert json.load(f)["id"] == "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_completions_autolog_streaming(client, monkeypatch):
    mlflow.openai.autolog(log_models=True)
    with mlflow.start_run() as run:
        client.completions._mlflow_session_id = "test_session_id"
        stream = client.completions.create(
            prompt="test",
            model="gpt-3.5-turbo",
            temperature=0,
            stream=True,
        )
        for _ in stream:
            pass

    artifact_dir = MlflowClient().download_artifacts(run.info.run_id, "artifacts-test_session_id-0")
    with open(f"{artifact_dir}/input.json") as f:
        assert json.load(f)["prompt"] == "test"

    with open(f"{artifact_dir}/output.json") as f:
        output = json.load(f)
        assert len(output) == 2
        assert output[0]["id"] == "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"
        assert output[0]["choices"][0]["text"] == "Hello"
        assert output[1]["id"] == "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"
        assert output[1]["choices"][0]["text"] == " world"

    trace = mlflow.get_last_active_trace()
    assert trace is not None
    assert isinstance(trace.data.response, str)

    response_data = json.loads(trace.data.response)

    assert response_data["result"] == "Hello world"

    stream_event_data = trace.data.spans[0].attributes["events"]

    assert stream_event_data[0]["id"] == "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"
    assert stream_event_data[0]["choices"][0]["text"] == "Hello"
    assert stream_event_data[1]["id"] == "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"
    assert stream_event_data[1]["choices"][0]["text"] == " world"


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_loaded_completions_autolog(client, monkeypatch):
    mlflow.openai.autolog(log_models=True)
    with mlflow.start_run() as run:
        client.completions.create(
            prompt="test",
            model="gpt-3.5-turbo-instruct",
            temperature=0,
        )

    loaded_model = mlflow.openai.load_model(f"runs:/{run.info.run_id}/model")
    assert loaded_model == {
        "model": "gpt-3.5-turbo-instruct",
        "task": "completions",
    }

    monkeypatch.setenvs(
        {
            "OPENAI_API_KEY": "test",
            "OPENAI_API_BASE": client.base_url,
        }
    )
    pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
    # expected output from mock_openai
    assert pyfunc_model.predict("test") == ["test"]


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_embeddings_autolog_artifacts(client, monkeypatch):
    mlflow.openai.autolog(log_models=True)
    with mlflow.start_run() as run:
        client.embeddings._mlflow_session_id = "test_session_id"
        client.embeddings.create(
            input="test",
            model="text-embedding-ada-002",
        )

    artifact_dir = MlflowClient().download_artifacts(run.info.run_id, "artifacts-test_session_id-0")
    with open(f"{artifact_dir}/input.json") as f:
        assert json.load(f)["input"] == "test"

    with open(f"{artifact_dir}/output.json") as f:
        assert len(json.load(f)["data"][0]["embedding"]) == 1536


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_loaded_embeddings_autolog(client, monkeypatch):
    mlflow.openai.autolog(log_models=True)
    with mlflow.start_run() as run:
        client.embeddings.create(
            input="test",
            model="text-embedding-ada-002",
        )

    loaded_model = mlflow.openai.load_model(f"runs:/{run.info.run_id}/model")
    assert loaded_model == {
        "model": "text-embedding-ada-002",
        "task": "embeddings",
    }

    monkeypatch.setenvs(
        {
            "OPENAI_API_KEY": "test",
            "OPENAI_API_BASE": client.base_url,
        }
    )
    pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
    # expected output from mock_openai
    output = pyfunc_model.predict("test")
    assert len(output) == 1
    assert len(output[0]) == 1536


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_autolog_with_registered_model_name(client):
    registered_model_name = "test_model"
    mlflow.openai.autolog(log_models=True, registered_model_name=registered_model_name)
    client.chat.completions.create(
        messages=[{"role": "user", "content": "test"}],
        model="gpt-3.5-turbo",
        temperature=0,
    )
    registered_model = MlflowClient().get_registered_model(registered_model_name)
    assert registered_model.name == registered_model_name
