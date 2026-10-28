# Regression test for https://github.com/mlflow/mlflow/issues/27005:
# mlflow-skinny users couldn't use mlflow.openai.autolog() because importing the flavor
# eagerly imported its model logging APIs, which depend on numpy and pandas. Tracing in the
# langchain and dspy flavors also imported mlflow.pyfunc, which depends on numpy.

import importlib
import importlib.util
import os

import pytest

import mlflow


@pytest.fixture(autouse=True)
def is_skinny():
    if "MLFLOW_SKINNY" not in os.environ:
        pytest.skip("This test is only valid for the skinny client")


@pytest.fixture
def tracking_uri(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")


def _assert_traced():
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    assert len(trace.data.spans) > 0


def _assert_numpy_and_pandas_not_installed():
    # Verify numpy and pandas are genuinely not installed (not just not yet imported)
    assert importlib.util.find_spec("numpy") is None
    assert importlib.util.find_spec("pandas") is None


@pytest.mark.parametrize("flavor", ["openai", "langchain", "llama_index", "dspy"])
def test_flavor_autolog_importable_without_numpy_and_pandas(flavor):
    _assert_numpy_and_pandas_not_installed()

    module = importlib.import_module(f"mlflow.{flavor}")
    assert callable(module.autolog)
    assert not hasattr(module, "log_model")


def test_openai_autolog_traces_without_numpy_and_pandas(tracking_uri):
    import httpx
    import openai

    _assert_numpy_and_pandas_not_installed()
    mlflow.openai.autolog()

    response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hi"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    client = openai.OpenAI(
        api_key="test",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response))
        ),
    )
    client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "Hello"}]
    )

    _assert_traced()


def test_langchain_autolog_traces_without_numpy_and_pandas(tracking_uri):
    from langchain_core.runnables import RunnableLambda

    _assert_numpy_and_pandas_not_installed()
    mlflow.langchain.autolog()

    assert RunnableLambda(lambda x: x + 1).invoke(1) == 2

    _assert_traced()


def test_dspy_autolog_traces_without_numpy_and_pandas(tracking_uri):
    import dspy
    from dspy.utils.dummies import DummyLM

    _assert_numpy_and_pandas_not_installed()
    mlflow.dspy.autolog()

    dspy.configure(lm=DummyLM([{"answer": "4"}]))
    assert dspy.Predict("question -> answer")(question="2 + 2?").answer == "4"

    _assert_traced()


def test_llama_index_autolog_traces_without_pandas(tracking_uri):
    from llama_index.core.llms.mock import MockLLM

    # llama-index-core requires numpy, so only pandas is missing in this case
    assert importlib.util.find_spec("pandas") is None
    mlflow.llama_index.autolog()

    MockLLM(max_tokens=5).complete("Hello")

    _assert_traced()
