"""Tests for MLflow Aider autologging integration."""

import sys
from types import ModuleType
from unittest import mock

import pytest

import mlflow
from mlflow.aider.autolog import _AIDER_PATCHED_ATTR, _get_chat_files, autolog
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

# ---------------------------------------------------------------------------
# Minimal stubs so tests run without aider-chat installed
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, name: str = "gpt-4o"):
        self.name = name


class _FakeCoder:
    """Minimal stub that mirrors the Aider Coder interface used by autolog."""

    main_model = _FakeModel()
    total_tokens_sent = 0
    total_tokens_received = 0
    total_cost = 0.0

    def run_one(self, user_message: str, preproc: bool = True) -> str:
        self.total_tokens_sent += 100
        self.total_tokens_received += 50
        self.total_cost += 0.002
        return f"Done: {user_message}"

    def get_inchat_relative_files(self) -> list[str]:
        return ["utils.py", "main.py"]


def _make_aider_module() -> ModuleType:
    """Build a fake `aider` package with a minimal `coders` sub-module."""
    aider_mod = ModuleType("aider")
    coders_mod = ModuleType("aider.coders")
    coders_mod.Coder = _FakeCoder
    aider_mod.coders = coders_mod
    return aider_mod, coders_mod


@pytest.fixture(autouse=True)
def _inject_fake_aider(monkeypatch):
    """Inject a fake aider package so tests never need aider-chat installed."""
    aider_mod, coders_mod = _make_aider_module()
    monkeypatch.setitem(sys.modules, "aider", aider_mod)
    monkeypatch.setitem(sys.modules, "aider.coders", coders_mod)
    # Reset patch state before each test
    _FakeCoder._mlflow_patched = False
    if hasattr(_FakeCoder.run_one, "__wrapped__"):
        _FakeCoder.run_one = _FakeCoder.run_one.__wrapped__
    yield
    # Clean up patch state after each test
    setattr(_FakeCoder, _AIDER_PATCHED_ATTR, False)


# ---------------------------------------------------------------------------
# Patching behaviour
# ---------------------------------------------------------------------------


def test_autolog_patches_coder():
    autolog()
    assert getattr(_FakeCoder, _AIDER_PATCHED_ATTR, False) is True


def test_autolog_is_idempotent():
    autolog()
    original_method = _FakeCoder.run_one
    autolog()
    assert _FakeCoder.run_one is original_method


def test_autolog_disable_unpatches_coder():
    autolog()
    assert getattr(_FakeCoder, _AIDER_PATCHED_ATTR, False) is True
    autolog(disable=True)
    assert getattr(_FakeCoder, _AIDER_PATCHED_ATTR, False) is False


def test_autolog_disable_when_not_patched_is_safe():
    autolog(disable=True)  # should not raise


# ---------------------------------------------------------------------------
# Trace content
# ---------------------------------------------------------------------------


def test_autolog_creates_trace_with_correct_inputs():
    autolog()
    coder = _FakeCoder()
    coder.run_one("Add type hints to utils.py")

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    root_span = trace.data.spans[0]
    assert root_span.span_type == SpanType.AGENT
    assert root_span.inputs["prompt"] == "Add type hints to utils.py"
    assert root_span.inputs["model"] == "gpt-4o"
    assert "utils.py" in root_span.inputs["files"]


def test_autolog_captures_response():
    autolog()
    coder = _FakeCoder()
    coder.run_one("Refactor main.py")

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    root_span = trace.data.spans[0]
    assert "Refactor main.py" in root_span.outputs["response"]


def test_autolog_captures_token_usage():
    autolog()
    coder = _FakeCoder()
    coder.run_one("Fix the bug")

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    root_span = trace.data.spans[0]
    usage = root_span.attributes.get(SpanAttributeKey.CHAT_USAGE)
    assert usage is not None
    assert usage[TokenUsageKey.INPUT_TOKENS] == 100
    assert usage[TokenUsageKey.OUTPUT_TOKENS] == 50
    assert usage[TokenUsageKey.TOTAL_TOKENS] == 150


def test_autolog_captures_total_cost():
    autolog()
    coder = _FakeCoder()
    coder.run_one("Update README")

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    root_span = trace.data.spans[0]
    assert root_span.attributes.get("total_cost_usd") is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_autolog_raises_if_aider_not_installed(monkeypatch):
    monkeypatch.delitem(sys.modules, "aider", raising=False)
    monkeypatch.delitem(sys.modules, "aider.coders", raising=False)

    with pytest.raises(ImportError, match="aider-chat"):
        autolog()


def test_get_chat_files_returns_empty_on_error():
    class BrokenCoder:
        def get_inchat_relative_files(self):
            raise RuntimeError("broken")

    assert _get_chat_files(BrokenCoder()) == []


def test_autolog_handles_none_response():
    autolog(disable=True)
    original_run_one = _FakeCoder.run_one
    _FakeCoder.run_one = mock.Mock(return_value=None)
    _FakeCoder.total_tokens_sent = 0
    _FakeCoder.total_tokens_received = 0
    autolog()
    coder = _FakeCoder()
    result = coder.run_one("What is 2+2?", preproc=True)
    assert result is None
    _FakeCoder.run_one = original_run_one
