from types import SimpleNamespace
from unittest.mock import patch

import pytest
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.yfinance import YFinanceTools

import mlflow
import mlflow.agno
from mlflow.entities import SpanType

from tests.tracing.helper import get_traces


def _safe_resp(content, *, calls=None):
    return SimpleNamespace(
        content=content,
        metrics={"token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}},
        tool_calls=calls or [],
        tool_executions=[],
        thinking="",
        redacted_thinking="",
        citations=SimpleNamespace(urls=[]),
        audio=[],
        image=[],
        created_at=[],
    )


@pytest.fixture(autouse=True)
def _reset_mlflow():
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    for integ in AUTOLOGGING_INTEGRATIONS.values():
        integ.clear()
    mlflow.utils.import_hooks._post_import_hooks = {}


@pytest.fixture
def simple_agent():
    return Agent(
        model=Claude(id="claude-sonnet-4-20250514"),
        instructions="Be concise.",
        markdown=True,
    )


@pytest.fixture
def agent_with_tool():
    return Agent(
        model=Claude(id="claude-sonnet-4-20250514"),
        tools=[YFinanceTools(stock_price=True)],
        instructions="Use YFinanceTools when needed.",
        markdown=True,
    )


def test_run_simple_autolog(simple_agent):
    with patch.object(Claude, "response", lambda self, messages, **kw: _safe_resp("Paris")):
        mlflow.agno.autolog(log_traces=True)
        resp = simple_agent.run("Capital of France?")
    assert resp.content == "Paris"

    spans = [s.span_type for s in get_traces()[0].data.spans]
    assert spans == [SpanType.AGENT]

    with patch.object(Claude, "response", lambda self, messages, **kw: _safe_resp("Paris")):
        mlflow.agno.autolog(disable=True)
        simple_agent.run("Again?")
    assert len(get_traces()) == 1


def test_run_failure_tracing(simple_agent):
    def _boom(self, messages, **kw):
        raise RuntimeError("bang")

    with patch.object(Claude, "response", new=_boom):
        mlflow.agno.autolog(log_traces=True)
        with pytest.raises(RuntimeError, match="bang"):
            simple_agent.run("fail")

    trace = get_traces()[0]
    assert trace.info.status == "ERROR"
    spans = [s.span_type for s in trace.data.spans]
    assert spans == [SpanType.AGENT]
