import sys
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

import httpx
from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport

import mlflow
from mlflow import MlflowClient
from mlflow.environment_variables import MLFLOW_DISABLE_TELEMETRY


def stdio_client(tracking_uri: str, tools: str = "genai") -> Client:
    return Client(
        StdioTransport(
            command=sys.executable,
            args=["-m", "mlflow", "mcp", "run"],
            env={
                "MLFLOW_TRACKING_URI": tracking_uri,
                "MLFLOW_MCP_TOOLS": tools,
                MLFLOW_DISABLE_TELEMETRY.name: "true",
            },
        )
    )


@asynccontextmanager
async def http_client(app, url: str = "http://testserver/mcp") -> AsyncIterator[Client]:
    # The MCP session manager is started by the server lifespan, so run it around the client.
    # Docker lookup is disabled to keep the unrelated sandbox cleanup out of these tests.
    with mock.patch("mlflow.server.fastapi_app.shutil.which", return_value=None):
        async with app.router.lifespan_context(app):
            transport = StreamableHttpTransport(
                url,
                httpx_client_factory=lambda **kwargs: httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), **kwargs
                ),
            )
            async with Client(transport) as client:
                yield client


@dataclass
class Scenario:
    """
    A valid call of one shared tool.

    Args:
        tool: The tool name.
        setup: Creates the resources the call needs, keyed by a per-transport suffix, and returns
            the call arguments.
        read_only: Whether the call leaves the store unchanged; read-only calls on both transports
            share one setup and must return identical results.
        generated: Result keys whose values the call generates (ids, random names, timestamps)
            and that differ between two otherwise identical calls.
    """

    tool: str
    setup: Callable[[str], dict[str, Any]]
    read_only: bool = False
    generated: frozenset[str] = field(default_factory=frozenset)


def _experiment(suffix: str) -> str:
    return MlflowClient().create_experiment(f"exp-{suffix}-{uuid.uuid4().hex[:8]}")


def _run(suffix: str) -> str:
    return MlflowClient().create_run(_experiment(suffix), run_name="run").info.run_id


def _trace(suffix: str, experiment_id: str | None = None) -> str:
    mlflow.set_experiment(experiment_id=experiment_id or _experiment(suffix))
    with mlflow.start_span("span") as span:
        span.set_inputs({"q": "hi"})
        span.set_outputs({"a": "hello"})
    mlflow.flush_trace_async_logging()
    return span.trace_id


def _feedback(suffix: str) -> tuple[str, str]:
    trace_id = _trace(suffix)
    assessment = mlflow.log_feedback(trace_id=trace_id, name="quality", value=0.5)
    return trace_id, assessment.assessment_id


def _experiment_with_runs(suffix: str) -> str:
    experiment_id = _experiment(suffix)
    client = MlflowClient()
    for name in ("a", "b"):
        client.create_run(experiment_id, run_name=name)
    return experiment_id


def _deleted_experiment(suffix: str) -> str:
    experiment_id = _experiment(suffix)
    MlflowClient().delete_experiment(experiment_id)
    return experiment_id


def _deleted_run(suffix: str) -> str:
    run_id = _run(suffix)
    MlflowClient().delete_run(run_id)
    return run_id


def _trace_with_tag(suffix: str) -> str:
    trace_id = _trace(suffix)
    MlflowClient().set_trace_tag(trace_id, "env", "prod")
    return trace_id


def _assessment_args(suffix: str) -> dict[str, Any]:
    trace_id, assessment_id = _feedback(suffix)
    return {"trace_id": trace_id, "assessment_id": assessment_id}


def _trace_and_experiment(suffix: str) -> dict[str, Any]:
    experiment_id = _experiment(suffix)
    return {"experiment_id": experiment_id, "trace_ids": [_trace(suffix, experiment_id)]}


_ASSESSMENT_TIMES = frozenset({"assessment_id", "create_time_ms", "last_update_time_ms"})
_TIMES = frozenset({"creation_time", "last_update_time", "start_time", "end_time"})

SCENARIOS: list[Scenario] = [
    # Experiments
    Scenario("search_experiments", lambda s: {"max_results": 100}, read_only=True),
    Scenario("get_experiment", lambda s: {"experiment_id": _experiment(s)}, read_only=True),
    Scenario(
        "create_experiment",
        lambda s: {"experiment_name": f"created-{s}", "trace_archival_retention": "1d"},
        generated=frozenset({"experiment_id"}),
    ),
    Scenario(
        "update_experiment",
        lambda s: {
            "experiment_id": _experiment(s),
            "trace_archival_retention": "2d",
            "trace_archive_now": True,
        },
    ),
    Scenario("delete_experiment", lambda s: {"experiment_id": _experiment(s)}),
    Scenario("restore_experiment", lambda s: {"experiment_id": _deleted_experiment(s)}),
    Scenario(
        "rename_experiment",
        lambda s: {"experiment_id": _experiment(s), "new_name": f"renamed-{s}"},
    ),
    # Runs
    Scenario("list_runs", lambda s: {"experiment_id": _experiment_with_runs(s)}, read_only=True),
    Scenario("describe_run", lambda s: {"run_id": _run(s)}, read_only=True),
    Scenario(
        "create_run",
        lambda s: {
            "experiment_id": _experiment(s),
            "run_name": "created",
            "tags": {"k": "v"},
            "status": "KILLED",
        },
        generated=frozenset({"run_id"}),
    ),
    Scenario("delete_run", lambda s: {"run_id": _run(s)}),
    Scenario("restore_run", lambda s: {"run_id": _deleted_run(s)}),
    Scenario(
        "link_traces_to_run",
        lambda s: {"run_id": _run(s), "trace_ids": [_trace(s)]},
    ),
    # Traces
    Scenario(
        "search_traces",
        lambda s: {"experiment_id": _trace_and_experiment(s)["experiment_id"]},
        read_only=True,
    ),
    Scenario("get_trace", lambda s: {"trace_id": _trace(s)}, read_only=True),
    Scenario("delete_traces", _trace_and_experiment),
    Scenario("set_trace_tag", lambda s: {"trace_id": _trace(s), "key": "env", "value": "prod"}),
    Scenario("delete_trace_tag", lambda s: {"trace_id": _trace_with_tag(s), "key": "env"}),
    Scenario(
        "log_trace_feedback",
        lambda s: {
            "trace_id": _trace(s),
            "name": "relevance",
            "value": 0.9,
            "source_type": "HUMAN",
            "source_id": "reviewer",
            "rationale": "on topic",
            "metadata": {"round": "1"},
        },
        generated=_ASSESSMENT_TIMES,
    ),
    Scenario(
        "log_trace_expectation",
        lambda s: {
            "trace_id": _trace(s),
            "name": "expected",
            "value": {"answer": "Paris"},
            "source_type": "HUMAN",
            "source_id": "annotator",
        },
        generated=_ASSESSMENT_TIMES,
    ),
    Scenario("get_trace_assessment", _assessment_args, read_only=True),
    Scenario(
        "update_trace_assessment",
        lambda s: {**_assessment_args(s), "value": 0.25, "rationale": "revised"},
        generated=_ASSESSMENT_TIMES,
    ),
    Scenario("delete_trace_assessment", _assessment_args),
    # Scorers
    Scenario("list_scorers", lambda s: {"builtin": True}, read_only=True),
    Scenario(
        "register_llm_judge_scorer",
        lambda s: {
            "name": f"judge_{s}",
            "instructions": "Is {{ outputs }} a good answer to {{ inputs }}?",
            "experiment_id": _experiment(s),
        },
    ),
]


def normalize(result: Any, arguments: dict[str, Any], generated: frozenset[str]) -> Any:
    """
    Replace what differs between two identical calls on separate resources: the per-transport
    argument values (resource ids and names) and the values the call generates.
    """
    placeholders: dict[str, str] = {}
    for key, value in arguments.items():
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str):
                placeholders[item] = f"<{key}>"

    def walk(value: Any, key: str | None = None) -> Any:
        if key in generated or key in _TIMES:
            return f"<{key}>"
        if isinstance(value, dict):
            return {k: walk(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str):
            return placeholders.get(value, value)
        return value

    return walk(result)
