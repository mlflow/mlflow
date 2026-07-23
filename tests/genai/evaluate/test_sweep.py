import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.sweep import _normalize_predict_fns, _percentile
from mlflow.genai.scorers.base import scorer
from mlflow.server import handlers
from mlflow.server.fastapi_app import app
from mlflow.server.handlers import initialize_backend_stores
from mlflow.utils.mlflow_tags import MLFLOW_RUN_TYPE

from tests.helper_functions import get_safe_port
from tests.tracking.integration_test_utils import ServerThread


@scorer
def exact_match(outputs, expectations):
    return outputs == expectations["expected"]


@scorer
def output_length(outputs):
    return float(len(str(outputs)))


ANSWERS = {"2+2": "4", "capital of France": "Paris", "color of sky": "blue"}


def good_model(question: str) -> str:
    return ANSWERS[question]


def bad_model(question: str) -> str:
    return "I don't know"


DATA = [
    {"inputs": {"question": "2+2"}, "expectations": {"expected": "4"}},
    {"inputs": {"question": "capital of France"}, "expectations": {"expected": "Paris"}},
    {"inputs": {"question": "color of sky"}, "expectations": {"expected": "blue"}},
]


@dataclass
class ServerConfig:
    host_type: Literal["local", "remote"]
    backend_type: Literal["file", "sqlalchemy"] | None = None


@pytest.fixture(
    params=[
        ServerConfig(host_type="local", backend_type="sqlalchemy"),
        ServerConfig(host_type="remote", backend_type="sqlalchemy"),
    ],
    ids=["local_sqlalchemy", "remote_sqlalchemy"],
)
def server_config(request, tmp_path: Path, db_uri: str):
    config = request.param
    backend_uri = db_uri

    match config.host_type:
        case "local":
            mlflow.set_tracking_uri(backend_uri)
            yield config
        case "remote":
            handlers._tracking_store = None
            handlers._model_registry_store = None
            initialize_backend_stores(backend_uri, default_artifact_root=tmp_path.as_uri())
            with ServerThread(app, get_safe_port()) as url:
                mlflow.set_tracking_uri(url)
                yield config


def test_sweep_multiple_configs_and_repeats(server_config):
    result = mlflow.genai.evaluate_sweep(
        data=DATA,
        scorers=[exact_match, output_length],
        predict_fns={"good": good_model, "bad": bad_model},
        n_repeats=3,
    )

    assert set(result.configs) == {"good", "bad"}
    for config in result.configs.values():
        assert len(config.child_run_ids) == 3
        assert len(config.results) == 3
        assert set(config.scorer_intervals) == {"exact_match", "output_length"}

    assert result.configs["good"].scorer_intervals["exact_match"].mean == 1.0
    assert result.configs["bad"].scorer_intervals["exact_match"].mean == 0.0
    assert result.best("exact_match") == "good"
    assert result.best("exact_match", higher_is_better=False) == "bad"


def test_sweep_boolean_scorer_uses_wilson_numeric_uses_t(server_config):
    result = mlflow.genai.evaluate_sweep(
        data=DATA,
        scorers=[exact_match, output_length],
        predict_fns={"good": good_model},
        n_repeats=3,
    )
    intervals = result.configs["good"].scorer_intervals
    assert intervals["exact_match"].method == "wilson"
    assert intervals["output_length"].method == "t"


def test_sweep_single_repeat_uses_bootstrap_for_numeric(server_config):
    result = mlflow.genai.evaluate_sweep(
        data=DATA,
        scorers=[output_length],
        predict_fns={"good": good_model},
        n_repeats=1,
    )
    assert result.configs["good"].scorer_intervals["output_length"].method == "bootstrap"
    assert len(result.configs["good"].child_run_ids) == 1


def test_sweep_flattens_summary_metrics_to_parent_run(server_config):
    result = mlflow.genai.evaluate_sweep(
        data=DATA,
        scorers=[exact_match],
        predict_fns={"good": good_model, "bad": bad_model},
        n_repeats=2,
    )
    parent = mlflow.get_run(result.parent_run_id)
    metrics = parent.data.metrics
    assert metrics["good/exact_match/mean"] == 1.0
    assert metrics["bad/exact_match/mean"] == 0.0
    assert "good/exact_match/ci_low" in metrics
    assert "good/exact_match/ci_high" in metrics
    assert parent.data.tags[MLFLOW_RUN_TYPE] == "genai_evaluate_sweep"


def test_sweep_child_runs_are_nested_under_parent(server_config):
    result = mlflow.genai.evaluate_sweep(
        data=DATA,
        scorers=[exact_match],
        predict_fns={"good": good_model},
        n_repeats=2,
    )
    from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

    for run_id in result.configs["good"].child_run_ids:
        child = mlflow.get_run(run_id)
        assert child.data.tags[MLFLOW_PARENT_RUN_ID] == result.parent_run_id


def test_sweep_comparison_df_shape(server_config):
    result = mlflow.genai.evaluate_sweep(
        data=DATA,
        scorers=[exact_match, output_length],
        predict_fns={"good": good_model, "bad": bad_model},
        n_repeats=2,
    )
    df = result.comparison_df
    # 2 configs x 2 scorers = 4 rows.
    assert len(df) == 4
    assert set(df["config"]) == {"good", "bad"}
    assert set(df["scorer"]) == {"exact_match", "output_length"}
    for col in ["mean", "ci_low", "ci_high", "ci_method", "latency_p50_ms"]:
        assert col in df.columns


def test_sweep_captures_cost_per_request(server_config):
    from mlflow.entities.span import SpanAttributeKey
    from mlflow.tracing.constant import TokenUsageKey

    def costed_model(question: str) -> str:
        span = mlflow.get_current_active_span()
        if span is not None:
            span.set_attribute(SpanAttributeKey.MODEL, "gpt-4o")
            span.set_attribute(
                SpanAttributeKey.CHAT_USAGE,
                {
                    TokenUsageKey.INPUT_TOKENS: 200,
                    TokenUsageKey.OUTPUT_TOKENS: 100,
                    TokenUsageKey.TOTAL_TOKENS: 300,
                },
            )
        return "answer"

    result = mlflow.genai.evaluate_sweep(
        data=DATA,
        scorers=[output_length],
        predict_fns={"gpt-4o": costed_model},
        n_repeats=2,
    )
    cost = result.configs["gpt-4o"].cost
    assert cost is not None
    assert cost.mean_per_request > 0
    assert cost.n_rows == len(DATA) * 2
    # Cost surfaces in both result views.
    assert "$" in result.summary_df().loc["gpt-4o", "cost/req"]
    assert result.comparison_df["cost_per_request_usd"].notna().all()


def test_sweep_cost_none_without_token_usage(server_config):
    result = mlflow.genai.evaluate_sweep(
        data=DATA,
        scorers=[output_length],
        predict_fns={"good": good_model},
        n_repeats=1,
    )
    # good_model reports no token usage, so cost is unavailable.
    assert result.configs["good"].cost is None
    assert result.summary_df().loc["good", "cost/req"] == "-"


def test_sweep_summary_df_wide_layout(server_config):
    result = mlflow.genai.evaluate_sweep(
        data=DATA,
        scorers=[exact_match, output_length],
        predict_fns={"good": good_model, "bad": bad_model},
        n_repeats=2,
    )
    df = result.summary_df()
    # One row per config, indexed by config name.
    assert list(df.index) == ["good", "bad"]
    # One column per scorer plus cost and latency columns.
    assert list(df.columns) == ["exact_match", "output_length", "cost/req", "latency_ms"]
    # Scorer cells are "mean +/- std" strings.
    assert df.loc["good", "exact_match"].startswith("1.000 +/- ")
    assert df.loc["bad", "exact_match"].startswith("0.000 +/- ")
    # Latency cell reports the four percentiles.
    for pct in ("p50=", "p90=", "p95=", "p99="):
        assert pct in df.loc["good", "latency_ms"]


def test_sweep_predict_once_reuses_predictions(server_config):
    calls = {"n": 0}

    def counting_model(question: str) -> str:
        calls["n"] += 1
        return "answer"

    result = mlflow.genai.evaluate_sweep(
        data=DATA,
        scorers=[output_length],
        predict_fns={"m": counting_model},
        n_repeats=3,
        predict_once=True,
    )
    # Predictions run only on repeat 0: one signature-validation probe + 3 rows = 4 calls.
    # Without predict_once this would be ~3x higher (predictions on every repeat).
    assert calls["n"] <= len(DATA) + 1
    assert len(result.configs["m"].child_run_ids) == 3


def test_sweep_latency_is_captured(server_config):
    def slow(question: str) -> str:
        time.sleep(0.05)
        return "x"

    result = mlflow.genai.evaluate_sweep(
        data=DATA,
        scorers=[output_length],
        predict_fns={"slow": slow},
        n_repeats=2,
    )
    latency = result.configs["slow"].latency
    assert latency is not None
    assert latency.n_rows == len(DATA) * 2
    assert latency.p50 >= 50.0


def test_sweep_list_of_predict_fns_named_by_function(server_config):
    result = mlflow.genai.evaluate_sweep(
        data=DATA,
        scorers=[exact_match],
        predict_fns=[good_model, bad_model],
        n_repeats=1,
    )
    assert set(result.configs) == {"good_model", "bad_model"}


def test_sweep_continues_when_a_config_fails(server_config):
    def broken_model(question: str) -> str:
        raise RuntimeError("endpoint down")

    result = mlflow.genai.evaluate_sweep(
        data=DATA,
        scorers=[exact_match],
        predict_fns={"good": good_model, "broken": broken_model},
        n_repeats=2,
    )
    # The sweep completes and still contains the healthy config's results.
    assert set(result.configs) == {"good", "broken"}
    assert result.configs["good"].scorer_intervals["exact_match"].mean == 1.0
    # A model whose every prediction errors scores lowest rather than crashing.
    assert result.configs["broken"].scorer_intervals["exact_match"].mean == 0.0


def test_sweep_traces_untraced_predict_fn(server_config):
    # A plain, undecorated predict_fn must still produce scored traces even though
    # the sweep defaults MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION on.
    def plain_model(question: str) -> str:
        return "good" if question in ANSWERS else "bad"

    result = mlflow.genai.evaluate_sweep(
        data=DATA,
        scorers=[output_length],
        predict_fns={"plain": plain_model},
        n_repeats=2,
    )
    assert "output_length" in result.configs["plain"].scorer_intervals


def test_sweep_does_not_set_skip_validation_permanently(server_config):
    from mlflow.environment_variables import MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION

    assert not MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION.is_set()
    mlflow.genai.evaluate_sweep(
        data=DATA,
        scorers=[exact_match],
        predict_fns={"good": good_model},
        n_repeats=1,
    )
    # The default is restored after the sweep, not leaked into the process.
    assert not MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION.is_set()


def test_sweep_rejects_invalid_n_repeats(server_config):
    with pytest.raises(MlflowException, match="n_repeats must be >= 1"):
        mlflow.genai.evaluate_sweep(
            data=DATA,
            scorers=[exact_match],
            predict_fns={"good": good_model},
            n_repeats=0,
        )


def test_sweep_rejects_empty_predict_fns(server_config):
    with pytest.raises(MlflowException, match="predict_fns must not be empty"):
        mlflow.genai.evaluate_sweep(
            data=DATA,
            scorers=[exact_match],
            predict_fns={},
        )


def test_normalize_predict_fns_dict_preserves_order():
    fns = {"b": good_model, "a": bad_model}
    assert list(_normalize_predict_fns(fns)) == ["b", "a"]


def test_normalize_predict_fns_list_dedupes_names():
    def m(question: str) -> str:
        return "x"

    other = lambda question: "y"  # noqa: E731
    other.__name__ = "m"
    named = _normalize_predict_fns([m, other])
    assert list(named) == ["m", "m_1"]


@pytest.mark.parametrize(
    ("values", "pct", "expected"),
    [
        ([10.0], 50, 10.0),
        ([10.0, 20.0], 50, 15.0),
        ([0.0, 100.0], 90, 90.0),
    ],
)
def test_percentile(values, pct, expected):
    assert _percentile(sorted(values), pct) == pytest.approx(expected)
