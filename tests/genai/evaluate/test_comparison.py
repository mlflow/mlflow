from typing import Any

import numpy as np
import pytest
from scipy import stats

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.comparison import (
    ComparisonResult,
    ScorerComparison,
    _compare_paired_values,
    _comparison_key,
)
from mlflow.genai.evaluation.context import NoneContext, _set_context
from mlflow.genai.scorers import scorer
from mlflow.tracking.client import MlflowClient


@pytest.fixture(autouse=True)
def reset_context():
    # `mlflow.genai.evaluate` leaves its run ID on the evaluation context of the calling thread.
    yield
    _set_context(NoneContext())


@scorer
def score(outputs) -> float:
    return outputs["score"]


@scorer
def passed(outputs) -> bool:
    return outputs["passed"]


@scorer
def rating(outputs) -> str:
    return "yes" if outputs["passed"] else "no"


def _rows(scores: list[float], passes: list[bool] | None = None) -> list[dict[str, Any]]:
    passes = passes or [s > 0.5 for s in scores]
    return [
        {"inputs": {"question": f"question {i}"}, "outputs": {"score": s, "passed": p}}
        for i, (s, p) in enumerate(zip(scores, passes))
    ]


def _evaluate(data: list[dict[str, Any]], scorers=(score, passed)) -> str:
    return mlflow.genai.evaluate(data=data, scorers=list(scorers)).run_id


def _binary_pairs(improved: int, regressed: int, ties: int) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.array([0.0] * improved + [1.0] * regressed + [1.0] * ties)
    candidate = np.array([1.0] * improved + [0.0] * regressed + [1.0] * ties)
    return baseline, candidate


def test_comparison_key():
    run_id = "0123456789abcdef0123456789abcdef"
    assert _comparison_key(run_id) == "compare/01234567"
    assert _comparison_key(run_id, "score", "p_value") == "compare/01234567/score/p_value"


def test_numeric_comparison_matches_scipy_and_detects_known_effect():
    rng = np.random.default_rng(1)
    baseline = rng.normal(0.6, 0.2, size=80)
    candidate = baseline + 0.1 + rng.normal(0, 0.1, size=80)
    deltas = candidate - baseline

    result = _compare_paired_values("score", baseline, candidate, binary=False)

    assert result.status == "ok"
    assert result.value_type == "numeric"
    assert result.method == "wilcoxon"
    assert result.n_paired == 80
    assert result.diff == pytest.approx(deltas.mean())
    assert result.baseline_mean == pytest.approx(baseline.mean())
    assert result.candidate_mean == pytest.approx(candidate.mean())
    assert result.p_value == pytest.approx(stats.wilcoxon(candidate, baseline).pvalue)
    assert result.t_test_p_value == pytest.approx(stats.ttest_rel(candidate, baseline).pvalue)
    assert result.effect_size == pytest.approx(deltas.mean() / deltas.std(ddof=1))
    assert result.ties == 0
    assert result.p_value < 0.001
    # The interval covers the true effect and excludes zero. It should also be close to the
    # normal-theory interval of the mean at this sample size.
    assert 0 < result.ci_low < 0.1 < result.ci_high
    t_low, t_high = stats.t.interval(0.95, df=79, loc=deltas.mean(), scale=stats.sem(deltas))
    assert result.ci_low == pytest.approx(t_low, abs=0.005)
    assert result.ci_high == pytest.approx(t_high, abs=0.005)


def test_bootstrap_interval_is_reproducible_and_widens_with_confidence_level():
    rng = np.random.default_rng(2)
    baseline = rng.normal(0.5, 0.2, size=40)
    candidate = baseline + rng.normal(0.05, 0.1, size=40)

    first = _compare_paired_values("score", baseline, candidate, binary=False)
    second = _compare_paired_values("score", baseline, candidate, binary=False)
    wide = _compare_paired_values("score", baseline, candidate, binary=False, confidence_level=0.99)

    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)
    assert wide.ci_low < first.ci_low < first.ci_high < wide.ci_high


@pytest.mark.parametrize(
    ("improved", "regressed", "ties", "expected_method"),
    [
        (12, 2, 26, "mcnemar_exact"),
        (3, 3, 10, "mcnemar_exact"),
        (40, 10, 50, "mcnemar_chi2"),
        (15, 10, 0, "mcnemar_chi2"),
    ],
)
def test_binary_comparison_matches_scipy(
    improved: int, regressed: int, ties: int, expected_method: str
):
    baseline, candidate = _binary_pairs(improved, regressed, ties)
    n = improved + regressed + ties

    result = _compare_paired_values("passed", baseline, candidate, binary=True)

    if expected_method == "mcnemar_exact":
        expected_p_value = stats.binomtest(improved, improved + regressed, 0.5).pvalue
    else:
        statistic = (abs(improved - regressed) - 1) ** 2 / (improved + regressed)
        expected_p_value = stats.chi2.sf(statistic, df=1)
    assert result.value_type == "binary"
    assert result.method == expected_method
    assert result.p_value == pytest.approx(expected_p_value)
    assert result.t_test_p_value is None
    assert result.diff == pytest.approx((improved - regressed) / n)
    assert result.ties == ties
    assert result.n_paired == n
    assert result.ci_low <= result.diff <= result.ci_high


@pytest.mark.parametrize("seed", range(5))
def test_null_comparison_does_not_reject(seed: int):
    # A/A: both runs draw each row's score from the same distribution.
    rng = np.random.default_rng(seed)
    quality = rng.uniform(0.2, 0.9, size=100)
    baseline = quality + rng.normal(0, 0.1, size=100)
    candidate = quality + rng.normal(0, 0.1, size=100)
    baseline_pass = (rng.uniform(size=100) < quality).astype(float)
    candidate_pass = (rng.uniform(size=100) < quality).astype(float)

    numeric = _compare_paired_values("score", baseline, candidate, binary=False)
    binary = _compare_paired_values("passed", baseline_pass, candidate_pass, binary=True)

    for comparison in (numeric, binary):
        assert comparison.p_value > 0.05
        assert comparison.ci_low < 0 < comparison.ci_high
    assert numeric.t_test_p_value > 0.05
    result = ComparisonResult("candidate", "baseline", {"score": numeric, "passed": binary})
    with pytest.raises(AssertionError, match="did not significantly improve"):
        result.assert_improved()


@pytest.mark.parametrize("binary", [True, False])
def test_identical_values_are_all_ties(binary: bool):
    values = np.array([1.0, 0.0, 1.0, 1.0])

    result = _compare_paired_values("s", values, values.copy(), binary=binary)

    assert result.status == "ok"
    assert result.diff == 0
    assert result.ties == 4
    assert result.p_value == 1.0
    assert (result.ci_low, result.ci_high) == (0.0, 0.0)
    assert np.isnan(result.effect_size)


@pytest.mark.parametrize(("n", "expected_p_value"), [(3, 0.25), (10, 2 / 2**10)])
def test_constant_nonzero_deltas(n: int, expected_p_value: float):
    baseline = np.arange(n, dtype=float)
    candidate = baseline + 0.5

    comparison = _compare_paired_values("score", baseline, candidate, binary=False)

    assert comparison.status == "ok"
    assert comparison.method == "wilcoxon"
    assert comparison.diff == 0.5
    assert comparison.ties == 0
    assert (comparison.ci_low, comparison.ci_high) == (0.5, 0.5)
    # The t statistic and d_z divide by a zero standard deviation.
    assert np.isnan(comparison.t_test_p_value)
    assert np.isnan(comparison.effect_size)
    # All n differences share one sign, so the exact two-sided p-value is 2 / 2**n.
    assert comparison.p_value == pytest.approx(expected_p_value)
    assert comparison.p_value == pytest.approx(stats.wilcoxon(candidate, baseline).pvalue)

    result = ComparisonResult("candidate", "baseline", {"score": comparison})
    if expected_p_value < 0.05:
        result.assert_improved(["score"])
    else:
        with pytest.raises(AssertionError, match="score: diff=\\+0.5"):
            result.assert_improved(["score"])


@pytest.mark.parametrize("n", [0, 1])
def test_insufficient_pairs(n: int):
    values = np.arange(n, dtype=float)

    result = _compare_paired_values("score", values, values + 1, binary=False)

    assert result == ScorerComparison(scorer="score", status="insufficient_pairs", n_paired=n)


def test_compare_evaluations_dispatches_on_value_type():
    baseline_scores = [0.1, 0.3, 0.4, 0.2, 0.6, 0.7, 0.3, 0.2]
    candidate_scores = [0.6, 0.7, 0.9, 0.8, 0.7, 0.9, 0.6, 0.4]
    scorers = (score, passed, rating)
    baseline_run_id = _evaluate(_rows(baseline_scores), scorers)
    candidate_run_id = _evaluate(_rows(candidate_scores), scorers)

    result = mlflow.genai.compare_evaluations(candidate_run_id, baseline_run_id)

    assert result.candidate_run_id == candidate_run_id
    assert result.baseline_run_id == baseline_run_id
    assert list(result.scorers) == ["passed", "rating", "score"]
    assert result.unpaired_baseline_trace_ids == []
    assert result.unpaired_candidate_trace_ids == []
    assert len(result.paired_rows) == 8

    numeric = result.scorers["score"]
    assert numeric.value_type == "numeric"
    assert numeric.method == "wilcoxon"
    assert numeric.n_paired == 8
    assert numeric.diff == pytest.approx(np.mean(candidate_scores) - np.mean(baseline_scores))
    assert numeric.p_value == pytest.approx(
        stats.wilcoxon(candidate_scores, baseline_scores).pvalue
    )
    assert numeric.t_test_p_value == pytest.approx(
        stats.ttest_rel(candidate_scores, baseline_scores).pvalue
    )

    # 5 rows flip from fail to pass, none regress.
    for name in ("passed", "rating"):
        binary = result.scorers[name]
        assert binary.value_type == "binary"
        assert binary.method == "mcnemar_exact"
        assert binary.diff == pytest.approx(5 / 8)
        assert binary.ties == 3
        assert binary.p_value == pytest.approx(stats.binomtest(5, 5, 0.5).pvalue)
        assert binary.t_test_p_value is None

    summary = result.summary()
    assert summary["scorer"].tolist() == ["passed", "rating", "score"]
    assert summary.set_index("scorer").loc["score", "n_paired"] == 8
    assert {"diff", "ci_low", "ci_high", "p_value", "effect_size", "ties"} <= set(summary.columns)


def test_compare_evaluations_reports_unmatched_rows_and_intersects_scorers():
    rows = _rows([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    extra_row = {"inputs": {"question": "candidate only"}, "outputs": {"score": 1.0}}
    baseline_run_id = _evaluate(rows, scorers=[score, passed])
    candidate_run_id = _evaluate([*rows[:4], extra_row], scorers=[score])

    result = mlflow.genai.compare_evaluations(candidate_run_id, baseline_run_id, log_results=False)

    assert list(result.scorers) == ["score"]
    assert result.scorers["score"].n_paired == 4
    assert result.scorers["score"].ties == 4
    assert len(result.paired_rows) == 4
    assert len(result.unpaired_baseline_trace_ids) == 2
    assert len(result.unpaired_candidate_trace_ids) == 1
    paired_trace_ids = {row["candidate_trace_id"] for row in result.paired_rows}
    assert paired_trace_ids.isdisjoint(result.unpaired_candidate_trace_ids)
    assert all(row["scores"]["score"]["delta"] == 0 for row in result.paired_rows)


def test_compare_evaluations_pairs_on_inputs_when_request_ids_differ():
    # `predict_fn` runs get a fresh eval request ID per row, so rows pair on the inputs hash.
    data = [{"inputs": {"question": f"question {i}"}} for i in range(5)]

    def run(offset: float) -> str:
        def predict_fn(question: str) -> dict[str, float]:
            return {"score": int(question.split()[-1]) / 10 + offset}

        return mlflow.genai.evaluate(data=data, predict_fn=predict_fn, scorers=[score]).run_id

    baseline_run_id = run(0.0)
    candidate_run_id = run(0.25)

    result = mlflow.genai.compare_evaluations(candidate_run_id, baseline_run_id, log_results=False)

    assert {row["paired_on"] for row in result.paired_rows} == {"inputs_hash"}
    assert result.unpaired_baseline_trace_ids == []
    assert result.unpaired_candidate_trace_ids == []
    assert result.scorers["score"].n_paired == 5
    assert result.scorers["score"].diff == pytest.approx(0.25)
    assert [row["scores"]["score"]["delta"] for row in result.paired_rows] == pytest.approx(
        [0.25] * 5
    )


def test_compare_evaluations_insufficient_pairs():
    baseline_run_id = _evaluate(_rows([0.1, 0.2, 0.3]))
    candidate_run_id = _evaluate(_rows([0.9]))

    result = mlflow.genai.compare_evaluations(candidate_run_id, baseline_run_id)

    comparison = result.scorers["score"]
    assert comparison.status == "insufficient_pairs"
    assert comparison.n_paired == 1
    assert comparison.p_value is None
    assert len(result.unpaired_baseline_trace_ids) == 2
    with pytest.raises(AssertionError, match="insufficient_pairs"):
        result.assert_improved(["score"])
    # Only the pair count is logged when there are no statistics.
    prefix = _comparison_key(baseline_run_id, "score")
    metrics = MlflowClient().get_run(candidate_run_id).data.metrics
    assert {k: v for k, v in metrics.items() if k.startswith(prefix)} == {f"{prefix}/n_paired": 1}


def test_compare_evaluations_logs_results_to_candidate_run():
    baseline_scores = [0.1, 0.3, 0.4, 0.2, 0.6, 0.7]
    candidate_scores = [0.6, 0.7, 0.9, 0.8, 0.7, 0.8]
    baseline_run_id = _evaluate(_rows(baseline_scores))
    candidate_run_id = _evaluate(_rows(candidate_scores))

    result = mlflow.genai.compare_evaluations(candidate_run_id, baseline_run_id)

    short_id = baseline_run_id[:8]
    metrics = MlflowClient().get_run(candidate_run_id).data.metrics
    for name, comparison in result.scorers.items():
        for stat in ("diff", "ci_low", "ci_high", "p_value", "effect_size", "n_paired"):
            assert metrics[f"compare/{short_id}/{name}/{stat}"] == pytest.approx(
                getattr(comparison, stat)
            )
    baseline_metrics = MlflowClient().get_run(baseline_run_id).data.metrics
    assert not any(key.startswith("compare/") for key in baseline_metrics)

    artifact = mlflow.artifacts.load_dict(f"runs:/{candidate_run_id}/compare/{short_id}.json")
    assert artifact["baseline_run_id"] == baseline_run_id
    assert artifact["candidate_run_id"] == candidate_run_id
    assert artifact["scorers"]["score"]["p_value"] == pytest.approx(result.scorers["score"].p_value)
    assert artifact["paired_rows"] == result.paired_rows
    assert sorted(row["scores"]["score"]["delta"] for row in artifact["paired_rows"]) == (
        pytest.approx(sorted(c - b for c, b in zip(candidate_scores, baseline_scores)))
    )
    assert {row["paired_on"] for row in artifact["paired_rows"]} == {"eval_request_id"}
    tags = MlflowClient().get_run(candidate_run_id).data.tags
    assert not any(key.startswith("mlflow.compare") for key in tags)


def test_compare_evaluations_keeps_results_of_multiple_baselines():
    first_baseline_run_id = _evaluate(_rows([0.1, 0.2, 0.3, 0.4]))
    second_baseline_run_id = _evaluate(_rows([0.5, 0.5, 0.6, 0.9]))
    candidate_run_id = _evaluate(_rows([0.6, 0.7, 0.8, 0.9]))

    first = mlflow.genai.compare_evaluations(candidate_run_id, first_baseline_run_id)
    second = mlflow.genai.compare_evaluations(candidate_run_id, second_baseline_run_id)

    metrics = MlflowClient().get_run(candidate_run_id).data.metrics
    first_key = _comparison_key(first_baseline_run_id, "score", "diff")
    second_key = _comparison_key(second_baseline_run_id, "score", "diff")
    assert first_key != second_key
    assert metrics[first_key] == pytest.approx(first.scorers["score"].diff) == pytest.approx(0.5)
    assert metrics[second_key] == pytest.approx(second.scorers["score"].diff)
    assert metrics[second_key] == pytest.approx(0.125)
    for baseline_run_id in (first_baseline_run_id, second_baseline_run_id):
        artifact = mlflow.artifacts.load_dict(
            f"runs:/{candidate_run_id}/{_comparison_key(baseline_run_id)}.json"
        )
        assert artifact["baseline_run_id"] == baseline_run_id


def test_compare_evaluations_does_not_log_when_disabled():
    baseline_run_id = _evaluate(_rows([0.1, 0.2, 0.3]))
    candidate_run_id = _evaluate(_rows([0.4, 0.6, 0.8]))

    mlflow.genai.compare_evaluations(candidate_run_id, baseline_run_id, log_results=False)

    client = MlflowClient()
    metrics = client.get_run(candidate_run_id).data.metrics
    assert not any(key.startswith("compare/") for key in metrics)
    assert client.list_artifacts(candidate_run_id, "compare") == []


def test_assert_improved():
    rng = np.random.default_rng(3)
    baseline = rng.normal(0.5, 0.1, size=50)
    improved = _compare_paired_values(
        "improved", baseline, baseline + 0.1 + rng.normal(0, 0.05, size=50), binary=False
    )
    regressed = _compare_paired_values(
        "regressed", baseline, baseline - 0.1 + rng.normal(0, 0.05, size=50), binary=False
    )
    noise = _compare_paired_values(
        "noise", baseline, baseline + rng.normal(0, 0.05, size=50), binary=False
    )
    latency = _compare_paired_values(
        "latency", baseline, baseline - 0.1, binary=False, greater_is_better=False
    )
    result = ComparisonResult(
        candidate_run_id="candidate",
        baseline_run_id="baseline",
        scorers={c.scorer: c for c in (improved, regressed, noise, latency)},
    )

    result.assert_improved(["improved"])
    result.assert_improved(["improved", "latency"], alpha=0.01)
    with pytest.raises(AssertionError, match="regressed: diff=-0"):
        result.assert_improved(["improved", "regressed"])
    with pytest.raises(AssertionError, match="noise: diff="):
        result.assert_improved(["noise"])
    with pytest.raises(AssertionError, match="did not significantly improve") as exc_info:
        result.assert_improved()
    assert "- improved:" not in str(exc_info.value)
    assert "- latency:" not in str(exc_info.value)
    # A significant p-value is not enough at a stricter alpha.
    with pytest.raises(AssertionError, match="improved: diff="):
        result.assert_improved(["improved"], alpha=1e-30)
    with pytest.raises(MlflowException, match="were not compared"):
        result.assert_improved(["missing"])


def test_compare_evaluations_greater_is_better_override():
    baseline_run_id = _evaluate(_rows([0.5, 0.6, 0.7, 0.8, 0.9, 0.7, 0.6, 0.8]))
    candidate_run_id = _evaluate(_rows([0.1, 0.2, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3]))

    result = mlflow.genai.compare_evaluations(
        candidate_run_id,
        baseline_run_id,
        greater_is_better={"score": False},
        log_results=False,
    )

    assert result.scorers["score"].greater_is_better is False
    assert result.scorers["passed"].greater_is_better is True
    assert result.scorers["score"].diff < 0
    result.assert_improved(["score"])
    with pytest.raises(AssertionError, match="passed: diff="):
        result.assert_improved(["passed"])


def test_compare_evaluations_rejects_invalid_arguments():
    with pytest.raises(MlflowException, match="must be different runs"):
        mlflow.genai.compare_evaluations("run", "run")
    with pytest.raises(MlflowException, match="`confidence_level` must be between 0 and 1"):
        mlflow.genai.compare_evaluations("candidate", "baseline", confidence_level=95)
