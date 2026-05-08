"""Performance benchmarks for assessment numeric filter queries (PR #21811).

The numeric comparison path (`feedback.score > 0.8`) applies a CASE/CAST expression
to every row in the assessments table, making the predicate non-sargable — no index
on the `value` column can be used. These benchmarks quantify that cost at scale and
compare it against the existing string-equality path.

Run via:
    uv run pytest dev/benchmarks/tracing/test_assessment_numeric_perf.py \\
        --benchmark-only \\
        --benchmark-json=benchmark-results.json \\
        -v
"""

import random
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from mlflow.entities.assessment import AssessmentSource, Feedback
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

from _data import generate_trace_data

# ---------------------------------------------------------------------------
# Dataset parameters
# ---------------------------------------------------------------------------

SEED_TRACES = 2_000
ASSESSORS_PER_TRACE = 3   # number of human raters per trace
SCORE_NAMES = ["score", "relevance", "faithfulness"]

# Fraction of traces that receive a numeric vs. string vs. boolean assessment.
# Matches a realistic production distribution.
NUMERIC_FRACTION = 0.70
STRING_FRACTION = 0.20
# remaining 10% are boolean (True → 1.0 via CASE)

HUMAN_SOURCE = AssessmentSource(source_type="HUMAN", source_id="reviewer@example.com")


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bench_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("bench_assessment")


@pytest.fixture(scope="session")
def store(bench_dir: Path) -> SqlAlchemyStore:
    db_uri = f"sqlite:///{bench_dir / 'mlflow_assessment.db'}"
    (bench_dir / "artifacts").mkdir()
    artifact_root = (bench_dir / "artifacts").as_uri()
    return SqlAlchemyStore(db_uri, artifact_root)


@pytest.fixture(scope="session")
def experiment_id(store: SqlAlchemyStore) -> str:
    return str(store.create_experiment("bench_assessment_numeric"))


@pytest.fixture(scope="session")
def seeded(store: SqlAlchemyStore, experiment_id: str) -> list[str]:
    """Seed SEED_TRACES traces with a mix of numeric, string, and boolean assessments."""
    rng = random.Random(42)
    trace_ids: list[str] = []

    for _ in range(SEED_TRACES):
        ti, sp = generate_trace_data(experiment_id, num_spans=5, rng=rng)
        store.start_trace(ti)
        store.log_spans(experiment_id, sp)
        trace_id = ti.trace_id
        trace_ids.append(trace_id)

        roll = rng.random()
        for score_name in SCORE_NAMES:
            if roll < NUMERIC_FRACTION:
                value = round(rng.uniform(0.0, 1.0), 2)
            elif roll < NUMERIC_FRACTION + STRING_FRACTION:
                value = rng.choice(["high", "medium", "low"])
            else:
                value = rng.choice([True, False])

            store.create_assessment(
                Feedback(
                    trace_id=trace_id,
                    name=score_name,
                    value=value,
                    source=HUMAN_SOURCE,
                )
            )

    return trace_ids


# ---------------------------------------------------------------------------
# Benchmarks — numeric comparison (new CASE/CAST path)
# ---------------------------------------------------------------------------


def test_numeric_gt_50pct_selectivity(
    benchmark: BenchmarkFixture,
    store: SqlAlchemyStore,
    experiment_id: str,
    seeded: list[str],
) -> None:
    """feedback.score > 0.5  — ~50% of numeric rows pass; tests mid-selectivity."""
    benchmark(
        store.search_traces,
        locations=[experiment_id],
        max_results=100,
        filter_string="feedback.`score` > 0.5",
    )


def test_numeric_gt_high_selectivity(
    benchmark: BenchmarkFixture,
    store: SqlAlchemyStore,
    experiment_id: str,
    seeded: list[str],
) -> None:
    """feedback.score > 0.95  — ~5% of rows pass; tests high-selectivity path.

    High selectivity is the worst case for a non-sargable scan: the DB cannot
    use an index range to stop early, so all rows are still evaluated.
    """
    benchmark(
        store.search_traces,
        locations=[experiment_id],
        max_results=100,
        filter_string="feedback.`score` > 0.95",
    )


def test_numeric_lt(
    benchmark: BenchmarkFixture,
    store: SqlAlchemyStore,
    experiment_id: str,
    seeded: list[str],
) -> None:
    """feedback.score < 0.3  — tests less-than path."""
    benchmark(
        store.search_traces,
        locations=[experiment_id],
        max_results=100,
        filter_string="feedback.`score` < 0.3",
    )


def test_numeric_gte_lte(
    benchmark: BenchmarkFixture,
    store: SqlAlchemyStore,
    experiment_id: str,
    seeded: list[str],
) -> None:
    """feedback.score >= 0.4 — tests >= operator."""
    benchmark(
        store.search_traces,
        locations=[experiment_id],
        max_results=100,
        filter_string="feedback.`score` >= 0.4",
    )


# ---------------------------------------------------------------------------
# Benchmarks — string equality (existing path, for comparison baseline)
# ---------------------------------------------------------------------------


def test_string_equality_baseline(
    benchmark: BenchmarkFixture,
    store: SqlAlchemyStore,
    experiment_id: str,
    seeded: list[str],
) -> None:
    """feedback.score = 'high'  — existing string-equality path (no CAST).

    This is the baseline to compare against the new numeric path.
    Both are non-sargable against SQLite's text storage, but the string path
    avoids the SUBSTRING + CAST overhead.
    """
    benchmark(
        store.search_traces,
        locations=[experiment_id],
        max_results=100,
        filter_string="feedback.`score` = 'high'",
    )


# ---------------------------------------------------------------------------
# Benchmarks — compound filters (numeric + string combined)
# ---------------------------------------------------------------------------


def test_compound_numeric_and_string(
    benchmark: BenchmarkFixture,
    store: SqlAlchemyStore,
    experiment_id: str,
    seeded: list[str],
) -> None:
    """feedback.score > 0.8 AND feedback.relevance = 'high'

    Exercises the JOIN + two assessment filter predicates path.  The planner
    must evaluate both assessments tables; the numeric CAST runs on the first,
    then surviving rows are string-compared on the second.
    """
    benchmark(
        store.search_traces,
        locations=[experiment_id],
        max_results=100,
        filter_string="feedback.`score` > 0.8 AND feedback.`relevance` = 'high'",
    )


def test_compound_two_numeric_filters(
    benchmark: BenchmarkFixture,
    store: SqlAlchemyStore,
    experiment_id: str,
    seeded: list[str],
) -> None:
    """feedback.score > 0.7 AND feedback.relevance >= 0.5

    Two numeric filters — two separate non-sargable full scans of the
    assessments table joined with traces.
    """
    benchmark(
        store.search_traces,
        locations=[experiment_id],
        max_results=100,
        filter_string="feedback.`score` > 0.7 AND feedback.`relevance` >= 0.5",
    )


# ---------------------------------------------------------------------------
# Scaling benchmark — vary dataset size at runtime
# ---------------------------------------------------------------------------


_SCALE_SIZES = [100, 500, 1_000, 2_000]


@pytest.fixture(scope="session", params=_SCALE_SIZES, ids=[f"n{n}" for n in _SCALE_SIZES])
def scaled_experiment(store: SqlAlchemyStore, request: pytest.FixtureRequest) -> Iterator[str]:
    n: int = request.param
    rng = random.Random(99)
    exp_id = str(store.create_experiment(f"bench_scale_{n}_{uuid.uuid4().hex[:6]}"))

    for _ in range(n):
        ti, sp = generate_trace_data(exp_id, num_spans=5, rng=rng)
        store.start_trace(ti)
        store.log_spans(exp_id, sp)
        store.create_assessment(
            Feedback(
                trace_id=ti.trace_id,
                name="score",
                value=round(rng.uniform(0.0, 1.0), 2),
                source=HUMAN_SOURCE,
            )
        )

    yield exp_id


def test_numeric_scaling(
    benchmark: BenchmarkFixture,
    store: SqlAlchemyStore,
    scaled_experiment: str,
) -> None:
    """Measures wall-clock time for feedback.score > 0.5 as dataset grows.

    Expected: linear scaling (O(n)) because the CASE/CAST predicate is
    non-sargable.  A plot of these results should be a straight line through
    the origin; any super-linear growth indicates additional join costs.
    """
    benchmark(
        store.search_traces,
        locations=[scaled_experiment],
        max_results=100,
        filter_string="feedback.`score` > 0.5",
    )


def test_string_scaling_baseline(
    benchmark: BenchmarkFixture,
    store: SqlAlchemyStore,
    scaled_experiment: str,
) -> None:
    """Same dataset sizes, string equality — gives a lower-bound scaling curve."""
    benchmark(
        store.search_traces,
        locations=[scaled_experiment],
        max_results=100,
        filter_string="feedback.`score` = '0.5'",
    )
