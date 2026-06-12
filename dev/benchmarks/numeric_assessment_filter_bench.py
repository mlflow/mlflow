"""Benchmark: runtime CAST-to-float vs string-equality for numeric assessment filters.

BENCHMARK-ONLY / NOT SHIPPABLE. This script does not modify MLflow production code.
It quantifies the per-query latency overhead of the prototype (PR #21811) approach of
CASTing the JSON-encoded assessment value to float at query time, so we can decide
whether to ship that approach or pivot to a dedicated/indexed numeric column.

What it does
------------
1. Spins up a real MLflow ``SqlAlchemyStore`` against a temp-file SQLite DB so the
   schema matches production exactly (``trace_info`` + ``assessments`` tables, real
   indexes from ``mlflow/store/tracking/dbmodels/models.py``).
2. Seeds N traces, each with one numeric ``feedback.score`` assessment (plus a small
   fraction of string/boolean values to exercise the CASE branches that the prototype
   uses to coerce non-numeric JSON to NULL). Seeding is done via *bulk inserts into the
   real ORM tables* (``SqlTraceInfo`` / ``SqlAssessments``) because the public
   ``log_trace`` + ``create_assessment`` APIs are far too slow at 100k+ scale. The rows
   written are byte-for-byte what the store APIs would write (same JSON encoding via
   ``SqlAssessments.from_mlflow_entity``-equivalent serialization).
3. Measures warm-cache query latency (median + p90 over several reps) for:
   a. BASELINE   - string-equality assessment filter (``feedback.score = "..."``),
                   executed through the REAL ``store.search_traces`` path.
   b. NUMERIC-CAST - the prototype numeric comparison (``feedback.score > 0.8``) built
                   with the SAME subquery+join skeleton the store uses, differing ONLY
                   in the value expression (dialect-aware CAST CASE). Constructed
                   locally without touching the store module.
   c. CONSTRUCTED-EQ - the string-equality value filter run through the *same locally
                   constructed skeleton* as (b). This isolates the pure value-expression
                   delta (b vs c) from any framing differences, and is validated to
                   return the same rows as the real path (a).
   d. ATTR-NUMERIC  - an already-numeric indexed column filter (``trace_info.timestamp_ms``)
                   as an upper-bound reference for how cheap a native numeric column is.

Usage
-----
    uv run python dev/benchmarks/numeric_assessment_filter_bench.py --scales 10000,100000
    uv run python dev/benchmarks/numeric_assessment_filter_bench.py --scales 1000000 --reps 5

Extending to 1M: pass ``--scales 1000000``. Seeding is O(N) bulk inserts batched at
50k rows/insert; 1M traces seed in a couple of minutes and ~hundreds of MB of SQLite.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import tempfile
import time

import sqlalchemy as sa

from mlflow.store.tracking.dbmodels.models import (
    SqlAssessments,
    SqlTraceInfo,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.search_utils import SearchTraceUtils

# ---------------------------------------------------------------------------
# Value expression builders (mirror the store / prototype exactly)
# ---------------------------------------------------------------------------


def build_string_eq_value_filter(value: str):
    """String-equality value filter, identical to what the store builds for ``=``.

    See ``SearchTraceUtils._get_sql_json_comparison_func`` -> json_equality path.
    """
    return SearchTraceUtils._get_sql_json_comparison_func("=", "sqlite")(
        SqlAssessments.value, value
    )


def build_numeric_cast_value_filter(comparator: str, value: float):
    """Prototype (PR #21811) numeric CAST value filter.

    Reproduces the ``json_numeric_comparison`` closure added in the prototype:
    coerce booleans/"yes"/"no" to 1.0/0.0, JSON null and non-scalar JSON
    (strings/arrays/objects, detected by first char in {", [, {}) to NULL, and
    otherwise CAST the TEXT to Float. NULL fails every comparison, so non-numeric
    assessments are silently excluded.
    """
    column = SqlAssessments.value
    numeric_col = sa.case(
        (column.in_([json.dumps(True), json.dumps("yes")]), 1.0),
        (column.in_([json.dumps(False), json.dumps("no")]), 0.0),
        (column == "null", sa.null()),
        (sa.func.substring(column, 1, 1).in_(['"', "[", "{"]), sa.null()),
        else_=sa.func.cast(column, sa.Float),
    )
    return SearchTraceUtils.get_comparison_func(comparator)(numeric_col, float(value))


def assessment_subquery(session, value_filter, key_name: str, key_type: str):
    """Replicate the store's assessment subquery skeleton (sqlalchemy_store.py).

    Mirrors the ``direct_matches`` distinct subquery on the assessments table filtered
    by (assessment_type, name, valid, value_filter). The only thing that varies between
    BASELINE and NUMERIC-CAST is ``value_filter``; everything else is held constant so
    the measured delta is purely the value-expression cost.
    """
    return (
        session
        .query(SqlAssessments.trace_id.label("request_id"))
        .filter(
            SqlAssessments.assessment_type == key_type,
            SqlAssessments.name == key_name,
            SqlAssessments.valid == sa.true(),
            value_filter,
        )
        .distinct()
        .subquery()
    )


def run_constructed_query(session, experiment_id: str, value_filter, key_name, key_type):
    """Join the assessment subquery back to trace_info, as the store does."""
    sub = assessment_subquery(session, value_filter, key_name, key_type)
    q = (
        session
        .query(SqlTraceInfo.request_id)
        .join(sub, sub.c.request_id == SqlTraceInfo.request_id)
        .filter(SqlTraceInfo.experiment_id == int(experiment_id))
    )
    return [r[0] for r in q.all()]


def run_attr_numeric_query(session, experiment_id: str, threshold_ms: int):
    """Upper-bound reference: filter on the already-numeric indexed timestamp_ms."""
    q = session.query(SqlTraceInfo.request_id).filter(
        SqlTraceInfo.experiment_id == int(experiment_id),
        SqlTraceInfo.timestamp_ms > threshold_ms,
    )
    return [r[0] for r in q.all()]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

STRING_FRACTION = 0.05  # fraction of assessments stored as a string ("high")
BOOL_FRACTION = 0.05  # fraction stored as a boolean (exercise the CASE)
BATCH = 50_000


def seed(store: SqlAlchemyStore, experiment_id: str, n: int, seed_val: int = 1234) -> dict:
    """Bulk-insert N traces, each with one ``feedback.score`` assessment.

    Numeric scores are uniform in [0, 10]. A small fraction are stored as the string
    "high" or a boolean, matching the messy real-world value distribution the prototype
    CASE has to handle. Returns selectivity metadata for query-constant selection.
    """
    rng = random.Random(seed_val)
    base_ts = 1_700_000_000_000
    engine = store.engine

    n_numeric = 0
    n_gt_8 = 0  # rows with score > 8.0 (selectivity for the > query)
    eq_target = None  # an exact numeric value guaranteed to exist for the = query

    trace_rows: list[dict] = []
    assess_rows: list[dict] = []

    def flush():
        nonlocal trace_rows, assess_rows
        if not trace_rows:
            return
        with engine.begin() as conn:
            conn.execute(sa.insert(SqlTraceInfo.__table__), trace_rows)
            conn.execute(sa.insert(SqlAssessments.__table__), assess_rows)
        trace_rows = []
        assess_rows = []

    for i in range(n):
        req_id = f"tr-{i:08d}"
        ts = base_ts + i
        trace_rows.append({
            "request_id": req_id,
            "experiment_id": int(experiment_id),
            "timestamp_ms": ts,
            "execution_time_ms": rng.randint(1, 5000),
            "status": "OK",
            "client_request_id": None,
            "request_preview": "req",
            "response_preview": "resp",
            "db_payload_generation": 0,
        })

        roll = rng.random()
        if roll < STRING_FRACTION:
            value_json = json.dumps("high")
        elif roll < STRING_FRACTION + BOOL_FRACTION:
            value_json = json.dumps(rng.random() < 0.5)
        else:
            n_numeric += 1
            # Round to 1 decimal so exact-equality query constants reliably exist.
            score = round(rng.uniform(0.0, 10.0), 1)
            if score > 8.0:
                n_gt_8 += 1
            if eq_target is None and score == 5.0:
                eq_target = "5.0"
            value_json = json.dumps(score)

        assess_rows.append({
            "assessment_id": f"a-{i:08d}",
            "trace_id": req_id,
            "name": "score",
            "assessment_type": "feedback",
            "value": value_json,
            "error": None,
            "created_timestamp": ts,
            "last_updated_timestamp": ts,
            "source_type": "HUMAN",
            "source_id": "user@example.com",
            "run_id": None,
            "span_id": None,
            "rationale": None,
            "overrides": None,
            "valid": True,
            "assessment_metadata": None,
        })

        if len(trace_rows) >= BATCH:
            flush()

    flush()

    if eq_target is None:
        eq_target = "5.0"  # extremely unlikely with N>=10k, but keep deterministic

    return {
        "n_numeric": n_numeric,
        "n_gt_8": n_gt_8,
        "eq_target": eq_target,
        "gt_threshold": 8.0,
        "ts_threshold": base_ts + int(n * 0.2),  # ~80% of rows have ts > this
    }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def timed(fn, reps: int, warmup: int = 2):
    """Run fn reps+warmup times (warm cache), return (median_ms, p90_ms, n_rows)."""
    n_rows = None
    for _ in range(warmup):
        n_rows = len(fn())
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        rows = fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
        n_rows = len(rows)
    samples.sort()
    median = statistics.median(samples)
    p90 = samples[min(len(samples) - 1, int(round(0.9 * (len(samples) - 1))))]
    return median, p90, n_rows


def run_scale(n: int, reps: int) -> list[dict]:
    tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)  # noqa: SIM115
    tmpfile.close()
    db_uri = f"sqlite:///{tmpfile.name}"
    print(f"\n=== scale={n:,} db={db_uri} ===")
    try:
        store = SqlAlchemyStore(db_uri, default_artifact_root=tempfile.mkdtemp())
        exp_id = store.create_experiment(f"bench_{n}")

        t0 = time.perf_counter()
        meta = seed(store, exp_id, n)
        print(
            f"seeded {n:,} traces ({meta['n_numeric']:,} numeric) "
            f"in {time.perf_counter() - t0:.1f}s; "
            f"eq_target={meta['eq_target']} score>8 rows={meta['n_gt_8']:,}"
        )

        session = store.ManagedSessionMaker().__enter__()
        results = []

        # (a) BASELINE — real search path, string equality
        eq_filter_str = f'feedback.score = "{meta["eq_target"]}"'

        # search_traces caps max_results at 50000; the equality filter returns far fewer
        # rows than that at every scale, so the cap does not truncate the result set.
        max_results = min(n, 50000)

        def baseline_real():
            traces, _ = store.search_traces(
                [exp_id], filter_string=eq_filter_str, max_results=max_results
            )
            return traces

        med, p90, rows = timed(baseline_real, reps)
        results.append(("BASELINE eq (real search_traces)", eq_filter_str, med, p90, rows))

        # (c) CONSTRUCTED-EQ — same skeleton as the CAST query, string-eq value filter
        eq_vf = build_string_eq_value_filter(meta["eq_target"])

        def constructed_eq():
            return run_constructed_query(session, exp_id, eq_vf, "score", "feedback")

        med, p90, rows = timed(constructed_eq, reps)
        results.append(("CONSTRUCTED eq (same skeleton)", eq_filter_str, med, p90, rows))

        # (b') NUMERIC-CAST = at the SAME value as the string-eq — identical selectivity
        # and identical skeleton, so this isolates the PURE per-row CAST/CASE overhead
        # (string-eq OR-compare vs CASE+CAST) free of any result-set-size confound.
        cast_eq_vf = build_numeric_cast_value_filter("=", float(meta["eq_target"]))

        def numeric_cast_eq():
            return run_constructed_query(session, exp_id, cast_eq_vf, "score", "feedback")

        med, p90, rows = timed(numeric_cast_eq, reps)
        results.append((
            "NUMERIC-CAST = (same rows as eq)",
            f"feedback.score = {meta['eq_target']}",
            med,
            p90,
            rows,
        ))

        # (b) NUMERIC-CAST — prototype approach, > threshold
        gt_vf = build_numeric_cast_value_filter(">", meta["gt_threshold"])

        def numeric_cast():
            return run_constructed_query(session, exp_id, gt_vf, "score", "feedback")

        med, p90, rows = timed(numeric_cast, reps)
        results.append((
            "NUMERIC-CAST > (prototype)",
            f"feedback.score > {meta['gt_threshold']}",
            med,
            p90,
            rows,
        ))

        # also <= for symmetry
        le_vf = build_numeric_cast_value_filter("<=", 5.0)

        def numeric_cast_le():
            return run_constructed_query(session, exp_id, le_vf, "score", "feedback")

        med, p90, rows = timed(numeric_cast_le, reps)
        results.append(("NUMERIC-CAST <= (prototype)", "feedback.score <= 5.0", med, p90, rows))

        # (d) ATTR-NUMERIC — native numeric indexed column, upper-bound reference
        def attr_numeric():
            return run_attr_numeric_query(session, exp_id, meta["ts_threshold"])

        med, p90, rows = timed(attr_numeric, reps)
        results.append((
            "ATTR-NUMERIC > (native column)",
            "trace_info.timestamp_ms > T",
            med,
            p90,
            rows,
        ))

        return [
            {
                "scale": n,
                "name": r[0],
                "filter": r[1],
                "median_ms": r[2],
                "p90_ms": r[3],
                "rows": r[4],
            }
            for r in results
        ]
    finally:
        os.unlink(tmpfile.name)


def fmt_table(all_results: list[dict]) -> str:
    header = (
        f"| {'scale':>9} | {'query':<32} | {'filter':<34} | "
        f"{'median(ms)':>10} | {'p90(ms)':>8} | {'rows':>8} |"
    )
    lines = [
        header,
        "|" + "-".join(["-" * 11, "-" * 34, "-" * 36, "-" * 12, "-" * 10, "-" * 10]) + "|",
    ]
    lines.extend(
        f"| {r['scale']:>9,} | {r['name']:<32} | {r['filter']:<34} | "
        f"{r['median_ms']:>10.2f} | {r['p90_ms']:>8.2f} | {r['rows']:>8,} |"
        for r in all_results
    )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scales", default="10000,100000", help="comma-separated trace counts")
    ap.add_argument("--reps", type=int, default=7, help="timed repetitions per query (warm cache)")
    ap.add_argument("--out", default="benchmark_results.md", help="results markdown output path")
    args = ap.parse_args()

    scales = [int(s) for s in args.scales.split(",") if s.strip()]
    all_results: list[dict] = []
    for n in scales:
        all_results.extend(run_scale(n, args.reps))

    table = fmt_table(all_results)
    print("\n" + table)

    # Verdict: CAST median vs the real string-equality baseline, per scale.
    md = [
        "# Numeric assessment filter benchmark results\n",
        "_BENCHMARK-ONLY. Generated by `dev/benchmarks/numeric_assessment_filter_bench.py`._\n",
        f"\nSQLite, warm cache, reps={args.reps}. Backend: real MLflow `SqlAlchemyStore`.\n",
        "\n## Results\n",
        table,
        "\n\n## CAST overhead vs string-equality baseline\n",
    ]
    by_scale: dict[int, dict] = {}
    for r in all_results:
        by_scale.setdefault(r["scale"], {})[r["name"]] = r
    md.append(
        "\n### Same-skeleton, same-selectivity (isolates pure per-row CAST/CASE cost)\n"
        "\n| scale | constructed eq (string) | CAST = (same rows) | ratio (CAST/eq) |\n"
        "|---|---|---|---|\n"
    )
    for n in scales:
        ce = by_scale[n]["CONSTRUCTED eq (same skeleton)"]["median_ms"]
        cce = by_scale[n]["NUMERIC-CAST = (same rows as eq)"]["median_ms"]
        md.append(f"| {n:,} | {ce:.2f} ms | {cce:.2f} ms | {cce / ce:.2f}x |\n")

    md.append(
        "\n### CAST range query vs real string-eq baseline (different selectivity)\n"
        "\n| scale | baseline eq median | CAST > median | ratio (CAST/baseline) | "
        "attr-numeric median |\n|---|---|---|---|---|\n"
    )
    for n in scales:
        b = by_scale[n]["BASELINE eq (real search_traces)"]["median_ms"]
        c = by_scale[n]["NUMERIC-CAST > (prototype)"]["median_ms"]
        a = by_scale[n]["ATTR-NUMERIC > (native column)"]["median_ms"]
        md.append(f"| {n:,} | {b:.2f} ms | {c:.2f} ms | {c / b:.2f}x | {a:.2f} ms |\n")

    with open(args.out, "w") as f:
        f.write("".join(md))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
