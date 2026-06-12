> ⚠️ **FOR REFERENCE ONLY — NOT FOR MERGE INTO MASTER.**
> This branch contains a throwaway performance benchmark used to decide whether to ship
> the runtime CAST-to-float approach for numeric assessment trace filtering (phase 1 of 2).
> It adds **no production code** and changes **no shippable surface** — only a script under
> `dev/benchmarks/` and a generated results file. Do not open this as a real PR.

Reference material for implementation PR mlflow/mlflow#23948.

## What this is

A self-contained benchmark that quantifies the per-query latency overhead of the
prototype (closed PR #21811) approach: implementing numeric assessment operators
(`>`, `>=`, `<`, `<=`) by **CASTing the JSON-encoded assessment value to float at query
time**, instead of adding a dedicated/indexed numeric column.

Assessment values are stored as JSON string primitives in the `assessments` table
(`SqlAssessments.value`, `Text`). The existing search path only supports equality /
string-pattern / null filters and builds the value comparison via
`SearchTraceUtils._get_sql_json_comparison_func`. The prototype adds a `CASE … CAST(value AS Float)` expression for the numeric operators. The concern is the runtime CAST cost at
scale; this benchmark measures it before we implement for real.

## How it works

- Spins up a **real** MLflow `SqlAlchemyStore` on a temp-file SQLite DB so the schema and
  indexes match production exactly.
- Seeds N traces, each with one `feedback.score` assessment (numeric in [0,10], with ~5%
  string and ~5% boolean values to exercise the prototype's coercion `CASE`). Seeding is
  done via **bulk inserts into the real ORM tables** (`SqlTraceInfo` / `SqlAssessments`)
  using the same JSON encoding the store APIs produce — the public `log_trace` +
  `create_assessment` APIs are far too slow at 100k+.
- Measures warm-cache median + p90 latency over repeated reps for:
  - **BASELINE** — string-equality filter through the **real** `store.search_traces` path.
  - **CONSTRUCTED-EQ** — string-equality value filter through a locally reconstructed copy
    of the store's subquery+join skeleton (validated to return identical rows to BASELINE).
  - **NUMERIC-CAST `=`** — the prototype CAST expression at the **same value/selectivity**
    as the string `=`, so the only difference is the value expression. **This is the
    decisive apples-to-apples comparison.**
  - **NUMERIC-CAST `>` / `<=`** — the prototype range queries (different selectivity).
  - **ATTR-NUMERIC** — a native indexed numeric column (`timestamp_ms`) as a reference.

The benchmark does **not** modify `mlflow/utils/search_utils.py` or the store; it imports
the real helpers and reconstructs the prototype's expression locally.

## Run it

```bash
uv run python dev/benchmarks/numeric_assessment_filter_bench.py --scales 10000,100000 --reps 9
# optional, completes in ~1 min:
uv run python dev/benchmarks/numeric_assessment_filter_bench.py --scales 1000000 --reps 9
```

## Results (SQLite, warm cache, reps=9)

Same-skeleton, **same selectivity** (isolates the pure per-row CAST/CASE cost):

| scale     | string `=` | CAST `=` (same rows) | ratio     |
| --------- | ---------- | -------------------- | --------- |
| 10,000    | 2.85 ms    | 6.48 ms              | **2.27x** |
| 100,000   | 29.94 ms   | 66.19 ms             | **2.21x** |
| 1,000,000 | 305.75 ms  | 665.18 ms            | **2.18x** |

Same-selectivity operator comparison (verifies numeric inequality operators are equivalent
to numeric `=` when matched on row selectivity):

| scale     | CAST `=`  | CAST `>`  | CAST `<=` |
| --------- | --------- | --------- | --------- |
| 10,000    | 6.48 ms   | 6.47 ms   | 6.48 ms   |
| 100,000   | 66.19 ms  | 66.12 ms  | 66.29 ms  |
| 1,000,000 | 665.18 ms | 664.37 ms | 666.60 ms |

Range query vs the real production string-eq baseline (different selectivity):

| scale     | baseline eq | CAST `>` | ratio | attr-numeric |
| --------- | ----------- | -------- | ----- | ------------ |
| 10,000    | 12.21 ms    | 9.14 ms  | 0.75x | 5.16 ms      |
| 100,000   | 141.57 ms   | 99.61 ms | 0.70x | 133.15 ms    |
| 1,000,000 | 1641 ms     | 1131 ms  | 0.69x | 1137 ms      |

Full table + verdict in `benchmark_results.md`.

## Verdict: GO

The CAST adds a **stable, bounded ~2.2x** overhead on the value comparison, flat across
10k → 1M (no super-linear cliff). Both the string-eq filter and the CAST filter are
already `O(N)` scans of the `assessments` table (no index on the JSON value), so the CAST
does **not** change the complexity class — it makes an already-linear comparison ~2x
heavier. On real range queries the absolute latency stays in the same ballpark as the
equality baseline already shipping in production.

At matched selectivity, the inequality operators (`>`, `<=`) are effectively equal to
numeric `=`; the operator choice does not add measurable overhead beyond the CAST itself.

**Recommend shipping the CAST approach in phase 2.** Separately (and orthogonally), if
sub-100 ms assessment-filter latency at 100k+ becomes a hard requirement, add a
generated/indexed numeric column — that would speed up _both_ string and numeric filters
and is not a reason to block the CAST.

### Caveats

- SQLite only; the prototype's MySQL/Postgres/MSSQL CAST branches are not exercised here.
  Re-verify on the production dialect before final sign-off.
- Bulk-insert seeding (documented above) rather than the public log APIs.
