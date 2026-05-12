# Tracing Benchmark

Per-commit tracing perf check. Runs on push to `master`; trend at https://mlflow.github.io/mlflow/dev/benchmarks/tracing/.

```bash
uv run pytest dev/benchmarks/tracing/ --benchmark-only
```

Add a scenario by writing a `test_*` function in `test_trace_perf.py` — it appears in the chart on the next master push. Renaming a test starts a new trend line.

Setup is modeled after [`opentelemetry-python`'s benchmark workflow](https://github.com/open-telemetry/opentelemetry-python/blob/main/.github/workflows/benchmarks.yml).
