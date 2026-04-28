window.BENCHMARK_DATA = {
  "lastUpdate": 1777363844308,
  "repoUrl": "https://github.com/mlflow/mlflow",
  "entries": {
    "MLflow Tracing Benchmark": [
      {
        "commit": {
          "author": {
            "email": "hkawamura0130@gmail.com",
            "name": "Harutaka Kawamura",
            "username": "harupy"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "56457d600bc2498eb4e2a12740b9d2061f397310",
          "message": "Show tracing benchmark chart in milliseconds instead of iter/sec (#22960)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-04-28T17:08:23+09:00",
          "tree_id": "fc0440e0c42541ff0b6c3246cc7c2e80a1b4687a",
          "url": "https://github.com/mlflow/mlflow/commit/56457d600bc2498eb4e2a12740b9d2061f397310"
        },
        "date": 1777363843144,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 59.447227000000424,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 18.344438948718224,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 17.970678942857685,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 18.77277874193706,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 16.846787492307577,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 5.123520799998005,
            "unit": "ms"
          }
        ]
      }
    ]
  }
}