window.BENCHMARK_DATA = {
  "lastUpdate": 1777387305280,
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
      },
      {
        "commit": {
          "author": {
            "email": "weichen.xu@databricks.com",
            "name": "WeichenXu",
            "username": "WeichenXu123"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0e9bf9ad163bb6e7c8b92f24831b40d243606fb5",
          "message": "Use `TaskContext.artifactDir` to get the correct unpacked artifacts directory (#22969)\n\nSigned-off-by: Weichen Xu <weichen.xu@databricks.com>",
          "timestamp": "2026-04-28T14:37:44Z",
          "tree_id": "52ca4e593e59d5a688fe1f82298f4e92aa8426cf",
          "url": "https://github.com/mlflow/mlflow/commit/0e9bf9ad163bb6e7c8b92f24831b40d243606fb5"
        },
        "date": 1777387304623,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 43.626549199998266,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.268143542856738,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 23.50392693548264,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.559361849056092,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 24.54516369491543,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 6.808702199998606,
            "unit": "ms"
          }
        ]
      }
    ]
  }
}