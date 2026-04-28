window.BENCHMARK_DATA = {
  "lastUpdate": 1777351242001,
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
          "id": "b47c6d43892d8eff24a2f6de95d6dbba3f307303",
          "message": "Add tracing benchmark CI workflow (#22602)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-04-28T04:36:50Z",
          "tree_id": "c2c7c70f9991dd2b68ec4f175a8e47e5182f986d",
          "url": "https://github.com/mlflow/mlflow/commit/b47c6d43892d8eff24a2f6de95d6dbba3f307303"
        },
        "date": 1777351240889,
        "tool": "pytest",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 24.565664587065047,
            "unit": "iter/sec",
            "range": "stddev: 0.0020055918541204447",
            "extra": "mean: 40.707223550001004 msec\nrounds: 20"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 53.83550741903573,
            "unit": "iter/sec",
            "range": "stddev: 0.01834179135574005",
            "extra": "mean: 18.57510122856963 msec\nrounds: 35"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 55.919405057991945,
            "unit": "iter/sec",
            "range": "stddev: 0.018074308220264606",
            "extra": "mean: 17.88287981538675 msec\nrounds: 65"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 57.85114067662642,
            "unit": "iter/sec",
            "range": "stddev: 0.013999842230891527",
            "extra": "mean: 17.285743864408012 msec\nrounds: 59"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 51.991104678998425,
            "unit": "iter/sec",
            "range": "stddev: 0.020745381118073447",
            "extra": "mean: 19.23405948333207 msec\nrounds: 60"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 113.20520617153127,
            "unit": "iter/sec",
            "range": "stddev: 0.0021557540169406834",
            "extra": "mean: 8.833516000004238 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}