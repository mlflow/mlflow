window.BENCHMARK_DATA = {
  "lastUpdate": 1777353551052,
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
      },
      {
        "commit": {
          "author": {
            "email": "pattara.sk127@gmail.com",
            "name": "Pat Sukprasert",
            "username": "PattaraS"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7686229ee4c11a3177426f3463664c3852f880ad",
          "message": "Aggregate role-based grants in workspace-level permission checks (#22954)\n\nSigned-off-by: Pat Sukprasert <pattara.sk127@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-04-28T05:15:38Z",
          "tree_id": "6cd489597b715baeeb44aa21b1b7d748d404d8cf",
          "url": "https://github.com/mlflow/mlflow/commit/7686229ee4c11a3177426f3463664c3852f880ad"
        },
        "date": 1777353550256,
        "tool": "pytest",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 23.58030780623103,
            "unit": "iter/sec",
            "range": "stddev: 0.0008200250213432247",
            "extra": "mean: 42.40826745000135 msec\nrounds: 20"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 39.848594094721356,
            "unit": "iter/sec",
            "range": "stddev: 0.032496170943215975",
            "extra": "mean: 25.094988235293037 msec\nrounds: 34"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 40.654476881605156,
            "unit": "iter/sec",
            "range": "stddev: 0.033520151741452155",
            "extra": "mean: 24.597537016949484 msec\nrounds: 59"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 40.841148324309856,
            "unit": "iter/sec",
            "range": "stddev: 0.03144925075196385",
            "extra": "mean: 24.48510977358515 msec\nrounds: 53"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 40.47858618466851,
            "unit": "iter/sec",
            "range": "stddev: 0.03107582982785137",
            "extra": "mean: 24.70442014545349 msec\nrounds: 55"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 47.647094968052436,
            "unit": "iter/sec",
            "range": "stddev: 0.014516964837542645",
            "extra": "mean: 20.987638400001174 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}