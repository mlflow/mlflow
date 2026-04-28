window.BENCHMARK_DATA = {
  "lastUpdate": 1777355564300,
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
      },
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
          "id": "cc5e6b33bc0bd29f44853265b34fae89c1159a76",
          "message": "Disable commit comments on tracing benchmark alerts (#22958)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-04-28T14:43:36+09:00",
          "tree_id": "2a315405ea7682459ac799a0a035df784603d7a2",
          "url": "https://github.com/mlflow/mlflow/commit/cc5e6b33bc0bd29f44853265b34fae89c1159a76"
        },
        "date": 1777355096099,
        "tool": "pytest",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 21.118873348317234,
            "unit": "iter/sec",
            "range": "stddev: 0.02511332508609094",
            "extra": "mean: 47.351010799999926 msec\nrounds: 20"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 42.409459937959774,
            "unit": "iter/sec",
            "range": "stddev: 0.028817671945298107",
            "extra": "mean: 23.579644764703122 msec\nrounds: 34"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 45.54466536461062,
            "unit": "iter/sec",
            "range": "stddev: 0.02738994680051884",
            "extra": "mean: 21.95646827118914 msec\nrounds: 59"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 41.51771528266485,
            "unit": "iter/sec",
            "range": "stddev: 0.0295889861050552",
            "extra": "mean: 24.08610380392334 msec\nrounds: 51"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 39.445416203358015,
            "unit": "iter/sec",
            "range": "stddev: 0.03161831080790349",
            "extra": "mean: 25.351488113208685 msec\nrounds: 53"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 14.375247599264963,
            "unit": "iter/sec",
            "range": "stddev: 0.07412911046751611",
            "extra": "mean: 69.56401919999848 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "82044803+serena-ruan@users.noreply.github.com",
            "name": "Serena Ruan",
            "username": "serena-ruan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5a36fcfa76d833f1e922cd1b76187f292b3396d9",
          "message": "Fix uv custom index URLs omitted from model `requirements.txt` (#22921)\n\nSigned-off-by: Serena Ruan <serena.rxy@gmail.com>",
          "timestamp": "2026-04-28T05:47:06Z",
          "tree_id": "4557e776f7b4687d12fb2ea3b9b1e7cf476d3ed3",
          "url": "https://github.com/mlflow/mlflow/commit/5a36fcfa76d833f1e922cd1b76187f292b3396d9"
        },
        "date": 1777355433498,
        "tool": "pytest",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 13.042975116762596,
            "unit": "iter/sec",
            "range": "stddev: 0.06349687243720512",
            "extra": "mean: 76.66962415000071 msec\nrounds: 20"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 45.15901016103713,
            "unit": "iter/sec",
            "range": "stddev: 0.025384227135216604",
            "extra": "mean: 22.14397517647083 msec\nrounds: 34"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 50.68380538741303,
            "unit": "iter/sec",
            "range": "stddev: 0.022548674288442202",
            "extra": "mean: 19.730168095238227 msec\nrounds: 63"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 51.05397271002534,
            "unit": "iter/sec",
            "range": "stddev: 0.019447656721053218",
            "extra": "mean: 19.587114320755543 msec\nrounds: 53"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 47.01414787801552,
            "unit": "iter/sec",
            "range": "stddev: 0.02366926910974108",
            "extra": "mean: 21.27019301922973 msec\nrounds: 52"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 118.63126004397932,
            "unit": "iter/sec",
            "range": "stddev: 0.0009358135010021776",
            "extra": "mean: 8.429481400006011 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "198982749+Copilot@users.noreply.github.com",
            "name": "Copilot",
            "username": "Copilot"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "06bdc1d9851c9330c6826e6f33e9c0f147614680",
          "message": "Add `MLFLOW_SKIP_PIP_REQUIREMENTS_CHECK` env var to bypass pip validation in air-gapped environments (#22920)\n\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>\nCo-authored-by: serena-ruan <82044803+serena-ruan@users.noreply.github.com>",
          "timestamp": "2026-04-28T05:49:00Z",
          "tree_id": "3ca4fa4c60781b2bc0929b3014bcaaa02bb452c3",
          "url": "https://github.com/mlflow/mlflow/commit/06bdc1d9851c9330c6826e6f33e9c0f147614680"
        },
        "date": 1777355563244,
        "tool": "pytest",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 27.194963608046432,
            "unit": "iter/sec",
            "range": "stddev: 0.0006634728844009041",
            "extra": "mean: 36.771514550000006 msec\nrounds: 20"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 43.873495518879544,
            "unit": "iter/sec",
            "range": "stddev: 0.028320741348213184",
            "extra": "mean: 22.792804361113244 msec\nrounds: 36"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 44.14758242868384,
            "unit": "iter/sec",
            "range": "stddev: 0.02903690743541827",
            "extra": "mean: 22.651296967742308 msec\nrounds: 62"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 62.39376924588999,
            "unit": "iter/sec",
            "range": "stddev: 0.0005703171713977128",
            "extra": "mean: 16.02724137500111 msec\nrounds: 8"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 46.1364612615418,
            "unit": "iter/sec",
            "range": "stddev: 0.025907786093426206",
            "extra": "mean: 21.674830983051034 msec\nrounds: 59"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 139.80212686569706,
            "unit": "iter/sec",
            "range": "stddev: 0.0015547790624029947",
            "extra": "mean: 7.1529669999989665 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}