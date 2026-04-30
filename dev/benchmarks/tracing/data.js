window.BENCHMARK_DATA = {
  "lastUpdate": 1777529774221,
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
      },
      {
        "commit": {
          "author": {
            "email": "veronica.lyu@databricks.com",
            "name": "veronicalyu320",
            "username": "veronicalyu320"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ac684f690937f038a4ebb8d2b7b6b56c141dd5f8",
          "message": "Support multiple assessments per trace in MemAlign optimizer (#22846)\n\nSigned-off-by: Veronica Lyu <veronica.lyu@databricks.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-04-29T01:48:09Z",
          "tree_id": "a9c663b0b8e26df195b8de1133890da630653137",
          "url": "https://github.com/mlflow/mlflow/commit/ac684f690937f038a4ebb8d2b7b6b56c141dd5f8"
        },
        "date": 1777427524918,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 45.40394645000134,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 21.971678555554302,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.390323863635896,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.265712118645972,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 20.89155449999917,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 8.415546800000584,
            "unit": "ms"
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
          "id": "0fd7dc79f163791f9c98a9da614d441e9dbbab7f",
          "message": "Allow workspace `USE` to create experiments and registered models (#22941)\n\nSigned-off-by: Pat Sukprasert <pattara.sk127@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-04-29T08:32:40Z",
          "tree_id": "b20f64b6cd2c8825ec89b265b744723c32dbb99b",
          "url": "https://github.com/mlflow/mlflow/commit/0fd7dc79f163791f9c98a9da614d441e9dbbab7f"
        },
        "date": 1777451841107,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 138.3697460999997,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.560151000001063,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 19.619112813333004,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 18.59108444285726,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 19.68303428378358,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 4.869955399999526,
            "unit": "ms"
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
          "id": "38a3c8af3ac122421ceb5fd0cf934347adff31cb",
          "message": "[Admin-UI-1/7] Add backend auth endpoints (#22928)\n\nSigned-off-by: Pat Sukprasert <pattara.sk127@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-04-29T09:21:09Z",
          "tree_id": "63e8adda61a662af851123e2ff99bb40e99428fb",
          "url": "https://github.com/mlflow/mlflow/commit/38a3c8af3ac122421ceb5fd0cf934347adff31cb"
        },
        "date": 1777454684987,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 42.89191469999594,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.824960194442824,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.552351140622484,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 22.194038965515237,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 21.85565677965696,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 34.85256199999185,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "58110562+4binas@users.noreply.github.com",
            "name": "Abi",
            "username": "4binas"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "7e76d07e2e6d674fc8b186c70a52808123b76119",
          "message": "add timeout env in genai llm eval (#22977)\n\nSigned-off-by: Abinas Kuganathan <58110562+4binas@users.noreply.github.com>",
          "timestamp": "2026-04-29T10:05:18Z",
          "tree_id": "cdabd7b70388d46115d2a94bda71944fc55dac8f",
          "url": "https://github.com/mlflow/mlflow/commit/7e76d07e2e6d674fc8b186c70a52808123b76119"
        },
        "date": 1777457342488,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 39.778510149999846,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.395564314285625,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 24.828266393443222,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 24.705134799999552,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 26.63495406896572,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 6.615782800005832,
            "unit": "ms"
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
          "id": "bfae9893e15b67d18353f8df787b60e2c077840c",
          "message": "Fix assessment logging silently dropped in distributed tracing (#22963)\n\nSigned-off-by: Serena Ruan <serena.rxy@gmail.com>\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>",
          "timestamp": "2026-04-30T03:55:34Z",
          "tree_id": "4269a258ce0e739bd1179f175df22ae81dea75aa",
          "url": "https://github.com/mlflow/mlflow/commit/bfae9893e15b67d18353f8df787b60e2c077840c"
        },
        "date": 1777521550161,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 41.35347240000158,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 19.342937485713005,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 17.94279071666741,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 20.50876266666585,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 19.67087555932128,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 9.4780921999984,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "31463517+B-Step62@users.noreply.github.com",
            "name": "Yuki Watanabe",
            "username": "B-Step62"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a2e1966a9588f7de44c19f4d4a92f952086105ff",
          "message": "Deprecate `enable_mlserver` in pyfunc serving backend (#22994)\n\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: Claude <noreply@anthropic.com>\nCo-authored-by: Tomu Hirata <tomu.hirata@gmail.com>",
          "timestamp": "2026-04-30T05:10:04Z",
          "tree_id": "d7938187d74b5ccffe6c792838b0899ecfc2269a",
          "url": "https://github.com/mlflow/mlflow/commit/a2e1966a9588f7de44c19f4d4a92f952086105ff"
        },
        "date": 1777526022418,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 45.01080690000663,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 21.60174705405964,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 20.65914119999661,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 20.780098327585275,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 20.34361567212656,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 8.765949999985878,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "31463517+B-Step62@users.noreply.github.com",
            "name": "Yuki Watanabe",
            "username": "B-Step62"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d6c6ec138109c0af844b62ca5701831943c4826c",
          "message": "Redact presigned URL credentials in urllib3 retry logs (#22995)\n\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nSigned-off-by: Tomu Hirata <tomu.hirata@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>\nCo-authored-by: Tomu Hirata <tomu.hirata@gmail.com>",
          "timestamp": "2026-04-30T06:12:44Z",
          "tree_id": "65dd50bdf822f7ae255aff8293f481283eda317a",
          "url": "https://github.com/mlflow/mlflow/commit/d6c6ec138109c0af844b62ca5701831943c4826c"
        },
        "date": 1777529773543,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 40.416961449996336,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 19.21101132352773,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 18.657637984373032,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 17.792766392859047,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 19.139962081965688,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 8.729021600004216,
            "unit": "ms"
          }
        ]
      }
    ]
  }
}