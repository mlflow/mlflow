window.BENCHMARK_DATA = {
  "lastUpdate": 1780072349794,
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
      },
      {
        "commit": {
          "author": {
            "email": "varun.bhandary@databricks.com",
            "name": "Varun Bhandary",
            "username": "vb-dbrks"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "39a00d1b6c67de25c47c37498db2d4898c3643bb",
          "message": "Fix Databricks unified auth support when MLFLOW_ENABLE_DB_SDK=true (#20599)\n\nSigned-off-by: Varun Bhandary <varun.bhandary@databricks.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-04-30T08:57:46Z",
          "tree_id": "ffe9ed9e9c90eae7d86e36ae42a71b588f15f254",
          "url": "https://github.com/mlflow/mlflow/commit/39a00d1b6c67de25c47c37498db2d4898c3643bb"
        },
        "date": 1777539703146,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 48.46527345000169,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 19.267387771429462,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 18.5969815079358,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 17.978854517242496,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 19.414924049179778,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 9.03619499999877,
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
          "id": "86f39f85486e678426ca6aba92acd343c810b89e",
          "message": "Add Helm charts for deploying mlflow to kubernetes cluster (#21973)\n\nSigned-off-by: Weichen Xu <weichen.xu@databricks.com>",
          "timestamp": "2026-05-02T11:28:52+08:00",
          "tree_id": "e18a2457b33a6c7a6ca88bfbcfb3b04705f19569",
          "url": "https://github.com/mlflow/mlflow/commit/86f39f85486e678426ca6aba92acd343c810b89e"
        },
        "date": 1777692634015,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 41.689458349998176,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.68467699999729,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.105003515625498,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 16.83660428571037,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.249702315785694,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 8.197710600006758,
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
          "id": "6ce9cfc128fc1c1ab25be2470e3908e935e31192",
          "message": "[Admin-UI-2/4] Add /account page and bottom-left account widget (#22973)\n\nSigned-off-by: Pat Sukprasert <pattara.sk127@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-04T03:19:46Z",
          "tree_id": "b2d984e4ee0b0ca0e43a00b2eb6383c32cbeff44",
          "url": "https://github.com/mlflow/mlflow/commit/6ce9cfc128fc1c1ab25be2470e3908e935e31192"
        },
        "date": 1777865036478,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 43.65999384999952,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.009179264706145,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 20.492520923075666,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 16.19755062499806,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 21.664821566665893,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 8.847649200001229,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "191841109+mlflow-app[bot]@users.noreply.github.com",
            "name": "mlflow-app[bot]",
            "username": "mlflow-app[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7bd32e3adb6699ffcf4d3d85e88208887ad9dff9",
          "message": "Update ML package versions for 3.12.0 (#23054)\n\nSigned-off-by: mlflow-app[bot] <mlflow-app[bot]@users.noreply.github.com>\nSigned-off-by: mlflow-app[bot] <191841109+mlflow-app[bot]@users.noreply.github.com>\nSigned-off-by: Daniel Lok <daniel.lok@databricks.com>\nCo-authored-by: mlflow-app[bot] <mlflow-app[bot]@users.noreply.github.com>\nCo-authored-by: mlflow-app[bot] <191841109+mlflow-app[bot]@users.noreply.github.com>\nCo-authored-by: Daniel Lok <daniel.lok@databricks.com>",
          "timestamp": "2026-05-04T09:17:38Z",
          "tree_id": "244ce7ea9089639fc4b1461239527160c30aeeb7",
          "url": "https://github.com/mlflow/mlflow/commit/7bd32e3adb6699ffcf4d3d85e88208887ad9dff9"
        },
        "date": 1777886477567,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 38.70798355000602,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.216752371430562,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.05947625396867,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 24.63013708771848,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.193943157891713,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.12106659998426,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "sai.ravuri@swiggy.in",
            "name": "Sai Nikitha Ravuri",
            "username": "sairavuri-sudo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2f79947e2baa9d4aa64381b0a97d35b0cf755aed",
          "message": "Fix `gateway_adapter` not forwarding workspace header to judge endpoints (#23047)\n\nSigned-off-by: Sai Ravuri <sai.ravuri@swiggy.in>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-05T02:46:54Z",
          "tree_id": "a5214e23258faa8c53093f2463ed91fa40d00553",
          "url": "https://github.com/mlflow/mlflow/commit/2f79947e2baa9d4aa64381b0a97d35b0cf755aed"
        },
        "date": 1777949450969,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 38.49618069999963,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.31878662857077,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.197824015873476,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 16.254108375001408,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.038917948275717,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 6.990167399999336,
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
          "id": "58c94e672e56994a04946b244f1d3bd581ffffd1",
          "message": "[Admin-UI-3/4] Add Platform Admin pages (#22929)\n\nSigned-off-by: Pat Sukprasert <pattara.sk127@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-05T10:06:14Z",
          "tree_id": "5b08f2836dff01d2c3c32b474b30b101f60f15fc",
          "url": "https://github.com/mlflow/mlflow/commit/58c94e672e56994a04946b244f1d3bd581ffffd1"
        },
        "date": 1777975893170,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 171.73413945000533,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.8233216590903,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 20.65228997468429,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.507094536231342,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 19.67058828571689,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 5.88274360000014,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "xshen.shc@gmail.com",
            "name": "Xiang Shen",
            "username": "xsh310"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3f06c16826aaa88eb501914b7515acbbaaf083bc",
          "message": "Add UC traces upsell message for set_experiment calls on Databricks (#23038)\n\nSigned-off-by: Xiang Shen <xshen.shc@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-06T01:06:21Z",
          "tree_id": "76e8a2da1a7bf9e1c57ce061359ef7d7adca7d92",
          "url": "https://github.com/mlflow/mlflow/commit/3f06c16826aaa88eb501914b7515acbbaaf083bc"
        },
        "date": 1778029805815,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 40.58284580000162,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.401910742857872,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.58039711290394,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 24.243511464287288,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.502310322033356,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 50.97342299999923,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "168796509+lavaFreak@users.noreply.github.com",
            "name": "Garion Milazzo",
            "username": "lavaFreak"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "92d951a179508647798d4d44c2282c28a974ec65",
          "message": "Add `session_count` trace metric for grouped traces (#23011)\n\nSigned-off-by: lavafreak <gemilazzo@gmail.com>",
          "timestamp": "2026-05-06T12:33:33+08:00",
          "tree_id": "7c6cd75e9da1b95748b3a23f3866f4063bc4c4b8",
          "url": "https://github.com/mlflow/mlflow/commit/92d951a179508647798d4d44c2282c28a974ec65"
        },
        "date": 1778042108479,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 39.58780530000112,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.880883914286755,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.89466143548244,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 17.272508666669257,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.949618913793472,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 6.83331580000015,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "rahul.rajesh.bhat@gmail.com",
            "name": "Rahul Rajesh",
            "username": "rrtheonlyone"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bab283da6c597220f7fd6118a7a99599aea17dc5",
          "message": "Add log levels for Trace Spans with UI switch to filter (#23017)\n\nSigned-off-by: Rahul Rajesh <rahul.rajesh.bhat@gmail.com>\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nSigned-off-by: Yuki Watanabe <31463517+B-Step62@users.noreply.github.com>\nCo-authored-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: Yuki Watanabe <31463517+B-Step62@users.noreply.github.com>",
          "timestamp": "2026-05-06T19:19:50+09:00",
          "tree_id": "6b05c8220c4cd5eedf7119431bf687634d11b16c",
          "url": "https://github.com/mlflow/mlflow/commit/bab283da6c597220f7fd6118a7a99599aea17dc5"
        },
        "date": 1778062900437,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 46.447564749998804,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.331289857143798,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.357503068965496,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.06912489090962,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.53338294999973,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 25.474908799998275,
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
          "id": "3df5fe11a0124ae73f273abb2710569bb898ec15",
          "message": "RBAC Phase 2: collapse legacy permission tables into `role_permissions` (#22855)\n\nSigned-off-by: Pat Sukprasert <pattara.sk127@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-06T11:40:43Z",
          "tree_id": "ca396a3ea0d09aa0d4c68a6cfaf48a89f9681af0",
          "url": "https://github.com/mlflow/mlflow/commit/3df5fe11a0124ae73f273abb2710569bb898ec15"
        },
        "date": 1778067855588,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 39.63598784999789,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.36087368571345,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 20.429056290323484,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 15.974242875003597,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 21.255562533331823,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 9.502556399991136,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "191841109+mlflow-app[bot]@users.noreply.github.com",
            "name": "mlflow-app[bot]",
            "username": "mlflow-app[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1d4023f13d66c7dc9820d2cadfc0293e67b055bb",
          "message": "Update model catalog from upstream sources (#23083)\n\nCo-authored-by: mlflow-app[bot] <mlflow-app[bot]@users.noreply.github.com>",
          "timestamp": "2026-05-07T11:50:02+09:00",
          "tree_id": "554e0be8240b803f77bc7266fd2495f56c3fea82",
          "url": "https://github.com/mlflow/mlflow/commit/1d4023f13d66c7dc9820d2cadfc0293e67b055bb"
        },
        "date": 1778122275200,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 45.76823724999883,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.644537771427867,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.46585468749851,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 22.39806703448329,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 21.575387233333508,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 10.482255799999507,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pdifranc@users.noreply.github.com",
            "name": "pdifranc",
            "username": "pdifranc"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8d75782b100873415d5f870a001e82a17bd95cb8",
          "message": "Feature/sagemaker build network option (#22996)\n\nSigned-off-by: Paolo Di Francesco <frpaolo@amazon.at>\nCo-authored-by: Kris Concepcion <84737625+kriscon-db@users.noreply.github.com>",
          "timestamp": "2026-05-07T04:08:28Z",
          "tree_id": "537c5cf9c462a735a486aac2a52a31387ec653af",
          "url": "https://github.com/mlflow/mlflow/commit/8d75782b100873415d5f870a001e82a17bd95cb8"
        },
        "date": 1778127245665,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 45.939873999998326,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.55336240000341,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.67251249180459,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 22.07698212000082,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.649271830507967,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.060376600003337,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "90125084+Genmin@users.noreply.github.com",
            "name": "Joey Roth",
            "username": "Genmin"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "143cf828ac88b14b28e8fdfef90891d16a311bc6",
          "message": "Fix Azure OpenAI streaming usage tracing (#23036)\n\nSigned-off-by: Genmin <joey@joeyroth.com>",
          "timestamp": "2026-05-07T13:44:53+09:00",
          "tree_id": "27ae5e732f2bed073c205adc2ba42f52855912f5",
          "url": "https://github.com/mlflow/mlflow/commit/143cf828ac88b14b28e8fdfef90891d16a311bc6"
        },
        "date": 1778129197532,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 39.5314711499978,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.929834277781598,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.82611281250002,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 22.64775324999643,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.001509728813378,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 6.863853599992353,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ktrk115@gmail.com",
            "name": "Kotaro Kikuchi",
            "username": "ktrk115"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "af7eed6d36e04448e5199df79966e271b012f06b",
          "message": "Trace `Runner.run_streamed()` in OpenAI Agents SDK autolog (#22962)\n\nSigned-off-by: Kotaro Kikuchi <ktrk115@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>\nCo-authored-by: Kris Concepcion <84737625+kriscon-db@users.noreply.github.com>",
          "timestamp": "2026-05-07T10:47:02Z",
          "tree_id": "4e42a600e88f7e03fd4dfa094dc368735d1ee8d0",
          "url": "https://github.com/mlflow/mlflow/commit/af7eed6d36e04448e5199df79966e271b012f06b"
        },
        "date": 1778151078946,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 36.35411079999926,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.27013448571701,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 23.2010630806453,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 24.831135892858317,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 24.041115016949156,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.3052239999924495,
            "unit": "ms"
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
          "id": "a3dde3f1a543d31fa75e023da5c15eac1ee0667b",
          "message": "Fix `sentence_transformers` pyfunc predict for v5.4+ (#23108)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-07T11:15:12Z",
          "tree_id": "413ec1c304fba56807db00ba4b8dbd7685cb680d",
          "url": "https://github.com/mlflow/mlflow/commit/a3dde3f1a543d31fa75e023da5c15eac1ee0667b"
        },
        "date": 1778152765913,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 46.915843300002535,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 31.53081985294355,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 24.540900714285637,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.398974313724853,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 17.098605166665948,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 6.649252600001887,
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
          "id": "6ec48571e2590c6e394c3d459824a6f20535ac48",
          "message": "Open `/admin` to workspace managers (scoped per their workspace) (#23086)\n\nSigned-off-by: Pat Sukprasert <pattara.sk127@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-07T17:48:03Z",
          "tree_id": "c579beec0e3bb771e8902ab6ec9ab62fb58cbd27",
          "url": "https://github.com/mlflow/mlflow/commit/6ec48571e2590c6e394c3d459824a6f20535ac48"
        },
        "date": 1778176345690,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 39.84896654999943,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 25.448478242425963,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.40428742372871,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 24.003684615386497,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.811578363636634,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 23.07762080000373,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "191841109+mlflow-app[bot]@users.noreply.github.com",
            "name": "mlflow-app[bot]",
            "username": "mlflow-app[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "5a73054b979c1927a2715bc58553f311cdbb7eeb",
          "message": "Update model catalog from upstream sources (#23116)\n\nCo-authored-by: mlflow-app[bot] <mlflow-app[bot]@users.noreply.github.com>",
          "timestamp": "2026-05-08T01:07:55Z",
          "tree_id": "59113915c8dbbd9110064752ea70cdf393088df8",
          "url": "https://github.com/mlflow/mlflow/commit/5a73054b979c1927a2715bc58553f311cdbb7eeb"
        },
        "date": 1778202716575,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 40.74104605000102,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.888483529412188,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.4618802580643,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 24.507460642857534,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 24.110849389830754,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 18.40011179999692,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "tomu.hirata@gmail.com",
            "name": "Tomu Hirata",
            "username": "TomeHirata"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f9b1eb510478570609ef451984a255775aa4b937",
          "message": "Fix trace API authorization vulnerability (#23014)\n\nSigned-off-by: Tomu Hirata <tomu.hirata@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-08T01:24:04Z",
          "tree_id": "fbb4f5c07690944295b627acbd0933a92c3f8f3a",
          "url": "https://github.com/mlflow/mlflow/commit/f9b1eb510478570609ef451984a255775aa4b937"
        },
        "date": 1778203691684,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 45.39481724999064,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.270084485708917,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 19.54808296773631,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 20.932811280703703,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 20.161382672131555,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 8.731590800005051,
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
          "id": "d33153ff4bc6596c9053df017d2c1509388d84ab",
          "message": "Skip re-alignment of unchanged traces in `MemAlignOptimizer` (#23008)\n\nSigned-off-by: Veronica Lyu <veronica.lyu@databricks.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-08T01:54:54Z",
          "tree_id": "b686d26a32a8a7de76e14a4a269d6bfdaa78bc5b",
          "url": "https://github.com/mlflow/mlflow/commit/d33153ff4bc6596c9053df017d2c1509388d84ab"
        },
        "date": 1778205591132,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 67.90749600000154,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 19.663484533334163,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 17.232099299997827,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 18.126543256758357,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 17.963455959458823,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 4.9364710000020295,
            "unit": "ms"
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
          "id": "0d361deb83e1a4c3d8e9c7de921ada8f42e5ccdc",
          "message": "Add per-image/video/audio pricing to `amazon.nova-2-multimodal-embeddings-v1:0` in Bedrock catalog (#23117)\n\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>\nCo-authored-by: TomeHirata <33407409+TomeHirata@users.noreply.github.com>",
          "timestamp": "2026-05-08T04:02:28Z",
          "tree_id": "512c8d0aefd197c02dd9f28725bf23bbf85a7a4a",
          "url": "https://github.com/mlflow/mlflow/commit/0d361deb83e1a4c3d8e9c7de921ada8f42e5ccdc"
        },
        "date": 1778213194853,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 36.051621999999384,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 21.15460462162037,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.058667921875607,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 20.776909189655843,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 21.821374111111492,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 12.704690399993979,
            "unit": "ms"
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
          "id": "b174540ff3b05b853a4a171f4d5df24cee245440",
          "message": "Catch `PermissionError` in `terminate_session_process` on Windows (#23118)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-08T05:18:29Z",
          "tree_id": "61a70d71d4c101d21642779b4c49ae254c30a96d",
          "url": "https://github.com/mlflow/mlflow/commit/b174540ff3b05b853a4a171f4d5df24cee245440"
        },
        "date": 1778217760636,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 45.778333949998995,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.04631081818281,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 20.342073196721717,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 15.992259249998142,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.27723794915107,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 52.223089400001754,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "debusinha2009@gmail.com",
            "name": "Debu Sinha",
            "username": "debu-sinha"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "598b7cb4d8ef7c90b4059d5a8c02b43881a27729",
          "message": "Add Google ADK  and  third-party scorers (#22299)\n\nSigned-off-by: debu-sinha <debusinha2009@gmail.com>",
          "timestamp": "2026-05-08T06:02:00Z",
          "tree_id": "b547d343a1ab4aa2b73a2d5985639d32c7596ec3",
          "url": "https://github.com/mlflow/mlflow/commit/598b7cb4d8ef7c90b4059d5a8c02b43881a27729"
        },
        "date": 1778220376918,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 38.239490499998396,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 25.742570764705874,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 26.359530303572093,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 25.048286905660717,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 24.302985963636946,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 6.6000075999937735,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "sai.ravuri@swiggy.in",
            "name": "Sai Nikitha Ravuri",
            "username": "sairavuri-sudo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a968b7371c1c0069016c2018e1655d468c4d27f8",
          "message": "Fix OTLP trace ingestion: double-encoded request ID and missing trace tags (#23067)\n\nSigned-off-by: Sai Ravuri <sai.ravuri@swiggy.in>\nCo-authored-by: Claude <noreply@anthropic.com>\nCo-authored-by: Pat Sukprasert <pattara.sk127@gmail.com>\nCo-authored-by: Yuki Watanabe <31463517+B-Step62@users.noreply.github.com>",
          "timestamp": "2026-05-08T07:09:54Z",
          "tree_id": "7dc392b41b61a2d74740b0f8a1b8020c0a883c16",
          "url": "https://github.com/mlflow/mlflow/commit/a968b7371c1c0069016c2018e1655d468c4d27f8"
        },
        "date": 1778224440188,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 43.093766199999095,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 28.344394058821617,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.201236967212065,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.559907781819078,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.29248961017144,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 6.246139400008133,
            "unit": "ms"
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
          "id": "925006e94c069e2c4de14187b4e66c9a1b3f8cb3",
          "message": "Pin `sentence-transformers<5.4` for `transformers<4.46` cross-version tests (#23123)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-08T07:32:46Z",
          "tree_id": "0853c6a6d7103aa4e0ba7b2b03242761fb617d28",
          "url": "https://github.com/mlflow/mlflow/commit/925006e94c069e2c4de14187b4e66c9a1b3f8cb3"
        },
        "date": 1778225807648,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 42.470415149999496,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.135053411763806,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.627536238094613,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 26.578297099999304,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.23082648275612,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 11.346669600010273,
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
          "id": "d18b37b19ef5083ec0f1abfd2608eaf0e0508170",
          "message": "RBAC Phase 2: remove legacy permission REST endpoints + client methods (#22859)\n\nSigned-off-by: Pat Sukprasert <pattara.sk127@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-08T08:43:06Z",
          "tree_id": "d5e9facd455db67de058b14acf9ba937e6fbf034",
          "url": "https://github.com/mlflow/mlflow/commit/d18b37b19ef5083ec0f1abfd2608eaf0e0508170"
        },
        "date": 1778230033340,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 48.32693649999982,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.559983181819,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.61656854385981,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 26.701774415092792,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.535893849056936,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 18.88633719999575,
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
          "id": "ffda95c23dec1cba23991a221a39109084ff22f9",
          "message": "RBAC Phase 2: collapse cascade helpers into 3 internal grant methods (#22861)\n\nSigned-off-by: Pat Sukprasert <pattara.sk127@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-08T14:09:45Z",
          "tree_id": "e85ee74036ffd26a6e98b2f1dceef1e6333e376d",
          "url": "https://github.com/mlflow/mlflow/commit/ffda95c23dec1cba23991a221a39109084ff22f9"
        },
        "date": 1778249646731,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 40.44907459999365,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.007342735296024,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.762571377046193,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.89400142105084,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.31496316666441,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 50.89755519999244,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shyamprasad.miryala@databricks.com",
            "name": "Shyamprasad Miryala",
            "username": "shyamspr"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2c7cae599d51d923ec8db042ae1ab527cd3162b9",
          "message": "Fix nested array items being stripped from function tool schemas (#23053)\n\nSigned-off-by: Shyamprasad Reddy <shyamspr@live.com>",
          "timestamp": "2026-05-08T22:32:49+08:00",
          "tree_id": "055afd3a18797c826dc8b3de46d48a0c69f3d78e",
          "url": "https://github.com/mlflow/mlflow/commit/2c7cae599d51d923ec8db042ae1ab527cd3162b9"
        },
        "date": 1778250945909,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 85.44189985000088,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 21.65148131110944,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 17.269781383561945,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 18.982604242423513,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 21.24118987500228,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 36.00335240000163,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "55605301+khaledsulayman@users.noreply.github.com",
            "name": "Khaled Sulayman",
            "username": "khaledsulayman"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e054abba2e1e16fa971e3e491b2250bca48affc8",
          "message": "Add `Link` entity and `LiveSpan.add_link()` for OpenTelemetry Span Links (#22797)\n\nSigned-off-by: Khaled Sulayman <ksulayma@redhat.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-08T20:54:30Z",
          "tree_id": "ccc08e179124bca5f72a45e6963a3a63eb54bbaa",
          "url": "https://github.com/mlflow/mlflow/commit/e054abba2e1e16fa971e3e491b2250bca48affc8"
        },
        "date": 1778273980855,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 44.43717890000087,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 20.142289147058282,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 19.221413890625485,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 18.58148380357148,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 20.69386855932226,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 6.970809399996369,
            "unit": "ms"
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
          "id": "085aec29f9b4f3fa866a6f583e0029afea229392",
          "message": "`gateway`: honor Anthropic `api_base` from secret `auth_config` (#23167)\n\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>\nCo-authored-by: TomeHirata <33407409+TomeHirata@users.noreply.github.com>",
          "timestamp": "2026-05-11T02:04:58Z",
          "tree_id": "cc1d8a144065c679253c6bdea9f9c39566baccaa",
          "url": "https://github.com/mlflow/mlflow/commit/085aec29f9b4f3fa866a6f583e0029afea229392"
        },
        "date": 1778465349754,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 49.63443855000094,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 31.894180303027902,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 27.47151691836704,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 27.11678772222378,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 27.33810715094446,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 51.73835279999821,
            "unit": "ms"
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
          "id": "fe81ae9aa850df938693ea22155f7b409576cbf3",
          "message": "Fix MySQL-incompatible `NULLS LAST` syntax in `list_endpoint_guardrail_configs` (#23168)\n\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>\nCo-authored-by: TomeHirata <33407409+TomeHirata@users.noreply.github.com>",
          "timestamp": "2026-05-11T11:54:34+09:00",
          "tree_id": "c86c84b587ff304aca749dda2e2a000ac298c30c",
          "url": "https://github.com/mlflow/mlflow/commit/fe81ae9aa850df938693ea22155f7b409576cbf3"
        },
        "date": 1778468172771,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 41.1894758499983,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.519805388881803,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 18.790832696967385,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.736494967742164,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 19.411697546881435,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 6.87585120000449,
            "unit": "ms"
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
          "id": "c97a236367d3955e5dfb60664ecd6d9cf753425d",
          "message": "Fix invalid stop-hook command when using `pixi` environment manager (#23030)\n\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>\nCo-authored-by: TomeHirata <33407409+TomeHirata@users.noreply.github.com>",
          "timestamp": "2026-05-11T03:13:12Z",
          "tree_id": "820eb529f38ddc53e87e8b0faf6a44c66ee85cec",
          "url": "https://github.com/mlflow/mlflow/commit/c97a236367d3955e5dfb60664ecd6d9cf753425d"
        },
        "date": 1778469460795,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 47.390266500002554,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 21.324525861110605,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.243699142857732,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.956759508771178,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 20.530902426230337,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 10.367769600003385,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "tomu.hirata@gmail.com",
            "name": "Tomu Hirata",
            "username": "TomeHirata"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3ca2130d8225188653df19812f0c92ac9f0ac9df",
          "message": "Fix Vertex AI gateway to use Anthropic API format for Claude models (#23175)\n\nSigned-off-by: Tomu Hirata <tomu.hirata@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-11T04:01:24Z",
          "tree_id": "148619cd831c3f6a700c95bc5be0e50d247f0495",
          "url": "https://github.com/mlflow/mlflow/commit/3ca2130d8225188653df19812f0c92ac9f0ac9df"
        },
        "date": 1778472321502,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 45.9279709499981,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.86110325714341,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.317488301589204,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 22.42191392856847,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.024112999999495,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 9.113775999998097,
            "unit": "ms"
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
          "id": "2f8345f1eea3729b979dd0dc894c5bce953047f4",
          "message": "Fix `runs:/<run_id>/<model_name>` loading by resolving logged-model artifacts via `models:/<model_id>` (#23130)\n\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>\nCo-authored-by: TomeHirata <33407409+TomeHirata@users.noreply.github.com>",
          "timestamp": "2026-05-11T13:12:44+09:00",
          "tree_id": "a0f5e267298c1d8d3474544374a89e1eb8dd67e6",
          "url": "https://github.com/mlflow/mlflow/commit/2f8345f1eea3729b979dd0dc894c5bce953047f4"
        },
        "date": 1778472859569,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 40.896227900005044,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.258146657142266,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.853505645162105,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.425730785712766,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.970447894737493,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 70.78159140000366,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "tomu.hirata@gmail.com",
            "name": "Tomu Hirata",
            "username": "TomeHirata"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "471fc1886396f5623d1530e37eb21a8f5921c04a",
          "message": "[Security] Add `MLFLOW_ALLOW_PICKLE_DESERIALIZATION` guard to `PickleEvaluationArtifact` (#23183)\n\nSigned-off-by: Tomu Hirata <tomu.hirata@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-11T04:37:12Z",
          "tree_id": "306c5f469ac683894633f6e63b7af6ebe46cf0b3",
          "url": "https://github.com/mlflow/mlflow/commit/471fc1886396f5623d1530e37eb21a8f5921c04a"
        },
        "date": 1778474491320,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 45.542460000002905,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.349872514287686,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.970532854838563,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.933846600000027,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.064776599998993,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 36.702497200002426,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "191841109+mlflow-app[bot]@users.noreply.github.com",
            "name": "mlflow-app[bot]",
            "username": "mlflow-app[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "ad90a4c0f9ba94fa3458ae051c0fbdd66687ccec",
          "message": "Update model catalog from upstream sources (#23164)\n\nSigned-off-by: Tomu Hirata <tomu.hirata@gmail.com>\nCo-authored-by: mlflow-app[bot] <mlflow-app[bot]@users.noreply.github.com>\nCo-authored-by: Tomu Hirata <tomu.hirata@gmail.com>",
          "timestamp": "2026-05-11T04:54:54Z",
          "tree_id": "1cd0b2e5c3a7e3b63ff6d37b9f4378cfa46fe2be",
          "url": "https://github.com/mlflow/mlflow/commit/ad90a4c0f9ba94fa3458ae051c0fbdd66687ccec"
        },
        "date": 1778475526421,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 45.164218150002,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.334010885713592,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.42019623437763,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.412787339286865,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 20.69273371666706,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 22.459907999996176,
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
          "id": "d1bf8becd02eb2777a5192c584187a6c6dac6fc3",
          "message": "Shell-quote `local_path` in `PyFuncBackend` serve commands (#23180)\n\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: sreelim <sreelim@users.noreply.github.com>",
          "timestamp": "2026-05-11T06:36:57Z",
          "tree_id": "a2468963d551840c256c7041cd889f422b55dcf4",
          "url": "https://github.com/mlflow/mlflow/commit/d1bf8becd02eb2777a5192c584187a6c6dac6fc3"
        },
        "date": 1778481656736,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 39.67462704998752,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 21.697041714293974,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.197565234373883,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.18738507017856,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 20.572033149994695,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 11.571294800046417,
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
          "id": "364f37717555872a3796051b34a513d87ff6e086",
          "message": "Properly escape host parameter in rfunc serve command (#23179)\n\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: sreelim <sreelim@users.noreply.github.com>",
          "timestamp": "2026-05-11T07:09:33Z",
          "tree_id": "b5ee959ed8083dd372beb1771f3f4ecbfeb3559f",
          "url": "https://github.com/mlflow/mlflow/commit/364f37717555872a3796051b34a513d87ff6e086"
        },
        "date": 1778483615795,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 40.03432970000631,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.74131611764908,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 20.326450177420032,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.92383073585212,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.3934546666679,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 48.29420539998637,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "tomu.hirata@gmail.com",
            "name": "Tomu Hirata",
            "username": "TomeHirata"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ddcb5fd52bac6795e3fbd50b0d04c9b9c0ff01c2",
          "message": "Fix dspy cross-version test failures for dspy 3.2.0 (#23174)\n\nSigned-off-by: Claude <noreply@anthropic.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-11T10:45:35Z",
          "tree_id": "eefbe60c49f1d0d2e7d70e980505d71ef7730a04",
          "url": "https://github.com/mlflow/mlflow/commit/ddcb5fd52bac6795e3fbd50b0d04c9b9c0ff01c2"
        },
        "date": 1778496607034,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 45.61156300000064,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.692896735295797,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 20.30133893548225,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 22.39071013207516,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.49845421666714,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 9.110440000000608,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mprahl@users.noreply.github.com",
            "name": "Matthew Prahl",
            "username": "mprahl"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6c92ae538ec0dcd4181c4a2796380c875dfd175a",
          "message": "Add workspace isolation on scorers when creating a guardrail (#23115)\n\nSigned-off-by: mprahl <mprahl@users.noreply.github.com>",
          "timestamp": "2026-05-11T11:30:01Z",
          "tree_id": "5138a988ae108bc0b1cfa0635b4891091b1f629d",
          "url": "https://github.com/mlflow/mlflow/commit/6c92ae538ec0dcd4181c4a2796380c875dfd175a"
        },
        "date": 1778499239975,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 48.57490244999099,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 29.10088241934949,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 27.380585160003648,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 27.481218679995436,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 17.841287166656155,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 29.551444999992782,
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
          "id": "b920d2d5ecd9649523c7944775afdac1216e075f",
          "message": "Warn on startup when default basic_auth admin password is in use (#23182)\n\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: sreelim <sreelim@users.noreply.github.com>",
          "timestamp": "2026-05-12T07:43:45Z",
          "tree_id": "1cc13a9f022cfe4754eb089069704cac23ace0d6",
          "url": "https://github.com/mlflow/mlflow/commit/b920d2d5ecd9649523c7944775afdac1216e075f"
        },
        "date": 1778572096403,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 40.93866219999853,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.91182908571474,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.568703949998756,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 16.303327428569705,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.872053894740986,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 18.316592200005744,
            "unit": "ms"
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
          "id": "1334930a4df657abe4159924b108c08aae62945b",
          "message": "Fix `AmazonBedrockProvider._build_converse_kwargs` tool-call history and validation for Bedrock Converse (#23223)\n\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>\nCo-authored-by: TomeHirata <33407409+TomeHirata@users.noreply.github.com>\nCo-authored-by: Tomu Hirata <tomu.hirata@gmail.com>",
          "timestamp": "2026-05-12T08:16:44Z",
          "tree_id": "7e61a1ae96760d774a96b59be8799e8271af93aa",
          "url": "https://github.com/mlflow/mlflow/commit/1334930a4df657abe4159924b108c08aae62945b"
        },
        "date": 1778574091445,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 37.33677924999981,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.40097511428446,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.213096344262528,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.824903615384038,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 24.803671576271437,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.437111000004393,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "60318530+SahilKumar75@users.noreply.github.com",
            "name": "Sahil Kumar Singh",
            "username": "SahilKumar75"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "744ea6a95e97008383cbf1f978a269d6cbe88534",
          "message": "Fix ended `LiveSpan` state mutation (#23152)\n\nSigned-off-by: Sahil Kumar Singh <sahilkumargreat12@gmail.com>",
          "timestamp": "2026-05-13T01:38:23Z",
          "tree_id": "fd221c0db6af89314a3d3ca08db6d6610b33398c",
          "url": "https://github.com/mlflow/mlflow/commit/744ea6a95e97008383cbf1f978a269d6cbe88534"
        },
        "date": 1778636532200,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 39.90910050000025,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.756521411760826,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 23.661810680002873,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.620312094339653,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.582412859654802,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 9.125598600019202,
            "unit": "ms"
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
          "id": "1d2ba941791dbc123869549eeed33175d50d90fb",
          "message": "Post release bump version to `3.12.1.dev0` (#23260)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-13T04:21:23Z",
          "tree_id": "d7d74675acf374db4307d9479f4077aac2520028",
          "url": "https://github.com/mlflow/mlflow/commit/1d2ba941791dbc123869549eeed33175d50d90fb"
        },
        "date": 1778646292699,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 41.38273429999657,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.628911882351,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.243464806452778,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.27593881131917,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 21.66245928069886,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 39.245362600001954,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "tomu.hirata@gmail.com",
            "name": "Tomu Hirata",
            "username": "TomeHirata"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "132b8da083d3761409aede9b1d55c576de965fba",
          "message": "Fix llama_index cross-version test failures for 0.12.x and 0.14.x (#23173)\n\nSigned-off-by: Claude <noreply@anthropic.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-13T14:15:40Z",
          "tree_id": "c18bb55bf5c92061c084cd257a962f4c795ed2c9",
          "url": "https://github.com/mlflow/mlflow/commit/132b8da083d3761409aede9b1d55c576de965fba"
        },
        "date": 1778681989958,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 32.359714649998494,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 20.076228659091363,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 17.103723402439364,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 18.586823534246946,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 18.089659586664954,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 13.778979599999275,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "74836691+SomtochiUmeh@users.noreply.github.com",
            "name": "Somtochi Umeh",
            "username": "SomtochiUmeh"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4b5c6691d545f06acae58b9c1d3e4c0106512dee",
          "message": "Unblock delete dataset records for managed datasets (#23214)\n\nSigned-off-by: SomtochiUmeh <somtochiumeh@gmail.com>",
          "timestamp": "2026-05-13T20:28:17Z",
          "tree_id": "45f51622b81ccc23ae43492e4c5a4053a46c0329",
          "url": "https://github.com/mlflow/mlflow/commit/4b5c6691d545f06acae58b9c1d3e4c0106512dee"
        },
        "date": 1778704326180,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 40.60354735000047,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.13021120000006,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.834119080645518,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.780969928570926,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.073717116667325,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 49.16387759999736,
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
          "id": "58895f77b7f24f6cc5d4722377e7327d796ca770",
          "message": "Switch default judge alignment optimizer to MemAlign (#23254)\n\nSigned-off-by: Veronica Lyu <veronica.lyu@databricks.com>\nCo-authored-by: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-13T23:13:49Z",
          "tree_id": "f81641a5c342d97177b911fe98d0d1c12d7ea0c7",
          "url": "https://github.com/mlflow/mlflow/commit/58895f77b7f24f6cc5d4722377e7327d796ca770"
        },
        "date": 1778714237262,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 42.76077499999644,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 25.09692967647184,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.86432633333959,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 24.881324471698676,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 24.287589109089613,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 9.873788599975342,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "james.fletcher@databricks.com",
            "name": "james-fletcher-db",
            "username": "james-fletcher-db"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a92ede784a6c91b59a42c1e9bbfc1217c5699f7e",
          "message": "Add warning when search_traces called on UC location without time constraint (#22832)\n\nSigned-off-by: James Fletcher <james.fletcher@databricks.com>\nCo-authored-by: Kris Concepcion <84737625+kriscon-db@users.noreply.github.com>",
          "timestamp": "2026-05-15T02:02:02Z",
          "tree_id": "8202cbc87e8b677414cfda90a8c0727b56bb4817",
          "url": "https://github.com/mlflow/mlflow/commit/a92ede784a6c91b59a42c1e9bbfc1217c5699f7e"
        },
        "date": 1778810745978,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 44.5124314500049,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 21.757563857132872,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 20.983178999992802,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 20.45819917240437,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 15.366523399984544,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 9.84653100003925,
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
          "id": "a66b82f2a950fd69eaaeb909683b18f58f19b39d",
          "message": "Promote `prompt` to a first-class RBAC `resource_type` (#23248)\n\nSigned-off-by: Pat Sukprasert <pattara.sk127@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-15T14:09:48Z",
          "tree_id": "16c202c845a48266d15ac546cd3f42680e35df37",
          "url": "https://github.com/mlflow/mlflow/commit/a66b82f2a950fd69eaaeb909683b18f58f19b39d"
        },
        "date": 1778854404280,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 47.37767845000107,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.06519157142759,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.728582968750686,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 20.76689041818001,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.019763763637577,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 9.846775999997703,
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
          "id": "0b920c608aeaec147aa1f5a224788ce14d6cd2ec",
          "message": "Replace Python hook used by `mlflow autolog claude` to the new official claude plugin (#23339)\n\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-15T19:52:51Z",
          "tree_id": "35755cadbd130db256abd9b4656cf3d457049736",
          "url": "https://github.com/mlflow/mlflow/commit/0b920c608aeaec147aa1f5a224788ce14d6cd2ec"
        },
        "date": 1778874986741,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 48.076762350000024,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.81097600000004,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.12949293442744,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.891323089286384,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.030281586207675,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 9.253522800000269,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mprahl@users.noreply.github.com",
            "name": "Matthew Prahl",
            "username": "mprahl"
          },
          "committer": {
            "email": "31463517+B-Step62@users.noreply.github.com",
            "name": "Yuki Watanabe",
            "username": "B-Step62"
          },
          "distinct": true,
          "id": "c56dbd4e48a69fddd96a109bf715724a4615bc4a",
          "message": "Show trace archival settings only when enabled on the server (#23366)\n\nSigned-off-by: mprahl <mprahl@users.noreply.github.com>",
          "timestamp": "2026-05-15T13:53:54-07:00",
          "tree_id": "ad41fc0165c506373a57bcfcad6b42138166d2d2",
          "url": "https://github.com/mlflow/mlflow/commit/c56dbd4e48a69fddd96a109bf715724a4615bc4a"
        },
        "date": 1778878523613,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 40.89029135000288,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 25.641176529408856,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 20.489760725806697,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.73853657894635,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 21.130977599998367,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 28.1300310000006,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "raviprakash.darbha@gmail.com",
            "name": "Raviprakash Darbha",
            "username": "ravidarbha"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f047b4faab3aca3a1db1d66ac5e33a4e1b0dc02b",
          "message": "feat(tracking): Add reader/writer instance routing for database replicas (#22910)\n\nSigned-off-by: Raviprakash Darbha <rdarbha@amazon.com>\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nSigned-off-by: Yuki Watanabe <31463517+B-Step62@users.noreply.github.com>\nCo-authored-by: Raviprakash Darbha <rdarbha@amazon.com>\nCo-authored-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: Claude <noreply@anthropic.com>\nCo-authored-by: Yuki Watanabe <31463517+B-Step62@users.noreply.github.com>",
          "timestamp": "2026-05-16T03:57:01Z",
          "tree_id": "aa4443e9fb3fb8b9a61cfc4c3c9b1a26a52bfb7d",
          "url": "https://github.com/mlflow/mlflow/commit/f047b4faab3aca3a1db1d66ac5e33a4e1b0dc02b"
        },
        "date": 1778904047193,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 47.014596749998816,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.639780657142087,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.719455126984183,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 22.446665232139246,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 21.30295886885241,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 9.207601999997905,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "samrajmoorjani@gmail.com",
            "name": "Samraj Moorjani",
            "username": "smoorjani"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ac15c22741f7aac50b16b9c74beff6a8e2fdd823",
          "message": "Surface mlflow version mismatch when deserializing scorers (#23215)\n\nSigned-off-by: Samraj Moorjani <samraj.moorjani@databricks.com>",
          "timestamp": "2026-05-16T15:01:03Z",
          "tree_id": "d0c60125522658847990205edea3d01578edd303",
          "url": "https://github.com/mlflow/mlflow/commit/ac15c22741f7aac50b16b9c74beff6a8e2fdd823"
        },
        "date": 1778943878542,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 48.622808149998775,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 21.80784737142777,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.470445142856956,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.725467410713556,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 21.19899780000007,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 24.592246200001,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "147849970+SuperSonnix71@users.noreply.github.com",
            "name": "Sonny",
            "username": "SuperSonnix71"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f7b2ddf7b8afb0bd36c7e6d5a8ec80985bd1b699",
          "message": "Add Ollama as assistant provider (#22098)\n\nSigned-off-by: SuperSonnix71 <sonnym@hotmail.se>\nSigned-off-by: SuperSonnix71 <sonnym@terranex.ai>\nSigned-off-by: Sonny M <sonny@Sonnys-MacBook-Pro-2.local>\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: Yuki Watanabe <31463517+B-Step62@users.noreply.github.com>\nCo-authored-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-16T22:02:33Z",
          "tree_id": "2d2dabf36a86f0fa144a92f6029d471dc5de355c",
          "url": "https://github.com/mlflow/mlflow/commit/f7b2ddf7b8afb0bd36c7e6d5a8ec80985bd1b699"
        },
        "date": 1778969178181,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 44.442580850002145,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.744243696971605,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.980597714284606,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.89933901785771,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.8247763636366,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 9.261022399999774,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "191841109+mlflow-app[bot]@users.noreply.github.com",
            "name": "mlflow-app[bot]",
            "username": "mlflow-app[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ee9f83604d715ff6b0ae453097c6bc90e784dc89",
          "message": "Update model catalog from upstream sources (#23298)\n\nCo-authored-by: mlflow-app[bot] <mlflow-app[bot]@users.noreply.github.com>\nCo-authored-by: Tomu Hirata <tomu.hirata@gmail.com>",
          "timestamp": "2026-05-18T10:06:00Z",
          "tree_id": "c50cda533830a7426decb7197660eac201ed2435",
          "url": "https://github.com/mlflow/mlflow/commit/ee9f83604d715ff6b0ae453097c6bc90e784dc89"
        },
        "date": 1779098973376,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 46.95425439999781,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.772447142857896,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.846970919354884,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 22.654882482143535,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 19.971375763635603,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 30.76207280000176,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "tomu.hirata@gmail.com",
            "name": "Tomu Hirata",
            "username": "TomeHirata"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "96b7d900e14227e0821981409d86f5fb6c59b386",
          "message": "Add /gateway/proxy/{endpoint_name}/{path} raw proxy endpoint (#23330)\n\nSigned-off-by: Tomu Hirata <tomu.hirata@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-18T10:09:48Z",
          "tree_id": "766329c9edd1ae5bc6be057e1f6f85090cf87db2",
          "url": "https://github.com/mlflow/mlflow/commit/96b7d900e14227e0821981409d86f5fb6c59b386"
        },
        "date": 1779099218587,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 38.74756644999877,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.254274756756132,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.116142924241494,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 22.787487344826644,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 20.24094293220337,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.728155800000991,
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
          "id": "4c8c9d9e0831bba77f98ae59cc7aad9cc4da6695",
          "message": "Add `mlflow.genai.test_agent` for automated agent stress-testing (#22990)\n\nSigned-off-by: Serena Ruan <serena.rxy@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-18T10:19:23Z",
          "tree_id": "57ddd4ccf89de08fb5c2710512ee1089ea0002d5",
          "url": "https://github.com/mlflow/mlflow/commit/4c8c9d9e0831bba77f98ae59cc7aad9cc4da6695"
        },
        "date": 1779099775879,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 43.163319300001035,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.513079583333706,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.23261198461553,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 22.133996596493898,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 21.56179468333524,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.220578799990562,
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
          "id": "40df9feda1a0699f996f2eabdb10dd4aac22ed19",
          "message": "Unified per-user permission APIs: `grant` / `revoke` / `get` / `list` under `/mlflow/users/permissions/*` (#23247)\n\nSigned-off-by: Pat Sukprasert <pattara.sk127@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-18T11:20:08Z",
          "tree_id": "9cee56be09800f8b1f11ca4a12aff14f24af0108",
          "url": "https://github.com/mlflow/mlflow/commit/40df9feda1a0699f996f2eabdb10dd4aac22ed19"
        },
        "date": 1779103455063,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 47.38826019999891,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.898389399998013,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.6060823396221,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 22.65966606976757,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.21086615517107,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 24.53036480000037,
            "unit": "ms"
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
          "id": "30dff22d2b02b922cf7be9c6a7837439410ce162",
          "message": "Skip `transformers==5.0.0` in cross-version tests (#23427)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-18T13:05:51Z",
          "tree_id": "1df0a11a34abad2faef42afe40c1e6b1d51f8f5b",
          "url": "https://github.com/mlflow/mlflow/commit/30dff22d2b02b922cf7be9c6a7837439410ce162"
        },
        "date": 1779109760890,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 42.073449899997684,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 21.988280057139978,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.564289968254673,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.121160017238854,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 20.906200694916656,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 29.339108599998553,
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
          "id": "fc45c72e11301bd0321d748a59dbc8f450e2eca2",
          "message": "Remove deprecated legacy per-resource permission methods + endpoints (#23337)\n\nSigned-off-by: Pat Sukprasert <pattara.sk127@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-18T14:14:39Z",
          "tree_id": "758221913af24fc41ee50f6cb22e7c07605b9c83",
          "url": "https://github.com/mlflow/mlflow/commit/fc45c72e11301bd0321d748a59dbc8f450e2eca2"
        },
        "date": 1779113928264,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 51.277693649999634,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 27.345944483871584,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 25.309009716981492,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 22.792212437500403,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.546960999999357,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 25.707418199999665,
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
          "distinct": false,
          "id": "ef409cab8a72d21763dc762b94791ef1ec40437a",
          "message": "RBAC: `default_permission` is a floor; workspace `USE` stops folding into resource lookups (#23379)\n\nSigned-off-by: Pat Sukprasert <pattara.sk127@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-18T15:21:29Z",
          "tree_id": "43fc7d45cf975e76697d9ff60e8acd792e7bede7",
          "url": "https://github.com/mlflow/mlflow/commit/ef409cab8a72d21763dc762b94791ef1ec40437a"
        },
        "date": 1779117906023,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 46.94128534999891,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.715206857143812,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 24.533220806450718,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.559006796296543,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 16.50596557142998,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 8.916497399999912,
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
          "id": "4a6175e278f8a4d24517606ebb6ef37076b08e84",
          "message": "RBAC: extend `prompt` resource_type to after-request handlers (#23426)\n\nSigned-off-by: Pat Sukprasert <pattara.sk127@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-19T01:08:54Z",
          "tree_id": "46d63a591b1547191c4bbc2a953bc858ea2862f3",
          "url": "https://github.com/mlflow/mlflow/commit/4a6175e278f8a4d24517606ebb6ef37076b08e84"
        },
        "date": 1779153152476,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 40.518698350002325,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 21.71640411428411,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.415448285714156,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.38102136842067,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 20.532174433334426,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 29.286863599998014,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "gkrumbac@redhat.com",
            "name": "Gage Krumbach",
            "username": "Gkrumbach07"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "18d14d32c6c04f34be92d366af97686a6f8698b6",
          "message": "Support settings.local.json for Claude Code tracing config (#23285)\n\nSigned-off-by: Gage Krumbach <gkrumbac@redhat.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-19T03:00:10Z",
          "tree_id": "47cc664603d8d7c31dbd6694f45bd7ae02b20bf2",
          "url": "https://github.com/mlflow/mlflow/commit/18d14d32c6c04f34be92d366af97686a6f8698b6"
        },
        "date": 1779159795966,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 38.086622550000016,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.803722399999675,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 25.044466983332825,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.964821410713352,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.760674385965388,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.494387000002689,
            "unit": "ms"
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
          "id": "3c23d8b20196bfa1519efb3346beb1a5f03703b7",
          "message": "Remove MLServer integration from pyfunc serving backend (#23356)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-19T06:05:47Z",
          "tree_id": "2b109b7ef74d2d2927523081bbafdd4d7fb6e2f6",
          "url": "https://github.com/mlflow/mlflow/commit/3c23d8b20196bfa1519efb3346beb1a5f03703b7"
        },
        "date": 1779170976688,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 45.027984749999206,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.59666891428469,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 24.61053657377034,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.638315800000573,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 16.40855850000141,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.575972200001502,
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
          "id": "ee39ed8110fa6e33f4772fc08f6372772f650c0a",
          "message": "Reject self-delete in `delete_user` (#23398)\n\nSigned-off-by: Pat Sukprasert <pattara.sk127@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-19T06:46:26Z",
          "tree_id": "e52ee397eecd250f862913c0d2c50bf5b533febf",
          "url": "https://github.com/mlflow/mlflow/commit/ee39ed8110fa6e33f4772fc08f6372772f650c0a"
        },
        "date": 1779173396718,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 45.274976449999116,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.51024931428622,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 24.199692370967888,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.342309785713404,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.012314377357498,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.75176199999521,
            "unit": "ms"
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
          "id": "7db6b4a0f437142d3daa4f909e71ff432ff0d86a",
          "message": "Release `_post_import_hooks_lock` before firing hooks (#23466)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-19T11:00:46Z",
          "tree_id": "2adb0a965b549234d070a1c0fc6dabe7207a8b6a",
          "url": "https://github.com/mlflow/mlflow/commit/7db6b4a0f437142d3daa4f909e71ff432ff0d86a"
        },
        "date": 1779188660531,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 44.19388605000165,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.544648756756413,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 15.306051888889746,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 25.667470517241988,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 25.4020580338992,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 28.614436399999477,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "arthur.jenoudet@databricks.com",
            "name": "Arthur Jenoudet",
            "username": "artjen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "92c9cf2f282a7eb6f1b8c26998708709ed977999",
          "message": "Make `mlflow.get_trace` V4 retry policy configurable (#23443)\n\nSigned-off-by: Arthur Jenoudet <arthur.jenoudet@databricks.com>\nCo-authored-by: Claude <noreply@anthropic.com>\nCo-authored-by: Yuki Watanabe <31463517+B-Step62@users.noreply.github.com>",
          "timestamp": "2026-05-19T12:30:44-07:00",
          "tree_id": "7553dc14a2685d36a25b88cb1902f88d5c5df101",
          "url": "https://github.com/mlflow/mlflow/commit/92c9cf2f282a7eb6f1b8c26998708709ed977999"
        },
        "date": 1779219133529,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 49.48736270000609,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.72906557575656,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.788055327867188,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.80283055357373,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.65577387719323,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 9.158226000005243,
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
          "id": "b4af789377b61f753a14bd0d10beff3a865f5923",
          "message": "Preserve pdfjs-dist bundles in webpack build (`craco.config.js`) (#23349)\n\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-19T20:20:25Z",
          "tree_id": "bca4ed73cddf9e984367ea8515405ad1cced5320",
          "url": "https://github.com/mlflow/mlflow/commit/b4af789377b61f753a14bd0d10beff3a865f5923"
        },
        "date": 1779222250112,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 52.91888359999746,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.16510562500296,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.091292531249085,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 22.686399473683565,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.91894565384758,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 60.13624920000211,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "1fannnw@gmail.com",
            "name": "Stefan Wang",
            "username": "1fanwang"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "78bd647bb5e5ecb698ad4d3dda8f9a8703872386",
          "message": "Fix UnicodeEncodeError on artifact download with non-ASCII filename (#23241)\n\nSigned-off-by: 1fanwang <1fannnw@gmail.com>\nSigned-off-by: Yuki Watanabe <31463517+B-Step62@users.noreply.github.com>\nCo-authored-by: Yuki Watanabe <31463517+B-Step62@users.noreply.github.com>",
          "timestamp": "2026-05-19T20:47:53Z",
          "tree_id": "4e141f04dfdaeb8c5c0630498957df21a1b68343",
          "url": "https://github.com/mlflow/mlflow/commit/78bd647bb5e5ecb698ad4d3dda8f9a8703872386"
        },
        "date": 1779223874931,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 44.69851955000337,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 25.384659911766757,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.80113913333442,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 24.45770582758606,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 24.143283448275344,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.606829600001674,
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
          "id": "d90c784b5701d756b5092e9298812e6c4f462312",
          "message": "Pin `transformers<5` for pytorch < 2.5 in cross-version tests (#23440)\n\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-19T20:52:14Z",
          "tree_id": "6ecc4bff1336dc6e8a9af0196ec53bba1edc3d6f",
          "url": "https://github.com/mlflow/mlflow/commit/d90c784b5701d756b5092e9298812e6c4f462312"
        },
        "date": 1779224144023,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 40.17700990000179,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 20.188407540543366,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 20.383333358209804,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 19.97779718644114,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 19.29860076562573,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.506053200000906,
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
          "id": "fe6ec1a7092be3ee8471f9f3c1591d993a483d7e",
          "message": "[Security] Extend MLFLOW_ALLOW_PICKLE_DESERIALIZATION guard to DSPy non-pkl branch (#23293)\n\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: Av7danger <Av7danger@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-19T21:36:41Z",
          "tree_id": "f7f0ae1069d5da7bc9bc0caade082a6d3acdcb22",
          "url": "https://github.com/mlflow/mlflow/commit/fe6ec1a7092be3ee8471f9f3c1591d993a483d7e"
        },
        "date": 1779226811779,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 48.47079680000377,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 25.722985794118358,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.721921951614377,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 24.40945705454438,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 26.635737175440045,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.069197399982841,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "58055473+xodn348@users.noreply.github.com",
            "name": "Junhyuk Lee",
            "username": "xodn348"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ee08e0166786b105cd0d17430dbe404cd9a552f9",
          "message": "fix(tracking): return `<console>` for `mlflow.source.name` when `sys.argv[0]` is empty (#23352)\n\nSigned-off-by: xodn348 <xodn348@tamu.edu>\nCo-authored-by: Yuki Watanabe <31463517+B-Step62@users.noreply.github.com>",
          "timestamp": "2026-05-19T21:38:26Z",
          "tree_id": "2d3ec3c94f2f5204416dbaced37679aa1dceaae2",
          "url": "https://github.com/mlflow/mlflow/commit/ee08e0166786b105cd0d17430dbe404cd9a552f9"
        },
        "date": 1779226943838,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 47.32755719999986,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.446481030304398,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.646982650000744,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 22.47045587719205,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.07835120338945,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 31.910170599996942,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "121050723+fenil210@users.noreply.github.com",
            "name": "Fenil Ramoliya",
            "username": "fenil210"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "301d0abbffdf3acb5e413dae75b999690d9a6b84",
          "message": "fenil/fix-response-format-json: Tighten response format JSON schema type (#23290)\n\nSigned-off-by: Fenil Ramoliya <fenilramoliya2103@gmail.com>",
          "timestamp": "2026-05-19T21:45:03Z",
          "tree_id": "8812884bc069546bd42ac89763ccc474051ccc78",
          "url": "https://github.com/mlflow/mlflow/commit/301d0abbffdf3acb5e413dae75b999690d9a6b84"
        },
        "date": 1779227331796,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 38.71868895000006,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.057291166667525,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.520984881355908,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.127377452830846,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 24.087446350877254,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.453104999996185,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "191841109+mlflow-app[bot]@users.noreply.github.com",
            "name": "mlflow-app[bot]",
            "username": "mlflow-app[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "53b53ddcd72cfce1bd431d29bdf5ac31029d7edb",
          "message": "Update model catalog from upstream sources (#23484)\n\nCo-authored-by: mlflow-app[bot] <mlflow-app[bot]@users.noreply.github.com>",
          "timestamp": "2026-05-20T01:40:27Z",
          "tree_id": "ead07c833abc5a5fe0db72407a9e39d0f9dd22d2",
          "url": "https://github.com/mlflow/mlflow/commit/53b53ddcd72cfce1bd431d29bdf5ac31029d7edb"
        },
        "date": 1779241459292,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 44.893953550000276,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.037162029413018,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.32441475806517,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 25.876251800000524,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 26.964268999999774,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.447239800001171,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "191841109+mlflow-app[bot]@users.noreply.github.com",
            "name": "mlflow-app[bot]",
            "username": "mlflow-app[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "44b622772b300328b78ec2dc9076c3ba90199ea3",
          "message": "Update ML package versions for 3.13.0rc0 (#23491)\n\nSigned-off-by: mlflow-app[bot] <mlflow-app[bot]@users.noreply.github.com>\nSigned-off-by: Kris Concepcion <kris.concepcion@databricks.com>\nCo-authored-by: mlflow-app[bot] <mlflow-app[bot]@users.noreply.github.com>\nCo-authored-by: Kris Concepcion <kris.concepcion@databricks.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-20T05:12:55-04:00",
          "tree_id": "6a1b842ca5444286bc5051606038b87700c8dc2a",
          "url": "https://github.com/mlflow/mlflow/commit/44b622772b300328b78ec2dc9076c3ba90199ea3"
        },
        "date": 1779268462479,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 37.68495230000042,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 27.37080888235255,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 28.1786350892845,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 29.633632169812742,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 29.294135490565267,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 8.14591380000138,
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
          "id": "66a84c10897e7e29dce5d7034829fbf6c45e9e1e",
          "message": "Reject same-password rotations in `update_user_password` (#23413)\n\nSigned-off-by: Pat Sukprasert <pattara.sk127@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-20T15:05:44Z",
          "tree_id": "14e81a16e95b5ec7083a206154a4ecd4d2a21401",
          "url": "https://github.com/mlflow/mlflow/commit/66a84c10897e7e29dce5d7034829fbf6c45e9e1e"
        },
        "date": 1779289782803,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 44.78129120000176,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.436293594594375,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.895473065572748,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.948673684210906,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 20.96371569354876,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 8.372532199996385,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "147849970+SuperSonnix71@users.noreply.github.com",
            "name": "Sonny",
            "username": "SuperSonnix71"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9e7d61e5422ab4d9cdd451d82df2c479305143ca",
          "message": "Add OpenAI Codex CLI as assistant provider (#22566)\n\nSigned-off-by: Sonny M <sonny@Sonnys-MacBook-Pro-2.local>\nSigned-off-by: SuperSonnix71 <sonnym@hotmail.se>\nSigned-off-by: Sonny <sonnym@terranex.ai>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-20T19:35:30Z",
          "tree_id": "f70c5c200e8401fde3e2ed34c3fd9d7523f2b89c",
          "url": "https://github.com/mlflow/mlflow/commit/9e7d61e5422ab4d9cdd451d82df2c479305143ca"
        },
        "date": 1779305979777,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 51.499552849998054,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 29.516302911762228,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 27.29573356000401,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 26.50510630188656,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 16.58241500000675,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 9.643898799993167,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "james.fletcher@databricks.com",
            "name": "james-fletcher-db",
            "username": "james-fletcher-db"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c9e3d40653a71fb40bb4540ce52ca7aa636c7e9d",
          "message": "Fix judge fallback on event-based traces grading itself (#23445)\n\nSigned-off-by: James Fletcher <james.fletcher@databricks.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-20T21:26:43Z",
          "tree_id": "7fa0ba1e07830353bc9ef07e5d22cc9132858a25",
          "url": "https://github.com/mlflow/mlflow/commit/c9e3d40653a71fb40bb4540ce52ca7aa636c7e9d"
        },
        "date": 1779312641575,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 51.293258100000116,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 35.11622619354908,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 31.138373379311865,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 30.94110258000228,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 30.247292884613312,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.345098799993366,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "tomu.hirata@gmail.com",
            "name": "Tomu Hirata",
            "username": "TomeHirata"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a37edf95cec2126ec1a566eb3150abed3a7b0e8e",
          "message": "Add 21 new models to Databricks model catalog (#23520)\n\nSigned-off-by: Tomu Hirata <tomu.hirata@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-21T15:40:29+09:00",
          "tree_id": "c8e209101d51f2c564873dafa36aa1336e303164",
          "url": "https://github.com/mlflow/mlflow/commit/a37edf95cec2126ec1a566eb3150abed3a7b0e8e"
        },
        "date": 1779345715021,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 46.62404860000038,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 25.853403799989596,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 24.824550698414196,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.04962243103589,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.449123900001194,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 27.206322799986538,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "191841109+mlflow-app[bot]@users.noreply.github.com",
            "name": "mlflow-app[bot]",
            "username": "mlflow-app[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1e1170ef51ac14d657e7df763f1c1c417c01a307",
          "message": "Update model catalog from upstream sources (#23527)\n\nCo-authored-by: mlflow-app[bot] <mlflow-app[bot]@users.noreply.github.com>",
          "timestamp": "2026-05-21T19:40:24+09:00",
          "tree_id": "f15697266030f3e89178da08606ff110e72b0d7b",
          "url": "https://github.com/mlflow/mlflow/commit/1e1170ef51ac14d657e7df763f1c1c417c01a307"
        },
        "date": 1779360109546,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 45.58110490000047,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 26.615914416666442,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 27.155887428571692,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 25.052267642853923,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 24.187515406777713,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.493198999992501,
            "unit": "ms"
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
          "id": "8d0ab52b30cd9bf18629969fca35063353bcf722",
          "message": "Add `flavors` CLI for `ml-package-versions.yml` (#23523)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-21T20:39:31+09:00",
          "tree_id": "21416a428c081eae7f085d15a93961f76a4c5f5a",
          "url": "https://github.com/mlflow/mlflow/commit/8d0ab52b30cd9bf18629969fca35063353bcf722"
        },
        "date": 1779363657801,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 38.63156115000024,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.77508938235385,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.598859559321205,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 24.97537091999959,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 24.41663684615437,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.6777213999946525,
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
          "id": "b5dd920c23f3a640598ab200747fb7023e205fa2",
          "message": "Reserve the entire `__user_` role-name prefix from admin authoring (#23496)\n\nSigned-off-by: Pat Sukprasert <pattara.sk127@gmail.com>",
          "timestamp": "2026-05-21T17:25:18Z",
          "tree_id": "fa9b947b620b05b5dc1203a7418b0e29bfa31c6a",
          "url": "https://github.com/mlflow/mlflow/commit/b5dd920c23f3a640598ab200747fb7023e205fa2"
        },
        "date": 1779384533622,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 51.06542600000097,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 32.76756565384504,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 30.519847348836436,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 34.642635250001774,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.00928400000161,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 11.165163399999756,
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
          "id": "1c491e70c19e1d7668b19419d8c9754a9632f4e3",
          "message": "[Security] Remove dead cloudpickle.load fallback in job subprocess entry (#23294)\n\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: jeongbeannnn <jeongbeannnn@users.noreply.github.com>",
          "timestamp": "2026-05-22T02:57:49Z",
          "tree_id": "63afcd211fba0fbb983e089ebbab93526fe81780",
          "url": "https://github.com/mlflow/mlflow/commit/1c491e70c19e1d7668b19419d8c9754a9632f4e3"
        },
        "date": 1779418893679,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 41.57006765000091,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.42881267647101,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 23.122036600001177,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.918751929823962,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 26.64147574999934,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 10.386014200000204,
            "unit": "ms"
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
          "distinct": false,
          "id": "377169ce5d8af0836fed320b89fc8d34fcb42477",
          "message": "Bump R `mlflow.connect.wait` in tests to reduce flakes (#23553)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-22T11:02:09Z",
          "tree_id": "dbdece3145c0fdb3f53ef9474f6778338c2ba02d",
          "url": "https://github.com/mlflow/mlflow/commit/377169ce5d8af0836fed320b89fc8d34fcb42477"
        },
        "date": 1779447950989,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 41.21091780000938,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.9179485999954,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.165202746033888,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.79174603703194,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 24.826544644064928,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 11.662411999998312,
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
          "distinct": false,
          "id": "9f54109b47eece5711c0c64990a32bd33fddd36d",
          "message": "Bring direct-grant picker to parity with role picker (#23420)\n\nSigned-off-by: Pat Sukprasert <pattara.sk127@gmail.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-22T18:39:57Z",
          "tree_id": "bf18410ed72c45da3abb4e1afd0a888b9337875e",
          "url": "https://github.com/mlflow/mlflow/commit/9f54109b47eece5711c0c64990a32bd33fddd36d"
        },
        "date": 1779475399033,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 46.95519964999946,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 28.03472061764806,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 26.216413052631736,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 24.362881425927466,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 26.70182820339077,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 9.204723199999876,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "indranil.dutta@jazzx.ai",
            "name": "Indranil",
            "username": "id-jazzx"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6ae7e215a7e46a481e50be0efda2f10eb7218bec",
          "message": "Add OpenAI `/responses/compact` passthrough route to AI Gateway (#23353)\n\nSigned-off-by: Indranil Dutta <indranil.dutta@jazzx.ai>",
          "timestamp": "2026-05-22T22:55:59Z",
          "tree_id": "0f75fe4f38cda5964273b2d5768570cd3935b22f",
          "url": "https://github.com/mlflow/mlflow/commit/6ae7e215a7e46a481e50be0efda2f10eb7218bec"
        },
        "date": 1779490783729,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 47.255457750003416,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 26.6300187941218,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 24.94040062295092,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 16.414735625001953,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 25.94665589830443,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 41.62405499999409,
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
          "id": "6ae1e6395d4fa2a304cf4834abece272ff39e829",
          "message": "Disable credentialed CORS when wildcard origins are configured (#23178)\n\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: sreelim <sreelim@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-24T19:30:10Z",
          "tree_id": "a804d9cfd263f2343b2e1767d9d3f4bf2c24d140",
          "url": "https://github.com/mlflow/mlflow/commit/6ae1e6395d4fa2a304cf4834abece272ff39e829"
        },
        "date": 1779651204282,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 45.694179100000554,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.481227694443202,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 24.09395962500005,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 22.937240526316987,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 21.89120849999971,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 9.75110819999827,
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
          "id": "46969b841f34ab074c549622f362754221660004",
          "message": "Record git branch and repo URL on runs in `GitRunContext` (#23310)\n\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-24T20:17:09Z",
          "tree_id": "92ce8baa20f5488ccf176606693f0d3d573188d9",
          "url": "https://github.com/mlflow/mlflow/commit/46969b841f34ab074c549622f362754221660004"
        },
        "date": 1779654031915,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 46.613249549999125,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.596075411766826,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 20.608383639341344,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.833440660713116,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 23.07793257894699,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 56.138221400004795,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "debusinha2009@gmail.com",
            "name": "Debu Sinha",
            "username": "debu-sinha"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "78c26e15886f83faa63e14d22f4a034d211b8d73",
          "message": "Add Google ADK LLM judge scorers (`Hallucination`, `Safety`, `ResponseEvaluation`) (#22496)\n\nSigned-off-by: debu-sinha <debusinha2009@gmail.com>\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-25T03:57:33Z",
          "tree_id": "a6e762ca937212e687cef3887e61dcbeca3f0608",
          "url": "https://github.com/mlflow/mlflow/commit/78c26e15886f83faa63e14d22f4a034d211b8d73"
        },
        "date": 1779681671704,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 45.0936986500011,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 25.559186242426016,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 23.70014682758563,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 25.788493040817208,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 17.003177285711704,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 10.387216200001603,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "kishor.radhakrishnan@outlook.com",
            "name": "kishor-rkrishnan",
            "username": "kishor-rkrishnan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fee05488d0b3757cf2d6426d8c07f1fb9a52e14e",
          "message": "Fix pydantic-ai >= 1.78.0 ToolManager module rename (#23508) (#23528)\n\nSigned-off-by: kishor-rkrishnan <286408206+kishor-rkrishnan@users.noreply.github.com>\nCo-authored-by: kishor-rkrishnan <286408206+kishor-rkrishnan@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-25T12:08:50+08:00",
          "tree_id": "02be545de284388aac2875e35778c804b06b98ca",
          "url": "https://github.com/mlflow/mlflow/commit/fee05488d0b3757cf2d6426d8c07f1fb9a52e14e"
        },
        "date": 1779682211540,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 46.761292250005226,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.881415823526236,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 20.558515475403006,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.413983896550313,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 21.45838316363221,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 9.169968800006245,
            "unit": "ms"
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
          "id": "eb785ecd7e3f5f8c19f4c66e0f1e0d207155d6c5",
          "message": "Pin `wrapt<2` for agno autologging cross-version tests (#23585)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-25T23:35:44+09:00",
          "tree_id": "7b17aa002f4debc238040ee5e25b4564912be884",
          "url": "https://github.com/mlflow/mlflow/commit/eb785ecd7e3f5f8c19f4c66e0f1e0d207155d6c5"
        },
        "date": 1779719858260,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 58.32747785000052,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.381480954544287,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 20.009450324323744,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 19.029854253731667,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 20.88481490411074,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 5.3170820000048025,
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
          "id": "196d7796fba6d62d5f16ba004c3c183c857760fd",
          "message": "Deprecate `validate_serving_input` in favor of `mlflow.models.predict` (#23376)\n\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-25T17:50:12Z",
          "tree_id": "b0dac1a90ae600988f330af29674b13ef358cae3",
          "url": "https://github.com/mlflow/mlflow/commit/196d7796fba6d62d5f16ba004c3c183c857760fd"
        },
        "date": 1779731630986,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 46.59846055000543,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.23516839999645,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 19.976280079366653,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 15.95310444444446,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 20.859758131146904,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 29.223204800001668,
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
          "id": "223a580cb4731d65830de08106eb30d7ad13db37",
          "message": "[Security] Register auth validator for /ajax-api/3.0/mlflow/get-trace-artifact (#23317)\n\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: eddieran <eddieran@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-25T19:16:06Z",
          "tree_id": "6946597f406463d632fe175c3d4ae8273c6e911b",
          "url": "https://github.com/mlflow/mlflow/commit/223a580cb4731d65830de08106eb30d7ad13db37"
        },
        "date": 1779736781768,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 46.208646200004466,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 21.53702628571799,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 19.282154203128954,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 19.24261418181872,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 18.75386494642685,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 8.658172200023273,
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
          "id": "607e88cc22e546c26428c7fd7ed708a625ac6d78",
          "message": "fix: UI does not show Judge costs (#23586)\n\nSigned-off-by: Weichen Xu <weichen.xu@databricks.com>\nSigned-off-by: WeichenXu <weichen.xu@databricks.com>\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>",
          "timestamp": "2026-05-26T04:05:57Z",
          "tree_id": "43f94afa30be0c05546f9fece8555a3162bb62d3",
          "url": "https://github.com/mlflow/mlflow/commit/607e88cc22e546c26428c7fd7ed708a625ac6d78"
        },
        "date": 1779768566506,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 38.55189885000172,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 25.20706745714116,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.99057138709564,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 25.275399127272635,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 16.93724633333223,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.259100999999646,
            "unit": "ms"
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
          "id": "2831e2163951e3ce98c90e8b9e4d22f4838245d9",
          "message": "Require top-level `permissions: {}` in workflows (#23606)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-26T23:52:53+09:00",
          "tree_id": "05bdbfcc138644660f5d7aae20b2d4b7287c96da",
          "url": "https://github.com/mlflow/mlflow/commit/2831e2163951e3ce98c90e8b9e4d22f4838245d9"
        },
        "date": 1779807266416,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 37.900562199995136,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.15352045714404,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.57808314285964,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 16.425133714294912,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.57700029999986,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.598029999996925,
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
          "id": "ef66c24eda2ddee99eac62d648d07750749f5871",
          "message": "Support AI Gateway as a backend of MLflow Assistant (#23559)\n\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-26T18:13:55-07:00",
          "tree_id": "bc7bdccf58b49e124a0a76f49ee60fc377ed00ed",
          "url": "https://github.com/mlflow/mlflow/commit/ef66c24eda2ddee99eac62d648d07750749f5871"
        },
        "date": 1779844524201,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 44.216034099999035,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.006889114286878,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.980165980391007,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.590176692308134,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.47418308621032,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 37.460954999994556,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "contact.rdudhat@gmail.com",
            "name": "Rudra Dudhat",
            "username": "RudraDudhat2509"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "60864d582fc522f5900250798a7b2fc6459b725f",
          "message": "fix: extend `mlflow.sourceRun` metrics filter to cover post-hoc linked OTLP traces (#23591)\n\nSigned-off-by: Rudra Dudhat <contact.rdudhat@gmail.com>\nCo-authored-by: WeichenXu <weichen.xu@databricks.com>",
          "timestamp": "2026-05-27T02:34:13Z",
          "tree_id": "8e8e7bac8af194c4673e0394b949d6bbd1bc59df",
          "url": "https://github.com/mlflow/mlflow/commit/60864d582fc522f5900250798a7b2fc6459b725f"
        },
        "date": 1779849473258,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 45.27946434999919,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 20.8609205405408,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 20.380925312500242,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 20.17934913793144,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 19.706245180327688,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 21.565112400003272,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "joshua.wong@databricks.com",
            "name": "joshuawong-db",
            "username": "joshuawong-db"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dddcbb70cccf54c1b2d91b3d5e665f431e7d4321",
          "message": "Flush async trace queue before generating MLFlow demo traces (#23614)\n\nSigned-off-by: Joshua Wong <joshua.wong@databricks.com>",
          "timestamp": "2026-05-27T18:43:22Z",
          "tree_id": "468f946ebcc9e3b4e448c308e3641ad9c17c6e3d",
          "url": "https://github.com/mlflow/mlflow/commit/dddcbb70cccf54c1b2d91b3d5e665f431e7d4321"
        },
        "date": 1779907618584,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 45.80656560000094,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 25.60210371428419,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 23.872180467742233,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 26.41092745283108,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 25.629062799999826,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.543341600000986,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "Mytolo@users.noreply.github.com",
            "name": "Mytolo",
            "username": "Mytolo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7a72efaf39d64b74f97b8cdd7def02d06b03b85d",
          "message": "Add ON DELETE CASCADE relationship for `SqlTraceInfo` to `SqlExperiment` (#23194)\n\nCo-authored-by: Panajiotis Kessler <Panajiotis.Kessler@digits.schwarz>\nCo-authored-by: Tomu Hirata <tomu.hirata@gmail.com>",
          "timestamp": "2026-05-27T21:10:03Z",
          "tree_id": "06d21785b479a7124f62662fcd2bd0169db2a471",
          "url": "https://github.com/mlflow/mlflow/commit/7a72efaf39d64b74f97b8cdd7def02d06b03b85d"
        },
        "date": 1779916428796,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 49.39138999999955,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 32.92424530303175,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 31.198209072726968,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 31.077405059999137,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 27.302003615381828,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 8.037728000005018,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "191841109+mlflow-app[bot]@users.noreply.github.com",
            "name": "mlflow-app[bot]",
            "username": "mlflow-app[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "044c89b850b2477e53abd040245ff5167aefc3f4",
          "message": "Update model catalog from upstream sources (#23575)\n\nCo-authored-by: mlflow-app[bot] <mlflow-app[bot]@users.noreply.github.com>\nCo-authored-by: Tomu Hirata <tomu.hirata@gmail.com>",
          "timestamp": "2026-05-27T21:19:16Z",
          "tree_id": "06a69373c34ee96c72cffe3f0fe3fcd7e8466e0f",
          "url": "https://github.com/mlflow/mlflow/commit/044c89b850b2477e53abd040245ff5167aefc3f4"
        },
        "date": 1779916962434,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 38.227857549999555,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 27.543756057142282,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 28.0498560645142,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 25.76195496296268,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 24.830565741378884,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.558771799989472,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "samrajmoorjani@gmail.com",
            "name": "Samraj Moorjani",
            "username": "smoorjani"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "705739916ddf6dc92f70787e39cc43b0dd1dcdd7",
          "message": "Forward MLflow client telemetry from inside Databricks workloads (#23483)\n\nSigned-off-by: Samraj Moorjani <samraj.moorjani@databricks.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-28T00:03:59Z",
          "tree_id": "bd7cc0889908879456f948ecc116fce1b7a59272",
          "url": "https://github.com/mlflow/mlflow/commit/705739916ddf6dc92f70787e39cc43b0dd1dcdd7"
        },
        "date": 1779926862845,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 45.91222144999989,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 26.790828628572125,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.664585435482913,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 16.444655375003947,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 25.61755910169369,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 10.080819000000929,
            "unit": "ms"
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
          "id": "494445fabf5495396047dc8eea9b180bddc46df4",
          "message": "Bump R serve test wait budget to ~30s (#23636)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-28T15:35:01+09:00",
          "tree_id": "68e17395e8286f04e5d22db7519f1a9240fb1047",
          "url": "https://github.com/mlflow/mlflow/commit/494445fabf5495396047dc8eea9b180bddc46df4"
        },
        "date": 1779950181996,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 37.559306950002735,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.130491081081118,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 23.08723452307635,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 22.89411927118699,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 24.03993163934371,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 6.7930768000024955,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "HumairAK@users.noreply.github.com",
            "name": "Humair Khan",
            "username": "HumairAK"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "61a19305e66075e6191c4a9b14578b5e56cbbb64",
          "message": "Clear archive-now requests for non-archivable leftovers (#23655)\n\nSigned-off-by: Humair Khan <HumairAK@users.noreply.github.com>",
          "timestamp": "2026-05-28T19:57:19Z",
          "tree_id": "12f073061292e72f2191c0ad6b39f0dc48059d12",
          "url": "https://github.com/mlflow/mlflow/commit/61a19305e66075e6191c4a9b14578b5e56cbbb64"
        },
        "date": 1779998486154,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 37.373896750001734,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.64051591428376,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 23.780668507935758,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.225955350875637,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.235230383333736,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 7.076338799998894,
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
          "id": "b37b43800472db5f2e203e7d17073da3276a19de",
          "message": "Reword `FileStore` maintenance-mode error and document escape hatch (#23637)\n\nSigned-off-by: B-Step62 <yuki.watanabe@databricks.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-28T22:55:54Z",
          "tree_id": "98f395f976a91bf61a9ae7cf763c012e77a246a8",
          "url": "https://github.com/mlflow/mlflow/commit/b37b43800472db5f2e203e7d17073da3276a19de"
        },
        "date": 1780009176193,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 39.30657405000062,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.406777028569895,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 23.063972095237148,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 22.625168232142556,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 21.70111390163766,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 47.38954100000399,
            "unit": "ms"
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
          "id": "3c276eed09ed38b507886909f86d630487f4a1ce",
          "message": "Forward OpenAI custom base URL in Detect Issues flow (#23650)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-29T00:19:32Z",
          "tree_id": "cf83e5204193d11b3613c1608edcfd045c88106b",
          "url": "https://github.com/mlflow/mlflow/commit/3c276eed09ed38b507886909f86d630487f4a1ce"
        },
        "date": 1780014189605,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 41.380290199998626,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 21.96346773529586,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 19.64672473437279,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 16.01256544443825,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 21.140087803277137,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 26.458166400004757,
            "unit": "ms"
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
          "id": "825247641d9370e7495a2662ed350dd51c379683",
          "message": "Pin `safetensors>=0.8.0rc0` for `diffusers>=0.38.0` cross-version tests (#23666)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-29T14:36:36+09:00",
          "tree_id": "cf3278396f2e533a3e7688cec4c738c373b730c8",
          "url": "https://github.com/mlflow/mlflow/commit/825247641d9370e7495a2662ed350dd51c379683"
        },
        "date": 1780033081354,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 40.907187349999674,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 24.12152064705917,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.943439476189717,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 17.05686771428816,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.565453196430393,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 33.185993600000074,
            "unit": "ms"
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
          "id": "0dab98c42523375569e12c4f9ff39ef377d084a3",
          "message": "Pin `opentelemetry-sdk<1.40` for mistralai 2.x cross-version tests (#23667)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-29T15:04:00+09:00",
          "tree_id": "b02686f73443ae830d1dd7c21137100502453873",
          "url": "https://github.com/mlflow/mlflow/commit/0dab98c42523375569e12c4f9ff39ef377d084a3"
        },
        "date": 1780034724678,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 38.31926205000045,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.053148888887836,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.847305875001055,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 21.555649672415235,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 21.218178799999993,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 38.49495320000074,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "60318530+SahilKumar75@users.noreply.github.com",
            "name": "Sahil Kumar Singh",
            "username": "SahilKumar75"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1bbb1d8b21d47fd7a9af2f3519fa1add356cb1ab",
          "message": "Unwrap JSON-encoded `session.id` / `user.id` span attributes on ingest (#23642)\n\nSigned-off-by: Sahil Kumar Singh <sahilkumargreat12@gmail.com>\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>\nCo-authored-by: Harutaka Kawamura <hkawamura0130@gmail.com>",
          "timestamp": "2026-05-29T09:43:02Z",
          "tree_id": "e95305a5b482469f2225ad5bff38e80597d7956a",
          "url": "https://github.com/mlflow/mlflow/commit/1bbb1d8b21d47fd7a9af2f3519fa1add356cb1ab"
        },
        "date": 1780048010790,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 39.43249514999678,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 25.566486352941556,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 22.849840403226967,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 24.519975339287672,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 17.02511733333741,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 6.648820799995292,
            "unit": "ms"
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
          "id": "53ea147198c63f3cf99c9239f8b45a4bc9742881",
          "message": "Restore `mlflow.crewai` autolog on crewai 1.14.5 (#23682)\n\nSigned-off-by: harupy <17039389+harupy@users.noreply.github.com>\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-29T22:55:21+09:00",
          "tree_id": "58b1d052b0ae2592a1791663b4aa927b22ef9c28",
          "url": "https://github.com/mlflow/mlflow/commit/53ea147198c63f3cf99c9239f8b45a4bc9742881"
        },
        "date": 1780063006995,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 41.733647000009455,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 23.8175296764713,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.025572171878615,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 16.42448724999923,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 22.162230033900723,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 23.613462399987384,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "HumairAK@users.noreply.github.com",
            "name": "Humair Khan",
            "username": "HumairAK"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "954379c3ba9814b45c6a8bd9b4a25feff28a017d",
          "message": "Prefer routed ASGI paths in FastAPI auth checks. (#23685)\n\nSigned-off-by: Humair Khan <HumairAK@users.noreply.github.com>",
          "timestamp": "2026-05-29T16:28:41Z",
          "tree_id": "543c40a3e50ece54e9c929485def8ccfdb9a9f28",
          "url": "https://github.com/mlflow/mlflow/commit/954379c3ba9814b45c6a8bd9b4a25feff28a017d"
        },
        "date": 1780072348675,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_ingest",
            "value": 41.02263725000057,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_tag",
            "value": 22.117319764705414,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_state",
            "value": 21.47641411475319,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_name_like",
            "value": 23.332870491227766,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_search_by_timestamp",
            "value": 15.916709777777607,
            "unit": "ms"
          },
          {
            "name": "dev/benchmarks/tracing/test_trace_perf.py::test_e2e_agent",
            "value": 23.20748899999785,
            "unit": "ms"
          }
        ]
      }
    ]
  }
}