window.BENCHMARK_DATA = {
  "lastUpdate": 1778176346612,
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
      }
    ]
  }
}