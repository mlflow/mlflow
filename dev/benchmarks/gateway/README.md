# MLflow AI Gateway Benchmark

Measures the **proxy overhead** of the MLflow tracking-server-backed AI Gateway under
concurrent load. A fake OpenAI server simulates the upstream provider at a fixed latency,
so results reflect pure MLflow processing time rather than provider variance.

## Prerequisites

- Python 3.10+ with [`uv`](https://docs.astral.sh/uv/)
- Docker (required for `--backend postgres` and `multi` mode)

## Quick start

```bash
cd dev/benchmarks/gateway

# Single instance, SQLite (no Docker needed)
UV_NO_SOURCES=1 uv run run.py single

# Single instance, PostgreSQL (auto-starts Docker container)
UV_NO_SOURCES=1 uv run run.py single --backend postgres

# 4 instances behind nginx load balancer
UV_NO_SOURCES=1 uv run run.py multi

# Benchmark an existing endpoint directly (skips all setup)
UV_NO_SOURCES=1 uv run run.py single --url http://your-server/gateway/my-endpoint/mlflow/invocations
```

## What is measured

Latency is measured **client-side** using `time.perf_counter()` around each `aiohttp` request.
Each sample covers the full round-trip: client serialization → loopback → full server processing → response deserialization. Only HTTP 200 responses count toward latency stats; errors are tracked separately.

Connection pooling and HTTP keep-alive are enabled, so TCP handshake cost is amortized after the warmup phase.

### What is NOT measured

| Factor | In this benchmark | In production |
|---|---|---|
| Network latency | ~0 ms (loopback) | 1–100 ms per hop |
| TLS/SSL | None (plain HTTP) | ~5–20 ms per new connection |
| Provider inference | Fixed fake delay (`--fake-delay-ms`) | Variable (50 ms – 60 s+) |
| Authentication | Disabled (`--disable-security-middleware`) | Token validation, RBAC |

### Isolating pure proxy overhead

Set `--fake-delay-ms 0` to remove simulated provider latency entirely. The resulting numbers
represent MLflow's irreducible per-request processing cost (~10–15 ms on a modern laptop with
config caching enabled).

## What MLflow does per request

Each invocation through the tracking-server gateway runs these steps:

```
1. Budget check          (in-memory)
2. Config resolution     (3–5 DB queries, cached after first hit)
3. Secret decryption     (cached, 60 s TTL)
4. Provider instantiation
5. Tracing               (if usage_tracking=True)
6. aiohttp upstream call → fake OpenAI server
7. Budget callback
```

Steps 2 (config DB queries) and 5 (tracing) have historically been the dominant bottlenecks.
With config caching enabled (default) and the batch span processor active, both are reduced
to near-zero overhead, leaving steps 1/3/4/6/7 at roughly 10–15 ms combined.

## Architecture

### Single instance (`run.py single`)

```
benchmark.py ──aiohttp──▶  MLflow server (:5731)  ──▶  fake_server.py (:9137)
                               │
                           SQLite or PostgreSQL
```

### Multi-instance (`run.py multi`)

```
benchmark.py ──aiohttp──▶  nginx LB (:5731)  ──round-robin──▶  MLflow :5800
                                                                 MLflow :5801
                                                                 MLflow :580N
                                                                    │
                                                               fake_server.py (:9137)
                                                               PostgreSQL (Docker)
```

MLflow instances are started **sequentially** (instance 0 first) to let it initialize the
DB schema before the others join. All instances share one PostgreSQL database.

## Options

### `single`

| Flag | Default | Description |
|---|---|---|
| `--url URL` | — | Benchmark this URL directly, skip all setup |
| `--workers N` | 4 | MLflow server worker processes |
| `--backend sqlite\|postgres` | `sqlite` | Database backend |
| `--mlflow-port N` | 5731 | MLflow server port |
| `--fake-server-port N` | 9137 | Fake OpenAI server port |
| `--no-usage-tracking` | — | Disable usage tracking (tracing) on the endpoint |
| `--requests N` | 2000 | Requests per run |
| `--max-concurrent N` | 50 | Max concurrent requests |
| `--runs N` | 3 | Number of benchmark runs |
| `--fake-delay-ms N` | 50 | Simulated provider latency in ms (0 = pure overhead) |

### `multi`

| Flag | Default | Description |
|---|---|---|
| `--url URL` | — | Benchmark this URL directly, skip all setup |
| `--instances N` | 4 | Number of MLflow instances |
| `--workers N` | 4 | Workers per instance |
| `--lb-port N` | 5731 | nginx load balancer port |
| `--base-port N` | 5800 | First MLflow instance port (rest are +1, +2, …) |
| `--fake-server-port N` | 9137 | Fake OpenAI server port |
| `--no-usage-tracking` | — | Disable usage tracking on the endpoint |
| `--requests N` | 10000 | Requests per run |
| `--max-concurrent N` | 200 | Max concurrent requests |
| `--runs N` | 3 | Number of benchmark runs |
| `--fake-delay-ms N` | 50 | Simulated provider latency in ms |

All flags can also be set via environment variables (same name, uppercased):
`REQUESTS`, `MAX_CONCURRENT`, `RUNS`, `FAKE_RESPONSE_DELAY_MS`, `MLFLOW_PORT`,
`FAKE_SERVER_PORT`, `TRACKING_SERVER_WORKERS`, `INSTANCES`, `WORKERS_PER_INSTANCE`,
`LB_PORT`, `BASE_PORT`.

## Known limitations

- **Loopback only** — all processes run on the same machine. Results don't include real
  network latency between client, gateway, and provider.
- **No TLS** — MLflow is started with `--disable-security-middleware`. Production deployments
  add TLS termination overhead.
- **Fixed provider latency** — `fake_server.py` always responds in exactly `--fake-delay-ms`.
  Real providers have high variance (P99 often 5–10× P50).
- **No auth** — token validation and RBAC are disabled. Auth middleware adds latency
  proportional to token lookup strategy.
- **Single machine resource contention** — in `multi` mode, all MLflow instances, nginx,
  PostgreSQL, and the benchmark client share CPU/memory. On a server with dedicated resources
  per instance, throughput will be higher.

## Files

| File | Purpose |
|---|---|
| `run.py` | Main entry point — orchestrates servers, Docker, endpoint setup, and benchmark |
| `benchmark.py` | Async HTTP benchmark client (standalone or imported by `run.py`) |
| `fake_server.py` | Fake OpenAI-compatible server for controlled latency simulation |
