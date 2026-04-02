# MLflow AI Gateway Benchmark

Measures the **proxy overhead** of the MLflow tracking-server-backed AI Gateway under
concurrent load. A fake OpenAI server simulates the upstream provider at a fixed latency,
so results reflect pure MLflow processing time rather than provider variance.

## Prerequisites

- Python 3.10+ with [`uv`](https://docs.astral.sh/uv/) — all scripts must be run via `uv run`, which handles dependency installation automatically via inline script metadata
- Docker (required for `--database postgres` and `multi` mode)

## Quick start

```bash
cd dev/benchmarks/gateway

# 4 instances behind nginx (default, requires Docker)
uv run run.py

# Single instance, SQLite (no Docker needed)
uv run run.py --instances 1

# Single instance, PostgreSQL
uv run run.py --instances 1 --database postgres

# Scale up
uv run run.py --instances 8 --workers 8

# Benchmark an existing endpoint directly (skips all setup)
uv run run.py --url http://your-server/gateway/my-endpoint/mlflow/invocations

```

## What is measured

Latency is measured **client-side** using `time.perf_counter()` around each `aiohttp` request.
Each sample covers the full round-trip: client serialization → loopback → full server processing → response deserialization. Only HTTP 200 responses count toward latency stats; errors are tracked separately.

Connection pooling and HTTP keep-alive are enabled, so TCP handshake cost is amortized after the warmup phase.

### What is NOT measured

| Factor             | In this benchmark                          | In production               |
| ------------------ | ------------------------------------------ | --------------------------- |
| Network latency    | ~0 ms (loopback)                           | 1–100 ms per hop            |
| TLS/SSL            | None (plain HTTP)                          | ~5–20 ms per new connection |
| Provider inference | Fixed fake delay (`--fake-delay-ms`)       | Variable (50 ms – 60 s+)    |
| Authentication     | Disabled (`--disable-security-middleware`) | Token validation, RBAC      |

## What MLflow does per request

Each invocation through the tracking-server gateway runs these steps:

```
1. Config resolution     (DB-backed, cached after first hit)
2. Secret decryption     (cached, 60 s TTL)
3. Provider instantiation
4. Tracing               (if usage_tracking=True)
5. HTTP call to LLM API
```

Steps 1 (config resolution) and 4 (tracing) have historically been the dominant bottlenecks.
Config caching (enabled by default) eliminates most of step 1's cost. Tracing overhead
depends on the span processor in use.

## Architecture

### Single instance (`--instances 1`)

```
benchmark.py ──aiohttp──▶  MLflow server (:5731)  ──▶  fake_server.py (:9137)
                               │
                           SQLite or PostgreSQL
```

### Multi-instance (`--instances N`, default)

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

| Flag                          | Default  | Description                                                             |
| ----------------------------- | -------- | ----------------------------------------------------------------------- |
| `--url URL`                   | —        | Benchmark this URL directly, skip all setup                             |
| `--instances N`               | 4        | MLflow instances. Use 1 for single-instance (no nginx, optional SQLite) |
| `--workers N`                 | 4        | MLflow worker processes per instance                                    |
| `--database sqlite\|postgres` | `sqlite` | Database to use — only applies when `--instances 1`                     |
| `--no-usage-tracking`         | —        | Disable usage tracking (tracing) on the endpoint                        |
| `--port N`                    | 5731     | Port to benchmark (MLflow port for single, nginx LB port for multi)     |
| `--base-port N`               | 5800     | First MLflow instance port in multi mode (rest are +1, +2, …)           |
| `--fake-server-port N`        | 9137     | Fake OpenAI server port                                                 |
| `--requests N`                | 2000     | Requests per run                                                        |
| `--max-concurrent N`          | 50       | Max concurrent requests                                                 |
| `--runs N`                    | 3        | Number of benchmark runs                                                |
| `--fake-delay-ms N`           | 50       | Simulated provider latency in ms                                        |
| `--min-rps N`                 | —        | Fail (exit 1) if average throughput falls below N req/s                 |
| `--max-p50-ms N`              | —        | Fail (exit 1) if average P50 latency exceeds N ms (CI threshold)        |
| `--max-p99-ms N`              | —        | Fail (exit 1) if average P99 latency exceeds N ms (CI threshold)        |

All flags can also be set via environment variables (same name, uppercased):
`INSTANCES`, `WORKERS_PER_INSTANCE`, `REQUESTS`, `MAX_CONCURRENT`, `RUNS`,
`FAKE_RESPONSE_DELAY_MS`, `MLFLOW_PORT`, `BASE_PORT`, `FAKE_SERVER_PORT`.

To avoid conflicts with a local PostgreSQL instance, override the port via `GATEWAY_BENCH_POSTGRES_PORT` (default: 5432).

## Known limitations

- **Loopback only** — all processes run on the same machine. Results don't include real
  network latency between client, gateway, and provider.
- **No TLS** — MLflow is started with `--disable-security-middleware`. Production deployments
  add TLS termination overhead.
- **Fixed provider latency** — `fake_server.py` always responds in exactly `--fake-delay-ms`.
  Real providers have high variance (P99 often 5–10× P50).
- **No auth** — token validation and RBAC are disabled. Auth middleware adds latency
  proportional to token lookup strategy.
- **Single machine resource contention** — with multiple instances, all MLflow instances, nginx,
  PostgreSQL, and the benchmark client share CPU/memory. On a server with dedicated resources
  per instance, throughput will be higher.

## Files

| File             | Purpose                                                                        |
| ---------------- | ------------------------------------------------------------------------------ |
| `run.py`         | Main entry point — orchestrates servers, Docker, endpoint setup, and benchmark |
| `benchmark.py`   | Async HTTP benchmark client (standalone or imported by `run.py`)               |
| `fake_server.py` | Fake OpenAI-compatible server for controlled latency simulation                |
