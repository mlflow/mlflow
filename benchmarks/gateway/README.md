# MLflow AI Gateway Benchmark

Benchmark suite for measuring the latency overhead of the MLflow AI Gateway under concurrent load.

## Quick Start

```bash
# If running inside the MLflow repo with uv (recommended):
uv sync

# Otherwise:
pip install mlflow[gateway]
```

### Single-instance benchmark

```bash
cd benchmarks/gateway

# SQLite backend (default)
bash run_tracking_server_benchmark.sh

# PostgreSQL backend (auto-starts Docker container)
BACKEND_STORE_URI=postgres bash run_tracking_server_benchmark.sh

# Without usage tracking (isolate pure proxy overhead)
USAGE_TRACKING=false bash run_tracking_server_benchmark.sh

# PostgreSQL with existing instance
BACKEND_STORE_URI="postgresql://user:pass@host:5432/mlflow" \
    bash run_tracking_server_benchmark.sh
```

### Multi-instance benchmark (requires Docker)

Runs N MLflow instances behind an nginx load balancer for sustained load testing.

```bash
cd benchmarks/gateway

# Default: 4 instances x 4 workers, 200 concurrency, 10K requests
bash run_multi_instance_benchmark.sh

# Sustained load test (200K requests, ~3.5 minutes)
INSTANCES=4 WORKERS_PER_INSTANCE=4 REQUESTS=200000 MAX_CONCURRENT=50 RUNS=1 \
    bash run_multi_instance_benchmark.sh

# Quick test
INSTANCES=2 REQUESTS=2000 RUNS=1 bash run_multi_instance_benchmark.sh
```

### Which script should I use?

| Script | What it tests | When to use |
| --- | --- | --- |
| `run_tracking_server_benchmark.sh` | Single instance (SQLite or PostgreSQL) | Quick iteration, comparing across branches |
| `run_multi_instance_benchmark.sh` | Multiple instances behind nginx (PostgreSQL) | Sustained load testing, horizontal scaling |

### Configuration

#### `run_tracking_server_benchmark.sh`

| Variable | Default | Description |
| --- | --- | --- |
| `TRACKING_SERVER_WORKERS` | 4 | Workers for `mlflow server` |
| `REQUESTS` | 2000 | Total requests per run |
| `MAX_CONCURRENT` | 50 | Max concurrent requests |
| `RUNS` | 3 | Number of benchmark runs |
| `FAKE_RESPONSE_DELAY_MS` | 50 | Simulated provider latency (ms) |
| `USAGE_TRACKING` | true | Enable usage tracking/tracing (set `false` to disable) |
| `BACKEND_STORE_URI` | SQLite (temp dir) | `postgres` to auto-start Docker, or a full `postgresql://...` URI |

#### `run_multi_instance_benchmark.sh`

| Variable | Default | Description |
| --- | --- | --- |
| `INSTANCES` | 4 | Number of MLflow server instances |
| `WORKERS_PER_INSTANCE` | 4 | Gunicorn workers per instance |
| `REQUESTS` | 10000 | Total requests per run |
| `MAX_CONCURRENT` | 200 | Max concurrent requests |
| `RUNS` | 3 | Number of benchmark runs |
| `FAKE_RESPONSE_DELAY_MS` | 50 | Simulated provider latency (ms) |
| `USAGE_TRACKING` | true | Enable usage tracking/tracing (set `false` to disable) |

## How It Works

The benchmark measures end-to-end gateway request latency by running a controlled setup:

1. **Fake OpenAI server** (`fake_openai_server.py`) - Returns canned responses with configurable delay (default 50ms)
2. **MLflow server** with a gateway endpoint configured to route to the fake server
3. **Benchmark client** (`benchmark_gateway.py`) - Fires concurrent requests via `aiohttp` and measures per-request latency

Single-instance:

```
aiohttp client  -->  MLflow Server  -->  Fake OpenAI Server (50ms delay)
(50 concurrent)      (4 workers)         (8 workers)
```

Multi-instance:

```
aiohttp client  -->  nginx (round-robin)  -->  MLflow Instance 1 (4 workers)  -->  Fake OpenAI Server
(200 concurrent)                           -->  MLflow Instance 2 (4 workers)  -->  (16 workers)
                                           -->  MLflow Instance N (4 workers)
                     All instances share PostgreSQL (Docker)
```

### What is measured

- **Latency**: Client-side `time.perf_counter()` per request (includes full HTTP round-trip)
- **Percentiles**: P50, P95, P99 across all runs
- **Throughput**: Requests per second (total requests / wall time)
- **Failures**: Count and breakdown by HTTP status code or exception type

### What is NOT measured (vs production)

| Factor | In benchmark | In production |
| --- | --- | --- |
| Network latency | ~0ms (loopback) | 1-100ms per hop |
| TLS/SSL | None (plain HTTP) | ~5-20ms per new connection |
| Provider delay | Fixed 50ms | Variable (50ms-60s+) |
| Authentication | Disabled | Token validation, RBAC |

## File Inventory

| File | Purpose |
| --- | --- |
| `common.sh` | Shared shell functions (server start/stop, health checks, cleanup) |
| `fake_openai_server.py` | FastAPI server returning canned OpenAI-compatible responses |
| `benchmark_gateway.py` | aiohttp-based benchmark client with failure breakdown logging |
| `setup_tracking_server.py` | Creates secret + model definition + endpoint via REST API |
| `run_tracking_server_benchmark.sh` | Single-instance benchmark (SQLite or PostgreSQL) |
| `run_multi_instance_benchmark.sh` | Multi-instance benchmark behind nginx (PostgreSQL, Docker) |
| `.gitignore` | Ignores `results/` directory |
