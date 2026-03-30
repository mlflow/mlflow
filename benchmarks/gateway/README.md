# MLflow AI Gateway Benchmark

Benchmark suite for measuring the latency overhead of the MLflow AI Gateway under concurrent load.

## Quick Start

```bash
# If running inside the MLflow repo with uv (recommended):
uv sync

# Otherwise:
pip install mlflow[gateway]
```

### Run the benchmark

```bash
cd benchmarks/gateway

# SQLite backend (default)
bash run_tracking_server_benchmark.sh

# PostgreSQL backend (auto-starts Docker container)
BACKEND_STORE_URI=postgres bash run_tracking_server_benchmark.sh

# PostgreSQL with existing instance
BACKEND_STORE_URI="postgresql://user:pass@host:5432/mlflow" \
    bash run_tracking_server_benchmark.sh
```

### Configuration

| Variable                  | Default           | Description                                                       |
| ------------------------- | ----------------- | ----------------------------------------------------------------- |
| `TRACKING_SERVER_WORKERS` | 4                 | Workers for `mlflow server`                                       |
| `REQUESTS`                | 2000              | Total requests per run                                            |
| `MAX_CONCURRENT`          | 50                | Max concurrent requests                                           |
| `RUNS`                    | 3                 | Number of benchmark runs                                          |
| `FAKE_RESPONSE_DELAY_MS`  | 50                | Simulated provider latency (ms)                                   |
| `USAGE_TRACKING`          | true              | Enable usage tracking/tracing (set `false` to disable)            |
| `BACKEND_STORE_URI`       | SQLite (temp dir) | `postgres` to auto-start Docker, or a full `postgresql://...` URI |

## How It Works

The benchmark measures end-to-end gateway request latency by running a controlled setup:

1. **Fake OpenAI server** (`fake_openai_server.py`) - Returns canned responses with configurable delay (default 50ms)
2. **MLflow server** with a gateway endpoint configured to route to the fake server
3. **Benchmark client** (`benchmark_gateway.py`) - Fires concurrent requests via `aiohttp` and measures per-request latency

```
aiohttp client  -->  MLflow Server  -->  Fake OpenAI Server (50ms delay)
(50 concurrent)      (4 workers)         (8 workers)
```

### What is measured

- **Latency**: Client-side `time.perf_counter()` per request (includes full HTTP round-trip)
- **Percentiles**: P50, P95, P99 across all runs
- **Throughput**: Requests per second (total requests / wall time)
- **Failures**: Count and breakdown by HTTP status code or exception type

### What is NOT measured (vs production)

| Factor          | In benchmark      | In production              |
| --------------- | ----------------- | -------------------------- |
| Network latency | ~0ms (loopback)   | 1-100ms per hop            |
| TLS/SSL         | None (plain HTTP) | ~5-20ms per new connection |
| Provider delay  | Fixed 50ms        | Variable (50ms-60s+)       |
| Authentication  | Disabled          | Token validation, RBAC     |

## File Inventory

| File                               | Purpose                                                                    |
| ---------------------------------- | -------------------------------------------------------------------------- |
| `common.sh`                        | Shared shell functions (server start/stop, health checks, cleanup)         |
| `fake_openai_server.py`            | FastAPI server returning canned OpenAI-compatible responses                |
| `benchmark_gateway.py`             | aiohttp-based benchmark client with failure breakdown logging              |
| `setup_tracking_server.py`         | Creates secret + model definition + endpoint via REST API                  |
| `run_tracking_server_benchmark.sh` | Orchestrates: fake server + MLflow server + endpoint setup + benchmark run |
| `.gitignore`                       | Ignores `results/` directory                                               |
