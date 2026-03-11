# MLflow AI Gateway Benchmark Suite

Benchmark suite for measuring the latency overhead and scalability of the MLflow AI Gateway. Includes a head-to-head comparison tool against [LiteLLM](https://docs.litellm.ai/docs/benchmarks) using the same methodology for direct comparability.

**Jira**: ML-61935

## Table of Contents

- [Architecture](#architecture)
- [File Inventory](#file-inventory)
- [Key Implementation Details](#key-implementation-details)
- [Quick Start](#quick-start)
- [Benchmark Configurations](#benchmark-configurations)
- [Head-to-Head Comparison (MLflow vs LiteLLM)](#head-to-head-comparison-mlflow-vs-litellm)
- [AI Gateway Per-Request Code Path](#ai-gateway-per-request-code-path)
- [Tracking Server Benchmark](#tracking-server-benchmark)
- [Full-Stack Comparison (Both on PostgreSQL)](#full-stack-comparison-both-on-postgresql)
- [Preliminary Results](#preliminary-results)
- [Deploying to a Server](#deploying-to-a-server)
- [Usage Tracking Benchmark](#usage-tracking-benchmark)
- [Known Limitations](#known-limitations)

---

## Architecture

### Gateway-only benchmark (`run_benchmark.sh`)

```
┌─────────┐     ┌──────────────────────┐     ┌──────────────────┐
│  Locust  │────▶│  MLflow AI Gateway   │────▶│  Fake OpenAI     │
│  (load)  │◀────│  + OverheadMiddleware │◀────│  Server          │
└─────────┘     └──────────────────────┘     └──────────────────┘
                         │
                ┌────────▼────────┐
                │ Resource Monitor │
                │ (CPU, Memory)    │
                └─────────────────┘
```

### Head-to-head comparison (`run_comparison.sh`)

```
                ┌──────────────────────┐
           ┌───▶│  MLflow AI Gateway   │───┐
           │    │  (gunicorn + uvicorn) │   │
┌─────────┐│    └──────────────────────┘   │    ┌──────────────────┐
│ aiohttp  ││                               ├───▶│  Fake OpenAI     │
│ benchmark││    ┌──────────────────────┐   │    │  Server           │
└─────────┘└───▶│  LiteLLM Proxy       │───┘    └──────────────────┘
                │  (gunicorn + uvicorn) │
                └──────────────────────┘
```

Both proxies run on the same machine, same number of workers, hitting the same fake backend. The benchmark client uses `aiohttp` with connection pooling (matching LiteLLM's own `benchmark_mock.py` methodology).

---

## File Inventory

| File                               | Purpose                                                                                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Core benchmark**                 |                                                                                                                                               |
| `fake_openai_server.py`            | FastAPI app returning canned OpenAI-compatible responses (chat, completions, embeddings) with configurable delay via `FAKE_RESPONSE_DELAY_MS` |
| `gateway_config.yaml`              | MLflow Gateway config with 3 endpoints pointing at fake server on `127.0.0.1:9000`                                                            |
| `overhead_middleware.py`           | Starlette `BaseHTTPMiddleware` that adds `X-MLflow-Gateway-Overhead-Ms` response header                                                       |
| `run_gateway_with_middleware.py`   | App factory (`create_app()`) wrapping `create_app_from_path` with `OverheadMiddleware` — no production code changes                           |
| `locustfile.py`                    | Locust `HttpUser` with weighted tasks: chat (8), completions (1), embeddings (1); captures overhead header as custom metric                   |
| `collect_resources.py`             | Polls `psutil` every 1s for CPU%, RSS, VMS, thread count across gateway + child workers; writes CSV                                           |
| `run_benchmark.sh`                 | Orchestration: starts fake server, gateway, resource monitor, runs Locust, analyzes results, cleans up                                        |
| `analyze_results.py`               | Reads Locust CSV + resource CSV, prints summary table with latency/RPS/overhead/resource stats                                                |
| **Head-to-head comparison**        |                                                                                                                                               |
| `litellm_config.yaml`              | LiteLLM proxy config pointing at the same fake server                                                                                         |
| `benchmark_compare.py`             | aiohttp-based benchmark (matches LiteLLM's `benchmark_mock.py`), runs both proxies sequentially with warmup, prints comparison table          |
| `run_comparison.sh`                | Starts fake server + MLflow AI Gateway (SQLite) + LiteLLM (YAML) + runs `benchmark_compare.py`                                                |
| **Tracking server benchmark**      |                                                                                                                                               |
| `setup_tracking_server.py`         | Creates secret + model definition + endpoint via REST API in a running tracking server                                                        |
| `run_tracking_server_benchmark.sh` | Starts fake server + `mlflow server` with SQLite + sets up endpoint + runs benchmark                                                          |
| **Full-stack comparison**          |                                                                                                                                               |
| `litellm_config_db.yaml`           | LiteLLM proxy config with PostgreSQL `database_url` for spend tracking                                                                        |
| `run_full_stack_comparison.sh`     | Starts PostgreSQL (Docker) + MLflow AI Gateway (PostgreSQL) + LiteLLM (PostgreSQL) + runs comparison benchmark                                |
| **Other**                          |                                                                                                                                               |
| `requirements.txt`                 | `locust>=2.20`, `psutil>=5.9`                                                                                                                 |
| `.gitignore`                       | Ignores `results/` directory                                                                                                                  |

### Modified existing files

| File                                                   | Change                                                                              |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `docs/docs/genai/governance/ai-gateway/index.mdx`      | Added Benchmarks tile card with Timer icon                                          |
| `docs/docs/genai/governance/ai-gateway/benchmarks.mdx` | New public-facing benchmark methodology & results page                              |
| `pyproject.toml`                                       | Added `"benchmarks/*" = ["T20"]` in ruff per-file-ignores (CLI scripts use `print`) |

---

## Key Implementation Details

### Overhead measurement

The `OverheadMiddleware` wraps the entire request lifecycle (including the upstream HTTP call to the fake server). To isolate the **pure proxy overhead**, subtract the fake delay from the measured latency:

```
proxy_overhead = measured_latency - FAKE_RESPONSE_DELAY_MS
```

With `FAKE_RESPONSE_DELAY_MS=50`, a measured p50 of 54ms means ~4ms of proxy overhead.

### No production code changes

The gateway is launched via a factory function that imports `create_app_from_path` and adds middleware externally:

```python
def create_app():
    gateway_app = create_app_from_path(config_path)
    gateway_app.add_middleware(OverheadMiddleware)
    return gateway_app
```

### `uv run` auto-detection

The shell scripts auto-detect whether `uv` is available and prefix all commands with `uv run --extra gateway` when running inside the MLflow repository. This ensures gateway dependencies (`slowapi`, `gunicorn`, etc.) are available without manual installation.

### macOS fork safety

Gunicorn workers on macOS hit an Objective-C runtime crash on `fork()`. The scripts set `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` automatically.

---

## Quick Start

### Prerequisites

```bash
# If running inside the MLflow repo with uv (recommended):
uv sync
uv pip install -r benchmarks/gateway/requirements.txt

# Otherwise:
pip install mlflow[gateway] -r benchmarks/gateway/requirements.txt
```

### Smoke test (5 users, 10 seconds)

```bash
cd benchmarks/gateway
LOCUST_USERS=5 LOCUST_RUN_TIME=10s bash run_benchmark.sh
```

### Full gateway benchmark

```bash
cd benchmarks/gateway
GATEWAY_WORKERS=4 LOCUST_USERS=50 LOCUST_RUN_TIME=15s bash run_benchmark.sh
```

### Head-to-head vs LiteLLM

```bash
cd benchmarks/gateway
pip install 'litellm[proxy]'  # or: uv pip install 'litellm[proxy]'
GATEWAY_WORKERS=4 REQUESTS=2000 MAX_CONCURRENT=50 RUNS=3 bash run_comparison.sh
```

---

## Benchmark Configurations

### `run_benchmark.sh` environment variables

| Variable                 | Default               | Description                                |
| ------------------------ | --------------------- | ------------------------------------------ |
| `GATEWAY_WORKERS`        | 2                     | Number of gunicorn workers for the gateway |
| `LOCUST_USERS`           | 1000                  | Concurrent simulated users                 |
| `LOCUST_SPAWN_RATE`      | 500                   | Users spawned per second                   |
| `LOCUST_RUN_TIME`        | 60s                   | Test duration                              |
| `FAKE_RESPONSE_DELAY_MS` | 50                    | Simulated provider latency (ms)            |
| `FAKE_SERVER_PORT`       | 9000                  | Port for fake OpenAI server                |
| `GATEWAY_PORT`           | 5000                  | Port for the gateway                       |
| `FAKE_SERVER_WORKERS`    | `GATEWAY_WORKERS * 2` | Workers for fake server (auto-scaled)      |

### `run_comparison.sh` environment variables

| Variable                 | Default | Description                     |
| ------------------------ | ------- | ------------------------------- |
| `GATEWAY_WORKERS`        | 4       | Workers for both proxies        |
| `REQUESTS`               | 2000    | Total requests per run          |
| `MAX_CONCURRENT`         | 50      | Max concurrent requests         |
| `RUNS`                   | 3       | Number of benchmark runs        |
| `FAKE_RESPONSE_DELAY_MS` | 50      | Simulated provider latency (ms) |

### Suggested test matrix

| Config    | Workers | Fake Delay | Purpose                     |
| --------- | ------- | ---------- | --------------------------- |
| Overhead  | 2       | 0 ms       | Isolate pure proxy overhead |
| Default   | 2       | 50 ms      | Default gateway behavior    |
| Scaled    | 4       | 50 ms      | Match LiteLLM's 4-CPU setup |
| Realistic | 4       | 100 ms     | Simulate real API latency   |

```bash
# Run the full matrix:
for delay in 0 50 100; do
  for workers in 2 4; do
    GATEWAY_WORKERS=$workers FAKE_RESPONSE_DELAY_MS=$delay \
      LOCUST_USERS=50 LOCUST_RUN_TIME=15s bash run_benchmark.sh
  done
done
```

### Output files

Results are saved to `results/<timestamp>_w<workers>_d<delay>ms_u<users>/`:

| File                       | Content                        |
| -------------------------- | ------------------------------ |
| `locust_stats.csv`         | Aggregated request statistics  |
| `locust_stats_history.csv` | Time-series stats (per-second) |
| `resources.csv`            | CPU/memory usage over time     |
| `config.txt`               | Benchmark configuration        |
| `locust_output.log`        | Full Locust console output     |
| `gateway.log`              | Gateway server logs            |
| `fake_server.log`          | Fake OpenAI server logs        |

---

## Head-to-Head Comparison (MLflow vs LiteLLM)

The `run_comparison.sh` script runs both proxies under identical conditions:

- Same fake OpenAI backend (8 gunicorn workers on port 9000)
- Same number of proxy workers
- Same aiohttp-based benchmark client with connection pooling
- Warmup phase (50 requests) before timed measurement
- Multiple runs to reduce variance

This mirrors LiteLLM's own `benchmark_mock.py` methodology for a fair comparison.

### What each side runs

- **MLflow AI Gateway**: `mlflow server` with SQLite backend, endpoint created via REST API (`/api/3.0/mlflow/gateway/endpoints/create`). This is the production AI Gateway code path — every request resolves endpoint config from the database.
- **LiteLLM Proxy**: `litellm --config litellm_config.yaml` — model list loaded from YAML, no database, `callbacks: []`.

Usage tracking is **disabled by default** (`USAGE_TRACKING=false`) to reduce DB overhead, but can be enabled with `USAGE_TRACKING=true`.

### Fairness note

This comparison is **not symmetric** in terms of architecture: MLflow resolves endpoint config from the database on every request (3-5 SQL queries), while LiteLLM loads its config from YAML at startup and keeps it in memory. This reflects the real production code paths for each proxy.

For a "full production stack" comparison where both sides use PostgreSQL, see [Full-Stack Comparison](#full-stack-comparison-both-on-postgresql).

---

## AI Gateway Per-Request Code Path

**Code path**: `mlflow/server/gateway_api.py` → `invocations()`

On **each request**:

1. **Get store** — cached, negligible
2. **Budget check** (`check_budget_limit`) — in-memory with periodic DB refresh (~5-10min intervals)
3. **Endpoint config resolution** (`get_endpoint_config` in `config_resolver.py`) — **3-5 DB queries, NOT cached**:
   - `SELECT FROM gateway_endpoint WHERE name = ?`
   - `SELECT FROM gateway_endpoint_model_mapping WHERE endpoint_id = ?`
   - `SELECT FROM gateway_model_definition WHERE model_definition_id = ?` (per model)
   - `SELECT FROM gateway_secret WHERE secret_id = ?` (per model)
4. **Secret decryption** — PBKDF2 key derivation (~1-2ms) + AES-256-GCM decrypt; cached for 60s via `SecretCache`
5. **Provider instantiation** — fresh per request (~1-5ms)
6. **Tracing** (`maybe_traced_gateway_call`) — if `usage_tracking=true`, wraps the call with `mlflow.trace()`, creates spans, records metadata; if `false`, returns raw function (no overhead)
7. **Budget callback** (`on_complete`) — runs after LLM response, records cost in-memory
8. **Upstream HTTP call** — `aiohttp`

### What's cached vs fresh

| Component                | Cached? | Strategy                             | TTL                        |
| ------------------------ | ------- | ------------------------------------ | -------------------------- |
| Store registry           | Yes     | LRU (maxsize=100)                    | Session lifetime           |
| Budget policies          | Yes     | Time-based refresh                   | ~5-10 min                  |
| Endpoint config          | **No**  | Fresh DB queries every request       | N/A                        |
| Secrets                  | Yes     | `SecretCache` (encrypted, ephemeral) | 60s (configurable 10-300s) |
| KEK (key encryption key) | **No**  | PBKDF2 derivation per decryption     | N/A                        |
| Provider instance        | **No**  | Created fresh every request          | N/A                        |

The **uncached endpoint config queries** are the dominant bottleneck. With SQLite (single-writer, no connection pooling), these serialize under concurrency. PostgreSQL alleviates the single-writer lock but adds network round-trip latency to each query.

---

## Tracking Server Benchmark

The `run_tracking_server_benchmark.sh` script benchmarks the tracking server gateway code path by running a real `mlflow server` with a SQLite backend.

### Quick start

```bash
cd benchmarks/gateway

# With usage tracking (default — traces enabled)
bash run_tracking_server_benchmark.sh

# Without usage tracking (tracing disabled)
USAGE_TRACKING=false bash run_tracking_server_benchmark.sh
```

### Architecture

```
                                    ┌───────────────────────────────┐
┌─────────┐     ┌──────────────────┤  MLflow Tracking Server       │     ┌──────────────────┐
│ aiohttp  │────▶│  FastAPI Router  │  gateway_api.py               │────▶│  Fake OpenAI     │
│ benchmark│◀────│  /gateway/...    │  ┌─────────────────────────┐  │◀────│  Server          │
└─────────┘     └──────────────────┤  │ SQLite: secrets, models, │  │     └──────────────────┘
                                    │  │ endpoints, configs       │  │
                                    │  └─────────────────────────┘  │
                                    └───────────────────────────────┘
```

### Configuration

| Variable                  | Default | Description                                            |
| ------------------------- | ------- | ------------------------------------------------------ |
| `TRACKING_SERVER_WORKERS` | 4       | Workers for `mlflow server`                            |
| `REQUESTS`                | 2000    | Total requests per run                                 |
| `MAX_CONCURRENT`          | 50      | Max concurrent requests                                |
| `RUNS`                    | 3       | Number of benchmark runs                               |
| `FAKE_RESPONSE_DELAY_MS`  | 50      | Simulated provider latency (ms)                        |
| `USAGE_TRACKING`          | true    | Enable usage tracking/tracing (set `false` to disable) |

### What it does

1. Starts a fake OpenAI server (gunicorn, 8 workers, port 9000)
2. Starts `mlflow server` with a fresh SQLite DB and `--disable-security-middleware`
3. Creates a gateway endpoint via REST API (`setup_tracking_server.py`): secret → model definition → endpoint
4. Runs `benchmark_compare.py --target mlflow` against `http://127.0.0.1:5000/gateway/benchmark-chat/mlflow/invocations`
5. Cleans up all processes and temp directory

### Comparing with and without usage tracking

Usage tracking is **enabled by default** when creating endpoints via the tracking server API. The `USAGE_TRACKING` env var controls this:

- `USAGE_TRACKING=true` (default): Endpoint is created with `usage_tracking=true`, which auto-creates an experiment and wraps every request with `mlflow.trace()`. This adds span creation, metadata recording, and async trace persistence.
- `USAGE_TRACKING=false`: Endpoint is created with `usage_tracking=false`, so `maybe_traced_gateway_call()` returns the raw function with no tracing overhead. The per-request DB queries for config resolution still happen.

---

## Full-Stack Comparison (Both on PostgreSQL)

The `run_full_stack_comparison.sh` script benchmarks both proxies with PostgreSQL and usage/spend tracking enabled — the production code path for each.

- **MLflow**: Tracking server with PostgreSQL, gateway endpoint created via REST API, usage tracking enabled (traces + budget callbacks)
- **LiteLLM**: Proxy with PostgreSQL (same Docker container, separate database), spend tracking auto-enabled by `database_url`

### Prerequisites

- **Docker**: Required for the PostgreSQL container. Install from https://docs.docker.com/get-docker/
- **litellm[proxy]**: `pip install 'litellm[proxy]'` or `uv pip install 'litellm[proxy]'`

### Quick start

```bash
cd benchmarks/gateway

# Default: 4 workers, 2000 requests, 50 concurrent, 3 runs
bash run_full_stack_comparison.sh

# Custom configuration
WORKERS=8 REQUESTS=5000 MAX_CONCURRENT=100 RUNS=5 bash run_full_stack_comparison.sh

# Without MLflow usage tracking (LiteLLM still has spend tracking via DB)
USAGE_TRACKING=false bash run_full_stack_comparison.sh
```

### Architecture

```
                ┌───────────────────────────────┐
           ┌───▶│  MLflow Tracking Server       │───┐
           │    │  (PostgreSQL, usage tracking)  │   │
┌─────────┐│    └──────────┬────────────────────┘   │    ┌──────────────────┐
│ aiohttp  ││              │                        ├───▶│  Fake OpenAI     │
│ benchmark││    ┌─────────┼────────────────────┐   │    │  Server          │
└─────────┘└───▶│  LiteLLM │Proxy               │───┘    └──────────────────┘
                │  (PostgreSQL, spend tracking)  │
                └──────────┼────────────────────┘
                           │
                   ┌───────▼───────┐
                   │  PostgreSQL   │
                   │  (Docker)     │
                   │  DB: mlflow   │
                   │  DB: litellm  │
                   └───────────────┘
```

### Configuration

| Variable                 | Default | Description                                           |
| ------------------------ | ------- | ----------------------------------------------------- |
| `WORKERS`                | 4       | Workers for both proxies                              |
| `REQUESTS`               | 2000    | Total requests per run                                |
| `MAX_CONCURRENT`         | 50      | Max concurrent requests                               |
| `RUNS`                   | 3       | Number of benchmark runs                              |
| `FAKE_RESPONSE_DELAY_MS` | 50      | Simulated provider latency (ms)                       |
| `USAGE_TRACKING`         | true    | Enable MLflow usage tracking (set `false` to disable) |

### What each side does per request

| Operation      | MLflow (tracking server)                   | LiteLLM (PostgreSQL)                  |
| -------------- | ------------------------------------------ | ------------------------------------- |
| Config lookup  | 3-5 SQL queries (PostgreSQL, uncached)     | In-memory from startup YAML + DB sync |
| Secret access  | Decrypt per request (60s cache)            | From YAML config (no encryption)      |
| Usage tracking | `mlflow.trace()` spans + async persistence | Spend tracking callbacks + DB writes  |
| Budget check   | In-memory with periodic DB refresh         | In-memory with DB persistence         |
| Upstream HTTP  | `aiohttp`                                  | `httpx` / `aiohttp`                   |

### Fairness notes

Both proxies use the same PostgreSQL instance (separate databases). Key architectural differences remain:

- **MLflow queries the DB on every request** for endpoint config (3-5 SQL queries, uncached). LiteLLM loads config from YAML at startup and uses DB primarily for spend tracking.
- Both sides have **usage/spend tracking enabled** by default, which is the intended production configuration for each.

---

## Preliminary Results

All results: MacBook Pro (Apple Silicon), 4 workers, 50 concurrent users, 2000 requests/run, 3 runs, 50ms fake delay.

### Head-to-head: MLflow AI Gateway vs LiteLLM (barebone)

MLflow AI Gateway with SQLite (usage tracking OFF) vs LiteLLM with YAML config (no DB, `callbacks: []`).

| Metric          | MLflow AI Gateway | LiteLLM |
| --------------- | ----------------- | ------- |
| **P50 latency** | 2,770 ms          | 77 ms   |
| **P95 latency** | 5,089 ms          | 116 ms  |
| **P99 latency** | 5,249 ms          | 265 ms  |
| **Throughput**  | 21 rps            | 602 rps |
| **Failures**    | 0                 | 0       |

The MLflow AI Gateway resolves endpoint config from the database on every request (3-5 SQL queries). With SQLite's single-writer lock, this serializes under concurrency, producing ~21 rps with multi-second latencies. LiteLLM keeps config in memory from startup.

### Full-stack: MLflow (PostgreSQL) vs LiteLLM (PostgreSQL)

Both proxies with PostgreSQL backend and usage/spend tracking enabled.

| Metric          | MLflow AI Gateway | LiteLLM |
| --------------- | ----------------- | ------- |
| **P50 latency** | 4,770 ms          | 113 ms  |
| **P95 latency** | 7,681 ms          | 152 ms  |
| **P99 latency** | 8,066 ms          | 315 ms  |
| **Throughput**  | 14 rps            | 430 rps |
| **Failures**    | 0                 | 0       |

PostgreSQL is actually _slower_ for MLflow than SQLite (14 rps vs 21 rps) because the per-request SQL queries now incur network round-trip latency. The bottleneck is the uncached endpoint config resolution pattern, not the DB engine. LiteLLM loads config from YAML at startup and only uses DB for spend tracking.

### MLflow AI Gateway: SQLite vs PostgreSQL

Same AI Gateway code path, different DB backends (usage tracking ON).

| Metric          | SQLite   | PostgreSQL |
| --------------- | -------- | ---------- |
| **P50 latency** | 1,504 ms | 4,770 ms   |
| **P95 latency** | 6,537 ms | 7,681 ms   |
| **P99 latency** | 6,850 ms | 8,066 ms   |
| **Throughput**  | 18 rps   | 14 rps     |

> **Key takeaway**: The dominant bottleneck in the MLflow AI Gateway is the **uncached per-request endpoint config resolution** (3-5 SQL queries per request). Switching from SQLite to PostgreSQL does not improve performance — it adds network latency to each query. Caching the endpoint config would be the most impactful optimization.

---

## Deploying to a Server

For production-grade benchmarks (recommended for publishable results), deploy to a server with dedicated resources.

### Option 1: Single server (simplest)

Run everything on one beefy machine (8+ CPU, 16+ GB RAM):

```bash
# SSH into the server
ssh benchmark-server

# Clone and setup
git clone https://github.com/mlflow/mlflow.git
cd mlflow
pip install -e '.[gateway]'
pip install -r benchmarks/gateway/requirements.txt
pip install 'litellm[proxy]'

# Increase file descriptor limit (critical for high concurrency)
ulimit -n 65536

# Run head-to-head comparison
cd benchmarks/gateway
GATEWAY_WORKERS=4 REQUESTS=10000 MAX_CONCURRENT=200 RUNS=5 bash run_comparison.sh

# Run the full Locust-based benchmark
GATEWAY_WORKERS=4 LOCUST_USERS=500 LOCUST_RUN_TIME=120s bash run_benchmark.sh
```

### Option 2: Separate machines (recommended for high fidelity)

Use 3 machines to eliminate resource contention:

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ Machine A    │────▶│ Machine B       │────▶│ Machine C        │
│ Load Gen     │     │ Proxy Under Test│     │ Fake OpenAI      │
│ (Locust /    │◀────│ (MLflow or      │◀────│ Server           │
│  aiohttp)    │     │  LiteLLM)       │     │                  │
└─────────────┘     └─────────────────┘     └──────────────────┘
```

**Machine C — Fake OpenAI server:**

```bash
# Start with plenty of workers to never be the bottleneck
FAKE_RESPONSE_DELAY_MS=50 gunicorn fake_openai_server:app \
    -k uvicorn.workers.UvicornWorker -w 16 -b 0.0.0.0:9000
```

**Machine B — Proxy under test:**

```bash
# MLflow AI Gateway
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES  # macOS only
mlflow server \
    --backend-store-uri "sqlite:///mlflow.db" \
    --host 0.0.0.0 --port 5000 --workers 4 \
    --disable-security-middleware
# Then create endpoint via setup_tracking_server.py

# OR LiteLLM
litellm --config litellm_config.yaml --port 4000 --num_workers 4
```

> Update `litellm_config.yaml` to point `api_base` at Machine C's IP instead of `127.0.0.1`.

**Machine A — Load generator:**

```bash
# aiohttp benchmark (apples-to-apples with LiteLLM methodology)
python benchmark_compare.py \
    --target both \
    --requests 10000 \
    --max-concurrent 200 \
    --runs 5 \
    --mlflow-url http://<machine-b>:5000/gateway/benchmark-chat/mlflow/invocations \
    --litellm-url http://<machine-b>:4000/chat/completions

# OR Locust benchmark (with overhead header tracking)
locust -f locustfile.py \
    --host http://<machine-b>:5000 \
    --headless --users 500 --spawn-rate 100 --run-time 120s \
    --csv results/server_run
```

### Option 3: Docker Compose

Create a `docker-compose.yaml` to standardize the environment:

```yaml
services:
  fake-openai:
    build: .
    command: gunicorn fake_openai_server:app -k uvicorn.workers.UvicornWorker -w 16 -b 0.0.0.0:9000
    environment:
      FAKE_RESPONSE_DELAY_MS: 50
    ports: ["9000:9000"]
    deploy:
      resources:
        limits: { cpus: "4", memory: 4G }

  mlflow-gateway:
    build: .
    command: >
      mlflow server
      --backend-store-uri sqlite:///mlflow.db
      --host 0.0.0.0 --port 5000 --workers 4
      --disable-security-middleware
    ports: ["5000:5000"]
    depends_on: [fake-openai]
    deploy:
      resources:
        limits: { cpus: "4", memory: 8G }

  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    command: --config /app/litellm_config.yaml --port 4000 --num_workers 4
    ports: ["4000:4000"]
    depends_on: [fake-openai]
    deploy:
      resources:
        limits: { cpus: "4", memory: 8G }
```

### Tips for server benchmarks

1. **Increase file descriptors**: `ulimit -n 65536` before starting anything. The gateway creates a new TCP connection per upstream request, which exhausts ephemeral ports under sustained load (see [Known Limitations](#known-limitations)).

2. **Pin CPU cores**: Use `taskset` on Linux to pin each service to specific cores and avoid cache contention.

3. **Disable power management**: `sudo cpupower frequency-set -g performance` on Linux.

4. **Multiple runs**: Use `--runs 5` or more for statistical significance. The comparison script reports run-to-run variance.

5. **Warm up the system**: The first run typically has higher latency. Use warmup runs or discard the first result.

6. **Monitor with `collect_resources.py`**: Run it against each proxy PID to track CPU/memory during the test.

---

## Usage Tracking Benchmark

The `run_tracking_server_benchmark.sh` script supports benchmarking both with and without usage tracking via the `USAGE_TRACKING` env var (default: `true`).

```bash
# With usage tracking (default)
bash run_tracking_server_benchmark.sh

# Without usage tracking
USAGE_TRACKING=false bash run_tracking_server_benchmark.sh
```

### What usage tracking adds to the request path

When `usage_tracking=true` (the server default), an experiment is auto-created and every request is wrapped with `mlflow.trace()`:

| Operation            | When                                  | Blocking? | Overhead                              |
| -------------------- | ------------------------------------- | --------- | ------------------------------------- |
| Trace span creation  | During request (via `mlflow.trace()`) | Yes       | Low (in-memory trace manager)         |
| Metadata recording   | During request                        | Yes       | Low (dict updates)                    |
| Token extraction     | During response parsing               | No        | Negligible (dict access)              |
| Cost calculation     | Post-request callback (`on_complete`) | Yes       | Low (LiteLLM pricing dict lookup)     |
| Budget recording     | Post-request callback                 | Yes       | Low (in-memory with `threading.Lock`) |
| Trace DB persistence | Background thread                     | No        | Async, but adds memory pressure       |
| Budget webhook       | Post-request (if threshold hit)       | Yes       | Rare, only on threshold crossing      |

When `usage_tracking=false`, `maybe_traced_gateway_call()` returns the raw provider function (line 189-190 in `tracing_utils.py`), bypassing all tracing overhead. The per-request DB queries for config resolution still happen.

Key: there are **no synchronous database writes** on the hot path. Traces go to an in-memory manager and are persisted asynchronously. Budget tracking is entirely in-memory.

---

## Known Limitations

### Ephemeral port exhaustion

The MLflow Gateway's `send_request()` in `mlflow/gateway/providers/utils.py` creates a new `aiohttp.ClientSession` per request without connection pooling. Under sustained high concurrency (>50 users for >20s on a single machine), this exhausts ephemeral TCP ports (`Can't assign requested address`).

**Impact**: Limits local laptop benchmarks to shorter durations or lower concurrency. On a server with separate machines (load gen / proxy / backend), this is less of an issue since TCP connections are distributed across network interfaces.

**Potential fix**: Add a persistent `aiohttp.ClientSession` with a `TCPConnector` pool to the provider base class. This would also improve real-world performance.

### CPU measurement on macOS

`psutil.cpu_percent()` returns 0% for gunicorn worker trees on macOS due to how the OS reports CPU for forked processes. On Linux, CPU tracking works correctly.

### LiteLLM `network_mock` mode

LiteLLM's published benchmarks use `network_mock: true`, which skips the real HTTP call entirely and returns a canned response inside the proxy process. Our comparison uses the real HTTP path (proxy -> fake server) for both sides, which is more representative of production behavior but produces different numbers than LiteLLM's published figures.
