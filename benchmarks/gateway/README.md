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
- [Preliminary Results](#preliminary-results)
- [Deploying to a Server](#deploying-to-a-server)
- [Usage Tracking Benchmark](#usage-tracking-benchmark-not-yet-covered)
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

| File                             | Purpose                                                                                                                                       |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Core benchmark**               |                                                                                                                                               |
| `fake_openai_server.py`          | FastAPI app returning canned OpenAI-compatible responses (chat, completions, embeddings) with configurable delay via `FAKE_RESPONSE_DELAY_MS` |
| `gateway_config.yaml`            | MLflow Gateway config with 3 endpoints pointing at fake server on `127.0.0.1:9000`                                                            |
| `overhead_middleware.py`         | Starlette `BaseHTTPMiddleware` that adds `X-MLflow-Gateway-Overhead-Ms` response header                                                       |
| `run_gateway_with_middleware.py` | App factory (`create_app()`) wrapping `create_app_from_path` with `OverheadMiddleware` — no production code changes                           |
| `locustfile.py`                  | Locust `HttpUser` with weighted tasks: chat (8), completions (1), embeddings (1); captures overhead header as custom metric                   |
| `collect_resources.py`           | Polls `psutil` every 1s for CPU%, RSS, VMS, thread count across gateway + child workers; writes CSV                                           |
| `run_benchmark.sh`               | Orchestration: starts fake server, gateway, resource monitor, runs Locust, analyzes results, cleans up                                        |
| `analyze_results.py`             | Reads Locust CSV + resource CSV, prints summary table with latency/RPS/overhead/resource stats                                                |
| **Head-to-head comparison**      |                                                                                                                                               |
| `litellm_config.yaml`            | LiteLLM proxy config pointing at the same fake server                                                                                         |
| `benchmark_compare.py`           | aiohttp-based benchmark (matches LiteLLM's `benchmark_mock.py`), runs both proxies sequentially with warmup, prints comparison table          |
| `run_comparison.sh`              | Starts fake server + both proxies + runs `benchmark_compare.py`                                                                               |
| **Other**                        |                                                                                                                                               |
| `requirements.txt`               | `locust>=2.20`, `psutil>=5.9`                                                                                                                 |
| `.gitignore`                     | Ignores `results/` directory                                                                                                                  |

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

---

## Preliminary Results

### Head-to-head: MLflow Gateway vs LiteLLM

**Setup**: MacBook Pro (Apple Silicon), 4 workers each, 50 concurrent users, 2000 requests/run, 3 runs, 50ms fake delay.

| Metric          | MLflow Gateway | LiteLLM  | MLflow Advantage |
| --------------- | -------------- | -------- | ---------------- |
| **P50 latency** | **54.4 ms**    | 98.4 ms  | 1.8x faster      |
| **P95 latency** | **68.4 ms**    | 136.2 ms | 2.0x faster      |
| **P99 latency** | **95.4 ms**    | 199.8 ms | 2.1x faster      |
| **Throughput**  | **866 rps**    | 505 rps  | 1.7x higher      |
| **Failures**    | 0              | 0        | —                |

### Pure proxy overhead (latency - 50ms fake delay)

| Metric       | MLflow Gateway | LiteLLM |
| ------------ | -------------- | ------- |
| P50 overhead | ~4 ms          | ~48 ms  |
| P95 overhead | ~18 ms         | ~86 ms  |
| P99 overhead | ~45 ms         | ~150 ms |

### Locust-based gateway benchmark (standalone)

**Setup**: 4 workers, 50 users, 15s, 50ms fake delay.

| Metric             | Value       |
| ------------------ | ----------- |
| Total requests     | 13,903      |
| Error rate         | 0%          |
| Median latency     | 53 ms       |
| P95 latency        | 55 ms       |
| P99 latency        | 60 ms       |
| RPS                | 1,864       |
| Memory (4 workers) | ~754 MB RSS |

### vs LiteLLM's published numbers

| Metric               | LiteLLM (published, 4 instances) | MLflow Gateway (local, 4 workers) |
| -------------------- | -------------------------------- | --------------------------------- |
| Median latency       | 100 ms                           | 54 ms                             |
| P99 latency          | 240 ms                           | 60 ms                             |
| RPS                  | 1,170                            | 1,864                             |
| Proxy overhead (p50) | 2 ms                             | ~4 ms                             |

> **Note**: LiteLLM's published numbers are from a multi-instance deployment (4 separate processes, 4 CPU / 8GB RAM each). Our local numbers use a single machine with 4 gunicorn workers. Server-grade hardware results will differ.

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
# MLflow Gateway
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES  # macOS only
BENCHMARK_GATEWAY_CONFIG=gateway_config.yaml \
PYTHONPATH="$(pwd)" \
    gunicorn "run_gateway_with_middleware:create_app()" \
    -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:5000

# OR LiteLLM
litellm --config litellm_config.yaml --port 4000 --num_workers 4
```

> Update `gateway_config.yaml` and `litellm_config.yaml` to point `api_base` at Machine C's IP instead of `127.0.0.1`.

**Machine A — Load generator:**

```bash
# aiohttp benchmark (apples-to-apples with LiteLLM methodology)
python benchmark_compare.py \
    --target both \
    --requests 10000 \
    --max-concurrent 200 \
    --runs 5 \
    --mlflow-url http://<machine-b>:5000/gateway/benchmark-chat/invocations \
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
      gunicorn "run_gateway_with_middleware:create_app()"
      -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:5000
    environment:
      BENCHMARK_GATEWAY_CONFIG: /app/gateway_config.yaml
      PYTHONPATH: /app
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

## Usage Tracking Benchmark (Not Yet Covered)

The current benchmarks test the **standalone YAML-config gateway** (`mlflow.gateway.app.create_app_from_path`), which does **not** include usage tracking. Usage tracking is only available when the gateway runs through the **MLflow tracking server** with database-backed endpoints (`mlflow/server/gateway_api.py`).

### What usage tracking adds to the request path

When an endpoint has `experiment_id` set and usage tracking enabled:

| Operation            | When                                   | Blocking? | Overhead                              |
| -------------------- | -------------------------------------- | --------- | ------------------------------------- |
| Token extraction     | During response parsing                | No        | Negligible (dict access)              |
| Trace span creation  | During request (via `@mlflow.trace()`) | No        | Low (in-memory trace manager)         |
| Cost calculation     | Post-request callback                  | Yes       | Low (LiteLLM pricing dict lookup)     |
| Budget recording     | Post-request callback                  | Yes       | Low (in-memory with `threading.Lock`) |
| Trace DB persistence | Background thread                      | No        | Async, but adds memory pressure       |
| Budget webhook       | Post-request (if threshold hit)        | Yes       | Rare, only on threshold crossing      |

Key: there are **no synchronous database writes** on the request path. Traces go to an in-memory manager and are persisted asynchronously. Budget tracking is entirely in-memory.

### How to benchmark with usage tracking (future work)

This requires running the full MLflow tracking server instead of the standalone gateway:

```bash
# 1. Start MLflow tracking server with gateway
mlflow server --host 0.0.0.0 --port 5000

# 2. Create an endpoint with usage tracking via REST API
curl -X POST http://localhost:5000/api/2.0/gateway/endpoints \
  -H "Content-Type: application/json" \
  -d '{
    "name": "benchmark-chat",
    "endpoint_type": "llm/v1/chat",
    "model": {"name": "gpt-4o-mini", "provider": "openai"},
    "openai_api_key": "fake-key",
    "openai_api_base": "http://127.0.0.1:9000/v1"
  }'

# 3. Enable usage tracking on the endpoint
curl -X PATCH http://localhost:5000/api/2.0/gateway/endpoints/benchmark-chat \
  -H "Content-Type: application/json" \
  -d '{"usage_tracking": true}'

# 4. Run benchmark against the tracking server
python benchmark_compare.py \
  --target mlflow \
  --mlflow-url http://localhost:5000/gateway/benchmark-chat/invocations \
  --requests 2000 --max-concurrent 50 --runs 3
```

The expected overhead from usage tracking is small (single-digit ms) since all hot-path operations are in-memory, but it should be measured to confirm. This is tracked as a P1 follow-up.

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
