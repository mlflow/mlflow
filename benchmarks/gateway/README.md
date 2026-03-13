# MLflow AI Gateway Benchmark Suite

Benchmark suite for measuring the latency overhead and scalability of the MLflow AI Gateway. Includes a head-to-head comparison tool against [LiteLLM](https://docs.litellm.ai/docs/benchmarks) using the same methodology for direct comparability.

**Jira**: ML-61935

## Table of Contents

- [Architecture](#architecture)
- [Request Lifecycle: Direct vs MLflow vs LiteLLM](#request-lifecycle-direct-vs-mlflow-vs-litellm)
- [MLflow and LiteLLM: Library vs Proxy](#mlflow-and-litellm-library-vs-proxy)
- [Measurement Methodology](#measurement-methodology)
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

## Request Lifecycle: Direct vs MLflow vs LiteLLM

To understand what the benchmark measures, it helps to compare the request lifecycle across three scenarios: calling the provider directly, going through the MLflow AI Gateway, and going through LiteLLM Proxy.

### 1. Direct to provider (baseline — not benchmarked)

```
Client ──HTTP──▶ OpenAI API ──HTTP──▶ Client
         (1 round-trip)
```

Steps: DNS + TLS handshake + send request + provider inference + receive response. This is the minimum latency for any LLM call. The gateway cannot reduce this.

### 2. Through MLflow AI Gateway

```
Client ──HTTP──▶ MLflow Server ──HTTP──▶ Provider ──HTTP──▶ Client
                 │                       ▲
                 │  (server-side steps)  │
                 ▼                       │
          ┌──────────────┐               │
          │ 1. Budget check (in-memory)  │
          │ 2. Config resolution (DB)  ◀─┤ 3-5 SQL queries
          │ 3. Secret decrypt (cached)   │
          │ 4. Provider instantiation    │
          │ 5. Tracing (if enabled)      │
          │ 6. litellm.acompletion()  ───┘  ◀── calls upstream provider
          │ 7. Budget callback           │
          └──────────────┘
```

**7 server-side steps** per request. Steps 2 (config resolution) and 5 (tracing) are the dominant bottlenecks — see [Bottleneck isolation](#bottleneck-isolation-config-cache--usage-tracking). When both are eliminated, steps 1/3/4/6/7 add only ~10ms of overhead (see [zero-delay results](#preliminary-results)).

### 3. Through LiteLLM Proxy

```
Client ──HTTP──▶ LiteLLM Proxy ──HTTP──▶ Provider ──HTTP──▶ Client
                 │                        ▲
                 │  (server-side steps)   │
                 ▼                        │
          ┌──────────────┐                │
          │ 1. Route lookup (in-memory)   │
          │ 2. Auth check                 │
          │ 3. httpx / aiohttp call  ─────┘  ◀── calls upstream provider
          │ 4. Spend tracking callback    │
          └──────────────┘
```

**4 server-side steps** per request. Config is loaded from YAML at startup and kept in memory — no per-request DB queries. When spend tracking is DB-backed (PostgreSQL mode), step 4 involves async DB writes.

### Side-by-side step comparison

| Step                   | MLflow AI Gateway                             | LiteLLM Proxy                         |
| ---------------------- | --------------------------------------------- | ------------------------------------- |
| Config lookup          | **3-5 SQL queries per request** (uncached)    | In-memory (loaded from YAML at start) |
| Secret/auth access     | AES-256-GCM decrypt (60s cache)               | Read from YAML config (no encryption) |
| Provider instantiation | Created fresh per request                     | Reused from startup                   |
| Upstream HTTP call     | `litellm.acompletion()` via aiohttp           | `httpx` / `aiohttp`                   |
| Usage/spend tracking   | `mlflow.trace()` spans + async DB persistence | Spend callbacks + optional async DB   |
| Budget/rate limiting   | In-memory with periodic DB refresh            | In-memory with DB persistence         |

The key architectural difference: MLflow treats the database as the source of truth for endpoint config and resolves it on every request. LiteLLM treats YAML as the source of truth and loads it once at startup. This is why MLflow is slower by default (3-5 extra SQL queries per request), but equally fast or faster when the config is cached.

---

## MLflow and LiteLLM: Library vs Proxy

The name "LiteLLM" appears in two different roles in this benchmark, which can be confusing:

```
┌──────────────────────────────────────────────────────┐
│  MLflow AI Gateway (the server being benchmarked)    │
│                                                      │
│  gateway_api.py                                      │
│       │                                              │
│       ▼                                              │
│  LiteLLMProvider                                     │
│       │                                              │
│       ▼                                              │
│  litellm.acompletion()  ◀── LiteLLM the LIBRARY     │
│       │                     (Python package)         │
│       ▼                                              │
│  HTTP call to upstream provider                      │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  LiteLLM Proxy (the server being compared against)   │
│                                                      │
│  litellm --config litellm_config.yaml                │
│       │                     ◀── LiteLLM the SERVER   │
│       ▼                         (CLI proxy process)  │
│  Route request to configured model                   │
│       │                                              │
│       ▼                                              │
│  HTTP call to upstream provider                      │
└──────────────────────────────────────────────────────┘
```

### LiteLLM the library (used inside MLflow)

MLflow uses `litellm` as a **Python client library** for provider abstraction. The `LiteLLMProvider` class in `mlflow/gateway/providers/litellm.py` calls functions like `litellm.acompletion()`, `litellm.aembedding()`, and `litellm.aresponses()` to normalize request/response formats across providers (OpenAI, Anthropic, Gemini, etc.). This is a lightweight in-process function call — no server, no HTTP, no proxy overhead.

### LiteLLM the proxy (benchmarked as competitor)

The benchmark scripts start `litellm --config litellm_config.yaml` as a **standalone proxy server** process (gunicorn + uvicorn workers). This is a separate product — an HTTP proxy that routes requests to LLM providers, with its own config, auth, spend tracking, and rate limiting.

### Why can MLflow be faster despite using litellm internally?

Because the bottlenecks are **not** in the litellm library call itself. They are in the server-side steps that happen _before_ litellm is called:

| Component                    | Overhead     | Where it happens              |
| ---------------------------- | ------------ | ----------------------------- |
| `litellm.acompletion()` call | ~1-2ms       | Both MLflow and LiteLLM Proxy |
| Config resolution (DB)       | **50-200ms** | MLflow only (uncached)        |
| Usage tracking / tracing     | **10-50ms**  | MLflow only (when enabled)    |
| Core proxy routing + HTTP    | ~5-10ms      | Both (similar)                |

When the two MLflow-specific bottlenecks are removed (config cached + tracing disabled), the remaining code path is faster than LiteLLM Proxy because MLflow's `aiohttp`-based upstream call has less overhead than LiteLLM Proxy's full request pipeline (middleware, callbacks, response transformation, logging).

---

## Measurement Methodology

### What is measured

Latency is measured **client-side** using `time.perf_counter()` (high-resolution monotonic clock) in the benchmark client (`benchmark_compare.py`):

```
           measured latency
    ◀──────────────────────────────▶
    │                              │
 t_start                        t_end
    │                              │
    ▼                              ▼
  POST request sent ─────▶ response body fully read
```

Each measurement includes: client-side serialization, network round-trip (loopback in local benchmarks), full server processing, and response deserialization. Only HTTP 200 responses are counted; failures are tracked separately.

### Connection pooling

The benchmark client uses `aiohttp.TCPConnector` with HTTP keep-alive (`force_close=False`), matching how production clients typically maintain persistent connections. This means the TCP handshake cost is amortized across requests after the warmup phase.

### Warmup phase

Before each timed run, `min(50, n_requests)` warmup requests are sent and discarded. This ensures connection pools are established, server caches are populated, and worker processes are initialized before measurement begins.

### What is NOT included (vs real-world deployments)

The benchmark runs everything on `127.0.0.1` (loopback), which eliminates several real-world costs:

| Factor                        | In benchmark                               | In production                 | Impact on results                                         |
| ----------------------------- | ------------------------------------------ | ----------------------------- | --------------------------------------------------------- |
| **Network latency**           | ~0ms (loopback)                            | 1-100ms per hop               | Adds to all latency numbers equally                       |
| **TLS/SSL handshake**         | None (plain HTTP)                          | ~5-20ms per new connection    | Adds to first-request latency                             |
| **DNS resolution**            | None (IP literal)                          | ~1-5ms (cached), ~50ms (cold) | Negligible with connection reuse                          |
| **Provider inference time**   | Fixed fake delay                           | Variable (50ms-60s+)          | Dominates real latency; proxy overhead becomes negligible |
| **Client-to-proxy network**   | ~0ms (loopback)                            | 1-50ms                        | Adds to client-measured latency                           |
| **Proxy-to-provider network** | ~0ms (loopback)                            | 10-200ms                      | Adds to server processing time                            |
| **TLS to provider**           | None                                       | ~10-30ms                      | Adds to upstream call time                                |
| **Authentication**            | Disabled (`--disable-security-middleware`) | Token validation, RBAC        | Adds ~1-5ms per request                                   |

### What the benchmark isolates

By controlling for network and provider latency, the benchmark measures **pure proxy overhead** — the extra time added by each gateway on top of the upstream provider call. The `FAKE_RESPONSE_DELAY_MS` parameter (default: 50ms) simulates a realistic provider response time.

```
measured latency = proxy overhead + fake delay + loopback RTT (~0ms)
                   ▲                ▲
                   │                └── controlled via FAKE_RESPONSE_DELAY_MS
                   └── this is what we're comparing
```

Setting `FAKE_RESPONSE_DELAY_MS=0` isolates the proxy overhead entirely. In the [zero-delay results](#preliminary-results), MLflow adds ~10ms and LiteLLM adds ~41ms of pure proxy overhead.

### Perspective: does proxy overhead matter in production?

With a real provider adding 200-2000ms of inference time, a 10-50ms proxy overhead is 0.5-5% of total latency — often negligible. The benchmark is useful for:

1. **Identifying bottlenecks** (config resolution, tracing) that could compound at scale
2. **Comparing architectures** under controlled conditions
3. **Validating optimizations** (e.g., config caching gives 3.4x throughput improvement)

---

## File Inventory

| File                               | Purpose                                                                                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Shared**                         |                                                                                                                                               |
| `common.sh`                        | Shared shell functions sourced by all benchmark scripts                                                                                       |
| `fake_openai_server.py`            | FastAPI app returning canned OpenAI-compatible responses (chat, completions, embeddings) with configurable delay via `FAKE_RESPONSE_DELAY_MS` |
| `benchmark_compare.py`             | aiohttp-based benchmark (matches LiteLLM's `benchmark_mock.py`), runs both proxies sequentially with warmup, prints comparison table          |
| `setup_tracking_server.py`         | Creates secret + model definition + endpoint via REST API in a running tracking server                                                        |
| **Head-to-head comparison**        |                                                                                                                                               |
| `litellm_config.yaml`              | LiteLLM proxy config pointing at the same fake server (YAML-only, no DB)                                                                      |
| `run_comparison.sh`                | Starts fake server + MLflow AI Gateway (SQLite) + LiteLLM (YAML) + runs `benchmark_compare.py`                                                |
| **Tracking server benchmark**      |                                                                                                                                               |
| `run_tracking_server_benchmark.sh` | Starts fake server + `mlflow server` with SQLite + sets up endpoint + runs benchmark                                                          |
| **Full-stack comparison**          |                                                                                                                                               |
| `litellm_config_db.yaml`           | LiteLLM proxy config with PostgreSQL `database_url` for spend tracking                                                                        |
| `run_full_stack_comparison.sh`     | Starts PostgreSQL (Docker) + MLflow AI Gateway (PostgreSQL) + LiteLLM (PostgreSQL) + runs comparison benchmark                                |
| **Other**                          |                                                                                                                                               |
| `.gitignore`                       | Ignores `results/` directory                                                                                                                  |

### Modified existing files

| File                                                   | Change                                                                              |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `docs/docs/genai/governance/ai-gateway/index.mdx`      | Added Benchmarks tile card with Timer icon                                          |
| `docs/docs/genai/governance/ai-gateway/benchmarks.mdx` | New public-facing benchmark methodology & results page                              |
| `pyproject.toml`                                       | Added `"benchmarks/*" = ["T20"]` in ruff per-file-ignores (CLI scripts use `print`) |

---

## Key Implementation Details

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

# Otherwise:
pip install mlflow[gateway]
```

### Head-to-head vs LiteLLM

```bash
cd benchmarks/gateway
pip install 'litellm[proxy]'  # or: uv pip install 'litellm[proxy]'
GATEWAY_WORKERS=4 REQUESTS=2000 MAX_CONCURRENT=50 RUNS=3 bash run_comparison.sh
```

### MLflow-only tracking server benchmark

```bash
cd benchmarks/gateway
bash run_tracking_server_benchmark.sh
```

### Full-stack comparison (both on PostgreSQL, requires Docker)

```bash
cd benchmarks/gateway
pip install 'litellm[proxy]'
bash run_full_stack_comparison.sh
```

---

## Benchmark Configurations

### `run_comparison.sh` environment variables

| Variable                 | Default | Description                                         |
| ------------------------ | ------- | --------------------------------------------------- |
| `GATEWAY_WORKERS`        | 4       | Workers for both proxies                            |
| `REQUESTS`               | 2000    | Total requests per run                              |
| `MAX_CONCURRENT`         | 50      | Max concurrent requests                             |
| `RUNS`                   | 3       | Number of benchmark runs                            |
| `FAKE_RESPONSE_DELAY_MS` | 50      | Simulated provider latency (ms)                     |
| `USAGE_TRACKING`         | false   | Enable MLflow usage tracking (set `true` to enable) |

### `run_tracking_server_benchmark.sh` environment variables

| Variable                  | Default | Description                                            |
| ------------------------- | ------- | ------------------------------------------------------ |
| `TRACKING_SERVER_WORKERS` | 4       | Workers for `mlflow server`                            |
| `REQUESTS`                | 2000    | Total requests per run                                 |
| `MAX_CONCURRENT`          | 50      | Max concurrent requests                                |
| `RUNS`                    | 3       | Number of benchmark runs                               |
| `FAKE_RESPONSE_DELAY_MS`  | 50      | Simulated provider latency (ms)                        |
| `USAGE_TRACKING`          | true    | Enable usage tracking/tracing (set `false` to disable) |

### `run_full_stack_comparison.sh` environment variables

| Variable                 | Default | Description                                           |
| ------------------------ | ------- | ----------------------------------------------------- |
| `WORKERS`                | 4       | Workers for both proxies                              |
| `REQUESTS`               | 2000    | Total requests per run                                |
| `MAX_CONCURRENT`         | 50      | Max concurrent requests                               |
| `RUNS`                   | 3       | Number of benchmark runs                              |
| `FAKE_RESPONSE_DELAY_MS` | 50      | Simulated provider latency (ms)                       |
| `USAGE_TRACKING`         | true    | Enable MLflow usage tracking (set `false` to disable) |

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

All results: MacBook Pro (Apple Silicon), 4 workers, 50 concurrent users, 50ms fake delay.

### Head-to-head: MLflow AI Gateway vs LiteLLM (barebone)

MLflow AI Gateway with SQLite (usage tracking OFF) vs LiteLLM with YAML config (no DB, `callbacks: []`). 2000 requests/run, 3 runs.

| Metric          | MLflow AI Gateway | LiteLLM |
| --------------- | ----------------- | ------- |
| **P50 latency** | 2,770 ms          | 77 ms   |
| **P95 latency** | 5,089 ms          | 116 ms  |
| **P99 latency** | 5,249 ms          | 265 ms  |
| **Throughput**  | 21 rps            | 602 rps |
| **Failures**    | 0                 | 0       |

The MLflow AI Gateway resolves endpoint config from the database on every request (3-5 SQL queries). With SQLite's single-writer lock, this serializes under concurrency, producing ~21 rps with multi-second latencies. LiteLLM keeps config in memory from startup.

### Full-stack: MLflow (PostgreSQL) vs LiteLLM (PostgreSQL)

Both proxies with PostgreSQL backend and usage/spend tracking enabled. 2000 requests/run, 3 runs.

| Metric          | MLflow AI Gateway | LiteLLM |
| --------------- | ----------------- | ------- |
| **P50 latency** | 4,770 ms          | 113 ms  |
| **P95 latency** | 7,681 ms          | 152 ms  |
| **P99 latency** | 8,066 ms          | 315 ms  |
| **Throughput**  | 14 rps            | 430 rps |
| **Failures**    | 0                 | 0       |

PostgreSQL is actually _slower_ for MLflow than SQLite (14 rps vs 21 rps) because the per-request SQL queries now incur network round-trip latency. The bottleneck is the uncached endpoint config resolution pattern, not the DB engine. LiteLLM loads config from YAML at startup and only uses DB for spend tracking.

### MLflow AI Gateway: SQLite vs PostgreSQL

Same AI Gateway code path, different DB backends (usage tracking ON). 2000 requests/run, 3 runs.

| Metric          | SQLite   | PostgreSQL |
| --------------- | -------- | ---------- |
| **P50 latency** | 1,504 ms | 4,770 ms   |
| **P95 latency** | 6,537 ms | 7,681 ms   |
| **P99 latency** | 6,850 ms | 8,066 ms   |
| **Throughput**  | 18 rps   | 14 rps     |

> **Key takeaway**: The dominant bottleneck in the MLflow AI Gateway is the **uncached per-request endpoint config resolution** (3-5 SQL queries per request). Switching from SQLite to PostgreSQL does not improve performance — it adds network latency to each query. Caching the endpoint config would be the most impactful optimization.

### Bottleneck isolation: config cache + usage tracking

To verify the bottleneck hypothesis, a benchmark-only config cache was added to `config_resolver.py` (enabled via `MLFLOW_GATEWAY_CACHE_CONFIG=true`). This caches the result of `get_endpoint_config()` after the first DB lookup, eliminating per-request SQL queries.

**Tracking server benchmark (SQLite, 4 workers, 50 concurrent, 1000 req/run, 2 runs):**

| Configuration                   | Throughput | P50      | P99      |
| ------------------------------- | ---------- | -------- | -------- |
| No cache, usage tracking ON     | 22 rps     | 1,211 ms | 5,414 ms |
| **Cache ON**, usage tracking ON | **75 rps** | 654 ms   | 1,565 ms |

Caching the endpoint config alone gives a **3.4x throughput improvement**.

**Full-stack comparison (PostgreSQL, 4 workers, 50 concurrent, 2000 req/run, 3 runs):**

| Configuration                           | MLflow RPS | MLflow P50 | LiteLLM RPS | LiteLLM P50 |
| --------------------------------------- | ---------- | ---------- | ----------- | ----------- |
| Cache OFF, usage tracking ON (baseline) | 14         | 4,770 ms   | 430         | 113 ms      |
| Cache ON, usage tracking ON             | 67         | 763 ms     | 521         | 86 ms       |
| **Cache ON, usage tracking OFF**        | **841**    | **56 ms**  | 539         | 88 ms       |

**With both optimizations, MLflow is 1.6x faster than LiteLLM** (841 vs 539 rps) — the core proxy path is not the bottleneck.

**Barebone comparison with cache (SQLite, usage tracking OFF, 50ms delay, 2000 req/run, 3 runs):**

| Metric          | MLflow AI Gateway | LiteLLM |
| --------------- | ----------------- | ------- |
| **P50 latency** | 55 ms             | 81 ms   |
| **P95 latency** | 74 ms             | 123 ms  |
| **P99 latency** | 105 ms            | 298 ms  |
| **Throughput**  | 844 rps           | 577 rps |
| **Failures**    | 0                 | 0       |

**Zero-delay full-stack comparison (PostgreSQL, cache ON, usage tracking OFF, 0ms delay, 2000 req/run, 3 runs):**

With `FAKE_RESPONSE_DELAY_MS=0`, the benchmark measures pure proxy overhead with no simulated provider latency.

| Metric          | MLflow AI Gateway | LiteLLM   |
| --------------- | ----------------- | --------- |
| **P50 latency** | 10 ms             | 41 ms     |
| **P95 latency** | 61 ms             | 98 ms     |
| **P99 latency** | 72 ms             | 254 ms    |
| **Throughput**  | 3,483 rps         | 1,062 rps |
| **Failures**    | 0                 | 0         |

MLflow's core proxy path adds ~10ms of overhead vs LiteLLM's ~41ms. At 3,483 rps, MLflow is **3.3x faster** than LiteLLM when both bottlenecks are removed.

### Bottleneck breakdown

| Bottleneck                        | Overhead factor | Evidence                             |
| --------------------------------- | --------------- | ------------------------------------ |
| **Usage tracking / tracing**      | ~12x            | 75 rps → 841 rps when disabled       |
| **Uncached config DB queries**    | ~5x             | 14 rps → 67 rps when cached          |
| **Core proxy path** (no overhead) | 1x (baseline)   | 841 rps — faster than LiteLLM at 539 |

The two bottlenecks are orthogonal and compound: together they reduce throughput from 841 rps to 14 rps (~60x total overhead). The usage tracking / tracing path is the dominant factor.

> **How to reproduce**: Set `MLFLOW_GATEWAY_CACHE_CONFIG=true` when launching `mlflow server` to enable the benchmark config cache. This is a benchmark-only change in `mlflow/store/tracking/gateway/config_resolver.py` and should not be used in production (it bypasses config updates).

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
pip install 'litellm[proxy]'

# Increase file descriptor limit (critical for high concurrency)
ulimit -n 65536

# Run head-to-head comparison
cd benchmarks/gateway
GATEWAY_WORKERS=4 REQUESTS=10000 MAX_CONCURRENT=200 RUNS=5 bash run_comparison.sh
```

### Option 2: Separate machines (recommended for high fidelity)

Use 3 machines to eliminate resource contention:

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ Machine A    │────▶│ Machine B       │────▶│ Machine C        │
│ Load Gen     │     │ Proxy Under Test│     │ Fake OpenAI      │
│ (aiohttp)    │◀────│ (MLflow or      │◀────│ Server           │
│              │     │  LiteLLM)       │     │                  │
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
python benchmark_compare.py \
    --target both \
    --requests 10000 \
    --max-concurrent 200 \
    --runs 5 \
    --mlflow-url http://<machine-b>:5000/gateway/benchmark-chat/mlflow/invocations \
    --litellm-url http://<machine-b>:4000/chat/completions
```

### Tips for server benchmarks

1. **Increase file descriptors**: `ulimit -n 65536` before starting anything. The gateway creates a new TCP connection per upstream request, which exhausts ephemeral ports under sustained load (see [Known Limitations](#known-limitations)).

2. **Pin CPU cores**: Use `taskset` on Linux to pin each service to specific cores and avoid cache contention.

3. **Disable power management**: `sudo cpupower frequency-set -g performance` on Linux.

4. **Multiple runs**: Use `--runs 5` or more for statistical significance. The comparison script reports run-to-run variance.

5. **Warm up the system**: The first run typically has higher latency. Use warmup runs or discard the first result.

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

### LiteLLM `network_mock` mode

LiteLLM's published benchmarks use `network_mock: true`, which skips the real HTTP call entirely and returns a canned response inside the proxy process. Our comparison uses the real HTTP path (proxy -> fake server) for both sides, which is more representative of production behavior but produces different numbers than LiteLLM's published figures.
