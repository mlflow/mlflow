# MLflow AI Gateway Benchmark Suite

Benchmark suite for measuring the latency overhead and scalability of the MLflow AI Gateway. Includes a head-to-head comparison tool against [LiteLLM](https://docs.litellm.ai/docs/benchmarks) and [Portkey AI Gateway](https://github.com/Portkey-AI/gateway) using the same methodology for direct comparability.

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
- [Head-to-Head Comparison (MLflow vs LiteLLM vs Portkey)](#head-to-head-comparison-mlflow-vs-litellm-vs-portkey)
- [AI Gateway Per-Request Code Path](#ai-gateway-per-request-code-path)
- [Tracking Server Benchmark](#tracking-server-benchmark)
- [Full-Stack Comparison (Both on PostgreSQL)](#full-stack-comparison-both-on-postgresql)
- [Preliminary Results](#preliminary-results)
- [Optimization Journey](#optimization-journey)
- [Deploying to a Server](#deploying-to-a-server)
- [Usage Tracking Benchmark](#usage-tracking-benchmark)
- [Known Limitations](#known-limitations)

---

## Architecture

### Head-to-head comparison (`run_comparison.sh`)

```mermaid
graph LR
    A[aiohttp\nbenchmark] --> B[MLflow AI Gateway\ngunicorn + uvicorn]
    A --> C[LiteLLM Proxy\ngunicorn + uvicorn]
    A --> E[Portkey AI Gateway\nNode.js, stateless]
    B --> D[Fake OpenAI\nServer]
    C --> D
    E --> D
```

All proxies run on the same machine, hitting the same fake backend. MLflow and LiteLLM use gunicorn workers; Portkey runs as a single Node.js process via `npx @portkey-ai/gateway`. The benchmark client uses `aiohttp` with connection pooling (matching LiteLLM's own `benchmark_mock.py` methodology).

---

## Request Lifecycle: Direct vs MLflow vs LiteLLM vs Portkey

To understand what the benchmark measures, it helps to compare the request lifecycle across four scenarios: calling the provider directly, going through the MLflow AI Gateway, going through LiteLLM Proxy, and going through Portkey AI Gateway.

### 1. Direct to provider (baseline — not benchmarked)

```mermaid
graph LR
    Client -- HTTP --> OpenAI[OpenAI API] -- HTTP --> Client
```

Steps: DNS + TLS handshake + send request + provider inference + receive response. This is the minimum latency for any LLM call. The gateway cannot reduce this.

### 2. Through MLflow AI Gateway

```mermaid
graph LR
    Client -- HTTP --> MLflow[MLflow Server] -- HTTP --> Provider -- HTTP --> Client

    subgraph MLflow["MLflow Server (server-side steps)"]
        S1["1. Budget check (in-memory)"]
        S2["2. Config resolution (DB) — 3-5 SQL queries"]
        S3["3. Secret decrypt (cached)"]
        S4["4. Provider instantiation"]
        S5["5. Tracing (if enabled)"]
        S6["6. litellm.acompletion() → calls upstream provider"]
        S7["7. Budget callback"]
        S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    end
```

**7 server-side steps** per request. Steps 2 (config resolution) and 5 (tracing) are the dominant bottlenecks — see [Bottleneck isolation](#bottleneck-isolation-config-cache--usage-tracking). When both are eliminated, steps 1/3/4/6/7 add only ~10ms of overhead (see [zero-delay results](#preliminary-results)).

### 3. Through LiteLLM Proxy

```mermaid
graph LR
    Client -- HTTP --> LiteLLM[LiteLLM Proxy] -- HTTP --> Provider -- HTTP --> Client

    subgraph LiteLLM["LiteLLM Proxy (server-side steps)"]
        L1["1. Route lookup (in-memory)"]
        L2["2. Auth check"]
        L3["3. httpx / aiohttp call → calls upstream provider"]
        L4["4. Spend tracking callback"]
        L1 --> L2 --> L3 --> L4
    end
```

**4 server-side steps** per request. Config is loaded from YAML at startup and kept in memory — no per-request DB queries. When spend tracking is DB-backed (PostgreSQL mode), step 4 involves async DB writes.

### 4. Through Portkey AI Gateway

```mermaid
graph LR
    Client -- HTTP --> Portkey[Portkey AI Gateway] -- HTTP --> Provider -- HTTP --> Client

    subgraph Portkey["Portkey AI Gateway (server-side steps)"]
        P1["1. Parse routing headers (x-portkey-provider, x-portkey-custom-host)"]
        P2["2. Transform request to provider format"]
        P3["3. HTTP call → calls upstream provider"]
        P1 --> P2 --> P3
    end
```

**3 server-side steps** per request. Portkey is stateless — all routing config is passed via per-request headers (`x-portkey-provider`, `x-portkey-custom-host`). No database, no config file, no spend tracking in the OSS version. This is the lightest-weight architecture of the three.

### Side-by-side step comparison

| Step                   | MLflow AI Gateway                             | LiteLLM Proxy                         | Portkey AI Gateway                |
| ---------------------- | --------------------------------------------- | ------------------------------------- | --------------------------------- |
| Config lookup          | **3-5 SQL queries per request** (uncached)    | In-memory (loaded from YAML at start) | Per-request headers (no lookup)   |
| Secret/auth access     | AES-256-GCM decrypt (60s cache)               | Read from YAML config (no encryption) | N/A (headers carry provider info) |
| Provider instantiation | Created fresh per request                     | Reused from startup                   | Stateless (per-request routing)   |
| Upstream HTTP call     | `litellm.acompletion()` via aiohttp           | `httpx` / `aiohttp`                   | Node.js HTTP client               |
| Usage/spend tracking   | `mlflow.trace()` spans + async DB persistence | Spend callbacks + optional async DB   | N/A in OSS (cloud-only feature)   |
| Budget/rate limiting   | In-memory with periodic DB refresh            | In-memory with DB persistence         | N/A in OSS (cloud-only feature)   |

The key architectural difference: MLflow treats the database as the source of truth for endpoint config and resolves it on every request. LiteLLM treats YAML as the source of truth and loads it once at startup. Portkey is fully stateless — routing is determined by per-request headers with no server-side state. This is why MLflow is slower by default (3-5 extra SQL queries per request), but equally fast or faster when the config is cached.

---

## MLflow and LiteLLM: Library vs Proxy

The name "LiteLLM" appears in two different roles in this benchmark, which can be confusing:

```mermaid
graph TD
    subgraph mlflow["MLflow AI Gateway (the server being benchmarked)"]
        direction TB
        A1[gateway_api.py] --> A2[LiteLLMProvider]
        A2 --> A3["litellm.acompletion()\n— LiteLLM the LIBRARY (Python package)"]
        A3 --> A4[HTTP call to upstream provider]
    end

    subgraph litellm_proxy["LiteLLM Proxy (the server being compared against)"]
        direction TB
        B1["litellm --config litellm_config.yaml\n— LiteLLM the SERVER (CLI proxy process)"]
        B1 --> B2[Route request to configured model]
        B2 --> B3[HTTP call to upstream provider]
    end
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

```mermaid
graph LR
    A["t_start\nPOST request sent"] -->|measured latency| B["t_end\nresponse body fully read"]
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

```mermaid
graph LR
    subgraph measured["measured latency"]
        A["proxy overhead\n⬆ this is what we're comparing"] --- B["fake delay\n⬆ controlled via FAKE_RESPONSE_DELAY_MS"] --- C["loopback RTT\n(~0ms)"]
    end
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
| `benchmark_compare.py`             | aiohttp-based benchmark (matches LiteLLM's `benchmark_mock.py`), runs proxies sequentially with warmup, prints comparison table               |
| `setup_tracking_server.py`         | Creates secret + model definition + endpoint via REST API in a running tracking server                                                        |
| **Head-to-head comparison**        |                                                                                                                                               |
| `litellm_config.yaml`              | LiteLLM proxy config pointing at the same fake server (YAML-only, no DB)                                                                      |
| `run_comparison.sh`                | Starts fake server + MLflow AI Gateway (SQLite) + LiteLLM (YAML) + Portkey (npx) + runs `benchmark_compare.py`                                |
| **Tracking server benchmark**      |                                                                                                                                               |
| `run_tracking_server_benchmark.sh` | Starts fake server + `mlflow server` with SQLite + sets up endpoint + runs benchmark                                                          |
| **Full-stack comparison**          |                                                                                                                                               |
| `litellm_config_db.yaml`           | LiteLLM proxy config with PostgreSQL `database_url` for spend tracking (metadata only)                                                        |
| `litellm_config_db_payload.yaml`   | LiteLLM proxy config with PostgreSQL + `store_prompts_in_spend_logs: true` for full payload logging                                           |
| `run_full_stack_comparison.sh`     | Starts PostgreSQL (Docker) + MLflow AI Gateway (PostgreSQL) + LiteLLM (PostgreSQL) + Portkey (npx) + runs comparison benchmark                |
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

### Head-to-head vs LiteLLM & Portkey

```bash
cd benchmarks/gateway
pip install 'litellm[proxy]'  # or: uv pip install 'litellm[proxy]'
# Portkey requires Node.js/npx — automatically skipped if not found
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

## Head-to-Head Comparison (MLflow vs LiteLLM vs Portkey)

The `run_comparison.sh` script runs all proxies under identical conditions:

- Same fake OpenAI backend (8 gunicorn workers on port 9000)
- Same aiohttp-based benchmark client with connection pooling
- Warmup phase (50 requests) before timed measurement
- Multiple runs to reduce variance

This mirrors LiteLLM's own `benchmark_mock.py` methodology for a fair comparison.

### What each side runs

- **MLflow AI Gateway**: `mlflow server` with SQLite backend, endpoint created via REST API (`/api/3.0/mlflow/gateway/endpoints/create`). This is the production AI Gateway code path — every request resolves endpoint config from the database.
- **LiteLLM Proxy**: `litellm --config litellm_config.yaml` — model list loaded from YAML, no database, `callbacks: []`.
- **Portkey AI Gateway**: `npx @portkey-ai/gateway` — stateless Node.js proxy. No config file; routing is via per-request headers (`x-portkey-provider: openai`, `x-portkey-custom-host: http://127.0.0.1:9000/v1`). Portkey requires Node.js/npx and is automatically skipped if not available.

Usage tracking is **disabled by default** (`USAGE_TRACKING=false`) to reduce DB overhead, but can be enabled with `USAGE_TRACKING=true`.

### Fairness note

This comparison is **not symmetric** in terms of architecture: MLflow resolves endpoint config from the database on every request (3-5 SQL queries), LiteLLM loads its config from YAML at startup and keeps it in memory, and Portkey is fully stateless with per-request header-based routing. Each reflects the real production code path for that proxy.

Portkey's OSS version has no database, spend tracking, or usage logging — it is a pure pass-through proxy. This gives it the lightest overhead but the fewest features.

For a "full production stack" comparison where MLflow and LiteLLM use PostgreSQL, see [Full-Stack Comparison](#full-stack-comparison-both-on-postgresql).

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

```mermaid
graph LR
    A[aiohttp\nbenchmark] <--> B

    subgraph B[MLflow Tracking Server]
        direction TB
        R[FastAPI Router\n/gateway/...] --- G[gateway_api.py]
        G --- DB["SQLite: secrets, models,\nendpoints, configs"]
    end

    B <--> C[Fake OpenAI\nServer]
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

```mermaid
graph LR
    A[aiohttp\nbenchmark] --> B[MLflow Tracking Server\nPostgreSQL, usage tracking]
    A --> C[LiteLLM Proxy\nPostgreSQL, spend tracking]
    A --> F[Portkey AI Gateway\nstateless, no DB]
    B --> D[Fake OpenAI\nServer]
    C --> D
    F --> D
    B --> E["PostgreSQL (Docker)\nDB: mlflow\nDB: litellm"]
    C --> E
```

### What each side does per request

| Operation      | MLflow (tracking server)                   | LiteLLM (PostgreSQL)                  | Portkey (stateless)               |
| -------------- | ------------------------------------------ | ------------------------------------- | --------------------------------- |
| Config lookup  | 3-5 SQL queries (PostgreSQL, uncached)     | In-memory from startup YAML + DB sync | Per-request headers (no lookup)   |
| Secret access  | Decrypt per request (60s cache)            | From YAML config (no encryption)      | N/A (headers carry provider info) |
| Usage tracking | `mlflow.trace()` spans + async persistence | Spend tracking callbacks + DB writes  | N/A in OSS                        |
| Budget check   | In-memory with periodic DB refresh         | In-memory with DB persistence         | N/A in OSS                        |
| Upstream HTTP  | `aiohttp`                                  | `httpx` / `aiohttp`                   | Node.js HTTP client               |

### Fairness notes

MLflow and LiteLLM use the same PostgreSQL instance (separate databases). Portkey runs stateless with no database. Key architectural differences remain:

- **MLflow queries the DB on every request** for endpoint config (3-5 SQL queries, uncached). LiteLLM loads config from YAML at startup and uses DB primarily for spend tracking.
- MLflow and LiteLLM have **usage/spend tracking enabled** by default, which is the intended production configuration for each.
- **Portkey has no DB mode in OSS** — it runs as a pure pass-through proxy. This gives it the lightest overhead but makes it not directly comparable on the "full-stack" dimension.

---

## Preliminary Results

All results: MacBook Pro (Apple Silicon), 4 workers, 50 concurrent users. Config caching is enabled (the default since [#21660](https://github.com/mlflow/mlflow/pull/21660)).

### Combined results

All benchmarks: 2000 requests/run, 3 runs, 4 workers, 50 concurrency.

| Configuration                | Metric          | MLflow Gateway | LiteLLM   | Portkey       |
| ---------------------------- | --------------- | -------------- | --------- | ------------- |
| **Barebone**                 | **P50 latency** | 56.7 ms        | 76.8 ms   | **52.5 ms**   |
| SQLite, no tracking          | **P95 latency** | 77.1 ms        | 104.0 ms  | **55.8 ms**   |
| 50ms delay                   | **P99 latency** | 110.8 ms       | 256.7 ms  | **59.0 ms**   |
|                              | **Throughput**  | **818 rps**    | 616 rps   | **941 rps**   |
|                              |                 |                |           |               |
| **Zero delay**               | **P50 latency** | 13.6 ms        | 40.1 ms   | **8.0 ms**    |
| SQLite, no tracking          | **P95 latency** | 57.5 ms        | 77.1 ms   | **14.2 ms**   |
| 0ms delay (pure overhead)    | **P99 latency** | 64.5 ms        | 232.5 ms  | **18.1 ms**   |
|                              | **Throughput**  | **3,306 rps**  | 1,181 rps | **5,575 rps** |
|                              |                 |                |           |               |
| **Full-stack, tracking OFF** | **P50 latency** | **56.2 ms**    | 92.0 ms   | 52.2 ms       |
| PostgreSQL, no tracking      | **P95 latency** | **79.9 ms**    | 128.6 ms  | 55.0 ms       |
| 50ms delay                   | **P99 latency** | **108.3 ms**   | 282.3 ms  | 60.6 ms       |
|                              | **Throughput**  | **816 rps**    | 530 rps   | **944 rps**   |
|                              |                 |                |           |               |
| **Full-stack, tracking ON**  | **P50 latency** | **54.8 ms**    | 103.0 ms  | **52.7 ms**   |
| PostgreSQL, tracking ON      | **P95 latency** | **72.1 ms**    | 154.7 ms  | **57.3 ms**   |
| 50ms delay                   | **P99 latency** | **102.4 ms**   | 298.8 ms  | **68.2 ms**   |
|                              | **Throughput**  | **852 rps**    | 461 rps   | **932 rps**   |

**Key observations:**

- **MLflow with full tracing (852 rps) is faster than LiteLLM without any tracking (602 rps)** — tracing is essentially free
- **MLflow is within 9% of Portkey** (852 vs 932 rps), despite Portkey being stateless with no DB
- **Portkey is consistently ~930 rps** across all configs — it's stateless with no features to slow it down
- **MLflow without tracking (816-818 rps) beats LiteLLM (530-616 rps)** by ~1.4x in every config
- **Pure proxy overhead** (zero delay): Portkey 8ms < MLflow 14ms < LiteLLM 40ms

### Usage/spend tracking overhead

The full-stack benchmark runs LiteLLM both with and without PostgreSQL spend tracking in the same invocation, allowing direct measurement of tracking overhead for each gateway.

| Gateway     | With tracking | Without tracking | Overhead               |
| ----------- | ------------- | ---------------- | ---------------------- |
| **MLflow**  | 852 rps       | 816 rps          | **1.04x slower** (~4%) |
| **LiteLLM** | 461 rps       | 602 rps          | **1.3x slower** (~30%) |

After the optimizations described in the [Optimization Journey](#optimization-journey) section below, MLflow's tracing overhead is comparable to LiteLLM's spend tracking overhead — both add only single-digit percentage impact.

> **Note**: Portkey's OSS version has no usage/spend tracking, so it runs at the same speed regardless of configuration.
>
> **Historical context**: Before optimizations, MLflow with tracking was **10.6x slower** (77 rps). See the [Optimization Journey](#optimization-journey) for the step-by-step path from 77 rps to 852 rps.

### Historical results (pre-config-cache)

These results were collected before endpoint config caching was added. They are preserved to show the impact of the config cache optimization.

#### Head-to-head: MLflow AI Gateway vs LiteLLM (barebone, no cache)

MLflow AI Gateway with SQLite (usage tracking OFF, **config cache OFF**) vs LiteLLM with YAML config (no DB, `callbacks: []`). 2000 requests/run, 3 runs.

| Metric          | MLflow AI Gateway | LiteLLM |
| --------------- | ----------------- | ------- |
| **P50 latency** | 2,770 ms          | 77 ms   |
| **P95 latency** | 5,089 ms          | 116 ms  |
| **P99 latency** | 5,249 ms          | 265 ms  |
| **Throughput**  | 21 rps            | 602 rps |
| **Failures**    | 0                 | 0       |

The MLflow AI Gateway resolves endpoint config from the database on every request (3-5 SQL queries). With SQLite's single-writer lock, this serializes under concurrency, producing ~21 rps with multi-second latencies. LiteLLM keeps config in memory from startup.

#### Full-stack: MLflow (PostgreSQL) vs LiteLLM (PostgreSQL, no cache)

Both proxies with PostgreSQL backend and usage/spend tracking enabled, **config cache OFF**. 2000 requests/run, 3 runs.

| Metric          | MLflow AI Gateway | LiteLLM |
| --------------- | ----------------- | ------- |
| **P50 latency** | 4,770 ms          | 113 ms  |
| **P95 latency** | 7,681 ms          | 152 ms  |
| **P99 latency** | 8,066 ms          | 315 ms  |
| **Throughput**  | 14 rps            | 430 rps |
| **Failures**    | 0                 | 0       |

#### Config cache impact

To verify the bottleneck hypothesis, endpoint config caching was added to `config_resolver.py` via the `SecretCache` ([#21660](https://github.com/mlflow/mlflow/pull/21660)). This caches the result of `get_endpoint_config()` after the first DB lookup, eliminating per-request SQL queries. This cache is now enabled by default.

**Full-stack comparison (PostgreSQL, 4 workers, 50 concurrent, 2000 req/run, 3 runs):**

| Configuration                           | MLflow RPS | MLflow P50 | LiteLLM RPS | LiteLLM P50 |
| --------------------------------------- | ---------- | ---------- | ----------- | ----------- |
| Cache OFF, usage tracking ON (baseline) | 14         | 4,770 ms   | 430         | 113 ms      |
| Cache ON, usage tracking ON             | 67         | 763 ms     | 521         | 86 ms       |
| **Cache ON, usage tracking OFF**        | **841**    | **56 ms**  | 539         | 88 ms       |

### Bottleneck breakdown

| Bottleneck                        | Overhead factor | Evidence                             |
| --------------------------------- | --------------- | ------------------------------------ |
| **Usage tracking / tracing**      | ~1.04x (solved) | 852 rps → 891 rps when disabled      |
| **Uncached config DB queries**    | ~5x             | 14 rps → 67 rps when cached          |
| **Core proxy path** (no overhead) | 1x (baseline)   | 816 rps — faster than LiteLLM at 530 |

Usage tracking used to be a 10.6x bottleneck (77 rps with tracking vs 816 without). After the optimizations described in the [Optimization Journey](#optimization-journey) section, it adds only ~4% overhead. Endpoint config caching resolved the other major bottleneck.

> **Note**: Endpoint config caching is now enabled by default via `SecretCache` in `config_resolver.py` ([#21660](https://github.com/mlflow/mlflow/pull/21660)). No special configuration is needed to reproduce the cached results above.

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

```mermaid
graph LR
    A["Machine A\nLoad Gen\n(aiohttp)"] <--> B["Machine B\nProxy Under Test\n(MLflow or LiteLLM)"] <--> C["Machine C\nFake OpenAI\nServer"]
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
| Trace DB persistence | Background thread (batch)             | No        | Batch export every 500ms              |
| Budget webhook       | Post-request (if threshold hit)       | Yes       | Rare, only on threshold crossing      |

When `usage_tracking=false`, `maybe_traced_gateway_call()` returns the raw provider function (line 189-190 in `tracing_utils.py`), bypassing all tracing overhead. The per-request DB queries for config resolution still happen.

Key: there are **no synchronous database writes** on the hot path. Traces go to an in-memory manager and are persisted asynchronously via a batch span processor (background thread, flushing every 500ms). Budget tracking is entirely in-memory.

---

## Optimization Journey

This section documents the step-by-step process of identifying and fixing performance bottlenecks in the MLflow AI Gateway's usage tracking path. Each optimization is described with its root cause, fix, and measured impact.

**Starting point**: 77 rps with usage tracking on PostgreSQL (10.6x slower than without tracking).
**End result**: 852 rps (within 9% of stateless Portkey at 932 rps).

### Summary

| Stage                      | RPS     | P99        | Key change                               | Impact                  |
| -------------------------- | ------- | ---------- | ---------------------------------------- | ----------------------- |
| Baseline (with tracking)   | 77      | ~1,700 ms  | Starting point                           | —                       |
| + DB contention fix        | 220     | 442 ms     | Consolidated span+trace writes           | 2.9x                    |
| + ASGI middleware          | 220     | 475 ms     | Eliminated BaseHTTPMiddleware overhead   | Latency fix (see below) |
| + Batch span processor     | **852** | **102 ms** | Decoupled trace export from request path | **3.9x**                |
| Without tracking (ceiling) | 891     | 85 ms      | No tracing at all                        | 1.05x headroom          |

Total improvement: **77 → 852 rps (11x)**. Tracing overhead reduced from 10.6x to 1.05x.

### Phase 1: Profiling infrastructure

Before optimizing, we needed visibility into where time was being spent. A per-phase profiling system was added to the gateway request handler (`gateway_api.py`), enabled with `MLFLOW_GATEWAY_PROFILE=1`.

The profiler wraps each phase of the `invocations()` handler in a timer and writes a summary to a file:

```
Phase                      Mean      P50      P90      P99
parse_body                30.21ms    7.77ms   81.12ms  378.22ms ***
get_config                 0.21ms    0.02ms    0.04ms    0.06ms
create_provider            0.05ms    0.03ms    0.06ms    0.10ms
provider_call            192.23ms  136.19ms  217.39ms 1544.71ms ***
```

This identified two hot spots: `parse_body` and `provider_call` (which includes tracing).

### Phase 2: Trace export DB contention fix (77 → 220 rps)

**Problem**: Server logs showed `DeadlockDetected` and `UniqueViolation` errors. For each gateway request, `MlflowV3SpanExporter.export()` enqueued **two independent async tasks** that raced on the same DB rows:

1. **`log_spans()`**: INSERT/MERGE span rows + UPDATE `trace_info` + MERGE `trace_tag`
2. **`start_trace()`**: INSERT `trace_info` + INSERT tags + INSERT metadata (14 rows) + INSERT metrics

With 10 async worker threads and 50 concurrent requests, both tasks target the same `trace_id` rows, causing PostgreSQL deadlocks and expensive INSERT → `IntegrityError` → rollback → SELECT → merge cycles.

```
Before:  export() → async task 1: log_spans()     ←─ RACE ─→  async task 2: start_trace()
After:   export() → async task: start_trace(spans=...)    (single transaction, no race)
```

**Fix**: Gateway traces are short-lived (2 spans, ~50-100ms) — the trace completes before the first `log_spans` task even runs. A `write_spans_with_trace` flag consolidates both writes into a single `start_trace()` transaction, eliminating the race.

Auto-enabled when `MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=true` (set by gateway at startup) with a non-Databricks SQL backend.

**Impact**: 77 → 220 rps, DB errors eliminated.

**Files**: `mlflow/tracing/export/mlflow_v3.py`, `mlflow/tracing/client.py`, `mlflow/store/tracking/sqlalchemy_store.py`, `mlflow/store/tracking/abstract_store.py`, `mlflow/tracing/provider.py`

**Review notes**: This is a targeted fix for the gateway's specific access pattern (short traces, async export). The flag defaults to `False` and is only enabled when async logging is explicitly set. Normal SDK usage is unaffected.

### Phase 3: Pure ASGI middleware (30ms → 0.02ms per request overhead)

**Problem**: Profiling showed `parse_body` at 30ms mean (P99=378ms) — but sub-phase timing revealed `parse_body.json` was 0.00ms. The bottleneck was `await request.body()` (Starlette's async receive), not JSON parsing.

**Root cause**: The workspace context middleware used `@app.middleware("http")` which wraps handlers in Starlette's `BaseHTTPMiddleware`. This spawns a **background task per request** and streams the response through a memory channel. Under 50 concurrent requests, this creates massive event loop scheduling overhead — each `await request.body()` waits for its ASGI message to be delivered through the extra task layer.

**Proof**: Temporarily disabling the middleware reduced `parse_body` from 30ms to 0.02ms.

**Fix**: Convert `WorkspaceContextMiddleware` from BaseHTTPMiddleware pattern:

```python
# Before — spawns background task per request via call_next
@app.middleware("http")
async def workspace_middleware(request, call_next):
    ...
    response = await call_next(request)


# After — pure ASGI passthrough, no extra task
class WorkspaceContextMiddleware:
    async def __call__(self, scope, receive, send):
        ...
        await self.app(scope, receive, send)
```

**Impact**: Per-request overhead dropped from 30ms to 0.37ms (80x reduction). RPS didn't change significantly because `provider_call` (tracing) was still the dominant cost.

**Files**: `mlflow/server/fastapi_app.py`

**Review notes**: This is a safe, general-purpose improvement. The pure ASGI pattern is what Starlette's own built-in middleware classes use (e.g., `HostValidationMiddleware` in `fastapi_security.py`). All 7 workspace middleware tests pass unchanged. Note: the auth middleware (`fastapi_permission_middleware` in `server/auth/__init__.py`) has the same `BaseHTTPMiddleware` pattern and should be converted as a follow-up when auth is enabled in production.

### Phase 4: Batch span processor (220 → 852 rps)

**Problem**: With the middleware fix eliminating `parse_body` overhead, profiling showed `provider_call` at 181ms mean (vs 53ms without tracing). The 130ms gap was traced to `SimpleSpanProcessor.on_end()`, which calls `self.span_exporter.export()` **synchronously inline** in the request's call stack. Even though the exporter queues an async DB task, the export machinery itself (creating tasks, scheduling on the event loop) runs before the response is returned.

**Root cause**: `BaseMlflowSpanProcessor` extends `SimpleSpanProcessor`, which exports every span inline during `on_end`. Under 50 concurrent requests, this creates 50+ export tasks competing for the same event loop, adding 130ms of scheduling overhead to each request.

**Fix**: Added a `use_batch_processor` flag that delegates to OTel's `BatchSpanProcessor`. Instead of exporting inline, `on_end` queues the span in memory. A background thread flushes batches every 500ms, completely decoupling trace export from the request path.

```python
# on_end with SimpleSpanProcessor (before):
#   update trace info → export(span) → [DB write blocks event loop] → return

# on_end with BatchSpanProcessor (after):
#   update trace info → queue(span) → return immediately
#   [background thread flushes batch every 500ms]
```

**Impact**: 220 → 852 rps. `provider_call` dropped from 181ms to 52.7ms (essentially the raw 50ms fake server delay).

**Files**: `mlflow/tracing/processor/base_mlflow.py`, `mlflow/tracing/processor/mlflow_v3.py`, `mlflow/tracing/provider.py`

**Review notes**: This is the highest-impact change and the one that needs the most scrutiny. Key considerations:

- **Trace delivery latency**: Traces are delivered up to 500ms after the request completes (configurable via `schedule_delay_millis`). This is fine for the gateway (traces are for observability, not real-time), but would change behavior for SDK users who call `mlflow.search_traces()` immediately after a traced function returns.
- **Activation scope**: Only enabled when `MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=true` AND the backend is not Databricks. The gateway sets this env var at startup; normal SDK usage doesn't.
- **Data integrity**: The batch processor's background thread calls the same `export()` method as before — just batched and deferred. The exporter, DB writes, and async queue logic are unchanged.
- **Shutdown behavior**: `shutdown()` and `force_flush()` are delegated to the batch processor, ensuring pending spans are flushed on process exit.
- **Memory**: Under sustained load, the batch queue can grow up to `max_queue_size` (OTel default: 2048 spans). With ~2 spans per trace and 852 rps, the queue cycles every ~1.2 seconds — well within limits.

### Additional optimizations (smaller impact)

These changes were made earlier in the process. They have modest individual impact but contribute to overall performance:

1. **Shared aiohttp session**: Replaced per-request `ClientSession` creation with a lazily-initialized shared session, reusing TCP connections across requests. Files: `mlflow/gateway/providers/utils.py`

2. **Per-trace deduplication locks**: Replaced the global lock in `BaseMlflowSpanProcessor` with per-trace locks, so concurrent spans from different traces don't serialize. Files: `mlflow/tracing/processor/base_mlflow.py`

3. **Cached `inspect.signature()`**: Added `@lru_cache` to `inspect.signature()` calls in the tracing utils, avoiding repeated introspection of the same functions. Files: `mlflow/tracing/utils/__init__.py`

4. **`model_dump()` over `jsonable_encoder()`**: Replaced FastAPI's internal `jsonable_encoder()` with Pydantic v2's native `model_dump()` in provider response serialization. Files: `mlflow/gateway/providers/openai.py`

---

## Known Limitations

### LiteLLM `network_mock` mode

LiteLLM's published benchmarks use `network_mock: true`, which skips the real HTTP call entirely and returns a canned response inside the proxy process. Our comparison uses the real HTTP path (proxy -> fake server) for both sides, which is more representative of production behavior but produces different numbers than LiteLLM's published figures.
