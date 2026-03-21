# OpenSearch Native Integration for MLflow — Implementation Plan

> **Status:** Draft / RFC
> **Author:** (contributor)
> **Date:** 2026-03-21
> **Tracking Issue:** opensearch-project/OpenSearch

---

## 1. Executive Summary

This document describes the plan to add **OpenSearch** as a native backend for storing and querying
MLflow telemetry data — experiments, runs, traces, spans, metrics, parameters, and tags. OpenSearch's
full-text search, nested-document support, and horizontal scalability make it well suited for the
high-cardinality, high-volume workloads that large ML teams generate.

The integration follows MLflow's existing **plugin architecture** (entry-point–based store registry)
so that users can adopt it without changes to core MLflow. A future second phase can contribute
selected pieces upstream.

---

## 2. Goals & Non-Goals

### Goals

| # | Goal |
|---|------|
| G1 | Implement an `OpenSearchTrackingStore` that satisfies the full `AbstractStore` interface |
| G2 | Translate MLflow's filter DSL (`metrics.accuracy > 0.8 AND tags.env = 'prod'`) to OpenSearch Query DSL |
| G3 | Support OpenSearch-native full-text search for trace/span content |
| G4 | Provide Docker Compose setup for local development and CI |
| G5 | Unit tests, integration tests, and Playwright end-to-end tests |
| G6 | Documentation for self-hosting MLflow with an OpenSearch backend |

### Non-Goals (v1)

* OpenSearch Dashboards integration (visualization layer)
* Distributed tracing correlation with OpenSearch APM
* Artifact storage in OpenSearch (artifacts remain in blob stores)
* Model registry backed by OpenSearch (continue using SQL)

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                      MLflow Client                       │
│  (mlflow.start_run, mlflow.log_metric, @mlflow.trace)    │
└────────────────────────┬─────────────────────────────────┘
                         │  REST / gRPC
                         ▼
┌──────────────────────────────────────────────────────────┐
│                    MLflow Server                         │
│                                                          │
│  TrackingStoreRegistry                                   │
│    ├── "sqlite://..."   → SqlAlchemyStore                │
│    ├── "postgresql://…" → SqlAlchemyStore                │
│    ├── "opensearch://…" → OpenSearchTrackingStore  ◄─NEW │
│    └── "file-plugin:…"  → PluginFileStore (example)      │
│                                                          │
└────────────┬─────────────────────────────────────────────┘
             │  opensearch-py client
             ▼
┌──────────────────────────────────────────────────────────┐
│                   OpenSearch Cluster                      │
│                                                          │
│  Index: mlflow_experiments                               │
│  Index: mlflow_runs                                      │
│  Index: mlflow_metrics        (time-series optimized)    │
│  Index: mlflow_params                                    │
│  Index: mlflow_tags                                      │
│  Index: mlflow_traces                                    │
│  Index: mlflow_spans          (full-text on content)     │
│  Index: mlflow_assessments                               │
│  Index: mlflow_trace_tags                                │
│  Index: mlflow_trace_metadata                            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 3.1 URI Scheme

```
opensearch://host:port[/index_prefix]
opensearch+https://user:pass@host:port[/index_prefix]
```

The index prefix (default `mlflow_`) allows multiple MLflow deployments to share one cluster.

### 3.2 Component Map

| Component | File(s) | Description |
|-----------|---------|-------------|
| **Store** | `mlflow/store/tracking/opensearch_store.py` | `AbstractStore` implementation |
| **Client wrapper** | `mlflow/store/tracking/_opensearch_client.py` | Connection pooling, retry, bulk helpers |
| **Index mappings** | `mlflow/store/tracking/opensearch_mappings.py` | Index creation & mapping definitions |
| **Query translator** | `mlflow/store/tracking/opensearch_query.py` | MLflow filter DSL → OpenSearch Query DSL |
| **Plugin entry point** | `pyproject.toml` entry | `mlflow.tracking_store → opensearch` scheme |
| **Docker Compose** | `tests/docker/docker-compose.opensearch-test.yaml` | OpenSearch + MLflow containers |
| **Unit tests** | `tests/store/tracking/test_opensearch_store.py` | Mocked OpenSearch client tests |
| **Integration tests** | `tests/docker/test_opensearch_integration.py` | End-to-end with real OpenSearch |
| **Playwright tests** | `tests/playwright/test_opensearch_traces.py` | UI verification with OpenSearch backend |
| **Documentation** | `docs/docs/opensearch-integration/` | Self-hosting guide |

---

## 4. Detailed Design

### 4.1 OpenSearch Index Mappings

Each MLflow entity maps to a dedicated OpenSearch index. Using separate indices (rather than a
single index with `_type`) provides better scaling, independent lifecycle management, and
simpler mapping definitions.

#### 4.1.1 `mlflow_experiments`

```json
{
  "mappings": {
    "properties": {
      "experiment_id":    { "type": "keyword" },
      "name":             { "type": "text", "fields": { "keyword": { "type": "keyword" } } },
      "artifact_location":{ "type": "keyword" },
      "lifecycle_stage":  { "type": "keyword" },
      "creation_time":    { "type": "long" },
      "last_update_time": { "type": "long" },
      "workspace":        { "type": "keyword" },
      "tags": {
        "type": "nested",
        "properties": {
          "key":   { "type": "keyword" },
          "value": { "type": "text", "fields": { "keyword": { "type": "keyword" } } }
        }
      }
    }
  },
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 1
  }
}
```

#### 4.1.2 `mlflow_runs`

```json
{
  "mappings": {
    "properties": {
      "run_id":          { "type": "keyword" },
      "run_name":        { "type": "text", "fields": { "keyword": { "type": "keyword" } } },
      "experiment_id":   { "type": "keyword" },
      "user_id":         { "type": "keyword" },
      "status":          { "type": "keyword" },
      "start_time":      { "type": "long" },
      "end_time":        { "type": "long" },
      "lifecycle_stage": { "type": "keyword" },
      "artifact_uri":    { "type": "keyword" },
      "deleted_time":    { "type": "long" }
    }
  }
}
```

#### 4.1.3 `mlflow_metrics`

```json
{
  "mappings": {
    "properties": {
      "run_id":    { "type": "keyword" },
      "key":       { "type": "keyword" },
      "value":     { "type": "double" },
      "timestamp": { "type": "long" },
      "step":      { "type": "long" },
      "is_nan":    { "type": "boolean" },
      "model_id":  { "type": "keyword" },
      "dataset_name":   { "type": "keyword" },
      "dataset_digest": { "type": "keyword" }
    }
  },
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "index.sort.field": ["run_id", "key", "timestamp"],
    "index.sort.order": ["asc", "asc", "desc"]
  }
}
```

#### 4.1.4 `mlflow_params`

```json
{
  "mappings": {
    "properties": {
      "run_id": { "type": "keyword" },
      "key":    { "type": "keyword" },
      "value":  { "type": "text", "fields": { "keyword": { "type": "keyword" } } }
    }
  }
}
```

#### 4.1.5 `mlflow_tags`

```json
{
  "mappings": {
    "properties": {
      "run_id": { "type": "keyword" },
      "key":    { "type": "keyword" },
      "value":  { "type": "text", "fields": { "keyword": { "type": "keyword" } } }
    }
  }
}
```

#### 4.1.6 `mlflow_traces`

```json
{
  "mappings": {
    "properties": {
      "trace_id":           { "type": "keyword" },
      "experiment_id":      { "type": "keyword" },
      "request_time":       { "type": "long" },
      "execution_duration": { "type": "long" },
      "status":             { "type": "keyword" },
      "client_request_id":  { "type": "keyword" },
      "request_preview":    { "type": "text" },
      "response_preview":   { "type": "text" }
    }
  }
}
```

#### 4.1.7 `mlflow_trace_tags`

```json
{
  "mappings": {
    "properties": {
      "trace_id": { "type": "keyword" },
      "key":      { "type": "keyword" },
      "value":    { "type": "text", "fields": { "keyword": { "type": "keyword" } } }
    }
  }
}
```

#### 4.1.8 `mlflow_trace_metadata`

```json
{
  "mappings": {
    "properties": {
      "trace_id": { "type": "keyword" },
      "key":      { "type": "keyword" },
      "value":    { "type": "text", "fields": { "keyword": { "type": "keyword" } } }
    }
  }
}
```

#### 4.1.9 `mlflow_spans`

```json
{
  "mappings": {
    "properties": {
      "trace_id":              { "type": "keyword" },
      "span_id":               { "type": "keyword" },
      "experiment_id":         { "type": "keyword" },
      "parent_span_id":        { "type": "keyword" },
      "name":                  { "type": "text", "fields": { "keyword": { "type": "keyword" } } },
      "type":                  { "type": "keyword" },
      "status":                { "type": "keyword" },
      "start_time_unix_nano":  { "type": "long" },
      "end_time_unix_nano":    { "type": "long" },
      "duration_ns":           { "type": "long" },
      "content":               { "type": "text" },
      "dimension_attributes":  { "type": "object", "enabled": false }
    }
  },
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "analysis": {
      "analyzer": {
        "span_content_analyzer": {
          "type": "standard",
          "max_token_length": 255
        }
      }
    }
  }
}
```

#### 4.1.10 `mlflow_assessments`

```json
{
  "mappings": {
    "properties": {
      "assessment_id":    { "type": "keyword" },
      "trace_id":         { "type": "keyword" },
      "experiment_id":    { "type": "keyword" },
      "name":             { "type": "keyword" },
      "source":           { "type": "object" },
      "create_time":      { "type": "long" },
      "last_update_time": { "type": "long" },
      "evaluation": {
        "type": "object",
        "properties": {
          "assessment_type": { "type": "keyword" },
          "value":           { "type": "text" },
          "boolean_value":   { "type": "boolean" },
          "numeric_value":   { "type": "double" }
        }
      },
      "rationale":        { "type": "text" },
      "error":            { "type": "object" },
      "metadata":         { "type": "object", "enabled": false }
    }
  }
}
```

### 4.2 OpenSearch Tracking Store — Class Skeleton

```python
# mlflow/store/tracking/opensearch_store.py

class OpenSearchTrackingStore(AbstractStore):
    """
    MLflow tracking store backed by OpenSearch.
    
    URI format: opensearch://host:port[/index_prefix]
                opensearch+https://user:pass@host:port[/index_prefix]
    """

    # --- Lifecycle --------------------------------------------------------
    def __init__(self, store_uri: str, artifact_uri: str | None = None):
        """Parse URI, create opensearch-py client, ensure indices exist."""

    def _ensure_indices(self):
        """Create indices with mappings if they don't already exist."""

    # --- Experiments ------------------------------------------------------
    def search_experiments(self, view_type, max_results, filter_string,
                           order_by, page_token) -> PagedList[Experiment]:
        """Translate filter DSL → OpenSearch bool query, execute, paginate."""

    def create_experiment(self, name, artifact_location, tags) -> str:
        """Index a new experiment document; return generated experiment_id."""

    def get_experiment(self, experiment_id) -> Experiment:
        """GET by _id from mlflow_experiments index."""

    def get_experiment_by_name(self, experiment_name) -> Experiment | None:
        """Term query on name.keyword."""

    def delete_experiment(self, experiment_id):
        """Update lifecycle_stage to 'deleted'."""

    def restore_experiment(self, experiment_id):
        """Update lifecycle_stage to 'active'."""

    def rename_experiment(self, experiment_id, new_name):
        """Partial doc update on name field."""

    def set_experiment_tag(self, experiment_id, tag):
        """Upsert into nested tags array using painless script."""

    # --- Runs -------------------------------------------------------------
    def create_run(self, experiment_id, user_id, start_time, tags,
                   run_name) -> Run:
        """Index run document + initial tag documents."""

    def get_run(self, run_id) -> Run:
        """GET run by _id; assemble RunInfo + RunData from sub-indices."""

    def update_run_info(self, run_id, run_status, end_time, run_name):
        """Partial update on mlflow_runs document."""

    def delete_run(self, run_id):
        """Set lifecycle_stage='deleted', record deleted_time."""

    def restore_run(self, run_id):
        """Set lifecycle_stage='active', clear deleted_time."""

    def search_runs(self, experiment_ids, filter_string, run_view_type,
                    max_results, order_by, page_token) -> PagedList[Run]:
        """
        Translate filter DSL to OpenSearch bool query with:
        - Terms on mlflow_runs for attribute filters
        - Nested queries joining mlflow_metrics / mlflow_params / mlflow_tags
        Use search_after for deep pagination.
        """

    # --- Metrics / Params / Tags ------------------------------------------
    def log_metric(self, run_id, metric):
        """Index metric document; update latest-metric via upsert script."""

    def log_batch(self, run_id, metrics, params, tags):
        """Bulk index metrics, params, tags using OpenSearch _bulk API."""

    def get_metric_history(self, run_id, metric_key, max_results,
                           page_token):
        """Range query on (run_id, key) ordered by (timestamp, step)."""

    def log_param(self, run_id, param):
        """Index param document with _id = run_id:key (idempotent upsert)."""

    def set_tag(self, run_id, tag):
        """Upsert tag document."""

    def delete_tag(self, run_id, key):
        """Delete tag document by composite _id."""

    # --- Traces -----------------------------------------------------------
    def start_trace(self, trace_info) -> TraceInfo:
        """Index trace info document with status IN_PROGRESS."""

    def end_trace(self, trace_id, trace_info) -> TraceInfo:
        """Update trace document with final status and duration."""

    def get_trace(self, trace_id) -> Trace:
        """Get trace info + span documents + tags + metadata."""

    def search_traces(self, experiment_ids, filter_string, max_results,
                      order_by, page_token) -> tuple[list[TraceInfo], str]:
        """
        Full-text search across trace fields and span content.
        Leverage OpenSearch's native text analysis for LIKE/ILIKE/RLIKE.
        """

    def set_trace_tag(self, trace_id, key, value):
        """Upsert trace tag document."""

    def delete_trace_tag(self, trace_id, key):
        """Delete trace tag document."""

    def delete_traces(self, experiment_id, max_timestamp, request_ids):
        """Delete by query on experiment_id + timestamp range."""

    # --- Spans ------------------------------------------------------------
    def log_spans(self, experiment_id, spans):
        """Bulk index span documents with full-text content field."""

    # --- Assessments ------------------------------------------------------
    def create_assessment(self, assessment) -> Assessment:
        """Index assessment document."""

    def update_assessment(self, assessment_id, updates) -> Assessment:
        """Partial update on assessment document."""

    def get_assessment(self, trace_id, assessment_id) -> Assessment:
        """GET by _id."""

    # --- Datasets / Inputs ------------------------------------------------
    def log_inputs(self, run_id, datasets):
        """Bulk index dataset input documents."""

    def search_datasets(self, *args, **kwargs):
        """Query dataset inputs by experiment."""

    # --- Logged Models ----------------------------------------------------
    def log_model(self, model):
        """Index logged model document."""

    def search_logged_models(self, *args, **kwargs):
        """Term/match queries on model attributes."""
```

### 4.3 Query Translation: MLflow Filter DSL → OpenSearch Query DSL

The `OpenSearchQueryTranslator` converts parsed filter expressions (from `search_utils.py`) into
OpenSearch Query DSL JSON.

**Translation rules:**

| MLflow Filter | OpenSearch Query DSL |
|---------------|---------------------|
| `attribute.status = 'FINISHED'` | `{"term": {"status": "FINISHED"}}` |
| `metrics.accuracy > 0.9` | Sub-query on `mlflow_metrics`: `{"range": {"value": {"gt": 0.9}}}` filtered by `{"term": {"key": "accuracy"}}` |
| `params.lr = '0.01'` | Sub-query on `mlflow_params`: `{"term": {"value.keyword": "0.01"}}` filtered by `{"term": {"key": "lr"}}` |
| `tags.env LIKE 'prod%'` | `{"wildcard": {"value.keyword": "prod*"}}` filtered by `{"term": {"key": "env"}}` |
| `tags.env ILIKE 'Prod%'` | `{"wildcard": {"value.keyword": {"value": "prod*", "case_insensitive": true}}}` |
| `attribute.start_time >= 171108` | `{"range": {"start_time": {"gte": 171108}}}` |
| `span.name = 'predict'` | Join query on `mlflow_spans`: `{"term": {"name.keyword": "predict"}}` |
| `span.content RLIKE '.*error.*'` | `{"regexp": {"content": ".*error.*"}}` on `mlflow_spans` |
| `A AND B` | `{"bool": {"must": [A_query, B_query]}}` |
| `IS NULL` | `{"bool": {"must_not": {"exists": {"field": "..."}}}}` |
| `IS NOT NULL` | `{"exists": {"field": "..."}}` |
| `IN (a, b, c)` | `{"terms": {"field": ["a", "b", "c"]}}` |
| `NOT IN (a, b)` | `{"bool": {"must_not": {"terms": {"field": ["a", "b"]}}}}` |

**Cross-index join strategy for `search_runs`:**

Since metrics, params, and tags live in separate indices, the translator uses a two-phase approach:

1. **Phase 1 — Sub-index filters:** Query `mlflow_metrics`, `mlflow_params`, or `mlflow_tags` to
   collect matching `run_id` sets.
2. **Phase 2 — Main query:** Combine the run_id sets with attribute filters in a single bool query
   on `mlflow_runs`.

For `search_traces`, span-level filters similarly use a two-phase approach:
1. Query `mlflow_spans` to get matching `trace_id` sets.
2. Filter `mlflow_traces` by the resulting trace_id set plus direct attribute/tag filters.

**Alternative: Denormalized document approach (future optimization):**

For workloads where search latency is critical, an optional denormalized index can embed latest
metrics, params, and tags directly in the run document. This trades write amplification for
single-query reads.

### 4.4 Pagination Strategy

OpenSearch's `search_after` is used for deep pagination (more efficient than `from`/`size` for
large result sets). The page token encodes:

```json
{
  "sort_values": [1711089570679, "run_abc123"],
  "pit_id": "optional_point_in_time_id"
}
```

The token is base64-encoded and URL-safe, consistent with MLflow's existing `PagedList` contract.

### 4.5 Connection Management

```python
# mlflow/store/tracking/_opensearch_client.py

class OpenSearchClientManager:
    """
    Manages OpenSearch client lifecycle with:
    - Connection pooling (urllib3 pool)
    - Retry with exponential backoff
    - Authentication (basic auth, AWS SigV4, certificate)
    - TLS/SSL configuration
    - Index prefix management
    """

    def __init__(self, store_uri: str):
        """Parse opensearch:// URI and configure client."""

    @property
    def client(self) -> OpenSearch:
        """Return the configured opensearch-py client."""

    def bulk_index(self, index: str, documents: list[dict]):
        """Use opensearch-py bulk helpers for efficient writes."""

    def ensure_index(self, index_name: str, mapping: dict):
        """Create index if not exists, update mapping if needed."""
```

**Supported authentication methods:**

| Method | URI Example | Config |
|--------|------------|--------|
| No auth (dev) | `opensearch://localhost:9200` | — |
| Basic auth | `opensearch://user:pass@host:9200` | Extracted from URI |
| AWS SigV4 | `opensearch+aws://host:443` | Uses `boto3` session |
| Client cert | `opensearch+https://host:9200` | Env vars for cert paths |

### 4.6 Plugin Registration

Add to `pyproject.toml`:

```toml
[project.entry-points."mlflow.tracking_store"]
opensearch = "mlflow.store.tracking.opensearch_store:OpenSearchTrackingStore"
"opensearch+https" = "mlflow.store.tracking.opensearch_store:OpenSearchTrackingStore"
"opensearch+aws" = "mlflow.store.tracking.opensearch_store:OpenSearchTrackingStore"
```

Add optional dependency group:

```toml
[project.optional-dependencies]
opensearch = ["opensearch-py>=2.4.0"]
```

---

## 5. Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

| Task | Files | Description |
|------|-------|-------------|
| 1.1 | `_opensearch_client.py` | Client wrapper with connection management |
| 1.2 | `opensearch_mappings.py` | All index mapping definitions |
| 1.3 | `opensearch_store.py` | Store skeleton with `__init__`, `_ensure_indices` |
| 1.4 | `pyproject.toml` | Entry point + optional dependency registration |
| 1.5 | Unit tests | Client wrapper and mapping tests |

**Milestone:** `OpenSearchTrackingStore("opensearch://localhost:9200")` connects and creates indices.

### Phase 2: Experiment & Run CRUD (Week 2-3)

| Task | Files | Description |
|------|-------|-------------|
| 2.1 | `opensearch_store.py` | Experiment CRUD methods |
| 2.2 | `opensearch_store.py` | Run CRUD methods |
| 2.3 | `opensearch_store.py` | Metric / param / tag logging |
| 2.4 | `opensearch_store.py` | `log_batch` with bulk API |
| 2.5 | Unit tests | CRUD operation tests with mocked client |

**Milestone:** `mlflow.start_run()` / `mlflow.log_metric()` / `mlflow.log_param()` work end-to-end.

### Phase 3: Search & Query Translation (Week 3-4)

| Task | Files | Description |
|------|-------|-------------|
| 3.1 | `opensearch_query.py` | Filter DSL → OpenSearch query translator |
| 3.2 | `opensearch_store.py` | `search_experiments` implementation |
| 3.3 | `opensearch_store.py` | `search_runs` with cross-index joins |
| 3.4 | `opensearch_store.py` | Pagination with `search_after` |
| 3.5 | Unit tests | Query translation tests (filter → JSON) |

**Milestone:** `mlflow.search_runs(filter_string="metrics.accuracy > 0.9")` returns results.

### Phase 4: Tracing Support (Week 4-5)

| Task | Files | Description |
|------|-------|-------------|
| 4.1 | `opensearch_store.py` | Trace CRUD (start, end, get, delete) |
| 4.2 | `opensearch_store.py` | Span logging with full-text content |
| 4.3 | `opensearch_store.py` | `search_traces` with span-level filters |
| 4.4 | `opensearch_store.py` | Trace tags and metadata management |
| 4.5 | `opensearch_store.py` | Assessment CRUD |
| 4.6 | Unit tests | Trace and span operation tests |

**Milestone:** `@mlflow.trace` decorator logs to OpenSearch; traces searchable in UI.

### Phase 5: Integration & E2E Tests (Week 5-6)

| Task | Files | Description |
|------|-------|-------------|
| 5.1 | `docker-compose.opensearch-test.yaml` | OpenSearch + MLflow Docker setup |
| 5.2 | `test_opensearch_integration.py` | Docker-based integration tests |
| 5.3 | `test_opensearch_traces.py` | Playwright E2E tests |
| 5.4 | Documentation | Self-hosting guide |
| 5.5 | CI workflow | GitHub Actions workflow for OpenSearch tests |

**Milestone:** Full CI pipeline green; documentation published.

---

## 6. Testing Strategy

### 6.1 Unit Tests

**Location:** `tests/store/tracking/test_opensearch_store.py`

Unit tests mock the `opensearch-py` client to verify store logic in isolation.

```python
# Example test structure

import pytest
from unittest.mock import MagicMock, patch
from mlflow.store.tracking.opensearch_store import OpenSearchTrackingStore


@pytest.fixture
def mock_client():
    """Create a mock OpenSearch client."""
    with patch("mlflow.store.tracking._opensearch_client.OpenSearch") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
def store(mock_client):
    """Create an OpenSearchTrackingStore with mocked client."""
    return OpenSearchTrackingStore("opensearch://localhost:9200")


class TestExperimentOperations:
    def test_create_experiment(self, store, mock_client):
        """Creating an experiment indexes a document."""
        mock_client.index.return_value = {"_id": "1", "result": "created"}
        exp_id = store.create_experiment("test-exp", artifact_location="/artifacts")
        mock_client.index.assert_called_once()
        assert exp_id == "1"

    def test_get_experiment(self, store, mock_client):
        """Getting an experiment retrieves from the experiments index."""
        mock_client.get.return_value = {
            "_id": "1",
            "_source": {
                "experiment_id": "1",
                "name": "test-exp",
                "lifecycle_stage": "active",
                "creation_time": 1711089570000,
                "last_update_time": 1711089570000,
            }
        }
        exp = store.get_experiment("1")
        assert exp.name == "test-exp"
        assert exp.lifecycle_stage == "active"

    def test_search_experiments_filter(self, store, mock_client):
        """Search with filter string translates to OpenSearch query."""
        mock_client.search.return_value = {"hits": {"hits": [], "total": {"value": 0}}}
        store.search_experiments(
            view_type=1, max_results=10,
            filter_string="name LIKE 'test%'", order_by=["name ASC"]
        )
        call_args = mock_client.search.call_args
        query = call_args[1]["body"]["query"]
        # Verify the query contains a wildcard clause
        assert "wildcard" in str(query) or "bool" in str(query)

    def test_delete_experiment(self, store, mock_client):
        """Deleting an experiment updates lifecycle_stage."""
        mock_client.get.return_value = {
            "_id": "1", "_source": {"lifecycle_stage": "active"}
        }
        mock_client.update.return_value = {"result": "updated"}
        store.delete_experiment("1")
        mock_client.update.assert_called_once()

    def test_create_experiment_duplicate_name_raises(self, store, mock_client):
        """Creating experiment with duplicate name raises MlflowException."""
        mock_client.search.return_value = {
            "hits": {"hits": [{"_source": {"name": "dup"}}], "total": {"value": 1}}
        }
        with pytest.raises(Exception):
            store.create_experiment("dup")


class TestRunOperations:
    def test_create_run(self, store, mock_client):
        """Creating a run indexes a run document and tag documents."""
        mock_client.index.return_value = {"_id": "run123", "result": "created"}
        mock_client.bulk.return_value = {"errors": False}
        run = store.create_run("exp1", "user1", 1711089570000, [], "my-run")
        assert mock_client.index.called
        assert run.info.run_id is not None

    def test_log_batch(self, store, mock_client):
        """log_batch uses OpenSearch bulk API."""
        mock_client.bulk.return_value = {"errors": False, "items": []}
        from mlflow.entities import Metric, Param, RunTag
        store.log_batch(
            "run123",
            metrics=[Metric("acc", 0.95, 1000, 0)],
            params=[Param("lr", "0.01")],
            tags=[RunTag("env", "prod")],
        )
        mock_client.bulk.assert_called_once()

    def test_search_runs_with_metric_filter(self, store, mock_client):
        """search_runs with metric filter does two-phase query."""
        # Phase 1: query metrics index returns run_ids
        # Phase 2: query runs index filtered by run_ids
        mock_client.search.side_effect = [
            # Metrics query
            {"hits": {"hits": [
                {"_source": {"run_id": "r1"}},
                {"_source": {"run_id": "r2"}},
            ], "total": {"value": 2}}},
            # Runs query
            {"hits": {"hits": [
                {"_source": {"run_id": "r1", "experiment_id": "1", "status": "FINISHED"}},
            ], "total": {"value": 1}}},
            # Empty for params/tags lookups
            {"hits": {"hits": [], "total": {"value": 0}}},
            {"hits": {"hits": [], "total": {"value": 0}}},
            {"hits": {"hits": [], "total": {"value": 0}}},
        ]
        results = store.search_runs(
            ["1"], "metrics.accuracy > 0.9", max_results=10
        )
        assert len(results) >= 0  # Validates query translation ran


class TestTraceOperations:
    def test_start_trace(self, store, mock_client):
        """Starting a trace indexes a trace info document."""
        mock_client.index.return_value = {"_id": "trace1", "result": "created"}
        from mlflow.entities import TraceInfo
        trace_info = TraceInfo(
            trace_id="trace1", experiment_id="1",
            request_time=1711089570000, state="IN_PROGRESS",
        )
        result = store.start_trace(trace_info)
        mock_client.index.assert_called()

    def test_search_traces_full_text(self, store, mock_client):
        """search_traces with span content filter uses full-text search."""
        mock_client.search.side_effect = [
            # Spans query for full-text
            {"hits": {"hits": [
                {"_source": {"trace_id": "t1"}},
            ], "total": {"value": 1}}},
            # Traces query
            {"hits": {"hits": [
                {"_source": {"trace_id": "t1", "experiment_id": "1", "status": "OK"}},
            ], "total": {"value": 1}}},
            # Tags
            {"hits": {"hits": [], "total": {"value": 0}}},
            # Metadata
            {"hits": {"hits": [], "total": {"value": 0}}},
        ]
        results, token = store.search_traces(
            ["1"], "span.content LIKE '%error%'", max_results=10
        )
        assert len(results) >= 0


class TestQueryTranslation:
    """Tests for MLflow filter DSL → OpenSearch Query DSL translation."""

    def test_simple_equality(self):
        from mlflow.store.tracking.opensearch_query import OpenSearchQueryTranslator
        translator = OpenSearchQueryTranslator()
        query = translator.translate("attribute.status = 'FINISHED'", entity_type="run")
        assert query["bool"]["must"][0] == {"term": {"status": "FINISHED"}}

    def test_numeric_range(self):
        from mlflow.store.tracking.opensearch_query import OpenSearchQueryTranslator
        translator = OpenSearchQueryTranslator()
        query = translator.translate("attribute.start_time >= 1711089570000", entity_type="run")
        assert "range" in str(query)

    def test_like_to_wildcard(self):
        from mlflow.store.tracking.opensearch_query import OpenSearchQueryTranslator
        translator = OpenSearchQueryTranslator()
        query = translator.translate("tags.env LIKE 'prod%'", entity_type="run")
        assert "wildcard" in str(query)

    def test_ilike_case_insensitive(self):
        from mlflow.store.tracking.opensearch_query import OpenSearchQueryTranslator
        translator = OpenSearchQueryTranslator()
        query = translator.translate("tags.env ILIKE 'Prod%'", entity_type="run")
        assert "case_insensitive" in str(query)

    def test_and_combines_must(self):
        from mlflow.store.tracking.opensearch_query import OpenSearchQueryTranslator
        translator = OpenSearchQueryTranslator()
        query = translator.translate(
            "attribute.status = 'FINISHED' AND attribute.start_time > 100",
            entity_type="run"
        )
        assert len(query["bool"]["must"]) == 2

    def test_in_to_terms(self):
        from mlflow.store.tracking.opensearch_query import OpenSearchQueryTranslator
        translator = OpenSearchQueryTranslator()
        query = translator.translate(
            "attribute.run_id IN ('r1', 'r2', 'r3')", entity_type="run"
        )
        assert "terms" in str(query)

    def test_is_null(self):
        from mlflow.store.tracking.opensearch_query import OpenSearchQueryTranslator
        translator = OpenSearchQueryTranslator()
        query = translator.translate("tags.env IS NULL", entity_type="run")
        assert "must_not" in str(query) and "exists" in str(query)

    def test_rlike_to_regexp(self):
        from mlflow.store.tracking.opensearch_query import OpenSearchQueryTranslator
        translator = OpenSearchQueryTranslator()
        query = translator.translate("span.content RLIKE '.*error.*'", entity_type="trace")
        assert "regexp" in str(query)
```

### 6.2 Integration Tests

**Location:** `tests/docker/test_opensearch_integration.py`

```python
# Example integration test structure

import os
from datetime import timedelta

import pytest
from testcontainers.compose import DockerCompose
from testcontainers.core.wait_strategies import HttpWaitStrategy

import mlflow


@pytest.mark.parametrize("compose_file", ["docker-compose.opensearch-test.yaml"])
def test_opensearch_backend_integration(compose_file):
    """
    End-to-end test with real OpenSearch cluster.

    Verifies:
    1. MLflow server starts with OpenSearch backend
    2. Experiments can be created and searched
    3. Runs with metrics/params/tags can be logged and queried
    4. Traces can be recorded and searched
    5. Full-text search on span content works
    """
    compose = DockerCompose(
        context=os.path.dirname(os.path.abspath(__file__)),
        compose_file_name=[compose_file],
    )
    compose.waiting_for({
        "mlflow": HttpWaitStrategy(5000, "/health")
        .for_status_code(200)
        .with_startup_timeout(timedelta(minutes=5))
    })

    with compose:
        base_url = "http://localhost:5000"
        mlflow.set_tracking_uri(base_url)

        # --- Experiment Operations ---
        exp_id = mlflow.create_experiment("opensearch-integration-test")
        mlflow.set_experiment(experiment_id=exp_id)

        exp = mlflow.get_experiment(exp_id)
        assert exp.name == "opensearch-integration-test"

        # --- Run Operations ---
        with mlflow.start_run(run_name="test-run") as run:
            mlflow.log_param("learning_rate", "0.01")
            mlflow.log_param("batch_size", "32")
            mlflow.log_metric("accuracy", 0.95)
            mlflow.log_metric("loss", 0.05)
            mlflow.set_tag("env", "integration-test")
            mlflow.set_tag("model_type", "transformer")

        # Verify run is searchable
        runs = mlflow.search_runs(
            experiment_ids=[exp_id],
            filter_string="metrics.accuracy > 0.9",
        )
        assert len(runs) == 1
        assert runs.iloc[0]["params.learning_rate"] == "0.01"

        # Search by tag
        runs = mlflow.search_runs(
            experiment_ids=[exp_id],
            filter_string="tags.env = 'integration-test'",
        )
        assert len(runs) == 1

        # --- Trace Operations ---
        @mlflow.trace
        def predict(model_input: list[str]) -> list[str]:
            return [f"prediction for {x}" for x in model_input]

        result = predict(["input1", "input2"])
        assert len(result) == 2

        # Search traces (allow time for async export)
        import time
        time.sleep(2)

        traces = mlflow.search_traces(experiment_ids=[exp_id])
        assert len(traces) >= 1


def test_opensearch_full_text_search():
    """Test OpenSearch-specific full-text search capabilities."""
    # This test would verify that LIKE/RLIKE on span content
    # leverages OpenSearch's text analysis capabilities.
    pass


def test_opensearch_bulk_operations():
    """Test high-volume logging with OpenSearch bulk API."""
    # This test would verify that logging many metrics/params
    # at once uses the bulk API efficiently.
    pass
```

### 6.3 Docker Compose for OpenSearch Testing

**Location:** `tests/docker/docker-compose.opensearch-test.yaml`

```yaml
services:
  opensearch:
    image: opensearchproject/opensearch:2.19.1
    container_name: mlflow-opensearch-test
    environment:
      - discovery.type=single-node
      - DISABLE_SECURITY_PLUGIN=true
      - OPENSEARCH_INITIAL_ADMIN_PASSWORD=M1flow_OpenSearch!
      - "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m"
    ulimits:
      memlock:
        soft: -1
        hard: -1
      nofile:
        soft: 65536
        hard: 65536
    expose:
      - 9200
    healthcheck:
      test: ["CMD-SHELL", "curl -s http://localhost:9200/_cluster/health | grep -q '\"status\":\"green\\|yellow\"'"]
      interval: 10s
      timeout: 5s
      retries: 30

  mlflow:
    container_name: mlflow-opensearch-test-server
    image: mlflow-integration-test
    build:
      context: ../..
      dockerfile: docker/Dockerfile.full.dev
    command: >
      mlflow server
        --backend-store-uri=opensearch://opensearch:9200
        --host=0.0.0.0
        --port=5000
    ports:
      - "5000:5000"
    depends_on:
      opensearch:
        condition: service_healthy
```

### 6.4 Playwright End-to-End Tests

**Location:** `tests/playwright/test_opensearch_traces.py`

These tests verify the UI works correctly when backed by OpenSearch. They exercise the
trace search and display functionality.

```python
"""
Playwright E2E tests for OpenSearch-backed MLflow.

Prerequisites:
  - MLflow server running with OpenSearch backend (docker-compose)
  - pip install pytest-playwright playwright
  - playwright install chromium

Run:
  pytest tests/playwright/test_opensearch_traces.py --base-url=http://localhost:5000
"""

import pytest
from playwright.sync_api import Page, expect


@pytest.fixture(scope="module")
def setup_data():
    """Seed MLflow with experiments, runs, and traces for UI testing."""
    import mlflow

    mlflow.set_tracking_uri("http://localhost:5000")
    exp_id = mlflow.create_experiment("playwright-opensearch-test")
    mlflow.set_experiment(experiment_id=exp_id)

    # Create runs with metrics
    for i in range(5):
        with mlflow.start_run(run_name=f"run-{i}"):
            mlflow.log_metric("accuracy", 0.8 + i * 0.04)
            mlflow.log_param("lr", str(0.001 * (i + 1)))
            mlflow.set_tag("env", "test")

    # Create traces
    @mlflow.trace
    def sample_prediction(text: str) -> str:
        return f"Predicted: {text}"

    for text in ["hello world", "opensearch test", "trace search"]:
        sample_prediction(text)

    import time
    time.sleep(3)  # Allow async trace export

    return exp_id


class TestExperimentPage:
    """Tests for the experiment page with OpenSearch backend."""

    def test_experiment_list_loads(self, page: Page):
        """Verify the experiment list page loads with OpenSearch data."""
        page.goto("/")
        # Expect the experiments table to be visible
        expect(page.locator("table")).to_be_visible(timeout=10000)

    def test_runs_table_loads(self, page: Page, setup_data):
        """Verify runs are displayed for an experiment."""
        exp_id = setup_data
        page.goto(f"/#/experiments/{exp_id}")
        # Wait for runs to load
        page.wait_for_load_state("networkidle")
        # Expect run rows to be visible
        expect(page.locator("[data-testid='runs-table']")).to_be_visible(timeout=10000)

    def test_search_runs_filter(self, page: Page, setup_data):
        """Verify search filter works with OpenSearch backend."""
        exp_id = setup_data
        page.goto(f"/#/experiments/{exp_id}")
        page.wait_for_load_state("networkidle")

        # Type a filter in the search box
        search_input = page.locator("[data-testid='search-input']")
        if search_input.is_visible():
            search_input.fill("metrics.accuracy > 0.9")
            search_input.press("Enter")
            page.wait_for_load_state("networkidle")


class TestTracesPage:
    """Tests for the traces page with OpenSearch backend."""

    def test_traces_tab_loads(self, page: Page, setup_data):
        """Verify the traces tab loads and shows traces."""
        exp_id = setup_data
        page.goto(f"/#/experiments/{exp_id}")
        page.wait_for_load_state("networkidle")

        # Click on Traces tab
        traces_tab = page.locator("text=Traces")
        if traces_tab.is_visible():
            traces_tab.click()
            page.wait_for_load_state("networkidle")

    def test_trace_detail_view(self, page: Page, setup_data):
        """Verify clicking a trace opens the detail view."""
        exp_id = setup_data
        page.goto(f"/#/experiments/{exp_id}")
        page.wait_for_load_state("networkidle")

        traces_tab = page.locator("text=Traces")
        if traces_tab.is_visible():
            traces_tab.click()
            page.wait_for_load_state("networkidle")
            # Click on the first trace row
            first_trace = page.locator("[data-testid='trace-row']").first
            if first_trace.is_visible():
                first_trace.click()
                page.wait_for_load_state("networkidle")

    def test_trace_search(self, page: Page, setup_data):
        """Verify trace search functionality with OpenSearch full-text."""
        exp_id = setup_data
        page.goto(f"/#/experiments/{exp_id}")
        page.wait_for_load_state("networkidle")

        traces_tab = page.locator("text=Traces")
        if traces_tab.is_visible():
            traces_tab.click()
            page.wait_for_load_state("networkidle")

            search_input = page.locator("[data-testid='trace-search-input']")
            if search_input.is_visible():
                search_input.fill("opensearch")
                search_input.press("Enter")
                page.wait_for_load_state("networkidle")
```

### 6.5 Manual Verification Steps

#### Step 1: Start OpenSearch locally

```bash
# Using Docker
docker run -d --name opensearch-dev \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  -e "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m" \
  opensearchproject/opensearch:2.19.1

# Verify OpenSearch is running
curl -s http://localhost:9200/ | python -m json.tool
```

#### Step 2: Install MLflow with OpenSearch support

```bash
# From MLflow repo root
pip install -e ".[opensearch]"
```

#### Step 3: Start MLflow server with OpenSearch backend

```bash
mlflow server \
  --backend-store-uri="opensearch://localhost:9200" \
  --host=0.0.0.0 \
  --port=5000

# Verify server health
curl http://localhost:5000/health
```

#### Step 4: Log experiments, runs, and metrics

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("opensearch-verification")

with mlflow.start_run(run_name="verification-run"):
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 32)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.05)
    mlflow.set_tag("env", "verification")

# Search runs
runs = mlflow.search_runs(filter_string="metrics.accuracy > 0.9")
print(f"Found {len(runs)} runs with accuracy > 0.9")
assert len(runs) == 1
```

#### Step 5: Log and search traces

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("opensearch-verification")

@mlflow.trace
def predict(text: str) -> str:
    return f"Prediction: {text}"

# Generate traces
predict("hello world")
predict("opensearch integration test")

import time
time.sleep(2)  # Allow async export

# Search traces
traces = mlflow.search_traces(
    experiment_ids=[mlflow.get_experiment_by_name("opensearch-verification").experiment_id]
)
print(f"Found {len(traces)} traces")
assert len(traces) >= 2
```

#### Step 6: Verify OpenSearch indices

```bash
# List all MLflow indices
curl -s 'http://localhost:9200/_cat/indices/mlflow_*?v'

# Check experiment index
curl -s 'http://localhost:9200/mlflow_experiments/_search?pretty' \
  -H 'Content-Type: application/json' \
  -d '{"query": {"match_all": {}}}'

# Check runs index
curl -s 'http://localhost:9200/mlflow_runs/_search?pretty' \
  -H 'Content-Type: application/json' \
  -d '{"query": {"match_all": {}}}'

# Check traces index
curl -s 'http://localhost:9200/mlflow_traces/_search?pretty' \
  -H 'Content-Type: application/json' \
  -d '{"query": {"match_all": {}}}'

# Check spans index with full-text search
curl -s 'http://localhost:9200/mlflow_spans/_search?pretty' \
  -H 'Content-Type: application/json' \
  -d '{"query": {"match": {"content": "opensearch"}}}'
```

#### Step 7: Verify UI

1. Open `http://localhost:5000` in a browser
2. Verify the "opensearch-verification" experiment appears
3. Click into the experiment and verify runs are listed
4. Verify metrics (accuracy, loss) are displayed
5. Click on the "Traces" tab and verify traces are listed
6. Try the search/filter functionality

#### Step 8: Run the full test suite

```bash
# Unit tests
pytest tests/store/tracking/test_opensearch_store.py -v

# Integration tests (requires Docker)
pytest tests/docker/test_opensearch_integration.py -v

# Playwright tests (requires running server)
pytest tests/playwright/test_opensearch_traces.py --base-url=http://localhost:5000 -v
```

---

## 7. Configuration Reference

### 7.1 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_OPENSEARCH_HOST` | OpenSearch host (override URI) | — |
| `MLFLOW_OPENSEARCH_PORT` | OpenSearch port | `9200` |
| `MLFLOW_OPENSEARCH_USERNAME` | Basic auth username | — |
| `MLFLOW_OPENSEARCH_PASSWORD` | Basic auth password | — |
| `MLFLOW_OPENSEARCH_INDEX_PREFIX` | Prefix for all index names | `mlflow_` |
| `MLFLOW_OPENSEARCH_VERIFY_CERTS` | TLS certificate verification | `true` |
| `MLFLOW_OPENSEARCH_CA_CERTS` | Path to CA certificate file | — |
| `MLFLOW_OPENSEARCH_CLIENT_CERT` | Path to client certificate | — |
| `MLFLOW_OPENSEARCH_CLIENT_KEY` | Path to client key | — |
| `MLFLOW_OPENSEARCH_USE_SSL` | Enable SSL/TLS | `false` |
| `MLFLOW_OPENSEARCH_NUM_SHARDS` | Default shard count for new indices | `1` |
| `MLFLOW_OPENSEARCH_NUM_REPLICAS` | Default replica count | `1` |
| `MLFLOW_OPENSEARCH_BULK_SIZE` | Max documents per bulk request | `500` |
| `MLFLOW_OPENSEARCH_TIMEOUT` | Request timeout (seconds) | `30` |

### 7.2 Server CLI

```bash
mlflow server \
  --backend-store-uri="opensearch://localhost:9200/my_prefix_" \
  --default-artifact-root="s3://my-bucket/artifacts" \
  --host=0.0.0.0 \
  --port=5000
```

---

## 8. Performance Considerations

### 8.1 Write Path

- **Bulk API:** All multi-document writes (`log_batch`, `log_spans`) use OpenSearch `_bulk` API
- **Refresh interval:** Indices use default `1s` refresh; configurable per-index
- **ID strategy:** Composite IDs (`run_id:key` for params/tags) enable idempotent upserts

### 8.2 Read Path

- **search_after:** Deep pagination without the O(n) cost of `from`/`size`
- **Index sorting:** Metrics index pre-sorted by `(run_id, key, timestamp)` for efficient history queries
- **Point-in-time:** Optional PIT for consistent pagination across refreshes

### 8.3 Scaling

- **Metrics index:** 3 shards by default (highest write volume)
- **Spans index:** 3 shards (large documents, full-text analysis)
- **Other indices:** 1 shard (smaller volume)
- **Index lifecycle:** Future work — ILM policies for automatic rollover of metrics/spans

---

## 9. Security Considerations

- Credentials parsed from URI are never logged
- SSL/TLS supported via `opensearch+https://` scheme
- AWS SigV4 auth supported for Amazon OpenSearch Service
- Index-level security can be configured in OpenSearch Security plugin
- `DISABLE_SECURITY_PLUGIN=true` should only be used in development

---

## 10. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opensearch-py` | `>=2.4.0` | OpenSearch client library |
| `urllib3` | (transitive) | HTTP connection pooling |
| `boto3` | (optional) | AWS SigV4 authentication |

The `opensearch-py` client is the official Python client maintained by the OpenSearch project.
It provides connection pooling, automatic retries, bulk helpers, and serialization.

---

## 11. Future Enhancements (v2+)

| Enhancement | Description |
|-------------|-------------|
| **Denormalized run index** | Embed latest metrics/params/tags in run docs for single-query reads |
| **OpenSearch Dashboards** | Pre-built dashboards for experiment comparison and trace analysis |
| **Index lifecycle management** | Automatic rollover and archival of old metrics/spans |
| **k-NN vector search** | Semantic search on embeddings stored in span attributes |
| **Cross-cluster search** | Federated search across multiple OpenSearch clusters |
| **Async store** | Async I/O with `opensearch-py[async]` for non-blocking server |
| **Model registry on OpenSearch** | Extend to support model registry storage |
| **OpenSearch APM correlation** | Link MLflow traces with OpenSearch Observability traces |

---

## 12. Open Questions

1. **Denormalized vs. normalized indices?**
   - Current plan: normalized (separate indices) for simplicity and write efficiency
   - Trade-off: read queries require cross-index lookups for metric/param filters

2. **Auto-increment experiment IDs?**
   - SQL stores use auto-increment; OpenSearch doesn't natively support this
   - Options: (a) Use OpenSearch sequence IDs, (b) UUID-based IDs, (c) Counter document with optimistic locking

3. **Index refresh timing?**
   - Default 1s refresh means searches may not see just-written data
   - Options: (a) Accept eventual consistency, (b) Force refresh after writes, (c) Configurable per-operation

4. **Migration from SQL to OpenSearch?**
   - Should we provide a migration tool for existing SQL-backed data?
   - Scope for v2

5. **OpenSearch version compatibility?**
   - Minimum version: 2.x (for `search_after` improvements)
   - Test against: 2.11+, 2.19.x (latest)

---

## 13. References

- [MLflow Tracking Store Plugin Guide](https://mlflow.org/docs/latest/plugins.html)
- [OpenSearch Python Client](https://opensearch.org/docs/latest/clients/python-low-level/)
- [OpenSearch Query DSL](https://opensearch.org/docs/latest/query-dsl/)
- [OpenSearch Bulk API](https://opensearch.org/docs/latest/api-reference/document-apis/bulk/)
- [MLflow AbstractStore Interface](mlflow/store/tracking/abstract_store.py)
- [MLflow Search Utils](mlflow/utils/search_utils.py)
