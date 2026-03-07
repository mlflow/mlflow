## MLflow OSS Design Template: Trace Data Archival from DB to Artifact Storage

| Author(s)                      | Edson Tirelli |
| :----------------------------- | :------------ |
| **Date Last Modified**         | 2026-02-26    |
| **Status or outcome**          | DRAFT         |
| **AI Assistant**               | Claude Code   |
| **MLflow Maintainer Sign off** |               |

## Intro

MLflow Tracing currently stores all span data (the full JSON content of each span) in the tracking database alongside trace metadata. While this approach supports real-time ingestion and SQL-based search, it becomes a significant cost and performance concern at scale—span content can be very large (up to 4 GB per span in MySQL) and dominates database storage. This proposal introduces the ability to archive span data from the database into the trace repository, retaining lightweight metadata in the DB for search while moving bulk span content to cheaper, scalable object stores. The design also introduces configurable retention policies so users can automatically manage trace lifecycle based on age or total storage size.

## Feature Request Information

**GitHub Issue:** [mlflow/mlflow#20574](https://github.com/mlflow/mlflow/issues/20574) — _\[FR\] Support archiving spans into artifact from DB for cost optimization_

When trace volume is large, storing complete span data in the tracking database (PostgreSQL, MySQL, SQLite) becomes expensive. The database must handle:

- **Storage costs:** The `spans.content` column (TEXT/LONGTEXT) stores full OTel span JSON, which can be orders of magnitude larger than the metadata columns.
- **Write amplification:** Each span creates a row with potentially large content, stressing DB write throughput.
- **Backup overhead:** Database backups grow linearly with span content, increasing operational burden.

Users need the ability to offload span content to a cheaper trace repository (e.g. S3, GCS, Azure Blob, local filesystem; optionally the same storage as artifacts) while retaining searchable metadata in the database.

## Requirements

- The feature **must** allow users to configure a separate trace repository location for trace span data
  - This configuration **must** be available as both a `mlflow server` CLI option and environment variable
  - This configuration **can** be set globally, but **must** respect per-workspace overrides (stored in the `workspaces` table; see schema changes below)
- The feature **must** support archiving span content from the DB to the trace repository based on configurable retention policies
  - Retention policies **must** be set globally; they **may** be overridden per workspace or per experiment
  - The archival process **must** support a time-based policy (e.g., archive traces older than N days)
  - The archival process **must** support a size-based policy (e.g., keep the latest N GB of span data in the DB)
  - The archival process **should** be triggerable via a CLI command (e.g., `mlflow traces archive`)
  - The archival process **could** be run automatically via a cron job or similar set by the admin
- The feature **must** store archived span data in OTLP-compatible protobuf format (`TracesData` message)
  - The implementation **should** also support OTLP JSON Lines (`.jsonl`) as a debug format
- The feature **must** maintain backward compatibility
  - Retrieving an archived trace **must** transparently fetch span data from the trace repository
  - Search/filter on trace metadata **must** continue to work for archived traces
  - The `search_traces` and `get_trace` APIs **must** be unaffected from the caller's perspective
- The feature **must** support the following repository backends (S3, GCS, Azure, local; similar to the supported integrations for artifact storage)
- The feature **should** allow two span storage modes at ingestion time:
  - `database` (default, current behavior): spans written to DB for real-time search on span-level attributes
  - `repository`: spans written by the server to the trace repository, only trace-level metadata stored in DB
  - Span storage mode **must** be configured on the server only; the client **must not** be able to change or affect this configuration
- The feature **should** record the storage location of span data in trace metadata so retrieval is transparent

### Out of Scope

- **Span-level attribute search on archived traces (JSON attributes):** Once span content is moved to the trace repository, SQL-based filtering that depends on the raw span payload in `spans.content` (e.g., `span.attributes.*`) will not be supported for archived traces. Column-backed span filters that use indexed span metadata (e.g., `span.type`, `span.status`, `span.duration_ns`) will continue to work as long as span rows and these columns are retained in the DB. Trace-level metadata search (timestamp, state, tags, trace metrics) will continue to work.
- **Automatic re-ingestion from trace repository to DB:** Restoring archived span data back into the database is not in scope. Users who need full, arbitrary span-level search over archived traces should query the OTLP files directly or use an external analytics tool.

## Proposal Sketch

### Why

MLflow's tracing adoption is growing rapidly, especially in GenAI/LLM observability workflows. A single LLM application can generate thousands of traces per day, each containing multiple spans with large input/output payloads. Storing all of this data in a relational database is:

1. **Expensive:** Managed database storage (RDS, Cloud SQL, Azure SQL) costs significantly more per GB than object storage (S3, GCS, Azure Blob).
2. **Operationally burdensome:** Large databases require more frequent backups, more careful capacity planning, and more expensive instance types.
3. **Unnecessary for historical data:** Users typically need real-time search only for recent traces. Older traces are accessed infrequently and primarily for debugging or auditing.

### What Problem Does This Solve

This proposal directly addresses the cost and performance gap between "hot" (recent, actively searched) and "cold" (historical, rarely accessed) trace data. By introducing a tiered storage model—database for hot data, trace repository for cold data—users can dramatically reduce database costs while retaining full trace history.

### Alternatives and Workarounds

1. **Manual database cleanup:** Users can run `mlflow gc` or `DELETE` queries to remove old traces, but this permanently loses data.
2. **External ETL pipelines:** Users can build custom scripts to export traces to Parquet/JSON and delete from DB, but this is error-prone and not integrated with MLflow's retrieval APIs.
3. **Database partitioning:** Some databases support table partitioning by time, which can help with performance but doesn't reduce storage costs or move data to cheaper storage.

### Breadth of Need

This serves a broad need. Any user running MLflow Tracing at moderate-to-large scale (>1000 traces/day) will benefit from cost-optimized storage. The feature aligns with MLflow's existing pattern of separating metadata storage (backend store) from bulk data storage (trace repository; may share backend with artifact store), already established for run artifacts.

### Example API

#### CLI: Configure Trace Repository

```bash
# Start server with a dedicated trace repository
mlflow server \
  --backend-store-uri postgresql://localhost/mlflow \
  --artifacts-destination s3://mlflow-artifacts/ \
  --trace-archival-location s3://mlflow-traces/ \
  ...
```

#### CLI: Archive Old Traces

Archival is performed **server-side**, analogous to `mlflow traces delete`. The client (CLI or Python) sends a request to the tracking server specified by `MLFLOW_TRACKING_URI`; the server executes the archival using its own configured backend store and trace repository. This is implemented with a **new REST API endpoint** (e.g. `POST /mlflow/traces/archive-traces`) that the server implements to run the archival workflow.

Alternatively, if maintainers would like to avoid creating the REST endpoint, it can be implemented by
defining a method on the AbstractStore class in mlflow/store/tracking/abstract_store.py. By default, it would not be implemented,
except in the `SqlAlchemyStore` (database). The admin then runs `mlflow traces archive` like an admin does for `mlflow gc`.

```bash
# Archive span data for traces older than 90 days in a specific workspace
# (on a workspace-enabled server, --workspace or --all-workspaces is required)
mlflow traces archive --workspace my-workspace --older-than 90d

# Archive across all workspaces (mirrors mlflow gc --all-workspaces)
mlflow traces archive --all-workspaces --older-than 90d

# Archive in one workspace, keeping only the latest 10 GB
mlflow traces archive \
  --workspace my-workspace \
  --max-db-size 10GB
```

#### Python API

No change is required in the Python API, given that these options are configured server-side
and cannot be changed by the client. Furthermore the existing APIs will continue to work
transparently, no matter if the traces are stored in the database or archived to the
traces repository.

## Decision Matrix

| Proposals                                                                                 | Storage Efficiency | Search Capability            | Backward Compat. | Implementation Effort | Standards Compliance |
| :---------------------------------------------------------------------------------------- | :----------------- | :--------------------------- | :--------------- | :-------------------- | :------------------- |
| **Option 1: Tiered storage + repository (configurable via `MLFLOW_TRACE_SPANS_STORAGE`)** | Medium to High     | High or trace-only (by mode) | Yes              | Moderate              | High (OTLP standard) |
| Option 2: Database partitioning + cold table                                              | Low                | High (SQL on cold table)     | Yes              | High                  | Low (custom format)  |

- **Storage Efficiency:** How much does the approach reduce database storage costs?
- **Search Capability:** Can users still search/filter on trace metadata after archival?
- **Backward Compat.:** Do existing APIs continue to work without changes?
- **Standards Compliance:** Does the storage format follow an existing standard (OTel, OTLP)?

## Decision details

### Option 1 (Recommended): Tiered Storage with OTLP Protobuf Archival and Repository Modes

We propose a single design that supports two configurable **ingestion modes** (server-side, via `MLFLOW_TRACE_SPANS_STORAGE` or `--trace-spans-storage`):

1. **`database` (default)** — Tiered storage: trace span content is stored in the database for a configurable retention window, then archived to the trace repository in OTLP protobuf format. Trace metadata remains in the database permanently; retrieval and retention policies apply as described below.
2. **`repository`** — Repository: span content is written by the server directly to the trace repository at ingestion time; only trace-level metadata is stored in the DB. No span content is written to the database, so there is no tiered archival step for these traces. Best for users who do not need span-level search and want maximum DB savings with minimal implementation surface.

Both modes use the same trace repository layout, OTLP protobuf format for span data, and transparent retrieval via `get_trace` / `search_traces`. Trace metadata remains in the database in all cases, enabling trace-level search and filtering.

NOTE: for cases where the traces storage is down, the server may save traces in the database meanwhile, until the connection is restored.

The key insight is that the `spans.content` column dominates database storage, while the metadata columns (`trace_info`, `trace_tags`, `trace_request_metadata`, `trace_metrics`, `span_metrics`) are compact and valuable for search. By archiving only the span content (or skipping DB for span content in `repository` mode), we achieve maximum storage savings with minimal impact on functionality.

#### Storage Format: OTLP Protobuf (`TracesData`)

The [OpenTelemetry specification](https://github.com/open-telemetry/opentelemetry-proto/blob/main/opentelemetry/proto/trace/v1/trace.proto) defines a `TracesData` message explicitly designed for persistent storage:

> _"TracesData represents the traces data that can be stored in a persistent storage, OR can be embedded by other protocols that transfer OTLP traces data but do not implement the OTLP protocol."_

MLflow already bundles the OTel protobuf definitions at `mlflow/protos/opentelemetry/proto/trace/v1/trace.proto` and has full serialization support via `Span.to_otel_proto()` and `Span.from_otel_proto()`. Using `TracesData` as the archival format:

- **Leverages existing code:** MLflow already converts between internal span representation and OTel proto.
- **Follows the standard:** `TracesData` is the OTel-recommended message for persistent storage, distinct from the wire protocol's `ExportTraceServiceRequest`.
- **Efficient encoding:** Protobuf binary encoding is ~3–10x more compact than equivalent JSON, with faster serialization/deserialization.
- **Interoperable:** Archived traces can be read or exported by any OTel-compatible tool.

**File layout in trace repository:**

```
<traces-root>/
└── <experiment_id>/
    └── traces/
        └── <trace_id>/
            └── artifacts/
                └── traces.pb          # TracesData protobuf binary
```

Each `traces.pb` file contains a single `TracesData` message with all spans for that trace. This enables efficient per-trace retrieval.

**Current OSS behavior:** In the current open-source version of MLflow, traces are always written to the database. The server persists the trace record and span content in the tracking store (e.g. `SqlAlchemyStore`); span location is `TRACKING_STORE` and full trace/span data is stored in the DB.

**Existing trace artifact upload path (V3 / artifact-backed traces):** When a trace is created (e.g. via the MLflow V3 exporter in `mlflow/tracing/export/mlflow_v3.py` or the inference-table exporter), the server creates the trace record in the tracking store and sets a tag `MLFLOW_ARTIFACT_LOCATION` on the trace with the URI where trace data should be stored. That URI is the experiment's artifact location plus `/traces/<trace_id>/artifacts/` (see `SqlAlchemyStore._get_trace_artifact_location_tag`). The **client** then uploads the full trace payload as a single JSON file named `traces.json` to that URI using the artifact repository (`ArtifactRepository.upload_trace_data` in `mlflow/store/artifact/artifact_repo.py`). The client uses `get_artifact_uri_for_trace(trace_info)` from `mlflow/tracing/utils/artifact_utils.py` to resolve the URI from the trace's tags and performs the upload; when not in proxy mode, the client therefore needs credentials for the artifact store. When the UI or API needs full trace data, the same URI is used to download `traces.json` and parse it. This path is used when the server marks the trace's span location as `ARTIFACT_REPO` (via the `mlflow.trace.spansLocation` tag), so span content is read from the artifact store rather than the DB. This mechanism exists in OSS today and coexists with DB-backed span storage (e.g. spans written via `log_spans`).

**Relationship to the proposed archival:** Today there are already two trace storage paths in OSS: (1) DB-backed spans written via `log_spans` (span content in the `spans` table), and (2) artifact-backed `traces.json` payloads uploaded by V3-style exporters to the artifact repository. The proposed archival mechanism is a third, separate path: it stores span content in OTLP protobuf format at a configurable trace repository (`--trace-archival-location`), which may or may not be the same as the artifact store. The proposed `traces.pb` archival is for offloading span content from the DB for traces that were initially written there via the DB-backed path. These three mechanisms can coexist.

#### Database Schema Changes

**Use the existing `SpansLocation` tag.** MLflow already stores where span data lives via the tag `TraceTagKey.SPANS_LOCATION` = `"mlflow.trace.spansLocation"` (see `mlflow/tracing/constant.py`). The enum `SpansLocation(str, Enum)` currently has `TRACKING_STORE` and `ARTIFACT_REPO`. Extend it with one new value for archival:

- **`TRACKING_STORE`** (existing): Span content is in the `spans` table (current behavior). Traces remain in this state until successfully archived to the trace repository (or artifact repo). Used for traces in `database` ingestion mode that have not yet been archived.
- **`TRACES_REPO`** (new): Span content has been archived to the trace repository (or written directly in `repository` ingestion mode). The `spans.content` column is cleared for archived traces; span metadata rows are retained for index-based filtering. Retrieval loads span data from the trace repository.
- **`ARTIFACT_REPO`** (existing): Retained for existing behavior (e.g. V3 API traces whose spans are stored in the artifact store). Distinct from `TRACES_REPO` for the proposed OTLP trace repository.

Traces stay in `TRACKING_STORE` until archival completes (export to repository + clear DB content + set tag to `TRACES_REPO`). If the process crashes mid-archival, the trace remains `TRACKING_STORE` and will be selected again on the next run (re-export overwrites the file, then clear and set tag).

The trace repository URI for archived traces (where to read `traces.pb`) continues to be recorded via the existing `mlflow.artifactLocation` tag when span data is in the trace repository, so retrieval can resolve the path. The **effective trace repository root** for a given trace is the workspace's `traces_destination` when set, otherwise the server's global `--trace-archival-location` (or default artifact root).

A `content_size` column (`BIGINT`, default 0) on `spans` stores the byte length of `content` at write time so size-based archival can use `SUM(content_size)` instead of `SUM(LENGTH(content))` for efficiency (see Size-based policy below). This column is mandatory (not optional) to keep the implementation simple and avoid a fallback code path for `LENGTH(content)` aggregation.

```sql
ALTER TABLE spans
ADD COLUMN content_size BIGINT NOT NULL DEFAULT 0;
```

Existing rows will have `content_size = 0` until backfilled (e.g. batched `UPDATE spans SET content_size = LENGTH(content) WHERE content_size = 0 AND content != ''`).

**Workspaces table: per-workspace trace repository override.** Add a column to the `workspaces` table to store an optional traces destination (server-side configuration) for each workspace. When set, it overrides the server's global trace repository for that workspace (for both archival write and retrieval). This supports multi-tenant deployments where different workspaces use different trace storage locations.

```sql
ALTER TABLE workspaces
ADD COLUMN traces_destination TEXT NULL;
```

- **Semantics:** For a trace in a given workspace, the trace repository root used for archival and for resolving `traces.pb` paths is `workspaces.traces_destination` for that workspace if non-NULL, else the server's `--trace-archival-location` (or `--artifacts-destination` when traces destination is unset).
- **Existing workspaces:** All existing rows have `traces_destination = NULL`, so they use the global trace repository; no backfill is required.

**NOTE:** traces URL resolution follows the same semantics as for artifact location on workspaces. The default, if not overridden at the workspace level, is <default traces root>/workspaces/<workspace name>.

Configuring the per-workspace traces repository override will require adding a **`traces_destination`** parameter (or equivalent name) in:

- **Python API:** `mlflow.create_workspace()` and `mlflow.update_workspace()` (and the corresponding `MlflowClient` methods) must accept an optional `traces_destination` argument; when provided, it is persisted to the workspace record. Add field to the Workspaces base class as well.
- **REST API:** The CreateWorkspace and UpdateWorkspace request messages (and responses) must include an optional `traces_destination` field so that clients can set or clear the workspace-level trace repository URI when creating or updating a workspace.
- **UI:** The create-workspace and edit-workspace flows must include a field for the optional traces destination (e.g. "Traces repository URI" or "Traces destination"), analogous to the existing default artifact root field, so that admins can configure it when creating or updating a workspace.

#### Archival Process

**Note:** Archival is exposed as a subcommand of `mlflow traces`: `mlflow traces archive`. This aligns with other trace lifecycle commands such as `mlflow traces delete`: the client calls the tracking server via `MLFLOW_TRACKING_URI`, and a **new REST API endpoint** (e.g. `POST /mlflow/traces/archive-traces`) is required so that the server performs the archival workflow using its own backend store and trace repository. The `archive` subcommand accepts policy options such as `--older-than` and `--max-db-size`, and **workspace targeting** mirrors `mlflow gc`: the user must specify either **`--workspace <workspace_name>`** (apply the policy to that workspace only) or **`--all-workspaces`** (apply the policy across all workspaces). If neither `--workspace` nor `--all-workspaces` is set and the server has workspaces enabled, the command **must fail** (active workspace is required). See previous examples above.

The archival process runs via `mlflow traces archive` and performs:

1. **Select traces to archive:** Query `trace_info` for traces matching the retention policy (older than N days, or exceeding the size budget ordered by oldest first). Only traces whose `mlflow.trace.spansLocation` tag is `TRACKING_STORE` (or missing, treated as TRACKING_STORE) are candidates.
2. **Export span content:** For each trace, read all spans from the `spans` table, convert to `TracesData` protobuf, and write to the trace repository. The trace repository root used for the write is the workspace's `traces_destination` (if set) for that trace's experiment/workspace, else the server's global traces destination. Record the trace repository location via the existing `mlflow.artifactLocation` tag (or equivalent) for the trace.
3. **Clear DB span content and set tag:** Update `spans.content` to an empty string, set `spans.content_size = 0`, and set the trace tag `mlflow.trace.spansLocation` = `SpansLocation.TRACES_REPO`.

**Crash recovery:** Traces remain in `TRACKING_STORE` until step 3 completes. If the process crashes after step 2 but before step 3, those traces are still `TRACKING_STORE` and will be selected again on the next run. The archival then re-exports (overwriting the file) and completes step 3. Re-running on traces already in `TRACES_REPO` is a no-op (they are not candidates). The process is thus crash-safe and idempotent without an intermediate state.

The archival is performed in batches to avoid locking the database for extended periods.

#### Retrieval Changes

The logic for loading span data based on `SpansLocation` (TRACKING_STORE, TRACES_REPO, or ARTIFACT_REPO) is implemented as a **method on the base class** (e.g. `AbstractStore` or the shared tracing mixin) so that all tracking store plugins inherit it and do not need to reimplement it. Plugins can **override** this method for special cases (e.g. custom repository resolution or format). The base class defines the dispatch and helpers; concrete stores implement the store-specific parts (e.g. `_get_spans_from_db`, `_get_traces_repository_uri`, `_load_spans_from_repository`, `_load_spans_from_artifact_repo`).

Concrete stores (e.g. `SqlAlchemyStore`, `FileStore`) implement the helpers (`_get_spans_from_db`, `_get_traces_repository_uri`, `_load_spans_from_repository`, `_get_artifact_uri_for_trace`, `_load_spans_from_artifact_repo`); the base class provides the shared dispatch so plugins do not reimplement the branching.

**Base class (e.g. AbstractStore):** shared method that all stores use unless they override it:

```python
def _load_spans_for_trace(self, trace_info: TraceInfo) -> list[Span]:
    """Load span data for a trace based on its spans location. Override in subclasses for special cases."""
    spans_location = trace_info.tags.get(
        TraceTagKey.SPANS_LOCATION, SpansLocation.TRACKING_STORE.value
    )
    if spans_location == SpansLocation.TRACES_REPO.value:
        traces_uri = self._get_traces_repository_uri(trace_info.trace_id)
        return self._load_spans_from_repository(traces_uri)
    if spans_location == SpansLocation.ARTIFACT_REPO.value:
        artifact_uri = self._get_artifact_uri_for_trace(trace_info)
        return self._load_spans_from_artifact_repo(artifact_uri)
    # TRACKING_STORE (default)
    return self._get_spans_from_db(trace_info.trace_id)


def get_trace(self, trace_id, allow_partial=False):
    trace_info = self.get_trace_info(trace_id)
    spans = self._load_spans_for_trace(trace_info)
    return Trace(trace_info, TraceData(spans))
```

**Impact on `search_traces`:** Today, `search_traces` supports both trace-level filters (timestamp, state, tags, metrics) and span-level filters (`span.name`, `span.type`, `span.status`, `span.duration_ns`, `span.attributes.*`, etc.). Span-level filters fall into two categories:

- **Column-based span filters** such as `span.name`, `span.type`, `span.status`, and `span.duration_ns` filter against dedicated columns on the `spans` table. Because archival retains span rows and these metadata columns while only clearing/moving `spans.content`, these filters continue to work for archived traces.
- **JSON-based span filters** such as `span.attributes.*` rely on data stored inside the `spans.content` JSON blob. After archival (or in `repository` mode), `spans.content` is no longer populated in the DB for archived traces, so these filters no longer match archived traces.

The implementation must handle this gracefully:

- When a query contains **only trace-level filters**, archived traces are included in results as usual.
- When a query contains **only column-based span-level filters**, archived traces are still included in results, since the relevant span metadata remains in the DB.
- When a query contains **any JSON-based span-level filters** (e.g. `span.attributes.*`), archived traces (where `mlflow.trace.spansLocation` = `TRACES_REPO`) are effectively excluded from results, because the SQL predicates referencing `spans.content` cannot match for those traces.

For `batch_get_traces`, the implementation should partition trace IDs by `SpansLocation` tag value: batch-read DB spans for `TRACKING_STORE` traces, and fetch from the trace repository (potentially in parallel) for `TRACES_REPO` traces.

#### Server Configuration

**New CLI option:**

```
--trace-archival-location URI             Destination URI for the trace repository (trace span data).
                              If not specified, traces use --default-artifact-root.
                              Can be overridden per workspace via workspaces.traces_destination.
                              Supports the same repository backends as artifacts (S3, GCS, Azure, etc.)
                              Env var: MLFLOW_TRACE_ARCHIVAL_LOCATION
```

**New environment variables:**

| Variable                               | Description                                                                                                                          | Default                           |
| :------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------- |
| `MLFLOW_TRACE_ARCHIVAL_LOCATION`       | Global destination URI for the trace repository (trace span storage). Overridable per workspace via `workspaces.traces_destination`. | Same as `--artifacts-destination` |
| `MLFLOW_TRACE_SPANS_STORAGE`           | Default span storage mode for new traces: `database` or `repository` (server-side only; clients cannot override)                     | `database`                        |
| `MLFLOW_TRACE_ARCHIVAL_OLDER_THAN`     | Default value for archiving traces older than this duration (e.g., `90d`, `24h`) (global default, overridable via CLI)               | None                              |
| `MLFLOW_TRACE_ARCHIVAL_MAX_DB_SIZE_MB` | Default value for archiving traces beyond total span content size in DB (global default, overridable via CLI)                        | None                              |

#### Strengths

- Maximizes cost savings by moving the largest data (span content) to cheap object storage while retaining compact, searchable metadata in the DB
- Follows the OTel `TracesData` standard for archival format, ensuring interoperability
- Fully backward compatible—existing APIs work transparently
- Leverages existing MLflow patterns (trace repository / artifact store abstraction, `mlflow traces` CLI subcommands such as `mlflow traces delete`, trace tag metadata)
- Protobuf binary format is 3–10x more compact than JSON, with fast serialization
- Idempotent archival process with clear state tracking (`SpansLocation` tag)
- Flexible retention policies (time-based, size-based, or both)
- Global policies with optional workspace overrides align with multi-tenant usage
- Repository ingestion mode (`MLFLOW_TRACE_SPANS_STORAGE=repository`) available for users who don't need span-level DB search: zero span data in DB, no archival job needed for those traces.
- Single implementation covers tiered archival and repository mode; one format (OTLP protobuf) and one retrieval path.

#### Risks

- **Increased retrieval latency for archived or `repository` mode traces:** Fetching span data from object storage (S3, GCS) adds latency compared to a DB read. This is acceptable for infrequent historical access but should be documented.
- **No span-level search on archived traces:** After archival, span-level filters (e.g., `span.type = 'LLM'`) won't work against archived span content. Trace-level filters (timestamp, state, tags, metrics) continue to work. This trade-off is inherent to the tiered storage model.
- **Trace repository availability:** If the trace repository is unavailable, archived traces cannot be retrieved. This is the same availability model as run artifacts today.
- **Size estimation accuracy:** The `content_size` column stores exact byte size at write time, avoiding the ambiguity of `LENGTH(content)` (which in some engines returns character count rather than bytes). Existing rows require a one-time backfill; until backfilled, their `content_size = 0` will cause them to be underestimated in size-based policies.
- **Resource intensive "size-based policy":** Computing and ranking trace sizes for the size-based retention policy can be expensive (full scan of span content length, JOINs, sort). It should be run as a scheduled or manual batch job, not inline with request handling. Prefer time-based policy where it is sufficient; use size-based policy when DB size is the primary constraint.
- **When using `repository` mode:** No span-level search (trace-level only); mode is chosen at server startup and applies to new traces. Retention policies do not apply to `repository` mode traces (there is no span content in DB to archive).

#### Trace Deletion and Archived File Cleanup

When traces are permanently deleted (via `mlflow traces delete` or `_delete_traces`), the implementation must also clean up archived span files in the trace repository. For traces whose `mlflow.trace.spansLocation` tag is `TRACES_REPO`, the corresponding `<traces-root>/<experiment_id>/traces/<trace_id>/artifacts/traces.pb` file must be deleted from the trace repository. The traces root is the workspace's `traces_destination` if set, else the server's global traces destination. Failure to do so would leave orphaned files. The deletion of repository files should be best-effort: if the repository is unavailable, the DB records should still be deleted (the orphaned file is harmless and can be cleaned up later by a separate sweep).

### Option 2: Database Partitioning + Cold Table

This approach keeps all data in the database but uses table partitioning (PostgreSQL/MySQL) to separate hot and cold trace data. Old partitions can be moved to cheaper storage tiers within the database (e.g., PostgreSQL tablespaces on cheaper disks).

#### Strengths

- Full SQL search capability on both hot and cold data
- No external dependency on trace repository for traces
- Database-native approach; familiar to DBAs

#### Risks

- **Database-specific:** Partitioning syntax and capabilities vary across PostgreSQL, MySQL, and SQLite. SQLite doesn't support partitioning at all.
- **Limited cost savings:** Data stays in the database; savings come from cheaper disk tiers, not from moving to object storage. Managed database services (RDS, Cloud SQL) often don't expose storage tier controls.
- **Operational complexity:** Partition management, maintenance windows, and monitoring add operational burden.
- **No standard format:** Data remains in database-specific format, not interoperable with OTel tools.
- **No retention policies:** Would need separate implementation for time/size-based management.

---

## Appendix A: OTLP File Format Reference

The [OpenTelemetry Protocol File Exporter specification](https://opentelemetry.io/docs/specs/otel/protocol/file-exporter/) defines one serialization format for persistent file storage:

1. **OTLP JSON Lines (`.jsonl`):** One `TracesData` JSON object per line. Human-readable but larger. This is the only format covered by the OTel File Exporter spec.

Additionally, the [OTLP wire protocol specification](https://opentelemetry.io/docs/specs/otlp/) defines binary protobuf encoding for the same `TracesData` message (used in gRPC and HTTP transports). While there is no OTel spec for a binary protobuf _file_ format, the `TracesData` protobuf message is explicitly designed for persistent storage (per the proto comments). This proposal uses binary protobuf as the primary archival format for storage efficiency, leveraging MLflow's existing full support for OTel protobuf serialization:

2. **OTLP Protobuf binary (`.pb`):** Binary-encoded `TracesData` message. Compact and fast. Not an OTel file spec, but uses the standard `TracesData` message.

For this proposal, protobuf binary is the primary format (maximizes storage savings), with JSON Lines as an optional debug/export format.

The `TracesData` protobuf message (from [`opentelemetry-proto`](https://github.com/open-telemetry/opentelemetry-proto)) is:

```protobuf
message TracesData {
  repeated ResourceSpans resource_spans = 1;
}

message ResourceSpans {
  Resource resource = 1;
  repeated ScopeSpans scope_spans = 2;
  string schema_url = 3;
}

message ScopeSpans {
  InstrumentationScope scope = 1;
  repeated Span spans = 2;
  string schema_url = 3;
}
```

MLflow already bundles these proto definitions at `mlflow/protos/opentelemetry/proto/trace/v1/trace.proto` and has full conversion support via `Span.to_otel_proto()` / `Span.from_otel_proto()`.

## Appendix B: Comparison of Storage Formats

| Format                         | Size (relative) | Serialization Speed | Human Readable | OTel Standard            | MLflow Support                     |
| :----------------------------- | :-------------- | :------------------ | :------------- | :----------------------- | :--------------------------------- |
| JSON (current `spans.content`) | 1.0x (baseline) | Moderate            | Yes            | Partial                  | Full                               |
| OTLP Protobuf binary           | ~0.15–0.3x      | Fast                | No             | Yes (`TracesData`)       | Full (existing proto + conversion) |
| OTLP JSON Lines                | ~0.9–1.1x       | Moderate            | Yes            | Yes (file exporter spec) | Straightforward to add             |

Protobuf binary offers the best combination of size reduction, speed, standards compliance, and existing MLflow support.

**Future enhancement: compression.** Applying gzip or zstd compression on top of protobuf binary could yield an additional 2–5x reduction for repetitive trace data (e.g., repeated attribute keys, similar payloads). MLflow's OTLP endpoint already supports gzip/deflate decompression (`mlflow/tracing/utils/otlp.py`), and most artifact store backends (S3, GCS) support server-side compression. This could be added as a configuration option in a future iteration.

## Appendix C: Impact on Existing MLflow Components

| Component            | Impact   | Changes Required                                                                                                                                                                                                 |
| :------------------- | :------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SqlAlchemyStore`    | Moderate | Add `SpansLocation` tag awareness to `get_trace`, `batch_get_traces`; add `_get_traces_repository_uri`, `_load_spans_from_repository`; add archival methods                                                      |
| `FileStore`          | Low      | Archive method writes to trace repository; retrieval checks trace repository location                                                                                                                            |
| `AbstractStore`      | Low      | Add `archive_traces` abstract method                                                                                                                                                                             |
| `mlflow traces` CLI  | Low      | Add `archive` subcommand with `--older-than`, `--max-db-size`, and workspace targeting: `--workspace <name>` or `--all-workspaces` (mirroring `mlflow gc`); fail when workspaces are enabled and neither is set. |
| `mlflow server` CLI  | Low      | Add `--trace-archival-location` option                                                                                                                                                                           |
| REST API handlers    | Low      | Add new endpoint (e.g. `POST /mlflow/traces/archive-traces`) so that archival is invoked by the client and executed server-side; handler calls tracking store's archive method.                                  |
| Proto definitions    | Low      | Extend `SpansLocation` enum (TRACES_REPO); trace tags already expose `mlflow.trace.spansLocation` in `TraceInfoV3`                                                                                               |
| DB migrations        | Low      | Alembic migration: add `content_size` column on `spans`; add `traces_destination` (TEXT, nullable) on `workspaces` for per-workspace trace repository override.                                                  |
| `search_traces`      | Low      | Span-level filters naturally exclude archived traces (empty `content`); add `has_excluded_archived_traces` flag to response                                                                                      |
| Python client        | None     | `get_trace` / `search_traces` APIs unchanged                                                                                                                                                                     |
| Workspace Python API | Low      | Add optional `traces_destination` parameter to `mlflow.create_workspace()` and `mlflow.update_workspace()` (and `MlflowClient` equivalents) for per-workspace trace repository override.                         |
| Workspace REST API   | Low      | Add optional `traces_destination` field to CreateWorkspace and UpdateWorkspace request/response messages and handlers.                                                                                           |
| UI (workspaces)      | Low      | Add optional "Traces destination" (or equivalent) field to create-workspace and edit-workspace flows, analogous to default artifact root.                                                                        |
| UI                   | None     | Trace detail view works transparently                                                                                                                                                                            |

---

## Open Questions

### Dropping size-based archival (time-based only)

**Current proposal:** The design supports both time-based archival (e.g. archive traces older than N days) and size-based archival (e.g. keep total span content in the DB under a cap, archive oldest first). Size-based policy requires aggregating per-trace span size (via a `content_size` column on `spans` or `SUM(LENGTH(content))`) and joining with `trace_info` / `trace_tags`, which is heavier than time-based (indexed filter on `timestamp_ms`).

**Alternative:** Support only **time-based** archival (e.g. `--older-than` / `MLFLOW_TRACE_ARCHIVAL_OLDER_THAN`). Drop the size-based policy and related config (`--max-db-size`, `MLFLOW_TRACE_ARCHIVAL_MAX_DB_SIZE_MB`).

**Pros of dropping size-based archival:**

- **Lower resource usage:** No need to compute or store per-span size; no heavy aggregation or join to select candidates. Time-based archival is a simple filter on `trace_info.timestamp_ms`.
- **No schema change for size:** The `content_size` column on `spans` is only needed for size-based policy. Dropping it avoids the migration and backfill; the only schema change would be the extended `SpansLocation` tag (no new columns).
- **Clearer semantics:** "Archive traces older than 90 days" is unambiguous. "Max DB size" can be misinterpreted as total table/database size (e.g. all tables) rather than only trace/span content size, leading to confusion or misconfiguration.
- **Simpler implementation and docs:** Fewer options, fewer edge cases (e.g. size estimation, units), and easier operator mental model.

**Cons / trade-offs:**

- **Less control for size-constrained deployments:** Users who care primarily about keeping DB size under a fixed budget (e.g. "keep only the latest 10 GB of span data") would need to translate that into a time window (e.g. "older than N days") or run ad-hoc archival; they lose a direct "cap size" knob.
- **Time-based may over- or under-archive:** If trace volume or size per trace varies a lot, a time-based policy might leave too much or too little data in the DB compared to a size target; size-based policy directly addresses storage cost.
- **Feature parity:** Some users may expect both policies (as in the current proposal); dropping size-based could be a follow-up or optional phase 2 to reduce scope for an initial release.

### Two repositories: separate trace repository vs. single (artifact) repository

**Current proposal:** The design allows a **separate** trace repository (e.g. `--trace-archival-location` / `MLFLOW_TRACE_ARCHIVAL_LOCATION`), which may be the same as or different from the default artifact root. Archived trace span data is written under this trace repository (e.g. `<traces-root>/<experiment_id>/traces/<trace_id>/artifacts/traces.pb`).

**Alternative:** Use a **single** repository: do not support a separate traces destination. Archive trace span data into the existing artifact repository (e.g. under the experiment's `artifact_location`, in a path such as `traces/<trace_id>/artifacts/traces.pb`). Reuse the same storage backend and configuration as run artifacts.

**Pros of having a separate trace repository (two repositories):**

- **Dedicated server credentials for archival:** In typical MLflow deployments the server may not have artifact repository credentials when not running in proxy mode, since clients upload artifacts directly. Trace archival, however, must be performed by the server and therefore requires server-side write access to the trace store. A separate trace repository allows admins to configure credentials specifically for traces (e.g. a service account with write access only to the trace bucket), without granting the server access to the artifact repository.
- **Isolation of trace volume:** Trace archival can be very high volume; a dedicated trace repository allows quotas, lifecycle rules, and cost attribution to be applied to traces independently of run artifacts. Operators can treat trace storage differently (e.g. different retention or storage class) without affecting run artifacts.
- **Flexibility for scale:** Large tracing deployments can use a dedicated bucket, region, or IAM for traces (e.g. different region, different team/project). The artifact root can stay focused on run artifacts.
- **Clear separation of concerns:** Run artifacts and archived trace data have distinct access patterns and lifecycle; a separate root makes it explicit and simplifies reasoning about capacity and policies per data type.
- **Optional colocation:** If desired, the traces destination can still be set to the same URI as the artifact root (one logical store, two config knobs); the design supports both separate and shared storage.

**Cons of having a separate trace repository (two repositories):**

- **More configuration:** Operators must configure and maintain an additional destination (`--trace-archival-location` or `MLFLOW_TRACE_ARCHIVAL_LOCATION`). Two env vars / CLI options instead of one for storage.
- **Higher implementation surface:** Code paths for writing/reading archived span data must resolve and use the trace repository; defaulting or falling back to the artifact root when traces destination is unset adds behavior to document and maintain.
