# FileStore Migration Tool - Proposal

Author: @harupy

> 🤖 This document was created with AI assistance (Claude Code).

## Context

GitHub Issue: https://github.com/mlflow/mlflow/issues/18534

MLflow is deprecating the filesystem backend (`./mlruns`) in favor of database backends. Users need a migration path to convert existing FileStore data to a database (SQLite, PostgreSQL, or MySQL).

## Existing Solution

https://github.com/mlflow/mlflow-export-import

See also: [ARCHITECTURE_mlflow_export_import.md](./ARCHITECTURE_mlflow_export_import.md)

An existing tool for copying MLflow objects between tracking servers. It exports data to files, then **recreates entities** on the target server via REST API.

```
Source Server → [Export to files] → [Recreate entities] → Target Server
```

#### What it does

- Supports experiments, runs, models, traces, prompts
- Works via REST API (creates new entities, not direct migration)

#### Issues for FileStore migration

- Doesn't preserve original IDs and timestamps ([ref](https://github.com/mlflow/mlflow/issues/18534#issuecomment-3772629460))
- Complex workflow (requires running servers, 4 commands)

## Comparison Matrix

| Criteria           | Option 1 (Use export-import) | Option 2 (New MLflow tool) ✅ |
| ------------------ | ---------------------------- | ----------------------------- |
| Preserve data      | No (new IDs/timestamps)      | Yes                           |
| Commands needed    | 4                            | 1                             |
| Learning cost\*    | High (new tool + multi-step) | Low (single command)          |
| Development effort | None                         | Medium\*\*                    |
| Performance        | Medium (API overhead)        | High (direct)                 |
| Maintenance        | External repo                | MLflow core (internal utils)  |
| Dependency         | External package             | None                          |

- \*For most users, this migration is a new process they haven't done before.
- \*\*Clear success criteria (`original data == migrated data`) enables AI-assisted development.

## Requirements

### Functional

- Migrate all FileStore data types (experiments, runs, params, metrics, tags, etc.)
- Preserve original run UUIDs and timestamps
- Preserve original experiment IDs
- Migrate deleted items (preserve complete history)
- Support SQLite initially (other backends like PostgreSQL, MySQL, MSSQL can be added later if requested)
- Provide verification that migrated data matches source data

### Non-Functional

- Single command execution (no multi-step workflow)
- Idempotent (can re-run to resume after failure)
- Progress reporting for large migrations
- Clear error messages for constraint violations or data issues
- Target: small to medium datasets (< 100K runs). FileStore doesn't scale well, so users with large datasets likely already use a database backend.

## Recommendation

**Option 2 (New MLflow tool)** - A built-in CLI command provides the best user experience with no external dependencies. Users can migrate with a single command without installing additional packages.

## Design Decisions

| Question              | Decision                                               |
| --------------------- | ------------------------------------------------------ |
| Artifacts handling    | Keep in place (only update URI references in database) |
| Incremental migration | No, full migration only                                |

## Artifact Location

FileStore stores artifacts at `mlruns/<exp_id>/<run_id>/artifacts/` with absolute path URIs. During migration:

- Copy original URIs - the absolute paths work with any backend
- Artifacts stay in place - no file copying needed
- No storage changes - users continue accessing artifacts from the same location

If users move the `mlruns` directory after migration, artifact URIs will break (same behavior as before migration).

## Constraint Analysis

Since SQLite is the only supported target in the initial version, most constraint risks are mitigated by SQLite's flexible type system.

### Rationale for SQLite-Only Initial Support

Supporting only SQLite initially allows us to **preserve original experiment IDs**. FileStore generates 18-digit experiment IDs via `_generate_unique_integer_id()`, which exceed the 32-bit integer limit (~2.1 billion) enforced by PostgreSQL, MySQL, and MSSQL. SQLite's dynamic typing handles large integers natively, so original IDs are preserved without modification.

**Verified:** The prototype successfully migrated an 18-digit experiment ID (`726979332293725066`) to SQLite and confirmed the ID was preserved exactly.

### Low Risk

- **Integer overflow**: FileStore generates 18-digit experiment IDs, but SQLite handles large integers due to dynamic typing (no strict 32-bit limit). Original experiment IDs are preserved.
- **Auto-increment**: `experiment_id` is auto-increment. SQLite handles explicit inserts gracefully.
- **Check (enums)**: Invalid `lifecycle_stage`, `status`, `source_type` values. FileStore validates enum values consistently during write.
- **String length**: Values exceeding column limits (e.g., 256 for experiment name). FileStore validates and truncates on write.
- **NOT NULL**: Missing required values (experiment name, run_uuid, etc.). FileStore requires these fields in its structure.
- **Primary key**: Duplicate entries for tags, params. FileStore enforces uniqueness via file system.
- **Foreign key**: Orphaned runs, metrics, tags without parent entities. FileStore stores entities hierarchically (run data inside run dir).

## Open Questions

1. What if users want to stay on an older MLflow version (e.g. 2.x) and can't upgrade to use the new tool?
2. Version compatibility: FileStore data may come from older MLflow versions with different field structures or missing fields (e.g., the run `name` field was added later). The migration tool should handle schema differences gracefully, using sensible defaults for missing fields.
3. Migrating to existing DB: Should we support migrating to a database that already has data? This could cause conflicts (e.g., experiment name collisions). Options: require empty target DB, or provide conflict resolution strategies.

---

## Data to Migrate

| Data Type       | FileStore                              | Database Tables                                         | Since  |
| --------------- | -------------------------------------- | ------------------------------------------------------- | ------ |
| Experiments     | `<exp_id>/meta.yaml`                   | `experiments`                                           | All    |
| Experiment Tags | `<exp_id>/tags/<key>`                  | `experiment_tags`                                       | All    |
| Runs            | `<exp_id>/<run_id>/meta.yaml`          | `runs`                                                  | All    |
| Params          | `<run_id>/params/<key>`                | `params`                                                | All    |
| Metrics         | `<run_id>/metrics/<key>`               | `metrics`, `latest_metrics`                             | All    |
| Tags            | `<run_id>/tags/<key>`                  | `tags`                                                  | All    |
| Datasets        | `<exp_id>/datasets/<id>/meta.yaml`     | `datasets`                                              | 2.10.0 |
| Inputs          | `<run_id>/inputs/<id>/meta.yaml`       | `inputs`, `input_tags`                                  | 2.10.0 |
| Traces          | `<exp_id>/traces/<id>/trace_info.yaml` | `trace_info`, `trace_tags`, `trace_request_metadata`    | 2.14.0 |
| Assessments     | `traces/<id>/assessments/<id>.yaml`    | `assessments`                                           | 3.2.0  |
| Logged Models   | `<exp_id>/models/<id>/meta.yaml`       | `logged_models`, `logged_model_{params\|metrics\|tags}` | 3.0.0  |

## Options

### Option 1: Use mlflow-export-import

Use the existing mlflow-export-import tool with its current export → import workflow.

#### How it works

```
FileStore (./mlruns) → [Export to files] → [Import via REST API] → Database
```

#### Pros

- Already exists and is well-tested
- Handles all MLflow object types

#### Cons

- Doesn't preserve original IDs and timestamps (recreates entities)
- Requires running MLflow servers
- Multi-step workflow
- External dependency (not part of core MLflow)

#### Example (4 commands)

```bash
# 1. Start source server
mlflow server --backend-store-uri ./mlruns --port 5000

# 2. Export data
export-all --output-dir /tmp/export

# 3. Start target server
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5001

# 4. Import data
MLFLOW_TRACKING_URI=http://localhost:5001 import-all --input-dir /tmp/export
```

---

### Option 2: New Standalone Tool in MLflow ✅ Recommended

Create a new migration tool built into MLflow core.

#### How it works

```
FileStore (./mlruns) → [mlflow migrate-filestore] → SQLite database
```

#### Pros

- Direct file-to-database conversion (no intermediate steps)
- No external dependencies
- Can leverage internal MLflow store implementations
- Ships with MLflow - always available
- Can be optimized for batch operations
- No changes needed to mlflow-export-import

#### Cons

- New code to write and maintain
- Duplicates some logic from mlflow-export-import
- Requires new MLflow release to ship bug fixes (can be mitigated by patch releases)

#### Implementation

- Read FileStore structure directly
- Use SQLAlchemy models to write to database
- Batch inserts for performance

#### Example

```bash
# 1. Upgrade MLflow (tool is only available in 3.<TBD>)
pip install --upgrade mlflow

# 2. Run migration (SQLite only in initial version)
mlflow migrate-filestore --source ./mlruns --target sqlite:///mlflow.db
```
