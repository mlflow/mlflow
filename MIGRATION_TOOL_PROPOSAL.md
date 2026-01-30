# FileStore to SQLite Migration Tool - Proposal

> 🤖 This document was created with AI assistance (Claude Code).

## Context

GitHub Issue: https://github.com/mlflow/mlflow/issues/18534

MLflow is deprecating the filesystem backend (`./mlruns`) in favor of SQLite (`sqlite:///mlflow.db`). Users need a migration path to convert existing file store data to SQLite.

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

## Recommendation

**Option 2 (New MLflow tool)** - A built-in CLI command provides the best user experience with no external dependencies. Users can migrate with a single command without installing additional packages.

## Design Decisions

| Question                  | Decision                                                                                                                         |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Artifacts handling        | Keep in place (only update URI references in database)                                                                           |
| Incremental migration     | No, full migration only                                                                                                          |
| Failure handling          | Idempotent (skip existing items, re-run to continue)                                                                             |
| Deleted items in `.trash` | Skip (only migrate active data)                                                                                                  |
| Performance target        | Small to medium (< 100K runs). FileStore doesn't scale well, so users with large datasets likely already use a database backend. |

## Artifact Location

FileStore stores artifacts at `mlruns/<exp_id>/<run_id>/artifacts/` with absolute path URIs. During migration:

- Copy original URIs - the absolute paths work with any backend
- Artifacts stay in place - no file copying needed
- No storage changes - users continue accessing artifacts from the same location

If users move the `mlruns` directory after migration, artifact URIs will break (same behavior as before migration).

## Constraint Analysis

| Constraint    | Issue                          | Why it's safe                              |
| ------------- | ------------------------------ | ------------------------------------------ |
| String length | Values exceeding column limits | FileStore validates and truncates on write |
| Uniqueness    | Duplicate entries              | FileStore also enforces uniqueness         |
| NOT NULL      | Missing required values        | FileStore requires these fields too        |

Note: This is not an exhaustive analysis; other constraint violations may exist.

## Open Questions

1. Target database support - SQLite only, or support PostgreSQL/MySQL too? (Recommendation: Start with SQLite, extend later if needed)
2. What if users want to stay on an older MLflow version (e.g. 2.x) and can't upgrade to use the new tool?

---

## Data to Migrate

| Data Type       | FileStore                              | SQLite Tables                                           | Since  |
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
FileStore (./mlruns) → [Export to files] → [Import via REST API] → SQLite (mlflow.db)
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
FileStore (./mlruns) → [mlflow db migrate] → SQLite (mlflow.db)
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

#### Implementation

- Read FileStore structure directly
- Use SQLAlchemy models to write to database
- Batch inserts for performance

#### Example

```bash
# 1. Upgrade MLflow (tool is only available in 3.<TBD>)
pip install --upgrade mlflow

# 2. Run migration
mlflow db migrate --source ./mlruns --target sqlite:///mlflow.db
```
