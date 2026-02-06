# Implementation Plan: FileStore to SQLite Migration Tool

## Overview

This plan prioritizes **test data generation** and **verification** as the foundation for implementation. The data shapes the implementation, not the other way around.

---

## Out of Scope

The following are explicitly **not** covered by this migration tool:

| Item                        | Reason                                                                 |
| --------------------------- | ---------------------------------------------------------------------- |
| PostgreSQL, MySQL, MSSQL    | SQLite-only initially; other backends can be added later if requested  |
| Incremental migration       | Full migration only; simplifies implementation and verification        |
| Artifact file copying       | Artifacts stay in place; only URI references are migrated              |
| Span migration to DB        | Spans remain as artifacts due to DB constraint risks                   |
| Non-empty target databases  | Requires empty target to avoid conflicts; simplifies failure handling  |
| Reverse migration (DB→File) | One-way migration; FileStore is being deprecated                       |
| Server-to-server migration  | Use mlflow-export-import for that use case                             |

---

## 1. Test Data Generation Strategy

### 1.1 MLflow Versions to Test

| Version          | Why                                        | Data Types Available     |
| ---------------- | ------------------------------------------ | ------------------------ |
| **2.x (latest)** | Last 2.x release before 3.0                | Core + Datasets + Traces |
| **3.2.0**        | First with assessments, has all data types | All data types           |

### 1.2 Data Generation Commands

#### Core Data (All Versions)

```python
import mlflow

# Experiments
exp_id = mlflow.create_experiment("test_experiment", tags={"team": "ml"})

# Runs with params, metrics, tags
with mlflow.start_run(experiment_id=exp_id):
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_metric("accuracy", 0.95, step=100)
    mlflow.set_tag("model_type", "sklearn")

# Deleted experiment (goes to .trash)
mlflow.delete_experiment(exp_id)
```

#### Datasets & Inputs (2.10.0+)

```python
import mlflow.data
import pandas as pd

dataset = mlflow.data.from_pandas(pd.DataFrame({"x": [1,2,3]}), name="train")
with mlflow.start_run():
    mlflow.log_input(dataset, context="training")
```

#### Traces (2.14.0+)

```python
from mlflow import MlflowClient

client = MlflowClient()
# Traces are typically created via auto-instrumentation or manually
# via client._start_tracked_trace() / client._end_trace()
```

#### Logged Models (3.0.0+)

```python
from mlflow import MlflowClient

client = MlflowClient()
model = client.create_logged_model(
    experiment_id=exp_id,
    name="my_model",
    tags=[LoggedModelTag("version", "1.0")],
    params=[LoggedModelParameter("n_estimators", "100")]
)
client.finalize_logged_model(model.model_id, LoggedModelStatus.READY)
```

#### Assessments (3.2.0+)

```python
from mlflow.entities import Feedback, AssessmentSource

feedback = Feedback(
    name="relevance",
    value=0.85,
    source=AssessmentSource(source_type="HUMAN", source_id="reviewer"),
    trace_id=trace.trace_id,
)
client.create_assessment(feedback)
```

#### Model Registry (All Versions)

```python
from mlflow import MlflowClient

client = MlflowClient()

# Create registered model
client.create_registered_model("my_model", tags={"team": "ml"}, description="A test model")

# Create model version (requires a logged model artifact)
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "model")
    model_uri = f"runs:/{run.info.run_id}/model"

client.create_model_version("my_model", model_uri, run_id=run.info.run_id)
client.set_model_version_tag("my_model", "1", "stage", "production")

# Create alias (2.3.0+)
client.set_registered_model_alias("my_model", "champion", "1")
```

#### Prompts (3.0.0+)

```python
import mlflow

# Create prompt
mlflow.register_prompt(
    name="qa_prompt",
    template="Answer the question: {{question}}",
    commit_message="Initial version",
)
```

### 1.3 Edge Cases to Generate

| Category            | Test Case                      | Command/Approach                                              |
| ------------------- | ------------------------------ | ------------------------------------------------------------- |
| **Large IDs**       | 18-digit experiment ID         | FileStore generates these via `_generate_unique_integer_id()` |
| **NaN/Inf Metrics** | `float('nan')`, `float('inf')` | `mlflow.log_metric("m", float('nan'))`                        |
| **Unicode**         | CJK, emoji in names/tags       | `mlflow.create_experiment("测试_🚀")`                         |
| **Deleted Items**   | Deleted exp/run                | `mlflow.delete_experiment()`, `mlflow.delete_run()`           |
| **Max Length**      | 8000 char param value          | `mlflow.log_param("k", "x"*8000)`                             |
| **Empty Runs**      | Run with no metrics/params     | `with mlflow.start_run(): pass`                               |

### 1.4 Test Data Generation with `uv`

Use `uv run` with version pinning to generate test data in CI (not committed to repo):

```bash
# Generate fixtures for 2.x (latest 2.x version)
uv run --with 'mlflow>=2,<3' --no-project python -I scripts/generate_migration_fixtures.py \
  --output /tmp/fixtures/v2.x/

# Generate fixtures for 3.2.0 (has all data types)
uv run --with mlflow==3.2.0 --no-project python -I scripts/generate_migration_fixtures.py \
  --output /tmp/fixtures/v3.2.0/
```

**Generator flags:**

```bash
# Quick iteration (minimal data, fast)
python scripts/generate_migration_fixtures.py --output /tmp/fixtures/ --size small

# Full test (comprehensive data)
python scripts/generate_migration_fixtures.py --output /tmp/fixtures/ --size full
```

| Size    | Use Case           |
| ------- | ------------------ |
| `small` | Quick iteration    |
| `full`  | CI / comprehensive |

**Entities to generate (24 total):**

| Entity                 | Parent           | small | full | Notes   |
| ---------------------- | ---------------- | ----- | ---- | ------- |
| Experiments            | -                | 2     | 5    |         |
| Experiment Tags        | Experiment       | 2     | 3    |         |
| Runs                   | Experiment       | 2     | 5    |         |
| Params                 | Run              | 3     | 10   |         |
| Metrics                | Run              | 3     | 10   |         |
| Latest Metrics         | Run              | 3     | 10   |         |
| Tags                   | Run              | 2     | 5    |         |
| Datasets               | Experiment       | 1     | 3    | 2.10.0+ |
| Inputs                 | Run              | 1     | 2    | 2.10.0+ |
| Input Tags             | Input            | 1     | 2    | 2.10.0+ |
| Traces                 | Experiment       | 1     | 3    | 2.14.0+ |
| Trace Tags             | Trace            | 1     | 3    | 2.14.0+ |
| Trace Request Metadata | Trace            | 1     | 3    | 2.14.0+ |
| Assessments            | Trace            | 1     | 2    | 3.2.0+  |
| Logged Models          | Experiment       | 1     | 2    | 3.0.0+  |
| Logged Model Params    | Logged Model     | 2     | 5    | 3.0.0+  |
| Logged Model Tags      | Logged Model     | 1     | 3    | 3.0.0+  |
| Logged Model Metrics   | Logged Model     | 2     | 5    | 3.0.0+  |
| Registered Models      | -                | 1     | 3    |         |
| Model Versions         | Registered Model | 2     | 5    |         |
| Registered Model Tags  | Registered Model | 1     | 3    |         |
| Model Version Tags     | Model Version    | 1     | 3    |         |
| Model Aliases          | Registered Model | 1     | 2    | 2.3.0+  |
| Prompts                | -                | 1     | 3    | 3.0.0+  |

**Not migrated (stay as artifacts):** Spans

**uv flags:**

- `--with 'mlflow>=2,<3'` - Install latest 2.x version
- `--with mlflow==3.2.0` - Pin exact version
- `--no-project` - Ignore local pyproject.toml
- `-I` - Isolated mode, ignore environment variables

**Dedicated CI workflow** (`.github/workflows/migration-tests.yml`):

```yaml
name: Migration Tool Tests

on:
  push:
  pull_request:
  workflow_dispatch:

jobs:
  migration-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        mlflow-version:
          - spec: ">=2,<3"
            name: v2.x
          - spec: "==3.2.0"
            name: v3.2.0
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4

      - name: Generate fixtures for MLflow ${{ matrix.mlflow-version.name }}
        run: |
          uv run --with 'mlflow${{ matrix.mlflow-version.spec }}' --no-project python -I \
            scripts/generate_migration_fixtures.py --output /tmp/fixtures/${{ matrix.mlflow-version.name }}/ --size full

      - name: Run migration tests
        run: |
          uv run pytest tests/store/tracking/migration/ \
            --fixtures-dir /tmp/fixtures/${{ matrix.mlflow-version.name }}
```

---

## 2. Verification Strategy

### 2.1 What to Verify (Field-by-Field)

#### Experiments

| Field               | Comparison    | Notes                                    |
| ------------------- | ------------- | ---------------------------------------- |
| `experiment_id`     | Exact match   | Critical: 18-digit IDs must be preserved |
| `name`              | Exact match   | Including unicode                        |
| `artifact_location` | Exact match   | Absolute path preserved                  |
| `lifecycle_stage`   | Exact match   | "active" or "deleted"                    |
| `creation_time`     | Exact match   | Milliseconds                             |
| `last_update_time`  | Exact match   | Milliseconds                             |
| `tags`              | Dict equality | Key-value pairs                          |

#### Runs

| Field             | Comparison  | Notes                          |
| ----------------- | ----------- | ------------------------------ |
| `run_id`          | Exact match | UUID preserved                 |
| `experiment_id`   | Exact match | Foreign key                    |
| `run_name`        | Exact match |                                |
| `user_id`         | Exact match |                                |
| `status`          | Exact match | RUNNING/FINISHED/FAILED/KILLED |
| `start_time`      | Exact match | Milliseconds                   |
| `end_time`        | Exact match | Milliseconds, can be None      |
| `lifecycle_stage` | Exact match | "active" or "deleted"          |
| `artifact_uri`    | Exact match | Path preserved                 |

#### Metrics

| Field       | Comparison  | Notes                             |
| ----------- | ----------- | --------------------------------- |
| `key`       | Exact match |                                   |
| `value`     | Special     | See floating-point handling below |
| `timestamp` | Exact match | Milliseconds                      |
| `step`      | Exact match | Integer                           |

**Floating-point comparison:**

```python
def compare_metric_values(src: float, dst: float) -> bool:
    # NaN: both must be NaN
    if math.isnan(src) and math.isnan(dst):
        return True
    # Inf: same sign infinity
    if math.isinf(src) and math.isinf(dst):
        return (src > 0) == (dst > 0)
    # Regular: use tolerance
    return math.isclose(src, dst, rel_tol=1e-9)
```

#### Params & Tags

| Field   | Comparison  | Notes             |
| ------- | ----------- | ----------------- |
| `key`   | Exact match |                   |
| `value` | Exact match | String comparison |

#### Traces (2.14.0+)

| Field               | Comparison    | Notes                |
| ------------------- | ------------- | -------------------- |
| `trace_id`          | Exact match   |                      |
| `experiment_id`     | Exact match   |                      |
| `timestamp_ms`      | Exact match   |                      |
| `execution_time_ms` | Exact match   |                      |
| `status`            | Exact match   | OK/ERROR/IN_PROGRESS |
| `tags`              | Dict equality |                      |
| `request_metadata`  | Dict equality |                      |

#### Assessments (3.2.0+)

| Field           | Comparison    | Notes                          |
| --------------- | ------------- | ------------------------------ |
| `assessment_id` | Exact match   |                                |
| `name`          | Exact match   |                                |
| `value`         | JSON equality | Can be float, bool, or complex |
| `source_type`   | Exact match   | HUMAN/CODE/LLM_JUDGE           |
| `source_id`     | Exact match   |                                |

#### Logged Models (3.0.0+)

| Field           | Comparison    | Notes |
| --------------- | ------------- | ----- |
| `model_id`      | Exact match   |       |
| `experiment_id` | Exact match   |       |
| `name`          | Exact match   |       |
| `status`        | Exact match   |       |
| `params`        | List equality |       |
| `tags`          | List equality |       |

#### Registered Models

| Field               | Comparison    | Notes                       |
| ------------------- | ------------- | --------------------------- |
| `name`              | Exact match   | Primary key                 |
| `creation_time`     | Exact match   | Milliseconds                |
| `last_updated_time` | Exact match   | Milliseconds                |
| `description`       | Exact match   |                             |
| `tags`              | Dict equality |                             |
| `aliases`           | Dict equality | alias_name → version number |

#### Model Versions

| Field               | Comparison  | Notes                                  |
| ------------------- | ----------- | -------------------------------------- |
| `name`              | Exact match | Registered model name                  |
| `version`           | Exact match | Integer version number                 |
| `creation_time`     | Exact match | Milliseconds                           |
| `last_updated_time` | Exact match | Milliseconds                           |
| `description`       | Exact match |                                        |
| `user_id`           | Exact match |                                        |
| `source`            | Exact match | Artifact path                          |
| `run_id`            | Exact match | Can be None                            |
| `status`            | Exact match | PENDING_REGISTRATION/READY/FAILED/etc. |
| `status_message`    | Exact match |                                        |
| `run_link`          | Exact match |                                        |
| `tags`              | Dict equality |                                      |

#### Prompts (3.0.0+)

| Field          | Comparison  | Notes            |
| -------------- | ----------- | ---------------- |
| `name`         | Exact match | Primary key      |
| `creation_time`| Exact match | Milliseconds     |
| `description`  | Exact match |                  |
| `tags`         | Dict equality |                |

### 2.2 Verification Implementation

```python
@dataclass
class VerificationResult:
    entity_type: str  # "experiment", "run", "metric", etc.
    entity_id: str
    passed: bool
    mismatches: list[FieldMismatch]

@dataclass
class FieldMismatch:
    field: str
    source_value: Any
    target_value: Any

def verify_migration(source_path: str, target_uri: str) -> VerificationReport:
    """Main verification entry point."""
    source = FileStore(source_path)
    target = SqlAlchemyStore(target_uri)

    results = []

    # 1. Verify experiments
    src_exps = {e.experiment_id: e for e in source.search_experiments(view_type=ViewType.ALL)}
    dst_exps = {e.experiment_id: e for e in target.search_experiments(view_type=ViewType.ALL)}

    for exp_id, src_exp in src_exps.items():
        dst_exp = dst_exps.get(exp_id)
        results.append(verify_experiment(src_exp, dst_exp))

    # 2. Verify runs (with params, metrics, tags)
    # 3. Verify traces
    # 4. Verify assessments
    # 5. Verify logged models

    return VerificationReport(results)
```

### 2.3 Verification Mode

Built-in flags:

```bash
# Basic migration
mlflow migrate-filestore --source ./mlruns --target sqlite:///mlflow.db

# With verification
mlflow migrate-filestore --source ./mlruns --target sqlite:///mlflow.db --verify

# Dry-run (preview without writing)
mlflow migrate-filestore --source ./mlruns --target sqlite:///mlflow.db --dry-run
```

| Flag | Description |
|------|-------------|
| `--verify` | Run verification after migration, fail if mismatches found |
| `--dry-run` | Preview what would be migrated without writing to target |
| `--verbose` | Show detailed progress (quiet by default for CI) |

### 2.4 Verification Report Format

```
Migration Verification Report
=============================

Experiments: 5/5 passed
Runs: 12/12 passed
Metrics: 48/48 passed
Params: 24/24 passed
Tags: 36/36 passed
Traces: 3/3 passed
Assessments: 6/6 passed
Logged Models: 2/2 passed
Registered Models: 3/3 passed
Model Versions: 8/8 passed
Prompts: 3/3 passed

Status: SUCCESS
```

On failure:

```
FAILED: Run abc123
  - Field 'end_time': source=1234567890, target=None

FAILED: Metric 'loss' in run xyz789
  - Field 'value': source=0.05, target=0.050000001
```

---

## 3. Implementation Tasks

### Phase 1: Test Infrastructure (Do First)

1. **Create test data generation script**
   - `scripts/generate_migration_fixtures.py`
   - Generates all entity types with edge cases
   - Deterministic (fixed seeds for reproducibility)
   - Must work standalone (no local project dependencies)

2. **Create verification module**
   - `mlflow/store/tracking/migration/verification.py`
   - Field-by-field comparison for all entity types
   - Report generation with clear mismatch details

### Phase 2: Core Migration

3. **Implement migration for core entities**
   - Experiments, Runs, Params, Metrics, Tags
   - Use SQLAlchemy models directly (like prototype)

4. **Add verification tests**
   - Test against each fixture directory
   - Verify all fields match exactly

### Phase 3: Extended Entities

5. **Implement migration for newer entities**
   - Datasets, Inputs (2.10.0+)
   - Traces, TraceMetadata, TraceTags (2.14.0+)
   - Assessments (3.2.0+)
   - LoggedModels, LoggedModelParams/Tags/Metrics (3.0.0+)
   - RegisteredModels, ModelVersions, ModelAliases (All + 2.3.0+)
   - Prompts (3.0.0+)

6. **Add version-aware tests**
   - Each fixture version tests only its available features

### Phase 4: CLI Integration

7. **Add CLI command**
   - `mlflow migrate-filestore --source --target [--verify]`
   - Follow existing CLI patterns in `mlflow/cli/`

---

## 4. Key Files to Modify/Create

### New Files

- `scripts/generate_migration_fixtures.py` - Test data generator (standalone, version-agnostic)
- `mlflow/store/tracking/migration/__init__.py`
- `mlflow/store/tracking/migration/filestore_to_sqlite.py` - Main migration logic
- `mlflow/store/tracking/migration/verification.py` - Verification logic
- `tests/store/tracking/migration/test_migration.py`
- `tests/store/tracking/migration/test_verification.py`
- `tests/store/tracking/migration/conftest.py` - Pytest fixtures for CI-generated data

### Files to Modify

- `mlflow/cli/__init__.py` - Add `migrate-filestore` command
- `mlflow/db.py` - Or create new `mlflow/migrate.py` for migration commands

### Reference Files (Read-Only)

- `mlflow/store/tracking/file_store.py` - Source store (2,884 lines)
- `mlflow/store/tracking/sqlalchemy_store.py` - Target store patterns (6,093 lines)
- `mlflow/store/tracking/dbmodels/models.py` - SQLAlchemy models (2,639 lines)
- `scripts/migrate_filestore_prototype.py` - Existing prototype (113 lines)

---

## 5. Documentation

Create a user-facing migration guide at `docs/source/tracking/filestore-migration.rst`:

### Outline

1. **Why migrate?** - FileStore deprecation notice, benefits of SQLite
2. **Prerequisites** - Backup recommendation, MLflow version requirement
3. **Quick start** - Single command example
4. **CLI reference** - All flags documented
5. **What gets migrated** - List of entity types
6. **What stays in place** - Artifacts (only URIs updated)
7. **Verification** - How to confirm migration succeeded
8. **Troubleshooting** - Common issues and solutions
9. **FAQ** - Can I migrate back? What if I have multiple mlruns directories?

---

## 6. Design Decisions

| Question             | Decision                                           |
| -------------------- | -------------------------------------------------- |
| **Fixture storage**  | Generate in CI using `uv run --with mlflow==X.Y.Z` |
| **Failure handling** | Require empty target + transaction rollback        |
| **Verification**     | Built-in `--verify` flag on migrate command        |

### Failure Handling Details

```python
def migrate(source_path: str, target_uri: str, verify: bool = False):
    engine = create_engine(target_uri)

    # Check target is empty
    with engine.connect() as conn:
        exp_count = conn.execute(text("SELECT COUNT(*) FROM experiments")).scalar()
        if exp_count > 0:
            raise MlflowException("Target database is not empty. Migration requires empty target.")

    # Use transaction for atomic migration
    with Session(engine) as session:
        try:
            # ... migrate all entities ...
            session.commit()
        except Exception as e:
            session.rollback()
            raise MlflowException(f"Migration failed, rolled back: {e}")

    if verify:
        report = verify_migration(source_path, target_uri)
        if not report.is_success():
            raise MlflowException(f"Verification failed: {report}")
```
