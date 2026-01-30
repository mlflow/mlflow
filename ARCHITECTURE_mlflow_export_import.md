# MLflow Export Import - Architecture Summary

https://github.com/mlflow/mlflow-export-import

## Overview

A Python package for copying MLflow objects (experiments, runs, models, traces, prompts) between MLflow tracking servers, including Databricks, open-source MLflow, and AWS SageMaker.

## Architecture

```
┌──────────────────────────────┐
│     Source MLflow Server     │
│  (FileStore, SQLite, etc.)   │
└──────────────┬───────────────┘
               │
               │ REST API
               │ (search_experiments, search_runs, ...)
               ▼
┌──────────────────────────────┐
│       export-experiment      │
│       export-model           │
│       export-all             │
└──────────────┬───────────────┘
               │
               │ Write to disk
               ▼
┌──────────────────────────────┐
│      Export Directory        │
│  ┌────────────────────────┐  │
│  │ experiment.json        │  │
│  │ runs/                  │  │
│  │   └─ <run_id>/         │  │
│  │       ├─ run.json      │  │
│  │       └─ artifacts/    │  │
│  └────────────────────────┘  │
└──────────────┬───────────────┘
               │
               │ Read from disk
               ▼
┌──────────────────────────────┐
│       import-experiment      │
│       import-model           │
│       import-all             │
└──────────────┬───────────────┘
               │
               │ REST API
               │ (create_experiment, create_run, ...)
               │ ⚠️  NEW IDs generated
               ▼
┌──────────────────────────────┐
│     Target MLflow Server     │
│  (SQLite, PostgreSQL, etc.)  │
└──────────────────────────────┘
```

## Supported Objects

| Object                    | Min MLflow Version |
| ------------------------- | ------------------ |
| Experiments, Runs, Models | All                |
| Traces                    | 2.14.0+            |
| Prompts                   | 2.21.0+            |
| Logged Models             | 3.0.0+             |

## Module Structure

```
┌─────────────────────────────────────────────────────┐
│                   CLI Commands                      │
│  ─────────────────────────────────────────────────  │
│  • Single: export-experiment, import-run, etc.      │
│  • Bulk: export-all, import-all                     │
│  • Copy: copy-run, copy-model-version               │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                   Core Modules                      │
│  ─────────────────────────────────────────────────  │
│  experiment/  run/  model/  trace/  prompt/         │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                     Support                         │
│  ─────────────────────────────────────────────────  │
│  • client/ - HTTP & MLflow clients                  │
│  • common/ - Utils, iterators, IO                   │
└─────────────────────────────────────────────────────┘
```

## Export File Format

```
export_dir/
├── experiment.json      # Metadata + system info
└── runs/
    └── {run_id}/
        ├── run.json     # Params, metrics, tags
        └── artifacts/   # Model files, etc.
```

## Key Features

- **Two modes**: Single object or bulk operations
- **Time-based filtering**: Incremental exports via `--run-start-time`
- **Governance**: Source tags (`mlflow_exim.*`) preserve lineage
- **Permissions**: Export/import Databricks ACLs
- **Multithreading**: Parallel processing support
- **Nested runs**: Preserves parent-child relationships

## What Gets Preserved

| Data                     | Preserved? | Notes                                     |
| ------------------------ | ---------- | ----------------------------------------- |
| Run IDs                  | No         | New IDs generated; mapping maintained     |
| Experiment IDs           | No         | Determined by destination experiment name |
| Model Version IDs        | No         | Auto-incremented in destination           |
| Run start/end time       | No         | Set when new run is created               |
| Experiment creation time | No         | Set when new experiment is created        |
| Model version timestamps | No         | Set when new version is created           |
| Metric timestamps        | Yes        | Exact timestamps preserved                |
| Metric step values       | Yes        | Step history preserved                    |
| Parent-child relations   | Yes        | Re-established after import               |
| Artifacts                | Yes        | Fully copied                              |
| Params, metrics, tags    | Yes        | Fully copied                              |

Use `--import-source-tags` to capture original IDs/timestamps as `mlflow_exim.*` tags for lineage tracking.

## Common CLI Usage

```bash
# Export experiment
export-experiment --experiment my_exp --output-dir /tmp/export

# Import to another server
export MLFLOW_TRACKING_URI=http://target:5000
import-experiment --experiment-name my_exp --input-dir /tmp/export

# Bulk export entire server
export-all --output-dir /tmp/backup

# Direct copy between servers
copy-run --run-id abc123 --src-mlflow-uri http://src:5000 --dst-mlflow-uri http://dst:5000
```

## Key Options

| Option                     | Purpose                                            |
| -------------------------- | -------------------------------------------------- |
| `--import-source-tags`     | Preserve original metadata as tags                 |
| `--export-permissions`     | Include Databricks ACLs                            |
| `--stages`                 | Filter models by stage (Production, Staging, etc.) |
| `--use-threads`            | Enable parallel processing                         |
| `--experiment-rename-file` | CSV mapping for renaming during import             |
