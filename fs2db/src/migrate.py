# ruff: noqa: T201
"""
Migrate MLflow FileStore data to a SQLite database.

Usage:
    uv run python fs2db/src/migrate.py \
        --source /tmp/fs2db/v3.5.1/ \
        --target sqlite:////tmp/migrated.db
"""

import argparse
import json
import math
import os
import uuid
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from mlflow.entities import RunStatus
from mlflow.store.db.utils import _initialize_tables
from mlflow.store.model_registry.dbmodels.models import (
    SqlModelVersion,
    SqlModelVersionTag,
    SqlRegisteredModel,
    SqlRegisteredModelAlias,
    SqlRegisteredModelTag,
)
from mlflow.store.tracking.dbmodels.models import (
    SqlAssessments,
    SqlDataset,
    SqlExperiment,
    SqlExperimentTag,
    SqlInput,
    SqlInputTag,
    SqlLatestMetric,
    SqlLoggedModel,
    SqlLoggedModelMetric,
    SqlLoggedModelParam,
    SqlLoggedModelTag,
    SqlMetric,
    SqlParam,
    SqlRun,
    SqlTag,
    SqlTraceInfo,
    SqlTraceMetadata,
    SqlTraceTag,
)
from mlflow.utils.file_utils import read_file, read_file_lines, read_yaml

# ── Summary counter ──────────────────────────────────────────────────────────

summary: dict[str, int] = {}


def bump(key: str, n: int = 1) -> None:
    summary[key] = summary.get(key, 0) + n


# ── File reading helpers ─────────────────────────────────────────────────────

META_YAML = "meta.yaml"


def safe_read_yaml(root: str, file_name: str) -> dict[str, Any] | None:
    try:
        return read_yaml(root, file_name)
    except Exception:
        return None


def list_subdirs(path: str) -> list[str]:
    if not os.path.isdir(path):
        return []
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def list_files(path: str) -> list[str]:
    if not os.path.isdir(path):
        return []
    return sorted(f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))


def read_tag_files(tag_dir: str) -> dict[str, str]:
    result = {}
    if not os.path.isdir(tag_dir):
        return result
    for root, _, files in os.walk(tag_dir):
        for name in files:
            abspath = os.path.join(root, name)
            key = os.path.relpath(abspath, tag_dir)
            # Normalize path separators for metric/tag keys that contain '/'
            key = key.replace(os.sep, "/")
            result[key] = read_file(tag_dir, key)
    return result


def read_metric_lines(metrics_dir: str) -> dict[str, list[str]]:
    result = {}
    if not os.path.isdir(metrics_dir):
        return result
    for root, _, files in os.walk(metrics_dir):
        for name in files:
            abspath = os.path.join(root, name)
            key = os.path.relpath(abspath, metrics_dir)
            key = key.replace(os.sep, "/")
            result[key] = read_file_lines(metrics_dir, key)
    return result


# ── Phase 1: Experiments + Tags ──────────────────────────────────────────────


def migrate_experiments(session: Session, mlruns: str) -> None:
    # Active experiments are direct subdirectories of mlruns that look like IDs
    for exp_id in _list_experiment_ids(mlruns):
        exp_dir = os.path.join(mlruns, exp_id)
        _migrate_one_experiment(session, exp_dir, exp_id)

    # Deleted experiments are in .trash/
    trash_dir = os.path.join(mlruns, ".trash")
    for exp_id in _list_experiment_ids(trash_dir):
        exp_dir = os.path.join(trash_dir, exp_id)
        _migrate_one_experiment(session, exp_dir, exp_id)


def _list_experiment_ids(root: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    result = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        # Experiment dirs are numeric IDs
        try:
            int(name)
        except ValueError:
            continue
        result.append(name)
    return result


def _migrate_one_experiment(session: Session, exp_dir: str, exp_id: str) -> None:
    meta = safe_read_yaml(exp_dir, META_YAML)
    if meta is None:
        return

    # experiment_id may be stored as int in legacy data
    db_exp_id = int(exp_id)

    session.add(
        SqlExperiment(
            experiment_id=db_exp_id,
            name=meta.get("name", f"experiment_{exp_id}"),
            artifact_location=meta.get("artifact_location"),
            lifecycle_stage=meta.get("lifecycle_stage", "active"),
            creation_time=meta.get("creation_time"),
            last_update_time=meta.get("last_update_time"),
        )
    )
    bump("experiments")

    # Experiment tags
    tags_dir = os.path.join(exp_dir, "tags")
    for key, value in read_tag_files(tags_dir).items():
        session.add(
            SqlExperimentTag(
                key=key,
                value=value,
                experiment_id=db_exp_id,
            )
        )
        bump("experiment_tags")


# ── Phase 2: Runs + Params + Tags + Metrics + LatestMetrics ──────────────────

RESERVED_FOLDERS = {"tags", "datasets", "traces", "models", ".trash"}


def migrate_runs(session: Session, mlruns: str) -> None:
    for exp_id in _list_experiment_ids(mlruns):
        exp_dir = os.path.join(mlruns, exp_id)
        _migrate_runs_in_dir(session, exp_dir, int(exp_id))

    trash_dir = os.path.join(mlruns, ".trash")
    for exp_id in _list_experiment_ids(trash_dir):
        exp_dir = os.path.join(trash_dir, exp_id)
        _migrate_runs_in_dir(session, exp_dir, int(exp_id))


def _migrate_runs_in_dir(session: Session, exp_dir: str, exp_id: int) -> None:
    for name in list_subdirs(exp_dir):
        if name in RESERVED_FOLDERS:
            continue
        run_dir = os.path.join(exp_dir, name)
        meta_path = os.path.join(run_dir, META_YAML)
        if not os.path.isfile(meta_path):
            continue
        _migrate_one_run(session, run_dir, exp_id)


def _migrate_one_run(session: Session, run_dir: str, exp_id: int) -> None:
    meta = safe_read_yaml(run_dir, META_YAML)
    if meta is None:
        return

    run_uuid = meta.get("run_uuid") or meta.get("run_id")
    if not run_uuid:
        return

    # Status is stored as integer enum in meta.yaml
    status_raw = meta.get("status", RunStatus.RUNNING)
    status = RunStatus.to_string(status_raw) if isinstance(status_raw, int) else str(status_raw)

    session.add(
        SqlRun(
            run_uuid=run_uuid,
            name=meta.get("run_name") or meta.get("name"),
            source_type=(
                meta.get("source_type", "LOCAL")
                if isinstance(meta.get("source_type"), str)
                else "LOCAL"
            ),
            source_name=meta.get("source_name", ""),
            entry_point_name=meta.get("entry_point_name", ""),
            user_id=meta.get("user_id"),
            status=status,
            start_time=meta.get("start_time"),
            end_time=meta.get("end_time"),
            deleted_time=meta.get("deleted_time"),
            source_version=meta.get("source_version", ""),
            lifecycle_stage=meta.get("lifecycle_stage", "active"),
            artifact_uri=meta.get("artifact_uri"),
            experiment_id=exp_id,
        )
    )
    bump("runs")

    # Params
    params_dir = os.path.join(run_dir, "params")
    for key, value in read_tag_files(params_dir).items():
        session.add(
            SqlParam(
                key=key,
                value=value,
                run_uuid=run_uuid,
            )
        )
        bump("params")

    # Tags
    tags_dir = os.path.join(run_dir, "tags")
    for key, value in read_tag_files(tags_dir).items():
        session.add(
            SqlTag(
                key=key,
                value=value,
                run_uuid=run_uuid,
            )
        )
        bump("tags")

    # Metrics + LatestMetrics
    metrics_dir = os.path.join(run_dir, "metrics")
    _migrate_run_metrics(session, metrics_dir, run_uuid, str(exp_id))


def _sanitize_metric_value(val: float) -> tuple[bool, float]:
    is_nan = math.isnan(val)
    if is_nan:
        return True, 0.0
    if math.isinf(val):
        return False, 1.7976931348623157e308 if val > 0 else -1.7976931348623157e308
    return False, val


def _parse_metric_line(metric_line: str) -> tuple[int, float, int]:
    parts = metric_line.strip().split(" ")
    ts = int(parts[0])
    val = float(parts[1])
    step = int(parts[2]) if len(parts) >= 3 else 0
    return ts, val, step


def _migrate_run_metrics(session: Session, metrics_dir: str, run_uuid: str, exp_id: str) -> None:
    all_metrics = read_metric_lines(metrics_dir)

    for key, lines in all_metrics.items():
        # Track the "latest" metric for this key: max by (step, timestamp, value)
        latest: tuple[int, int, float] | None = None  # (step, timestamp, value)
        latest_is_nan = False

        for line in lines:
            ts, raw_val, step = _parse_metric_line(line)
            is_nan, db_val = _sanitize_metric_value(raw_val)

            session.add(
                SqlMetric(
                    key=key,
                    value=db_val,
                    timestamp=ts,
                    step=step,
                    is_nan=is_nan,
                    run_uuid=run_uuid,
                )
            )
            bump("metrics")

            # For latest_metrics: NaN comparison uses 0 as proxy value
            cmp_val = 0.0 if is_nan else db_val
            cmp_tuple = (step, ts, cmp_val)
            if latest is None or cmp_tuple > latest:
                latest = cmp_tuple
                latest_is_nan = is_nan

        if latest is not None:
            l_step, l_ts, l_val = latest
            session.add(
                SqlLatestMetric(
                    key=key,
                    value=0.0 if latest_is_nan else l_val,
                    timestamp=l_ts,
                    step=l_step,
                    is_nan=latest_is_nan,
                    run_uuid=run_uuid,
                )
            )
            bump("latest_metrics")


# ── Phase 3: Datasets + Inputs + InputTags ───────────────────────────────────


def migrate_datasets(session: Session, mlruns: str) -> None:
    for exp_id in _list_experiment_ids(mlruns):
        exp_dir = os.path.join(mlruns, exp_id)
        _migrate_datasets_for_experiment(session, exp_dir, int(exp_id))

    trash_dir = os.path.join(mlruns, ".trash")
    for exp_id in _list_experiment_ids(trash_dir):
        exp_dir = os.path.join(trash_dir, exp_id)
        _migrate_datasets_for_experiment(session, exp_dir, int(exp_id))


def _migrate_datasets_for_experiment(session: Session, exp_dir: str, exp_id: int) -> None:
    datasets_dir = os.path.join(exp_dir, "datasets")
    if not os.path.isdir(datasets_dir):
        return

    # Build a map: dataset_dir_name -> SqlDataset info (for input linking)
    dataset_uuid_map: dict[str, str] = {}  # dataset_dir_name -> dataset_uuid

    for ds_dir_name in list_subdirs(datasets_dir):
        ds_dir = os.path.join(datasets_dir, ds_dir_name)
        meta = safe_read_yaml(ds_dir, META_YAML)
        if meta is None:
            continue

        ds_uuid = meta.get("dataset_uuid") or str(uuid.uuid4())
        dataset_uuid_map[ds_dir_name] = ds_uuid

        session.add(
            SqlDataset(
                dataset_uuid=ds_uuid,
                experiment_id=exp_id,
                name=meta.get("name", ""),
                digest=meta.get("digest", ""),
                dataset_source_type=meta.get("source_type", ""),
                dataset_source=meta.get("source", ""),
                dataset_schema=meta.get("schema"),
                dataset_profile=meta.get("profile"),
            )
        )
        bump("datasets")

    # Now scan runs in this experiment for inputs
    for run_name in list_subdirs(exp_dir):
        if run_name in RESERVED_FOLDERS:
            continue
        run_dir = os.path.join(exp_dir, run_name)
        inputs_dir = os.path.join(run_dir, "inputs")
        if not os.path.isdir(inputs_dir):
            continue

        for input_dir_name in list_subdirs(inputs_dir):
            input_dir = os.path.join(inputs_dir, input_dir_name)
            input_meta = safe_read_yaml(input_dir, META_YAML)
            if input_meta is None:
                continue

            source_type = input_meta.get("source_type", "DATASET")
            source_id = input_meta.get("source_id", "")

            # Check if source_id maps to a known dataset
            ds_uuid = dataset_uuid_map.get(source_id)
            if source_type == "DATASET" and ds_uuid is None:
                continue

            input_uuid = str(uuid.uuid4())
            session.add(
                SqlInput(
                    input_uuid=input_uuid,
                    source_type=source_type,
                    source_id=ds_uuid if source_type == "DATASET" else source_id,
                    destination_type="RUN",
                    destination_id=run_name,
                )
            )
            bump("inputs")

            # Input tags
            input_tags = input_meta.get("tags", {})
            for tag_name, tag_value in input_tags.items():
                session.add(
                    SqlInputTag(
                        input_uuid=input_uuid,
                        name=tag_name,
                        value=str(tag_value),
                    )
                )
                bump("input_tags")


# ── Phase 4: Traces + TraceTags + TraceMetadata ──────────────────────────────


def migrate_traces(session: Session, mlruns: str) -> None:
    for exp_id in _list_experiment_ids(mlruns):
        exp_dir = os.path.join(mlruns, exp_id)
        _migrate_traces_for_experiment(session, exp_dir, int(exp_id))

    trash_dir = os.path.join(mlruns, ".trash")
    for exp_id in _list_experiment_ids(trash_dir):
        exp_dir = os.path.join(trash_dir, exp_id)
        _migrate_traces_for_experiment(session, exp_dir, int(exp_id))


def _migrate_traces_for_experiment(session: Session, exp_dir: str, exp_id: int) -> None:
    traces_dir = os.path.join(exp_dir, "traces")
    if not os.path.isdir(traces_dir):
        return

    for trace_dir_name in list_subdirs(traces_dir):
        trace_dir = os.path.join(traces_dir, trace_dir_name)
        trace_info_path = os.path.join(trace_dir, "trace_info.yaml")
        if not os.path.isfile(trace_info_path):
            continue

        meta = safe_read_yaml(trace_dir, "trace_info.yaml")
        if meta is None:
            continue

        # V2 uses request_id, V3 uses trace_id
        trace_id = meta.get("trace_id") or meta.get("request_id") or trace_dir_name

        # V2 uses timestamp_ms, V3 uses request_time (proto timestamp string)
        timestamp_ms = meta.get("timestamp_ms")
        if timestamp_ms is None:
            # V3 format - request_time may be a protobuf timestamp string
            request_time = meta.get("request_time")
            if isinstance(request_time, int):
                timestamp_ms = request_time
            elif isinstance(request_time, str):
                # Protobuf timestamp format, try parsing
                try:
                    from google.protobuf.timestamp_pb2 import Timestamp

                    ts = Timestamp()
                    ts.FromJsonString(request_time)
                    timestamp_ms = ts.ToMilliseconds()
                except Exception:
                    timestamp_ms = 0
            else:
                timestamp_ms = 0

        # V2 uses execution_time_ms, V3 uses execution_duration_ms
        execution_time_ms = meta.get("execution_time_ms") or meta.get("execution_duration_ms")

        # Status: V2 has status as string like "OK", V3 has state
        status = meta.get("status") or meta.get("state", "OK")

        session.add(
            SqlTraceInfo(
                request_id=trace_id,
                experiment_id=exp_id,
                timestamp_ms=timestamp_ms,
                execution_time_ms=execution_time_ms,
                status=status,
                client_request_id=meta.get("client_request_id"),
                request_preview=meta.get("request_preview"),
                response_preview=meta.get("response_preview"),
            )
        )
        bump("traces")

        # Trace tags
        trace_tags_dir = os.path.join(trace_dir, "tags")
        for key, value in read_tag_files(trace_tags_dir).items():
            session.add(
                SqlTraceTag(
                    key=key,
                    value=value,
                    request_id=trace_id,
                )
            )
            bump("trace_tags")

        # Trace request metadata
        metadata_dir = os.path.join(trace_dir, "request_metadata")
        for key, value in read_tag_files(metadata_dir).items():
            session.add(
                SqlTraceMetadata(
                    key=key,
                    value=value,
                    request_id=trace_id,
                )
            )
            bump("trace_metadata")


# ── Phase 5: Assessments ─────────────────────────────────────────────────────


def migrate_assessments(session: Session, mlruns: str) -> None:
    for exp_id in _list_experiment_ids(mlruns):
        exp_dir = os.path.join(mlruns, exp_id)
        _migrate_assessments_for_experiment(session, exp_dir)

    trash_dir = os.path.join(mlruns, ".trash")
    for exp_id in _list_experiment_ids(trash_dir):
        exp_dir = os.path.join(trash_dir, exp_id)
        _migrate_assessments_for_experiment(session, exp_dir)


def _migrate_assessments_for_experiment(session: Session, exp_dir: str) -> None:
    traces_dir = os.path.join(exp_dir, "traces")
    if not os.path.isdir(traces_dir):
        return

    for trace_dir_name in list_subdirs(traces_dir):
        trace_dir = os.path.join(traces_dir, trace_dir_name)
        assessments_dir = os.path.join(trace_dir, "assessments")
        if not os.path.isdir(assessments_dir):
            continue

        # Read trace_info to get the actual trace_id
        trace_meta = safe_read_yaml(trace_dir, "trace_info.yaml")
        if trace_meta is None:
            continue
        trace_id = trace_meta.get("trace_id") or trace_meta.get("request_id") or trace_dir_name

        for filename in list_files(assessments_dir):
            if not filename.endswith(".yaml"):
                continue
            assessment_id = filename[:-5]  # strip .yaml
            meta = safe_read_yaml(assessments_dir, filename)
            if meta is None:
                continue

            _migrate_one_assessment(session, meta, trace_id, assessment_id)


def _migrate_one_assessment(
    session: Session, meta: dict[str, Any], trace_id: str, assessment_id: str
) -> None:
    # Determine assessment type and value
    feedback_data = meta.get("feedback")
    expectation_data = meta.get("expectation")

    if feedback_data is not None:
        assessment_type = "feedback"
        value_json = json.dumps(feedback_data.get("value"))
        error_data = feedback_data.get("error")
        error_json = json.dumps(error_data) if error_data else None
    elif expectation_data is not None:
        assessment_type = "expectation"
        value_json = json.dumps(expectation_data.get("value"))
        error_json = None
    else:
        return

    # Source info
    source = meta.get("source", {})
    source_type = source.get("source_type", "CODE")
    source_id = source.get("source_id")

    # Timestamps
    create_time = meta.get("create_time_ms", 0)
    last_update_time = meta.get("last_update_time_ms", create_time)

    # Metadata
    assessment_metadata = meta.get("metadata")
    metadata_json = json.dumps(assessment_metadata) if assessment_metadata else None

    session.add(
        SqlAssessments(
            assessment_id=meta.get("assessment_id") or assessment_id,
            trace_id=trace_id,
            name=meta.get("assessment_name", meta.get("name", "")),
            assessment_type=assessment_type,
            value=value_json,
            error=error_json,
            created_timestamp=create_time,
            last_updated_timestamp=last_update_time,
            source_type=source_type,
            source_id=source_id,
            run_id=meta.get("run_id"),
            span_id=meta.get("span_id"),
            rationale=meta.get("rationale"),
            overrides=meta.get("overrides"),
            valid=meta.get("valid", True),
            assessment_metadata=metadata_json,
        )
    )
    bump("assessments")


# ── Phase 6: Logged Models + Params + Tags + Metrics ────────────────────────


def migrate_logged_models(session: Session, mlruns: str) -> None:
    for exp_id in _list_experiment_ids(mlruns):
        exp_dir = os.path.join(mlruns, exp_id)
        _migrate_logged_models_for_experiment(session, exp_dir, int(exp_id))

    trash_dir = os.path.join(mlruns, ".trash")
    for exp_id in _list_experiment_ids(trash_dir):
        exp_dir = os.path.join(trash_dir, exp_id)
        _migrate_logged_models_for_experiment(session, exp_dir, int(exp_id))


def _migrate_logged_models_for_experiment(session: Session, exp_dir: str, exp_id: int) -> None:
    models_dir = os.path.join(exp_dir, "models")
    if not os.path.isdir(models_dir):
        return

    for model_dir_name in list_subdirs(models_dir):
        model_dir = os.path.join(models_dir, model_dir_name)
        meta = safe_read_yaml(model_dir, META_YAML)
        if meta is None:
            continue

        model_id = meta.get("model_id", model_dir_name)

        # Status may be stored as an integer enum or string
        status_raw = meta.get("status", 1)  # 1 = PENDING typically
        if isinstance(status_raw, str):
            # LoggedModelStatus string -> try to map
            from mlflow.entities.logged_model_status import LoggedModelStatus

            try:
                status_raw = LoggedModelStatus[status_raw].value
            except (KeyError, AttributeError):
                status_raw = 1
        status = int(status_raw)

        session.add(
            SqlLoggedModel(
                model_id=model_id,
                experiment_id=exp_id,
                name=meta.get("name", ""),
                artifact_location=meta.get("artifact_location", ""),
                creation_timestamp_ms=meta.get("creation_timestamp", 0),
                last_updated_timestamp_ms=meta.get("last_updated_timestamp", 0),
                status=status,
                lifecycle_stage=meta.get("lifecycle_stage", "active"),
                model_type=meta.get("model_type"),
                source_run_id=meta.get("source_run_id"),
                status_message=meta.get("status_message"),
            )
        )
        bump("logged_models")

        # Logged model params
        params_dir = os.path.join(model_dir, "params")
        for key, value in read_tag_files(params_dir).items():
            session.add(
                SqlLoggedModelParam(
                    model_id=model_id,
                    experiment_id=exp_id,
                    param_key=key,
                    param_value=value,
                )
            )
            bump("logged_model_params")

        # Logged model tags
        tags_dir = os.path.join(model_dir, "tags")
        for key, value in read_tag_files(tags_dir).items():
            session.add(
                SqlLoggedModelTag(
                    model_id=model_id,
                    experiment_id=exp_id,
                    tag_key=key,
                    tag_value=value,
                )
            )
            bump("logged_model_tags")

        # Logged model metrics
        metrics_dir = os.path.join(model_dir, "metrics")
        _migrate_logged_model_metrics(session, metrics_dir, model_id, exp_id)


def _migrate_logged_model_metrics(
    session: Session, metrics_dir: str, model_id: str, exp_id: int
) -> None:
    all_metrics = read_metric_lines(metrics_dir)

    for key, lines in all_metrics.items():
        for line in lines:
            parts = line.strip().split(" ")
            # Logged model metric format: timestamp value step run_id [dataset_name dataset_digest]
            if len(parts) not in (4, 6):
                continue

            ts = int(parts[0])
            val = float(parts[1])
            step = int(parts[2])
            run_id = parts[3]
            dataset_name = parts[4] if len(parts) == 6 else None
            dataset_digest = parts[5] if len(parts) == 6 else None

            session.add(
                SqlLoggedModelMetric(
                    model_id=model_id,
                    metric_name=key,
                    metric_timestamp_ms=ts,
                    metric_step=step,
                    metric_value=val,
                    experiment_id=exp_id,
                    run_id=run_id,
                    dataset_uuid=None,
                    dataset_name=dataset_name,
                    dataset_digest=dataset_digest,
                )
            )
            bump("logged_model_metrics")


# ── Phase 7: Model Registry ─────────────────────────────────────────────────


def migrate_model_registry(session: Session, mlruns: str) -> None:
    models_dir = os.path.join(mlruns, "models")
    if not os.path.isdir(models_dir):
        return

    for model_name in list_subdirs(models_dir):
        model_dir = os.path.join(models_dir, model_name)
        meta = safe_read_yaml(model_dir, META_YAML)
        if meta is None:
            continue

        session.add(
            SqlRegisteredModel(
                name=meta.get("name", model_name),
                creation_time=meta.get("creation_timestamp"),
                last_updated_time=meta.get("last_updated_timestamp"),
                description=meta.get("description"),
            )
        )
        bump("registered_models")

        # Registered model tags
        tags_dir = os.path.join(model_dir, "tags")
        for key, value in read_tag_files(tags_dir).items():
            session.add(
                SqlRegisteredModelTag(
                    name=meta.get("name", model_name),
                    key=key,
                    value=value,
                )
            )
            bump("registered_model_tags")

        # Model versions
        for version_dir_name in list_subdirs(model_dir):
            if not version_dir_name.startswith("version-"):
                continue
            version_dir = os.path.join(model_dir, version_dir_name)
            _migrate_model_version(session, version_dir, meta.get("name", model_name))

        # Aliases
        aliases_dir = os.path.join(model_dir, "aliases")
        for alias_name in list_files(aliases_dir):
            version_str = read_file(aliases_dir, alias_name).strip()
            try:
                version_int = int(version_str)
            except ValueError:
                continue
            session.add(
                SqlRegisteredModelAlias(
                    name=meta.get("name", model_name),
                    alias=alias_name,
                    version=version_int,
                )
            )
            bump("registered_model_aliases")


def _migrate_model_version(session: Session, version_dir: str, model_name: str) -> None:
    meta = safe_read_yaml(version_dir, META_YAML)
    if meta is None:
        return

    version = meta.get("version")
    if version is None:
        # Try extracting from directory name
        dir_name = os.path.basename(version_dir)
        try:
            version = int(dir_name.replace("version-", ""))
        except ValueError:
            return

    session.add(
        SqlModelVersion(
            name=model_name,
            version=int(version),
            creation_time=meta.get("creation_timestamp"),
            last_updated_time=meta.get("last_updated_timestamp"),
            description=meta.get("description"),
            user_id=meta.get("user_id"),
            current_stage=meta.get("current_stage", "None"),
            source=meta.get("source"),
            storage_location=meta.get("storage_location"),
            run_id=meta.get("run_id"),
            run_link=meta.get("run_link"),
            status=meta.get("status", "READY"),
            status_message=meta.get("status_message"),
        )
    )
    bump("model_versions")

    # Model version tags
    tags_dir = os.path.join(version_dir, "tags")
    for key, value in read_tag_files(tags_dir).items():
        session.add(
            SqlModelVersionTag(
                name=model_name,
                version=int(version),
                key=key,
                value=value,
            )
        )
        bump("model_version_tags")


# ── Orchestration ────────────────────────────────────────────────────────────


def _assert_empty_db(engine) -> None:
    with engine.connect() as conn:
        for table in ("experiments", "runs", "registered_models"):
            try:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            except Exception:
                continue
            if count > 0:
                raise SystemExit(
                    f"Target database is not empty: table '{table}' has {count} rows. "
                    "Migration requires an empty database."
                )


def _print_summary() -> None:
    print()
    print("=" * 50)
    print("Migration summary:")
    print("=" * 50)
    for key, count in sorted(summary.items()):
        print(f"  {key}: {count}")
    print("=" * 50)


def migrate(source: str, target_uri: str) -> None:
    mlruns = os.path.join(source, "mlruns")
    if not os.path.isdir(mlruns):
        # Source may be the mlruns directory itself — check if it has experiment-like subdirs
        has_experiment_dirs = any(
            d.isdigit() or d in {".trash", "models"}
            for d in os.listdir(source)
            if os.path.isdir(os.path.join(source, d))
        )
        if has_experiment_dirs:
            mlruns = source
        else:
            raise SystemExit(f"Cannot find mlruns directory in '{source}'")

    print(f"Source: {mlruns}")
    print(f"Target: {target_uri}")
    print()

    engine = create_engine(target_uri)

    print("Initializing database schema...")
    _initialize_tables(engine)
    _assert_empty_db(engine)

    with Session(engine) as session:
        try:
            # Phase 1
            print("[1/7] Migrating experiments + tags...")
            migrate_experiments(session, mlruns)
            session.flush()

            # Phase 2
            print("[2/7] Migrating runs + params + tags + metrics...")
            migrate_runs(session, mlruns)
            session.flush()

            # Phase 3
            has_datasets = os.path.isdir(mlruns) and any(
                os.path.isdir(os.path.join(mlruns, d, "datasets"))
                for d in _list_experiment_ids(mlruns)
                if os.path.isdir(os.path.join(mlruns, d))
            )
            if has_datasets:
                print("[3/7] Migrating datasets + inputs...")
                migrate_datasets(session, mlruns)
                session.flush()
            else:
                print("[3/7] Skipping datasets (not found)")

            # Phase 4
            has_traces = any(
                os.path.isdir(os.path.join(mlruns, d, "traces"))
                for d in _list_experiment_ids(mlruns)
                if os.path.isdir(os.path.join(mlruns, d))
            )
            if has_traces:
                print("[4/7] Migrating traces + tags + metadata...")
                migrate_traces(session, mlruns)
                session.flush()

                # Phase 5
                print("[5/7] Migrating assessments...")
                migrate_assessments(session, mlruns)
                session.flush()
            else:
                print("[4/7] Skipping traces (not found)")
                print("[5/7] Skipping assessments (not found)")

            # Phase 6
            has_models = any(
                os.path.isdir(os.path.join(mlruns, d, "models"))
                for d in _list_experiment_ids(mlruns)
                if os.path.isdir(os.path.join(mlruns, d))
            )
            if has_models:
                print("[6/7] Migrating logged models...")
                migrate_logged_models(session, mlruns)
                session.flush()
            else:
                print("[6/7] Skipping logged models (not found)")

            # Phase 7
            has_registry = os.path.isdir(os.path.join(mlruns, "models"))
            if has_registry:
                print("[7/7] Migrating model registry...")
                migrate_model_registry(session, mlruns)
            else:
                print("[7/7] Skipping model registry (not found)")

            session.commit()
            print()
            print("Migration completed successfully!")

        except Exception:
            session.rollback()
            print()
            print("Migration FAILED — transaction rolled back.")
            raise

    _print_summary()


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate MLflow FileStore data to a SQLite database"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Root directory containing mlruns/ FileStore data",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="SQLite URI (e.g. sqlite:////tmp/migrated.db)",
    )
    args = parser.parse_args()

    if not args.target.startswith("sqlite:///"):
        raise SystemExit("--target must be a SQLite URI starting with 'sqlite:///'")

    source = os.path.abspath(args.source)
    if not os.path.isdir(source):
        raise SystemExit(f"--source directory does not exist: {source}")

    migrate(source, args.target)


if __name__ == "__main__":
    main()
