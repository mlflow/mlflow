"""
Migrate tracking store entities from FileStore to DB.

FileStore layout:

    <mlruns>/
    ├── <experiment_id>/
    │   ├── meta.yaml                    -> experiments
    │   ├── tags/<key>                   -> experiment_tags
    │   ├── <run_uuid>/
    │   │   ├── meta.yaml                -> runs
    │   │   ├── params/<key>             -> params
    │   │   ├── tags/<key>               -> tags
    │   │   ├── metrics/<key>            -> metrics, latest_metrics
    │   │   └── inputs/<id>/meta.yaml    -> inputs, input_tags
    │   ├── datasets/<id>/meta.yaml      -> datasets
    │   ├── traces/<trace_id>/
    │   │   ├── trace_info.yaml          -> trace_info
    │   │   ├── tags/<key>               -> trace_tags
    │   │   ├── request_metadata/<key>   -> trace_request_metadata
    │   │   └── assessments/<id>.yaml    -> assessments
    │   └── models/<model_id>/
    │       ├── meta.yaml                -> logged_models
    │       ├── params/<key>             -> logged_model_params
    │       ├── tags/<key>               -> logged_model_tags
    │       └── metrics/<key>            -> logged_model_metrics
    └── .trash/
        └── <experiment_id>/...          (same structure, deleted experiments)
"""

import json
import logging
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from mlflow.entities import RunStatus
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.store.fs2db._utils import (
    MigrationStats,
    for_each_experiment,
    list_files,
    list_subdirs,
    read_metric_lines,
    read_tag_files,
    safe_read_yaml,
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
from mlflow.store.tracking.file_store import FileStore

_logger = logging.getLogger(__name__)


def migrate_experiments(session: Session, mlruns: Path, stats: MigrationStats) -> None:
    for exp_dir, exp_id in for_each_experiment(mlruns):
        _migrate_one_experiment(session, exp_dir, exp_id, stats)


def _migrate_one_experiment(
    session: Session, exp_dir: Path, exp_id: str, stats: MigrationStats
) -> None:
    meta = safe_read_yaml(exp_dir, FileStore.META_DATA_FILE_NAME)
    if meta is None:
        return

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
    stats.experiments += 1

    for key, value in read_tag_files(exp_dir / FileStore.TAGS_FOLDER_NAME).items():
        session.add(
            SqlExperimentTag(
                key=key,
                value=value,
                experiment_id=db_exp_id,
            )
        )
        stats.experiment_tags += 1


RESERVED_FOLDERS = {
    FileStore.TAGS_FOLDER_NAME,
    FileStore.DATASETS_FOLDER_NAME,
    FileStore.TRACES_FOLDER_NAME,
    FileStore.MODELS_FOLDER_NAME,
    FileStore.TRASH_FOLDER_NAME,
}


def migrate_runs(session: Session, mlruns: Path, stats: MigrationStats) -> None:
    for exp_dir, exp_id in for_each_experiment(mlruns):
        _migrate_runs_in_dir(session, exp_dir, int(exp_id), stats)


def _migrate_runs_in_dir(
    session: Session,
    exp_dir: Path,
    exp_id: int,
    stats: MigrationStats,
    *,
    batch_size: int = 1000,
) -> None:
    count = 0
    for name in list_subdirs(exp_dir):
        if name in RESERVED_FOLDERS:
            continue
        run_dir = exp_dir / name
        if not (run_dir / FileStore.META_DATA_FILE_NAME).is_file():
            continue
        _migrate_one_run(session, run_dir, exp_id, stats)
        count += 1
        if count % batch_size == 0:
            session.flush()
            session.expunge_all()


def _migrate_one_run(session: Session, run_dir: Path, exp_id: int, stats: MigrationStats) -> None:
    meta = safe_read_yaml(run_dir, FileStore.META_DATA_FILE_NAME)
    if meta is None:
        return

    run_uuid = meta.get("run_uuid") or meta.get("run_id")
    if not run_uuid:
        _logger.warning("Skipping run in %s: missing run_uuid/run_id", run_dir)
        return

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
    stats.runs += 1

    # Params
    for key, value in read_tag_files(run_dir / FileStore.PARAMS_FOLDER_NAME).items():
        session.add(
            SqlParam(
                key=key,
                value=value,
                run_uuid=run_uuid,
            )
        )
        stats.params += 1

    # Tags
    for key, value in read_tag_files(run_dir / FileStore.TAGS_FOLDER_NAME).items():
        session.add(
            SqlTag(
                key=key,
                value=value,
                run_uuid=run_uuid,
            )
        )
        stats.tags += 1

    # Metrics + LatestMetrics
    _migrate_run_metrics(session, run_dir / FileStore.METRICS_FOLDER_NAME, run_uuid, stats)


def _sanitize_metric_value(val: float) -> tuple[bool, float]:
    is_nan = math.isnan(val)
    if is_nan:
        return True, 0.0
    if math.isinf(val):
        return False, 1.7976931348623157e308 if val > 0 else -1.7976931348623157e308
    return False, val


def _parse_metric_line(metric_line: str) -> tuple[int, float, int]:
    match metric_line.strip().split(" "):
        case [ts, val]:
            return int(ts), float(val), 0
        case [ts, val, step, *_]:
            return int(ts), float(val), int(step)
        case _:
            raise ValueError(f"Malformed metric line: {metric_line!r}")


def _migrate_run_metrics(
    session: Session,
    metrics_dir: Path,
    run_uuid: str,
    stats: MigrationStats,
    *,
    batch_size: int = 5000,
) -> None:
    all_metrics = read_metric_lines(metrics_dir)
    count = 0

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
            stats.metrics += 1
            count += 1
            if count % batch_size == 0:
                session.flush()
                session.expunge_all()

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
            stats.latest_metrics += 1


def migrate_datasets(session: Session, mlruns: Path, stats: MigrationStats) -> None:
    for exp_dir, exp_id in for_each_experiment(mlruns):
        _migrate_datasets_for_experiment(session, exp_dir, int(exp_id), stats)


def _migrate_datasets_for_experiment(
    session: Session, exp_dir: Path, exp_id: int, stats: MigrationStats
) -> None:
    datasets_dir = exp_dir / FileStore.DATASETS_FOLDER_NAME
    if not datasets_dir.is_dir():
        return

    dataset_uuid_map: dict[str, str] = {}  # dataset_dir_name -> dataset_uuid

    for ds_dir_name in list_subdirs(datasets_dir):
        meta = safe_read_yaml(datasets_dir / ds_dir_name, FileStore.META_DATA_FILE_NAME)
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
        stats.datasets += 1

    # Scan runs in this experiment for inputs (run dirs are named by run UUID)
    for run_uuid in list_subdirs(exp_dir):
        if run_uuid in RESERVED_FOLDERS:
            continue
        inputs_dir = exp_dir / run_uuid / FileStore.INPUTS_FOLDER_NAME
        if not inputs_dir.is_dir():
            continue

        for input_dir_name in list_subdirs(inputs_dir):
            input_meta = safe_read_yaml(inputs_dir / input_dir_name, FileStore.META_DATA_FILE_NAME)
            if input_meta is None:
                continue

            source_type = input_meta.get("source_type", "DATASET")
            source_id = input_meta.get("source_id", "")

            if source_type == "DATASET":
                ds_uuid = dataset_uuid_map.get(source_id)
                if ds_uuid is None:
                    continue
                # FileStore doesn't persist input UUIDs; generate for the DB
                input_uuid = str(uuid.uuid4())
                session.add(
                    SqlInput(
                        input_uuid=input_uuid,
                        source_type="DATASET",
                        source_id=ds_uuid,
                        destination_type="RUN",
                        destination_id=run_uuid,
                    )
                )
            elif source_type == "MODEL":
                # FileStore: source_type=MODEL, source_id=model_id, destination_type=RUN
                # DB store:  source_type=RUN_INPUT, source_id=run_id, destination_type=MODEL_INPUT
                input_uuid = str(uuid.uuid4())
                session.add(
                    SqlInput(
                        input_uuid=input_uuid,
                        source_type="RUN_INPUT",
                        source_id=run_uuid,
                        destination_type="MODEL_INPUT",
                        destination_id=source_id,
                    )
                )
            else:
                continue
            stats.inputs += 1

            input_tags = input_meta.get("tags", {})
            for tag_name, tag_value in input_tags.items():
                session.add(
                    SqlInputTag(
                        input_uuid=input_uuid,
                        name=tag_name,
                        value=str(tag_value),
                    )
                )
                stats.input_tags += 1


def _migrate_outputs_for_experiment(session: Session, exp_dir: Path, stats: MigrationStats) -> None:
    for run_uuid in list_subdirs(exp_dir):
        if run_uuid in RESERVED_FOLDERS:
            continue
        outputs_dir = exp_dir / run_uuid / FileStore.OUTPUTS_FOLDER_NAME
        if not outputs_dir.is_dir():
            continue

        for model_id in list_subdirs(outputs_dir):
            meta = safe_read_yaml(outputs_dir / model_id, FileStore.META_DATA_FILE_NAME)
            if meta is None:
                continue

            # FileStore doesn't persist input UUIDs; generate for the DB
            session.add(
                SqlInput(
                    input_uuid=str(uuid.uuid4()),
                    source_type="RUN_OUTPUT",
                    source_id=run_uuid,
                    destination_type="MODEL_OUTPUT",
                    destination_id=model_id,
                    step=meta.get("step", 0),
                )
            )
            stats.outputs += 1


def migrate_traces(session: Session, mlruns: Path, stats: MigrationStats) -> None:
    for exp_dir, exp_id in for_each_experiment(mlruns):
        _migrate_traces_for_experiment(session, exp_dir, int(exp_id), stats)


def _parse_timestamp_ms(request_time: str) -> int:
    try:
        dt = datetime.fromisoformat(request_time.replace("Z", "+00:00"))
        return int(dt.replace(tzinfo=dt.tzinfo or timezone.utc).timestamp() * 1000)
    except Exception:
        return 0


def _migrate_traces_for_experiment(
    session: Session,
    exp_dir: Path,
    exp_id: int,
    stats: MigrationStats,
    *,
    batch_size: int = 1000,
) -> None:
    traces_dir = exp_dir / FileStore.TRACES_FOLDER_NAME
    if not traces_dir.is_dir():
        return

    count = 0
    for trace_dir_name in list_subdirs(traces_dir):
        trace_dir = traces_dir / trace_dir_name
        if not (trace_dir / FileStore.TRACE_INFO_FILE_NAME).is_file():
            continue

        meta = safe_read_yaml(trace_dir, FileStore.TRACE_INFO_FILE_NAME)
        if meta is None:
            continue

        # V2 uses request_id, V3 uses trace_id
        trace_id = meta.get("trace_id") or meta.get("request_id") or trace_dir_name

        # V2 uses timestamp_ms, V3 uses request_time (proto timestamp string)
        timestamp_ms = meta.get("timestamp_ms")
        if timestamp_ms is None:
            request_time = meta.get("request_time")
            if isinstance(request_time, int):
                timestamp_ms = request_time
            elif isinstance(request_time, str):
                timestamp_ms = _parse_timestamp_ms(request_time)
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
        stats.traces += 1

        # Trace tags
        for key, value in read_tag_files(trace_dir / FileStore.TRACE_TAGS_FOLDER_NAME).items():
            session.add(
                SqlTraceTag(
                    key=key,
                    value=value,
                    request_id=trace_id,
                )
            )
            stats.trace_tags += 1

        # Trace request metadata
        for key, value in read_tag_files(
            trace_dir / FileStore.TRACE_TRACE_METADATA_FOLDER_NAME
        ).items():
            session.add(
                SqlTraceMetadata(
                    key=key,
                    value=value,
                    request_id=trace_id,
                )
            )
            stats.trace_metadata += 1

        count += 1
        if count % batch_size == 0:
            session.flush()
            session.expunge_all()


def migrate_assessments(session: Session, mlruns: Path, stats: MigrationStats) -> None:
    for exp_dir, _exp_id in for_each_experiment(mlruns):
        _migrate_assessments_for_experiment(session, exp_dir, stats)


def _migrate_assessments_for_experiment(
    session: Session, exp_dir: Path, stats: MigrationStats
) -> None:
    traces_dir = exp_dir / FileStore.TRACES_FOLDER_NAME
    if not traces_dir.is_dir():
        return

    for trace_dir_name in list_subdirs(traces_dir):
        trace_dir = traces_dir / trace_dir_name
        assessments_dir = trace_dir / FileStore.ASSESSMENTS_FOLDER_NAME
        if not assessments_dir.is_dir():
            continue

        trace_meta = safe_read_yaml(trace_dir, FileStore.TRACE_INFO_FILE_NAME)
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

            _migrate_one_assessment(session, meta, trace_id, assessment_id, stats)


def _migrate_one_assessment(
    session: Session,
    meta: dict[str, Any],
    trace_id: str,
    assessment_id: str,
    stats: MigrationStats,
) -> None:
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

    source = meta.get("source", {})
    source_type = source.get("source_type", "CODE")
    source_id = source.get("source_id")

    create_time = meta.get("create_time_ms", 0)
    last_update_time = meta.get("last_update_time_ms", create_time)

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
    stats.assessments += 1


def migrate_logged_models(session: Session, mlruns: Path, stats: MigrationStats) -> None:
    for exp_dir, exp_id in for_each_experiment(mlruns):
        _migrate_logged_models_for_experiment(session, exp_dir, int(exp_id), stats)


def _migrate_logged_models_for_experiment(
    session: Session, exp_dir: Path, exp_id: int, stats: MigrationStats
) -> None:
    models_dir = exp_dir / FileStore.MODELS_FOLDER_NAME
    if not models_dir.is_dir():
        return

    for model_dir_name in list_subdirs(models_dir):
        model_dir = models_dir / model_dir_name
        meta = safe_read_yaml(model_dir, FileStore.META_DATA_FILE_NAME)
        if meta is None:
            continue

        model_id = meta.get("model_id", model_dir_name)

        # Status may be stored as an integer enum or string
        status_raw = meta.get("status", 1)  # 1 = PENDING typically
        if isinstance(status_raw, str):
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
        stats.logged_models += 1

        # Logged model params
        for key, value in read_tag_files(model_dir / FileStore.PARAMS_FOLDER_NAME).items():
            session.add(
                SqlLoggedModelParam(
                    model_id=model_id,
                    experiment_id=exp_id,
                    param_key=key,
                    param_value=value,
                )
            )
            stats.logged_model_params += 1

        # Logged model tags
        for key, value in read_tag_files(model_dir / FileStore.TAGS_FOLDER_NAME).items():
            session.add(
                SqlLoggedModelTag(
                    model_id=model_id,
                    experiment_id=exp_id,
                    tag_key=key,
                    tag_value=value,
                )
            )
            stats.logged_model_tags += 1

        # Logged model metrics
        _migrate_logged_model_metrics(
            session, model_dir / FileStore.METRICS_FOLDER_NAME, model_id, exp_id, stats
        )


def _migrate_logged_model_metrics(
    session: Session, metrics_dir: Path, model_id: str, exp_id: int, stats: MigrationStats
) -> None:
    all_metrics = read_metric_lines(metrics_dir)

    for key, lines in all_metrics.items():
        for line in lines:
            # Format: timestamp value step run_id [dataset_name dataset_digest]
            match line.strip().split(" "):
                case [ts, val, step, run_id]:
                    dataset_name = None
                    dataset_digest = None
                case [ts, val, step, run_id, dataset_name, dataset_digest]:
                    pass
                case _:
                    _logger.warning(
                        "Skipping malformed logged model metric line in %s: %s", key, line
                    )
                    continue

            session.add(
                SqlLoggedModelMetric(
                    model_id=model_id,
                    metric_name=key,
                    metric_timestamp_ms=int(ts),
                    metric_step=int(step),
                    metric_value=float(val),
                    experiment_id=exp_id,
                    run_id=run_id,
                    dataset_uuid=None,
                    dataset_name=dataset_name,
                    dataset_digest=dataset_digest,
                )
            )
            stats.logged_model_metrics += 1
