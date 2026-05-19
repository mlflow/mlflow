import logging
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from mlflow.store.tracking.file_store import FileStore

_logger = logging.getLogger(__name__)


@dataclass
class MigrationStats:
    experiments: int = 0
    experiment_tags: int = 0
    runs: int = 0
    params: int = 0
    tags: int = 0
    metrics: int = 0
    latest_metrics: int = 0
    datasets: int = 0
    inputs: int = 0
    input_tags: int = 0
    outputs: int = 0
    traces: int = 0
    trace_tags: int = 0
    trace_metadata: int = 0
    assessments: int = 0
    logged_models: int = 0
    logged_model_params: int = 0
    logged_model_tags: int = 0
    logged_model_metrics: int = 0
    registered_models: int = 0
    registered_model_tags: int = 0
    registered_model_aliases: int = 0
    model_versions: int = 0
    model_version_tags: int = 0

    def items(self) -> Iterator[tuple[str, int]]:
        for f in fields(self):
            val = getattr(self, f.name)
            if val > 0:
                yield f.name, val

    def summary(self, source: str, target_uri: str, db_counts: dict[str, int] | None = None) -> str:
        sep = "=" * 50
        lines = [sep, "Migration summary:", sep]
        if db_counts:
            lines.append(f"  {'entity':<25} {'migrated':>10} {'in DB':>10}")
            lines.append(f"  {'-' * 25} {'-' * 10} {'-' * 10}")
            for key, count in self.items():
                db_val = db_counts.get(key, "")
                lines.append(f"  {key:<25} {count:>10} {db_val:>10}")
        else:
            for key, count in self.items():
                lines.append(f"  {key}: {count}")
        lines.append(sep)
        lines.append(f"  source: {source}")
        lines.append(f"  target: {target_uri}")
        lines.append(sep)
        lines.append("")
        lines.append("To start a server with the migrated data:")
        lines.append(f"  mlflow server --backend-store-uri {target_uri}")
        return "\n".join(lines)


def safe_read_yaml(root: Path, file_name: str) -> dict[str, Any] | None:
    try:
        return yaml.safe_load((root / file_name).read_text())
    except Exception as e:
        _logger.warning("Failed to read %s: %s", root / file_name, e)
        return None


def list_subdirs(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    return sorted(d.name for d in path.iterdir() if d.is_dir())


def list_files(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    return sorted(f.name for f in path.iterdir() if f.is_file())


def read_tag_files(tag_dir: Path) -> dict[str, str]:
    result = {}
    if not tag_dir.is_dir():
        return result
    for p in tag_dir.rglob("*"):
        if not p.is_file():
            continue
        key = p.relative_to(tag_dir).as_posix()
        result[key] = (tag_dir / key).read_text()
    return result


def read_metric_lines(metrics_dir: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if not metrics_dir.is_dir():
        return result
    for p in metrics_dir.rglob("*"):
        if not p.is_file():
            continue
        key = p.relative_to(metrics_dir).as_posix()
        result[key] = (metrics_dir / key).read_text().splitlines()
    return result


def list_experiment_ids(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    result = []
    for d in sorted(root.iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue
        try:
            int(d.name)
        except ValueError:
            continue
        result.append(d.name)
    return result


def for_each_experiment(mlruns: Path) -> Iterator[tuple[Path, str]]:
    """Yield (exp_dir, exp_id) for all experiments in both mlruns and .trash."""
    for exp_id in list_experiment_ids(mlruns):
        yield mlruns / exp_id, exp_id

    trash_dir = mlruns / FileStore.TRASH_FOLDER_NAME
    for exp_id in list_experiment_ids(trash_dir):
        yield trash_dir / exp_id, exp_id
