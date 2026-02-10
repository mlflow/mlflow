from collections.abc import Iterator
from pathlib import Path

from mlflow.utils.file_utils import read_file, read_file_lines, read_yaml

META_YAML = "meta.yaml"


summary: dict[str, int] = {}


def bump(key: str, n: int = 1) -> None:
    summary[key] = summary.get(key, 0) + n


def safe_read_yaml(root: Path, file_name: str) -> dict[str, object] | None:
    try:
        return read_yaml(str(root), file_name)
    except Exception:
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
        result[key] = read_file(str(tag_dir), key)
    return result


def read_metric_lines(metrics_dir: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if not metrics_dir.is_dir():
        return result
    for p in metrics_dir.rglob("*"):
        if not p.is_file():
            continue
        key = p.relative_to(metrics_dir).as_posix()
        result[key] = read_file_lines(str(metrics_dir), key)
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

    trash_dir = mlruns / ".trash"
    for exp_id in list_experiment_ids(trash_dir):
        yield trash_dir / exp_id, exp_id
