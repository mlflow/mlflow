from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from flavors._schema import FlavorConfig

VERSIONS_YAML_PATH = "mlflow/ml-package-versions.yml"


def load(path: str | Path) -> dict[str, FlavorConfig]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {name: FlavorConfig(**cfg) for name, cfg in raw.items()}


def load_or_default(path: str | Path, default: Any) -> Any:
    try:
        return load(path)
    except Exception as e:
        print(f"Failed to read '{path}' due to: `{e}`")
        return default


def load_raw(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)
