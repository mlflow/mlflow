"""
Lint `mlflow/ml-package-versions.yml`.

Currently enforces block-style sequences under `requirements:` so the
`flavors update` prune step's `ruamel.yaml` round-trip doesn't reformat
unrelated entries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

from flavors._loader import VERSIONS_YAML_PATH


def _check_block_style(yml_path: Path) -> list[str]:
    yaml = YAML()
    yaml.preserve_quotes = True
    with yml_path.open() as f:
        data = yaml.load(f)

    violations: list[str] = []
    for flavor, cfg in data.items():
        if not isinstance(cfg, dict):
            continue
        for cat in ("models", "autologging"):
            if cat not in cfg:
                continue
            cat_cfg = cfg[cat]
            if not isinstance(cat_cfg, dict):
                continue
            req = cat_cfg.get("requirements")
            if not req:
                continue
            for spec in req:
                value = req[spec]
                if isinstance(value, CommentedSeq) and value.fa.flow_style():
                    violations.append(f"{flavor} / {cat} / {spec!r}")
    return violations


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--versions-yaml",
        default=VERSIONS_YAML_PATH,
        help=f"Local path of the config yaml. Defaults to '{VERSIONS_YAML_PATH}'.",
    )


def run(args: argparse.Namespace) -> None:
    violations = _check_block_style(Path(args.versions_yaml))
    if not violations:
        return
    print(
        "Found flow-style sequences inside `requirements:`. Convert them to "
        "block style (e.g. `- foo` on each line):",
        file=sys.stderr,
    )
    for v in violations:
        print(f"  - {v}", file=sys.stderr)
    sys.exit(1)
