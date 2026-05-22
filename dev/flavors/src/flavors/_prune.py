"""
Drop unused `requirements:` specifiers from `mlflow/ml-package-versions.yml`.

A specifier like `"< 1.0.0"` becomes unused when bumping `minimum`/`maximum`
moves the matrix beyond its range. `flavors update` calls this after a bump
so the YAML stays in sync without manual cleanup.
"""

from __future__ import annotations

from pathlib import Path

from packaging.specifiers import SpecifierSet
from pypi import get_packages
from ruamel.yaml import YAML

from flavors._loader import load
from flavors._matrix import compute_matrix_versions, get_flavor
from flavors._releases import get_released_versions
from flavors._schema import PackageInfo, Version


def _is_unused_specifier(
    specifier: str, package_info: PackageInfo, versions: list[Version]
) -> bool:
    if "dev" in specifier and package_info.install_dev:
        return False
    return not any(map(SpecifierSet(specifier).contains, versions))


async def prune_unused_requirements(yml_path: str | Path) -> None:
    config = load(yml_path)
    pip_releases = list({fc.package_info.pip_release for fc in config.values()})
    packages = dict(zip(pip_releases, await get_packages(pip_releases)))

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(yml_path) as f:
        raw = yaml.load(f)

    changed = False
    for name, flavor_config in config.items():
        package = packages[flavor_config.package_info.pip_release]
        all_versions = get_released_versions(package)
        flavor = get_flavor(name)
        for category, cfg in flavor_config.categories:
            if not cfg.requirements:
                continue
            versions = compute_matrix_versions(flavor, cfg, all_versions)
            unused = [
                s
                for s in cfg.requirements
                if _is_unused_specifier(s, flavor_config.package_info, versions)
            ]
            if not unused:
                continue
            requirements_node = raw[name][category]["requirements"]
            for specifier in unused:
                del requirements_node[specifier]
            if not requirements_node:
                del raw[name][category]["requirements"]
            changed = True

    if changed:
        with open(yml_path, "w") as f:
            yaml.dump(raw, f)
