"""
Drop unused `requirements:` specifiers from `mlflow/ml-package-versions.yml`.

A specifier like `"< 1.0.0"` becomes unused when bumping `minimum`/`maximum`
moves the matrix beyond its range. `flavors update` calls this after a bump
so the YAML stays in sync without manual cleanup.
"""

from __future__ import annotations

from pathlib import Path

from packaging.specifiers import SpecifierSet
from pypi import Package, get_packages
from ruamel.yaml import YAML

from flavors._loader import load
from flavors._matrix import compute_matrix_versions, get_flavor
from flavors._releases import get_released_versions
from flavors._schema import DEV_NUMERIC, DEV_VERSION, PackageInfo, Version


def _is_unused_specifier(
    specifier: str, package_info: PackageInfo, versions: list[Version]
) -> bool:
    # Preserve the dev sentinel when install_dev is set; matrix dev testing
    # uses `"== dev"` even though it's currently disabled in expand_config.
    if package_info.install_dev and specifier.strip() == f"== {DEV_VERSION}":
        return False
    # Match `_matrix._find_matches`: normalize the dev token so SpecifierSet
    # can parse specifiers like `"< dev"` or `"< 4.52.0.dev0"`.
    spec_set = SpecifierSet(specifier.replace(DEV_VERSION, DEV_NUMERIC))
    return not any(map(spec_set.contains, versions))


async def prune_unused_requirements(
    yml_path: str | Path, *, packages: dict[str, Package] | None = None
) -> None:
    config = load(yml_path)
    pip_releases = list({fc.package_info.pip_release for fc in config.values()})
    packages = dict(packages or {})
    if missing := [p for p in pip_releases if p not in packages]:
        packages.update(zip(missing, await get_packages(missing)))

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
