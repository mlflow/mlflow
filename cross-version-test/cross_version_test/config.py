import typing as t
import itertools
import os
from concurrent.futures import ThreadPoolExecutor

from .models import Flavor, Job
from .utils.string import get_pip_install_cmd, remove_comments
from .utils.version import (
    DEV_VERSION,
    Version,
    filter_versions,
    get_extra_requirements,
    get_released_versions,
    select_latest_micro_versions,
)


def expand_flavor(group: str, flavor: Flavor) -> t.List[Job]:
    """
    Expand a flavor configuration into a list of jobs.
    """
    flavor_name = group.split("-")[0]
    all_versions = get_released_versions(flavor.package_info.pip_release)
    jobs = []
    for category, cfg in flavor.categories.items():
        if cfg is None:
            continue

        min_ver = Version(cfg.minimum)
        max_ver = Version(cfg.maximum)
        exclude = list(map(Version, cfg.unsupported or []))
        versions = filter_versions(
            all_versions,
            min_ver=min_ver,
            max_ver=max_ver,
            exclude=exclude,
        )
        versions = select_latest_micro_versions(versions)

        # Always include the minimum version
        if min_ver not in versions:
            versions.append(min_ver)

        pip_release = flavor.package_info.pip_release
        for version in versions:
            requirements = ["{}=={}".format(pip_release, version)]
            requirements.extend(get_extra_requirements(cfg.requirements, version))
            install = get_pip_install_cmd(requirements)
            run = remove_comments(cfg.run)
            jobs.append(
                Job(
                    group=group,
                    category=category,
                    flavor=flavor_name,
                    install=install,
                    run=run,
                    package=pip_release,
                    version=str(version),
                    supported=version <= max_ver,
                )
            )

        # Development version
        install_dev = flavor.package_info.install_dev
        if install_dev:
            requirements = get_extra_requirements(cfg.requirements, Version(DEV_VERSION))
            install_commands = [remove_comments(install_dev)]
            if requirements:
                install_commands.append(get_pip_install_cmd(requirements))
            run = remove_comments(cfg.run)
            jobs.append(
                Job(
                    group=group,
                    category=category,
                    flavor=flavor_name,
                    install="\n".join(install_commands),
                    run=run,
                    package=pip_release,
                    version=DEV_VERSION,
                    supported=False,
                )
            )
    return jobs


def expand_config(config: t.Dict[str, t.Any]) -> t.Set[Job]:
    """
    Expand a cross-version test configuration into a set of jobs.
    """
    with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 0) + 4)) as executor:
        futures = [
            executor.submit(expand_flavor, group, Flavor.parse_obj(flavor))
            for group, flavor in config.items()
        ]
        return set(itertools.chain.from_iterable(f.result() for f in futures))
