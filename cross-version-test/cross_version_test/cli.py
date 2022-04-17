import json
import os
import re
import typing as t
import logging
from pathlib import Path
from distutils.dir_util import copy_tree

import click

import cross_version_test.constants as const
from cross_version_test.models import Job
from .version import version
from .config import expand_config
from .templates import DOCKERFILE_TEMPLATE, DOCKER_COMPOSE_TEMPLATE, SHELL_SCRIPT_TEMPLATE
from .utils.docker import Docker, DockerCompose
from .utils.process import run_cmd
from .utils.file import read_yaml
from .utils.version import DEV_VERSION
from .utils.flavor import get_changed_flavors
from .utils.reporter import Reporter, Result
from .utils.string import split


ROOT_DIR = Path(const.ROOT_DIR)
DOCKER_COMPOSE_FILE = ROOT_DIR.joinpath(const.DOCKER_COMPOSE_FILE)

_logger = logging.getLogger(__name__)


def sort_jobs_by_name(jobs: t.Set[Job]) -> t.List[Job]:
    return sorted(jobs, key=lambda x: x.name)


def on_github_actions() -> bool:
    return "GITHUB_ACTIONS" in os.environ


def save_as_github_actions_matrix(jobs: t.Sequence[Job], path: Path) -> None:
    if jobs:
        job_names = [j.name for j in jobs]
        matrix = {"name": job_names, "include": [j.dict() for j in jobs]}
    else:
        matrix = {}
    path.write_text(json.dumps(matrix))


def get_python_version(job: Job) -> str:
    if (job.version == "dev") and job.package in ["scikit-learn", "statsmodels"]:
        return "3.8"
    return "3.7"


def build_docker_files(jobs: t.Sequence[Job]) -> None:
    mlflow_requirements = run_cmd(
        [
            "python",
            "setup.py",
            "-q",
            "dependencies",
        ],
        capture_output=True,
    ).stdout

    entrypoint_cmd = """
source $VENV_DIR/bin/activate
pip install -e .
exec "$@"
"""
    entrypoint = SHELL_SCRIPT_TEMPLATE.render(cmd=entrypoint_cmd)
    for job in jobs:
        job_dir = ROOT_DIR.joinpath(job.name)
        job_dir.mkdir(parents=True, exist_ok=True)
        # Create shell scripts
        requirements_file = job_dir.joinpath(const.MLFLOW_REQUIREMENTS_FILE)
        install_script = job_dir.joinpath(const.INSTALL_SCRIPT)
        run_script = job_dir.joinpath(const.RUN_SCRIPT)
        entrypoint_script = job_dir.joinpath(const.ENTRYPOINT_SCRIPT)
        requirements_file.write_text(mlflow_requirements)
        install_script.write_text(SHELL_SCRIPT_TEMPLATE.render(cmd=job.install))
        run_script.write_text(SHELL_SCRIPT_TEMPLATE.render(cmd=job.run))
        entrypoint_script.write_text(entrypoint)
        # Modify file permission
        os.chmod(install_script, 0o777)
        os.chmod(run_script, 0o777)
        os.chmod(entrypoint_script, 0o777)
        # Create Dockerfile
        content = DOCKERFILE_TEMPLATE.render(
            base_image_name=const.DOCKER_BASE_IMAGE,
            mlflow_requirements_file=const.MLFLOW_REQUIREMENTS_FILE,
            small_requirements_file=const.SMALL_REQUIREMENTS_FILE,
            install_script=const.INSTALL_SCRIPT,
            run_script=const.RUN_SCRIPT,
            entrypoint_script=const.ENTRYPOINT_SCRIPT,
            python_version=get_python_version(job),
        )
        job_dir.joinpath(const.DOCKERFILE).write_text(content)
        copy_tree("requirements", str(job_dir.joinpath("requirements")))

    # Create docker-compose.yml
    docker_compose_content = DOCKER_COMPOSE_TEMPLATE.render(jobs=[j.name for j in jobs])
    DOCKER_COMPOSE_FILE.write_text(docker_compose_content)


def build_docker_images(jobs: t.Sequence[Job], no_cache: bool) -> bool:
    no_cache_opt = ("--no-cache",) if no_cache else ()
    reporter = Reporter()
    reporter.write_separator("Building base image")
    this_file_dir = Path(__file__).parent
    base_dockerfile = this_file_dir.joinpath("base_image", "Dockerfile")
    docker = Docker(base_dockerfile)
    docker.build(
        *no_cache_opt,
        "-t",
        const.DOCKER_BASE_IMAGE,
        this_file_dir,
        check=True,
    )

    docker_compose = DockerCompose(DOCKER_COMPOSE_FILE)
    for job in jobs:
        reporter.write_separator(f"Building {job.name}")
        prc = docker_compose.build(*no_cache_opt, job.name, check=False)
        reporter.add_result(Result(job.name, prc.success))

    reporter.write_separator("Results")
    reporter.report()
    return reporter.all_success()


# Options used in multiple commands
PATTERN = click.option(
    "-p",
    "--pattern",
    required=False,
    help="If specified, only select jobs that matches the given pattern (regular expression). "
    "For example, 'xgboost_.+_autologging' selects xgboost autologging jobs.",
)

VERSIONS_YAML = click.option(
    "--versions-yaml",
    required=False,
    default="mlflow/ml-package-versions.yml",
    help=(
        "URL or local file path of the config yaml. Defaults to "
        "'mlflow/ml-package-versions.yml' on the branch where this script is running."
    ),
)

NO_CACHE = click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Do not use cache when building images",
)


@click.group(help="CLI for cross-version test")
@click.version_option(version)
def cli() -> None:
    pass


@cli.command(help="Build jobs")
@VERSIONS_YAML
@PATTERN
@NO_CACHE
def build(versions_yaml: str, pattern: t.Optional[str], no_cache: bool) -> None:
    all_jobs = expand_config(read_yaml(versions_yaml))
    matched_jobs = sort_jobs_by_name(all_jobs)
    if pattern:
        matched_jobs = [j for j in matched_jobs if re.compile(pattern).search(j.name)]

    if on_github_actions():
        ROOT_DIR.mkdir(parents=True, exist_ok=True)
        json_file = ROOT_DIR.joinpath(const.MATRIX_JSON)
        save_as_github_actions_matrix(matched_jobs, json_file)
        # The cross-version-tests workflow only requires a matrix JSON file so we can exit here
        return

    build_docker_files(sort_jobs_by_name(all_jobs))
    success = build_docker_images(matched_jobs, no_cache)
    exit(int(not success))


@cli.command(help="Build jobs relevant to configuration / flavor changes")
@VERSIONS_YAML
@click.option(
    "--ref-versions-yaml",
    required=False,
    help=(
        "URL or local file path of the reference config yaml which will be compared with the "
        "config specified by `--versions-yaml` in order to identify the config updates."
    ),
)
@click.option(
    "--changed-files",
    required=False,
    help="A string that represents a list of changed files",
)
@click.option(
    "--exclude-dev-versions/--include-dev-versions",
    is_flag=True,
    default=True,
    help="If True, exclude dev versions from the test matrix.",
)
@NO_CACHE
def diff(
    versions_yaml: str,
    ref_versions_yaml: t.Optional[str],
    changed_files: t.Optional[str],
    exclude_dev_versions: bool,
    no_cache: bool,
) -> None:
    all_jobs = expand_config(read_yaml(versions_yaml))
    if ref_versions_yaml is None and changed_files is None:
        jobs_changed = all_jobs
    else:
        # Select jobs relevant to configuration file changes. For example, if we update
        # the "sklearn.autologging.run" field, we should run all the sklearn autologging jobs
        # to ensure the updated run command works properly.
        if ref_versions_yaml:
            try:
                # `ref_yaml` may not have the format that `expand_config` expects.
                ref_yaml = read_yaml(ref_versions_yaml)
                jobs_ref = expand_config(ref_yaml)
            except Exception as e:
                _logger.warning("Failed to parse the ref versions yaml: %s", repr(e))
                jobs_ref = set()
            jobs_changed_config = set(all_jobs).difference(jobs_ref)
        else:
            jobs_changed_config = set()

        # Select jobs relevant to the flavor file changes. For example, if we update a file in
        # mlflow/sklearn, we should run all the sklearn jobs should to ensure the update is
        # compatible with all the supported sklearn versions.
        if changed_files:
            changed_flavors = get_changed_flavors(split(changed_files, sep="\n"))
            jobs_changed_file = {j for j in all_jobs if j.flavor in changed_flavors}
        else:
            jobs_changed_file = set()

        jobs_changed = jobs_changed_config.union(jobs_changed_file)

    if exclude_dev_versions:
        jobs_changed = {j for j in jobs_changed if j.version != DEV_VERSION}

    sorted_jobs = sort_jobs_by_name(jobs_changed)
    if on_github_actions():
        ROOT_DIR.mkdir(parents=True, exist_ok=True)
        json_file = ROOT_DIR.joinpath(const.MATRIX_JSON)
        save_as_github_actions_matrix(sorted_jobs, json_file)
        return

    build_docker_files(sort_jobs_by_name(all_jobs))
    success = build_docker_images(sorted_jobs, no_cache)
    exit(int(not success))


def list_jobs() -> t.List[str]:
    docker_compose = DockerCompose(DOCKER_COMPOSE_FILE)
    return docker_compose.config("--services", capture_output=True, check=True).stdout.splitlines()


@cli.command(name="list", help="List jobs")
def list_command() -> None:
    click.echo("\n".join(list_jobs()))


@cli.command(
    help="Run jobs",
    context_settings=dict(ignore_unknown_options=True),
)
@PATTERN
@click.argument("cmd", nargs=-1)
def run(pattern: t.Optional[str], cmd: t.Tuple[str, ...]) -> None:
    jobs = list_jobs()

    if pattern:
        jobs = [j for j in jobs if re.compile(pattern).search(j)]

    docker_compose = DockerCompose(DOCKER_COMPOSE_FILE)
    reporter = Reporter()
    for job in jobs:
        reporter.write_separator(f"Running {job}")
        prc = docker_compose.run("--rm", job, *cmd, check=False)
        reporter.add_result(Result(job, prc.success))

    reporter.write_separator("Results")
    reporter.report()
    exit(int(not reporter.all_success()))


@cli.command(
    help="Remove containers, volumes, network, and images",
    context_settings=dict(ignore_unknown_options=True),
)
@click.option(
    "--rmi",
    is_flag=True,
    default=False,
    help="If True, remove images created by cross-version test",
)
def down(rmi: bool) -> None:
    docker_compose = DockerCompose(DOCKER_COMPOSE_FILE)
    args = ["--volumes", "--remove-orphans"]
    if rmi:
        args.extend(["--rmi", "all"])
    docker_compose.down(*args, check=True)
