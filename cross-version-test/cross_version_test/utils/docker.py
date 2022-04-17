import typing as t

from ..types import PathLike
from .process import CompletedProcess, run_cmd


class Docker:
    def __init__(self, docker_file: PathLike) -> None:
        self.docker_file = docker_file

    def _run(self, cmd: str, *args: PathLike, **kwargs: t.Any) -> CompletedProcess:
        return run_cmd(
            [
                "docker",
                cmd,
                "--file",
                self.docker_file,
                *args,
            ],
            **kwargs,
        )

    def build(self, *args: PathLike, **kwargs: t.Any) -> CompletedProcess:
        return self._run("build", *args, **kwargs)


class DockerCompose:
    def __init__(self, compose_file: PathLike) -> None:
        self.compose_file = compose_file

    def _run(self, cmd: str, *args: PathLike, **kwargs: t.Any) -> CompletedProcess:
        return run_cmd(
            [
                "docker",
                "compose",
                "--file",
                self.compose_file,
                cmd,
                *args,
            ],
            **kwargs,
        )

    def build(self, *args: PathLike, **kwargs: t.Any) -> CompletedProcess:
        return self._run("build", *args, **kwargs)

    def run(self, *args: PathLike, **kwargs: t.Any) -> CompletedProcess:
        return self._run("run", *args, **kwargs)

    def config(self, *args: PathLike, **kwargs: t.Any) -> CompletedProcess:
        return self._run("config", *args, **kwargs)

    def down(self, *args: PathLike, **kwargs: t.Any) -> CompletedProcess:
        return self._run("down", *args, **kwargs)
