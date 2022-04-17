import typing as t
from dataclasses import dataclass

import click

from .string import separator


@dataclass
class Result:
    name: str
    success: bool


class Reporter:
    def __init__(self) -> None:
        self.results: t.List[Result] = []

    def add_result(self, result: Result) -> None:
        self.results.append(result)

    def _report_success(self, message: str) -> None:
        click.secho(message, fg="green")

    def _report_error(self, message: str) -> None:
        click.secho(message, fg="red")

    def report(self) -> None:
        for result in self.results:
            click.echo(result.name + " ", nl=False)
            if result.success:
                self._report_success("PASSED")
            else:
                self._report_error("FAILED")

    def all_success(self) -> bool:
        return all(r.success for r in self.results)

    def write_separator(self, title: str) -> None:
        click.secho(separator(title), fg="blue")
