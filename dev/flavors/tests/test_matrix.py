import asyncio
import functools
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar
from unittest import mock

import pytest

pytestmark = pytest.mark.skip(
    reason="Disabled by #21985: dev installs use git+ which doesn't respect UV_EXCLUDE_NEWER"
)

from flavors._matrix import generate_matrix

_P = ParamSpec("_P")
_R = TypeVar("_R")


class MockResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def json(self) -> dict[str, Any]:
        return self.data

    def raise_for_status(self) -> None:
        pass

    @classmethod
    def from_versions(cls, versions: list[str]) -> "MockResponse":
        return cls({
            "releases": {
                v: [
                    {
                        "filename": v + ".whl",
                        "upload_time": "2023-10-04T16:38:57",
                    }
                ]
                for v in versions
            }
        })


def mock_pypi_api(
    mock_responses: dict[str, MockResponse],
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def requests_get_patch(url: str, *args: Any, **kwargs: Any) -> MockResponse:
        match = re.search(r"https://pypi\.org/pypi/(.+)/json", url)
        assert match is not None
        return mock_responses[match.group(1)]

    def decorator(test_func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(test_func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with mock.patch("requests.get", new=requests_get_patch):
                return test_func(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def mock_ml_package_versions_yml(src_base: str, src_ref: str) -> Iterator[list[str]]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yml_base = Path(tmp_dir).joinpath("base.yml")
        yml_ref = Path(tmp_dir).joinpath("ref.yml")
        yml_base.write_text(src_base)
        yml_ref.write_text(src_ref)
        yield ["--versions-yaml", str(yml_base), "--ref-versions-yaml", str(yml_ref)]


MOCK_YAML_SOURCE = """
foo:
  package_info:
    pip_release: foo
    install_dev: "pip install git+https://github.com/foo/foo.git"

  autologging:
    minimum: "1.0.0"
    maximum: "1.2.0"
    run: pytest tests/foo

bar:
  package_info:
    pip_release: bar
    install_dev: "pip install git+https://github.com/bar/bar.git"

  autologging:
    minimum: "1.3"
    maximum: "1.4"
    run: pytest/tests bar
"""

MOCK_PYPI_API_RESPONSES = {
    "foo": MockResponse.from_versions(["1.0.0", "1.1.0", "1.1.1", "1.2.0"]),
    "bar": MockResponse.from_versions(["1.3", "1.4"]),
}


@pytest.mark.parametrize(
    ("flavors", "expected"),
    [
        ("foo", {"foo"}),
        ("foo,bar", {"foo", "bar"}),
        ("foo, bar", {"foo", "bar"}),  # Contains a space after a comma
        ("", {"foo", "bar"}),
        (None, {"foo", "bar"}),
    ],
)
@mock_pypi_api(MOCK_PYPI_API_RESPONSES)
def test_flavors(flavors: str | None, expected: set[str]) -> None:
    with mock_ml_package_versions_yml(MOCK_YAML_SOURCE, "{}") as path_args:
        flavors_args = [] if flavors is None else ["--flavors", flavors]
        matrix = asyncio.run(generate_matrix([*path_args, *flavors_args]))
        flavor_names = {x.flavor for x in matrix}
        assert flavor_names == expected


@pytest.mark.parametrize(
    ("versions", "expected"),
    [
        ("1.0.0", {"1.0.0"}),
        ("1.0.0,1.1.1", {"1.0.0", "1.1.1"}),
        ("1.3, 1.4", {"1.3", "1.4"}),  # Contains a space after a comma
        ("", {"1.0.0", "1.1.1", "1.2.0", "1.3", "1.4", "dev"}),
        (None, {"1.0.0", "1.1.1", "1.2.0", "1.3", "1.4", "dev"}),
    ],
)
@mock_pypi_api(MOCK_PYPI_API_RESPONSES)
def test_versions(versions: str | None, expected: set[str]) -> None:
    with mock_ml_package_versions_yml(MOCK_YAML_SOURCE, "{}") as path_args:
        versions_args = [] if versions is None else ["--versions", versions]
        matrix = asyncio.run(generate_matrix([*path_args, *versions_args]))
        version_strs = {str(x.version) for x in matrix}
        assert version_strs == expected


@mock_pypi_api(MOCK_PYPI_API_RESPONSES)
def test_flavors_and_versions() -> None:
    with mock_ml_package_versions_yml(MOCK_YAML_SOURCE, "{}") as path_args:
        matrix = asyncio.run(
            generate_matrix([*path_args, "--flavors", "foo,bar", "--versions", "dev"])
        )
        flavors = {x.flavor for x in matrix}
        versions = {str(x.version) for x in matrix}
        assert flavors == {"foo", "bar"}
        assert versions == {"dev"}


@mock_pypi_api(MOCK_PYPI_API_RESPONSES)
def test_no_dev() -> None:
    with mock_ml_package_versions_yml(MOCK_YAML_SOURCE, "{}") as path_args:
        matrix = asyncio.run(generate_matrix([*path_args, "--no-dev"]))
        flavors = {x.flavor for x in matrix}
        versions = {str(x.version) for x in matrix}
        assert flavors == {"foo", "bar"}
        assert versions == {"1.0.0", "1.1.1", "1.2.0", "1.3", "1.4"}


@mock_pypi_api(MOCK_PYPI_API_RESPONSES)
def test_changed_files() -> None:
    with mock_ml_package_versions_yml(MOCK_YAML_SOURCE, MOCK_YAML_SOURCE) as path_args:
        matrix = asyncio.run(
            generate_matrix([*path_args, "--changed-files", "mlflow/foo/__init__.py"])
        )
        flavors = {x.flavor for x in matrix}
        versions = {str(x.version) for x in matrix}
        assert flavors == {"foo"}
        assert versions == {"1.0.0", "1.1.1", "1.2.0", "dev"}
