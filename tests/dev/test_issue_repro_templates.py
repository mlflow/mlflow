import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

from issue_repro_templates import (
    NO_REPRO,
    TARGETED_PYTEST,
    InvalidTemplate,
    command_for,
    validate_template,
)


def test_targeted_pytest_has_a_fixed_argv_shape():
    template = validate_template({
        "id": TARGETED_PYTEST,
        "test_path": "tests/dev/test_classify_flaky_tests.py",
    })

    assert command_for(template) == [
        "uv",
        "run",
        "pytest",
        "tests/dev/test_classify_flaky_tests.py",
    ]


@pytest.mark.parametrize(
    "path",
    [
        "../tests/a.py",
        "/tests/a.py",
        "tests/a.py -k x",
        "tests/a.py;id",
        "tests\\a.py",
        "tests/a.txt",
    ],
)
def test_rejects_unsafe_paths(path):
    with pytest.raises(InvalidTemplate, match="test path"):
        validate_template({"id": TARGETED_PYTEST, "test_path": path})


def test_rejects_unknown_fields_and_accepts_no_repro():
    with pytest.raises(InvalidTemplate, match="template"):
        validate_template({"id": NO_REPRO, "command": "id"})

    assert command_for(validate_template({"id": NO_REPRO, "test_path": None})) is None
