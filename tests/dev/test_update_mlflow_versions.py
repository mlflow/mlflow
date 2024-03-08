import os
import shutil
from pathlib import Path

import pytest

from dev.update_mlflow_versions import update_versions


def setup_test_directory(path):
    shutil.copytree("tests/resources/mlflow_version_files/before", path, dirs_exist_ok=True)


@pytest.mark.parametrize(
    ("version", "directory"),
    # these are hardcoded in the expectation files
    [("2.11.1", "after-pre"), ("2.11.2.dev0", "after-post")],
)
def test_update_mlflow_versions_pre(tmp_path, version, directory):
    old_cwd = os.getcwd()
    setup_test_directory(tmp_path)

    os.chdir(tmp_path)
    update_versions(version)
    os.chdir(old_cwd)

    # check that the files are as expected
    root = Path(f"tests/resources/mlflow_version_files/{directory}")
    entries = root.rglob("*")
    expect_files = [entry for entry in entries if entry.is_file()]
    for expect_file in expect_files:
        relative_path = expect_file.relative_to(root)
        actual_file = tmp_path / relative_path

        assert expect_file.read_text() == actual_file.read_text()
