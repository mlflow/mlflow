import filecmp
import hashlib
import io
import os
import shutil
import stat
import tarfile
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.pyfunc.dbconnect_artifact_cache import extract_archive_to_dir
from mlflow.utils import file_utils
from mlflow.utils.file_utils import (
    TempDir,
    _copy_file_or_tree,
    _handle_readonly_on_windows,
    check_tarfile_security,
    get_parent_dir,
    get_total_file_size,
    local_file_uri_to_path,
)
from mlflow.utils.os import is_windows

from tests.helper_functions import random_int
from tests.projects.utils import TEST_PROJECT_DIR


@pytest.fixture(scope="module")
def spark_session():
    with SparkSession.builder.master("local[*]").getOrCreate() as session:
        yield session


def test_mkdir(tmp_path):
    temp_dir = str(tmp_path)
    new_dir_name = f"mkdir_test_{random_int()}"
    file_utils.mkdir(temp_dir, new_dir_name)
    assert os.listdir(temp_dir) == [new_dir_name]

    with pytest.raises(OSError, match="bad directory"):
        file_utils.mkdir("/   bad directory @ name ", "ouch")

    # does not raise if directory exists already
    file_utils.mkdir(temp_dir, new_dir_name)

    # raises if it exists already but is a file
    dummy_file_path = str(tmp_path.joinpath("dummy_file"))
    with open(dummy_file_path, "a"):
        pass

    with pytest.raises(OSError, match="exists"):
        file_utils.mkdir(dummy_file_path)


def test_make_tarfile(tmp_path):
    # Tar a local project
    tarfile0 = str(tmp_path.joinpath("first-tarfile"))
    file_utils.make_tarfile(
        output_filename=tarfile0, source_dir=TEST_PROJECT_DIR, archive_name="some-archive"
    )
    # Copy local project into a temp dir
    dst_dir = str(tmp_path.joinpath("project-directory"))
    shutil.copytree(TEST_PROJECT_DIR, dst_dir)
    # Tar the copied project
    tarfile1 = str(tmp_path.joinpath("second-tarfile"))
    file_utils.make_tarfile(
        output_filename=tarfile1, source_dir=dst_dir, archive_name="some-archive"
    )
    # Compare the archives & explicitly verify their SHA256 hashes match (i.e. that
    # changes in file modification timestamps don't affect the archive contents)
    assert filecmp.cmp(tarfile0, tarfile1, shallow=False)
    with open(tarfile0, "rb") as first_tar, open(tarfile1, "rb") as second_tar:
        assert (
            hashlib.sha256(first_tar.read()).hexdigest()
            == hashlib.sha256(second_tar.read()).hexdigest()
        )
    # Extract the TAR and check that its contents match the original directory
    extract_dir = str(tmp_path.joinpath("extracted-tar"))
    os.makedirs(extract_dir)
    with tarfile.open(tarfile0, "r:gz") as handle:
        handle.extractall(path=extract_dir)
    dir_comparison = filecmp.dircmp(os.path.join(extract_dir, "some-archive"), TEST_PROJECT_DIR)
    assert len(dir_comparison.left_only) == 0
    assert len(dir_comparison.right_only) == 0
    assert len(dir_comparison.diff_files) == 0
    assert len(dir_comparison.funny_files) == 0


def test_get_parent_dir(tmp_path):
    child_dir = tmp_path.joinpath("dir")
    child_dir.mkdir()
    assert str(tmp_path) == get_parent_dir(str(child_dir))


def test_file_copy():
    with TempDir() as tmp:
        file_path = tmp.path("test_file.txt")
        copy_path = tmp.path("test_dir1/")
        os.mkdir(copy_path)
        with open(file_path, "a") as f:
            f.write("testing")
        _copy_file_or_tree(file_path, copy_path, "")
        assert filecmp.cmp(file_path, os.path.join(copy_path, "test_file.txt"))


def test_dir_create():
    with TempDir() as tmp:
        file_path = tmp.path("test_file.txt")
        create_dir = tmp.path("test_dir2/")
        with open(file_path, "a") as f:
            f.write("testing")
        name = _copy_file_or_tree(file_path, file_path, create_dir)
        assert filecmp.cmp(file_path, name)


def test_dir_copy():
    with TempDir() as tmp:
        dir_path = tmp.path("test_dir1/")
        copy_path = tmp.path("test_dir2")
        os.mkdir(dir_path)
        with open(os.path.join(dir_path, "test_file.txt"), "a") as f:
            f.write("testing")
        _copy_file_or_tree(dir_path, copy_path, "")
        assert filecmp.dircmp(dir_path, copy_path)


@pytest.mark.skipif(not is_windows(), reason="requires Windows")
def test_handle_readonly_on_windows(tmp_path):
    tmp_path = tmp_path.joinpath("file")
    with open(tmp_path, "w"):
        pass

    # Make the file read-only
    os.chmod(tmp_path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    # Ensure the file can't be removed
    with pytest.raises(PermissionError, match="Access is denied") as exc:
        os.unlink(tmp_path)

    _handle_readonly_on_windows(
        os.unlink,
        tmp_path,
        (exc.type, exc.value, exc.traceback),
    )
    assert not os.path.exists(tmp_path)


@pytest.mark.skipif(not is_windows(), reason="This test only passes on Windows")
@pytest.mark.parametrize(
    ("input_uri", "expected_path"),
    [
        (r"\\my_server\my_path\my_sub_path", r"\\my_server\my_path\my_sub_path"),
    ],
)
def test_local_file_uri_to_path_on_windows(input_uri, expected_path):
    assert local_file_uri_to_path(input_uri) == expected_path


def test_shutil_copytree_without_file_permissions(tmp_path):
    src_dir = tmp_path.joinpath("src-dir")
    src_dir.mkdir()
    dst_dir = tmp_path.joinpath("dst-dir")
    dst_dir.mkdir()
    # Test copying empty directory
    mlflow.utils.file_utils.shutil_copytree_without_file_permissions(src_dir, dst_dir)
    assert len(os.listdir(dst_dir)) == 0
    # Test copying directory with contents
    src_dir.joinpath("subdir").mkdir()
    src_dir.joinpath("subdir/subdir-file.txt").write_text("testing 123")
    src_dir.joinpath("top-level-file.txt").write_text("hi")
    mlflow.utils.file_utils.shutil_copytree_without_file_permissions(src_dir, dst_dir)
    assert set(os.listdir(dst_dir)) == {"top-level-file.txt", "subdir"}
    assert set(os.listdir(dst_dir.joinpath("subdir"))) == {"subdir-file.txt"}
    assert dst_dir.joinpath("subdir/subdir-file.txt").read_text() == "testing 123"
    assert dst_dir.joinpath("top-level-file.txt").read_text() == "hi"


def test_get_total_size_basic(tmp_path):
    subdir = tmp_path.joinpath("subdir")
    subdir.mkdir()

    def generate_file(path, size_in_bytes):
        with path.open("wb") as fp:
            fp.write(b"\0" * size_in_bytes)

    file_size_map = {"file1.txt": 11, "file2.txt": 23}
    for name, size in file_size_map.items():
        generate_file(tmp_path.joinpath(name), size)
    generate_file(subdir.joinpath("file3.txt"), 22)
    assert get_total_file_size(tmp_path) == 56
    assert get_total_file_size(subdir) == 22

    path_not_exists = tmp_path.joinpath("does_not_exist")
    assert get_total_file_size(path_not_exists) is None

    path_file = tmp_path.joinpath("file1.txt")
    assert get_total_file_size(path_file) is None


def test_check_tarfile_security(tmp_path):
    def create_tar_with_escaped_path(tar_path: str, escaped_path: str, content: bytes) -> None:
        """Create tar with path traversal entry."""
        with tarfile.open(tar_path, "w:gz") as tar:
            # Add traversal file
            data = io.BytesIO(content)
            info = tarfile.TarInfo(name=escaped_path)
            info.size = len(content)
            tar.addfile(info, data)

    tar1_path = str(tmp_path.joinpath("file1.tar"))
    create_tar_with_escaped_path(tar1_path, "../pwned2.txt", b"ABX")
    with pytest.raises(
        MlflowException, match="Escaped path destination in the archive file is not allowed"
    ):
        check_tarfile_security(tar1_path)

    def create_tar_with_symlink(
        tar_path: str, link_name: str, link_target: str, file_via_link: str, content: bytes
    ) -> None:
        """Create tar with symlink that points outside, then file through symlink."""
        with tarfile.open(tar_path, "w:gz") as tar:
            # First: create a symlink pointing to parent directory
            link_info = tarfile.TarInfo(name=link_name)
            link_info.type = tarfile.SYMTYPE
            link_info.linkname = link_target
            tar.addfile(link_info)
            # Second: create a file that goes through the symlink
            data = io.BytesIO(content)
            file_info = tarfile.TarInfo(name=file_via_link)
            file_info.size = len(content)
            tar.addfile(file_info, data)

    tar2_path = str(tmp_path.joinpath("file2.tar"))
    create_tar_with_symlink(
        tar2_path,
        link_name="escape",
        link_target="..",
        file_via_link="escape/pwned.txt",
        content=b"XYZ",
    )
    with pytest.raises(
        MlflowException,
        match="Destination path in the archive file can not go through a symlink",
    ):
        check_tarfile_security(tar2_path)

    def create_tar_with_abs_path(tar_path: str, abs_path: str, content: bytes) -> None:
        """Create tar with path traversal entry."""
        with tarfile.open(tar_path, "w:gz") as tar:
            # Add traversal file
            data = io.BytesIO(content)
            info = tarfile.TarInfo(name=abs_path)
            info.size = len(content)
            tar.addfile(info, data)

    tar3_path = str(tmp_path.joinpath("file3.tar"))
    create_tar_with_abs_path(tar3_path, "/tmp/pwned2.txt", b"ABX")
    with pytest.raises(
        MlflowException, match="Absolute path destination in the archive file is not allowed"
    ):
        check_tarfile_security(tar3_path)

    # Symlink with safe target but file going through it
    tar2b_path = str(tmp_path.joinpath("file2b.tar"))
    create_tar_with_symlink(
        tar2b_path,
        link_name="link_dir",
        link_target="subdir",
        file_via_link="link_dir/pwned.txt",
        content=b"XYZ",
    )
    with pytest.raises(
        MlflowException, match="Destination path in the archive file can not go through a symlink"
    ):
        check_tarfile_security(tar2b_path)

    # Backslash-based path traversal in tar (Windows tar slip / path traversal)
    tar4_path = str(tmp_path.joinpath("file4.tar"))
    create_tar_with_escaped_path(tar4_path, "..\\..\\pwned.txt", b"ABX")
    with pytest.raises(
        MlflowException, match="Escaped path destination in the archive file is not allowed"
    ):
        check_tarfile_security(tar4_path)

    def create_tar_with_symlink_only(tar_path: Path, link_name: str, link_target: str) -> None:
        with tarfile.open(tar_path, "w:gz") as tar:
            link_info = tarfile.TarInfo(name=link_name)
            link_info.type = tarfile.SYMTYPE
            link_info.linkname = link_target
            tar.addfile(link_info)

    # Symlinks with absolute/escaping targets are allowed (virtualenvs use them).
    # Security is enforced by _safe_extractall at extraction time.
    tar5_path = tmp_path / "file5.tar"
    create_tar_with_symlink_only(tar5_path, "python", "/usr/bin/python3")
    check_tarfile_security(tar5_path)  # should not raise

    tar6_path = tmp_path / "file6.tar"
    create_tar_with_symlink_only(tar6_path, "lib", "../../shared/lib")
    check_tarfile_security(tar6_path)  # should not raise

    # Symlink with absolute path as its own name
    tar7_path = tmp_path / "file7.tar"
    create_tar_with_symlink_only(tar7_path, "/tmp/escape", "foo")
    with pytest.raises(
        MlflowException, match="Absolute path destination in the archive file is not allowed"
    ):
        check_tarfile_security(tar7_path)

    # Symlink whose name escapes with ..
    tar8_path = tmp_path / "file8.tar"
    create_tar_with_symlink_only(tar8_path, "../escape", "foo")
    with pytest.raises(
        MlflowException, match="Escaped path destination in the archive file is not allowed"
    ):
        check_tarfile_security(tar8_path)

    # Hard link with escaping target
    def create_tar_with_hardlink(tar_path: Path, name: str, linkname: str) -> None:
        with tarfile.open(tar_path, "w:gz") as tar:
            info = tarfile.TarInfo(name=name)
            info.type = tarfile.LNKTYPE
            info.linkname = linkname
            tar.addfile(info)

    tar9_path = tmp_path / "file9.tar"
    create_tar_with_hardlink(tar9_path, "legit.txt", "../../etc/passwd")
    with pytest.raises(
        MlflowException, match="Escaped path destination in the archive file is not allowed"
    ):
        check_tarfile_security(tar9_path)

    # Hard link with absolute target
    tar10_path = tmp_path / "file10.tar"
    create_tar_with_hardlink(tar10_path, "legit.txt", "/etc/passwd")
    with pytest.raises(
        MlflowException, match="Absolute path destination in the archive file is not allowed"
    ):
        check_tarfile_security(tar10_path)

    # Windows drive-letter absolute path
    tar11_path = tmp_path / "file11.tar"
    create_tar_with_escaped_path(tar11_path, "C:/Windows/System32/evil.dll", b"ABX")
    with pytest.raises(
        MlflowException, match="Absolute path destination in the archive file is not allowed"
    ):
        check_tarfile_security(tar11_path)


def test_extract_archive_to_dir_blocks_traversal(tmp_path):
    # Test that check_tarfile_security blocks path traversal
    mal_tar = tmp_path / "malicious.tar.gz"
    with tarfile.open(mal_tar, "w:gz") as tar:
        info = tarfile.TarInfo("../../escape.txt")
        data = b"owned via tar traversal"
        info.size = len(data)
        tar.addfile(info, fileobj=io.BytesIO(data))

    dest = tmp_path / "extracted"
    escape_target = tmp_path.parent.parent / "escape.txt"
    with pytest.raises(MlflowException, match="Escaped path destination in the archive file"):
        extract_archive_to_dir(mal_tar, dest)
    assert not escape_target.exists()


def test_safe_extractall_blocks_symlink_escape(tmp_path):
    """Test that _safe_extractall blocks extraction when a filesystem symlink
    inside dest_dir would cause a member to resolve outside dest_dir.
    """
    from mlflow.pyfunc.dbconnect_artifact_cache import _safe_extractall

    dest = tmp_path / "extracted"
    dest.mkdir()
    # Create a symlink inside dest_dir pointing outside
    escape_link = dest / "escape_link"
    escape_link.symlink_to(tmp_path.parent)

    # Create a tar with a file that goes through the symlink
    mal_tar = tmp_path / "symlink_escape.tar.gz"
    with tarfile.open(mal_tar, "w:gz") as tar:
        info = tarfile.TarInfo("escape_link/pwned.txt")
        data = b"escaped via filesystem symlink"
        info.size = len(data)
        tar.addfile(info, fileobj=io.BytesIO(data))

    with tarfile.open(mal_tar, "r") as tar:
        with pytest.raises(MlflowException, match="would be extracted outside"):
            _safe_extractall(tar, dest)
    assert not (tmp_path.parent / "pwned.txt").exists()
