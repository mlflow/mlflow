from __future__ import print_function

import filecmp
import os

from mlflow import pyfunc
from mlflow.utils.file_utils import TempDir

# Tests for mlflow/pyfunc/__init__.py _copy_file_or_tree(src, dst, dst_dir)


# Tests whether copying a file works.
def test_file_copy():
    with TempDir() as tmp:
        file_path = tmp.path("test_file.txt")
        copy_path = tmp.path("test_dir1/")
        os.mkdir(copy_path)
        with open(file_path, 'a') as f:
            f.write("testing")
        pyfunc._copy_file_or_tree(file_path, copy_path, "")
        assert filecmp.cmp(file_path, os.path.join(copy_path, "test_file.txt"))


# Tests whether creating a directory works.
def test_dir_create():
    with TempDir() as tmp:
        file_path = tmp.path("test_file.txt")
        create_dir = tmp.path("test_dir2/")
        with open(file_path, 'a') as f:
            f.write("testing")
        name = pyfunc._copy_file_or_tree(file_path, file_path, create_dir)
        assert filecmp.cmp(file_path, name)


# Tests whether copying a directory works.
def test_dir_copy():
    with TempDir() as tmp:
        dir_path = tmp.path("test_dir1/")
        copy_path = tmp.path("test_dir2")
        os.mkdir(dir_path)
        with open(os.path.join(dir_path, "test_file.txt"), 'a') as f:
            f.write("testing")
        pyfunc._copy_file_or_tree(dir_path, copy_path, "")
        assert filecmp.dircmp(dir_path, copy_path)
