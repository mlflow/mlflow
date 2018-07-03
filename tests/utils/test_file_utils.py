import os
import shutil
import tempfile
import unittest

import mlflow
from mlflow.utils import file_utils
from mlflow.utils.file_utils import TempDir

from tests.helper_functions import random_int, random_file


class TestFileUtils(unittest.TestCase):
    TEST_ROOT = "/tmp"

    def setUp(self):
        self.test_folder = os.path.join(TestFileUtils.TEST_ROOT, "test_folder_%d" % random_int())
        os.mkdir(self.test_folder)

    def tearDown(self):
        shutil.rmtree(self.test_folder, ignore_errors=True)

    def test_yaml_read_and_write(self):
        yaml_file = random_file("yaml")
        data = {"a": random_int(), "B": random_int()}
        file_utils.write_yaml(self.test_folder, yaml_file, data)
        read_data = file_utils.read_yaml(self.test_folder, yaml_file)
        self.assertEqual(data, read_data)

    def test_mkdir(self):
        new_dir_name = "mkdir_test_%d" % random_int()
        file_utils.mkdir(self.test_folder, new_dir_name)
        self.assertEqual(os.listdir(self.test_folder), [new_dir_name])

        with self.assertRaises(OSError):
            file_utils.mkdir("/   bad directory @ name ", "ouch")

    def test_make_tarfile(self):
        with TempDir() as tmp:
            dst_dir = tmp.path()
            mlflow.projects._fetch_project(uri=TEST_PROJECT_DIR, version=None, dst_dir=dst_dir)
            dir_comparison = filecmp.dircmp(TEST_PROJECT_DIR, dst_dir)
            assert len(dir_comparison.left_only) == 0
            assert len(dir_comparison.right_only) == 0
            assert len(dir_comparison.diff_files) == 0
            assert len(dir_comparison.funny_files) == 0
            # Make a tarfile of a project, fetch the project into a working directory, tar it again,
            # verify they're the same
            temp_file = tempfile.mktemp()
            file_utils.make_tarfile(output_filename=temp_file, source_dir="")

        pass