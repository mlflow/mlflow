#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import unittest
import six

from mlflow.utils import file_utils
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
        long_value = long(1) if six.PY2 else 1
        data = {"a": random_int(), "B": random_int(), "text_value": u"中文",
                "long_value": long_value, "int_value": 32}
        file_utils.write_yaml(self.test_folder, yaml_file, data)
        read_data = file_utils.read_yaml(self.test_folder, yaml_file)
        self.assertEqual(data, read_data)
        with open(file_utils.build_path(self.test_folder, yaml_file)) as handle:
            contents = handle.read()
        self.assertNotIn("!!python/unicode", contents)

    def test_mkdir(self):
        new_dir_name = "mkdir_test_%d" % random_int()
        file_utils.mkdir(self.test_folder, new_dir_name)
        self.assertEqual(os.listdir(self.test_folder), [new_dir_name])

        with self.assertRaises(OSError):
            file_utils.mkdir("/   bad directory @ name ", "ouch")
