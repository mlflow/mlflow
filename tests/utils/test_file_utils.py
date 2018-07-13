#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
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
        long_value = long(1) if six.PY2 else 1  # pylint: disable=undefined-variable
        data = {"a": random_int(), "B": random_int(), "text_value": u"中文",
                "long_value": long_value, "int_value": 32, "text_value_2": u"hi"}
        file_utils.write_yaml(self.test_folder, yaml_file, data)
        read_data = file_utils.read_yaml(self.test_folder, yaml_file)
        self.assertEqual(data, read_data)
        yaml_path = file_utils.build_path(self.test_folder, yaml_file)
        with codecs.open(yaml_path, encoding="utf-8") as handle:
            contents = handle.read()
        self.assertNotIn("!!python", contents)
        # Check that UTF-8 strings are written properly to the file (rather than as ASCII
        # representations of their byte sequences).
        self.assertIn(u"中文", contents)

    def test_mkdir(self):
        new_dir_name = "mkdir_test_%d" % random_int()
        file_utils.mkdir(self.test_folder, new_dir_name)
        self.assertEqual(os.listdir(self.test_folder), [new_dir_name])

        with self.assertRaises(OSError):
            file_utils.mkdir("/   bad directory @ name ", "ouch")
