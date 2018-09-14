import unittest

from mlflow.entities import FileInfo
from tests.helper_functions import random_str, random_int


class TestFileInfo(unittest.TestCase):
    def _check(self, fi, path, is_dir, size_in_bytes):
        self.assertIsInstance(fi, FileInfo)
        self.assertEqual(fi.path, path)
        self.assertEqual(fi.is_dir, is_dir)
        self.assertEqual(fi.file_size, size_in_bytes)

    def test_creation_and_hydration(self):
        path = random_str(random_int(10, 50))
        is_dir = random_int(10, 2500) % 2 == 0
        size_in_bytes = random_int(1, 10000)
        fi1 = FileInfo(path, is_dir, size_in_bytes)
        self._check(fi1, path, is_dir, size_in_bytes)

        as_dict = {"path": path, "is_dir": is_dir, "file_size": size_in_bytes}
        self.assertEqual(dict(fi1), as_dict)

        proto = fi1.to_proto()
        fi2 = FileInfo.from_proto(proto)
        self._check(fi2, path, is_dir, size_in_bytes)

        fi3 = FileInfo.from_dictionary(as_dict)
        self._check(fi3, path, is_dir, size_in_bytes)
