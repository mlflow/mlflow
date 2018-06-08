import filecmp
import os
import unittest

from mlflow import pyfunc
from mlflow.utils.file_utils import TempDir

class TestCopyFileOrTree(unittest.TestCase):
    def setUp(self):
    	with TempDir(chdr=True, remove_on_exit=True) as tmp:
    		file_path = tmp.path("test_file")
    		copy_path = tmp.path("copy_file")
    		with open(file_path, 'a') as f:
    			f.write("testing")
    		pyfunc._copy_file_or_tree(file_path, copy_path, None)
    		assert filecmp.cmp(file_path, copy_path, shallow=False)
    		