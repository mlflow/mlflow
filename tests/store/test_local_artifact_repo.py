import os
import shutil
import unittest

from mlflow.store.artifact_repo import ArtifactRepository, LocalArtifactRepository
from mlflow.utils.file_utils import TempDir
from tests.helper_functions import random_int


class TestLocalArtifactRepo(unittest.TestCase):
    ROOT_LOCATION = "/tmp"

    def setUp(self):
        self._create_root(TestLocalArtifactRepo.ROOT_LOCATION)

    def _create_root(self, root):
        self.test_root = os.path.abspath(os.path.join(root, "test_local_repo_%d" % random_int()))
        os.mkdir(self.test_root)

    def test_basic_functions(self):
        repo = ArtifactRepository.from_artifact_uri(self.test_root)
        self.assertIsInstance(repo, LocalArtifactRepository)
        self.assertListEqual(repo.list_artifacts(), [])
        with self.assertRaises(Exception):
            open(repo.download_artifacts("test.txt")).read()

        with TempDir() as tmp:
            # Create and log a test.txt file directly
            with open(tmp.path("test.txt"), "w") as f:
                f.write("Hello world!")
            repo.log_artifact(tmp.path("test.txt"))
            text = open(repo.download_artifacts("test.txt")).read()
            self.assertEqual(text, "Hello world!")
            # Check that it actually got written in the expected place
            text = open(os.path.join(self.test_root, "test.txt")).read()
            self.assertEqual(text, "Hello world!")

            # Create a subdirectory for log_artifacts
            os.mkdir(tmp.path("subdir"))
            os.mkdir(tmp.path("subdir", "nested"))
            with open(tmp.path("subdir", "a.txt"), "w") as f:
                f.write("A")
            with open(tmp.path("subdir", "b.txt"), "w") as f:
                f.write("B")
            with open(tmp.path("subdir", "nested", "c.txt"), "w") as f:
                f.write("C")
            repo.log_artifacts(tmp.path("subdir"))
            text = open(repo.download_artifacts("a.txt")).read()
            self.assertEqual(text, "A")
            text = open(repo.download_artifacts("b.txt")).read()
            self.assertEqual(text, "B")
            text = open(repo.download_artifacts("nested/c.txt")).read()
            self.assertEqual(text, "C")
            paths = sorted([f.path for f in repo.list_artifacts()])
            self.assertListEqual(paths, ["a.txt", "b.txt", "nested", "test.txt"])
            paths = sorted([f.path for f in repo.list_artifacts("nested")])
            self.assertListEqual(paths, ["nested/c.txt"])

    def tearDown(self):
        shutil.rmtree(self.test_root, ignore_errors=True)
