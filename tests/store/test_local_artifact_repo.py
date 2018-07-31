import os
import unittest

from mlflow.store.artifact_repo import ArtifactRepository, LocalArtifactRepository
from mlflow.utils.file_utils import TempDir


class TestLocalArtifactRepo(unittest.TestCase):
    def test_basic_functions(self):
        with TempDir() as test_root, TempDir() as tmp:
            repo = ArtifactRepository.from_artifact_uri(test_root.path())
            self.assertIsInstance(repo, LocalArtifactRepository)
            self.assertListEqual(repo.list_artifacts(), [])
            with self.assertRaises(Exception):
                open(repo.download_artifacts("test.txt")).read()

            # Create and log a test.txt file directly
            with open(tmp.path("test.txt"), "w") as f:
                f.write("Hello world!")
            repo.log_artifact(tmp.path("test.txt"))
            text = open(repo.download_artifacts("test.txt")).read()
            self.assertEqual(text, "Hello world!")
            # Check that it actually got written in the expected place
            text = open(os.path.join(test_root.path(), "test.txt")).read()
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
            infos = sorted([(f.path, f.is_dir, f.file_size) for f in repo.list_artifacts()])
            self.assertListEqual(infos, [
                ("a.txt", False, 1),
                ("b.txt", False, 1),
                ("nested", True, None),
                ("test.txt", False, 12)
            ])
            infos = sorted([(f.path, f.is_dir, f.file_size) for f in repo.list_artifacts("nested")])
            self.assertListEqual(infos, [("nested/c.txt", False, 1)])

            # Download a subdirectory
            downloaded_dir = repo.download_artifacts("nested")
            self.assertEqual(os.path.basename(downloaded_dir), "nested")
            text = open(os.path.join(downloaded_dir, "c.txt")).read()
            self.assertEqual(text, "C")
