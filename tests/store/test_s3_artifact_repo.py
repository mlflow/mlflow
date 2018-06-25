import os
import unittest

import boto3
from moto import mock_s3

from mlflow.store.artifact_repo import ArtifactRepository, S3ArtifactRepository
from mlflow.utils.file_utils import TempDir
from tests.helper_functions import random_int


class TestS3ArtifactRepo(unittest.TestCase):
    @mock_s3
    def test_basic_functions(self):
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket="test_bucket")

        repo = ArtifactRepository.from_artifact_uri("s3://test_bucket/some/path")
        self.assertIsInstance(repo, S3ArtifactRepository)
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
            # Check that it actually made it to S3
            obj = s3.get_object(Bucket="test_bucket", Key="some/path/test.txt")
            text = obj["Body"].read().decode('utf-8')
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
