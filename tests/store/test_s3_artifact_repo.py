import os
import unittest

import boto3
from mock import Mock
from moto import mock_s3

from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.store.s3_artifact_repo import S3ArtifactRepository
from mlflow.utils.file_utils import TempDir


class TestS3ArtifactRepo(unittest.TestCase):
    @mock_s3
    def test_basic_functions(self):
        with TempDir() as tmp:
            # Create a mock S3 bucket in moto
            # Note that we must set these as environment variables in case users
            # so that boto does not attempt to assume credentials from the ~/.aws/config
            # or IAM role. moto does not correctly pass the arguments to boto3.client().
            os.environ["AWS_ACCESS_KEY_ID"] = "a"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "b"
            s3 = boto3.client("s3")
            s3.create_bucket(Bucket="test_bucket")

            repo = ArtifactRepository.from_artifact_uri("s3://test_bucket/some/path", Mock())
            self.assertIsInstance(repo, S3ArtifactRepository)
            self.assertListEqual(repo.list_artifacts(), [])
            with self.assertRaises(Exception):
                open(repo.download_artifacts("test.txt")).read()

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

            # Download the root directory
            downloaded_dir = repo.download_artifacts("")
            dir_contents = os.listdir(downloaded_dir)
            assert "nested" in dir_contents
            assert os.path.isdir(os.path.join(downloaded_dir, "nested"))
            assert "a.txt" in dir_contents
            assert "b.txt" in dir_contents
