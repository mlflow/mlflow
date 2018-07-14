import os
import mock
import unittest

from google.cloud.storage import client as gcs_client

from mlflow.store.artifact_repo import ArtifactRepository, GCSArtifactRepository
from mlflow.utils.file_utils import TempDir

class TestGCSArtifactRepo(unittest.TestCase):
    def setUp(self):
        # Make sure that the environment variable isn't set to actually make calls
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/dev/null'

        self.gcs = mock.MagicMock(autospec=gcs_client)

    def tearDown(self):
        self.gcs.stop()

    def test_artifact_uri_factory(self):
        repo = ArtifactRepository.from_artifact_uri("gs://test_bucket/some/path")
        self.assertIsInstance(repo, GCSArtifactRepository)

    def test_list_artifacts_empty(self):
        repo = GCSArtifactRepository("gs://test_bucket/some/path", self.gcs)
        self.gcs.Client.return_value.get_bucket.return_value\
            .list_blobs.return_value = []
        self.assertListEqual(repo.list_artifacts(), [])

    def test_list_artifacts(self):
        repo = GCSArtifactRepository("gs://test_bucket/some/path", self.gcs)
        mockobj = mock.Mock()
        mockobj.configure_mock(
                name='/some/path/mockeryname',
                f='/mockeryname',
                size=1,
        )
        self.gcs.Client.return_value.get_bucket.return_value\
            .list_blobs.return_value = [mockobj]
        self.assertEqual(repo.list_artifacts()[0].path, mockobj.f)
        self.assertEqual(repo.list_artifacts()[0].file_size, mockobj.size)

    def test_log_artifact(self):
        repo = GCSArtifactRepository("gs://test_bucket/some/path", self.gcs)
        with TempDir() as tmp:
            if not os.path.exists(tmp.path()):
                os.makedirs(tmp.path())
            with open(tmp.path("test.txt"), "w") as f:
                f.write("Hello world!")

            self.gcs.Client.return_value.get_bucket.return_value\
                .upload_from_filename.side_effect = lambda f: os.path.isfile(f)
            repo.log_artifact(tmp.path('test.txt'))
