import os
import shutil
import six
import tempfile
import unittest
import warnings

import mock
import pytest
import sqlalchemy
import time
import mlflow
import uuid

import mlflow.db
from mlflow.entities import ViewType, RunTag, SourceType, RunStatus, Experiment, Metric, Param
from mlflow.protos.databricks_pb2 import ErrorCode, RESOURCE_DOES_NOT_EXIST, \
    INVALID_PARAMETER_VALUE, INTERNAL_ERROR
from mlflow.store import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.db.utils import _get_schema_version
from mlflow.store.dbmodels import initial_artifact_store_models
from mlflow import entities
from mlflow.exceptions import MlflowException
from mlflow.store.db_artifact_repo import DBArtifactRepository
from mlflow.utils import extract_db_type_from_uri
from tests.resources.db.initial_models import Base as InitialBase
from tests.integration.utils import invoke_cli_runner
from mlflow.utils.file_utils import TempDir

DB_URI = 'sqlite:///'


class TestSqlAlchemyStoreSqlite(unittest.TestCase):

    def _get_store(self, db_uri=''):
        return DBArtifactRepository(db_uri)

    def setUp(self):
        self.maxDiff = None  # print all differences on assert failures
        fd, self.temp_dbfile = tempfile.mkstemp()
        # Close handle immediately so that we can remove the file later on in Windows
        os.close(fd)
        self.db_url = "%s%s" % (DB_URI, self.temp_dbfile)
        self.store = self._get_store(self.db_url)

    def test_log_artifact(self):
        with TempDir() as tmp_dir:
            local_file = tmp_dir.path('model')
            with open(local_file, "w") as f:
                f.write('DB store Test')

            self.store.log_artifact(local_file, 'more_path/some')
            with self.store.ManagedSessionMaker() as session:
                result = session.query(initial_artifact_store_models.SqlArtifact).all()
                self.assertEqual(len(result), 1)

                test_art = session.query(initial_artifact_store_models.SqlArtifact).\
                    filter_by(artifact_name='model').first()
                self.assertEqual(str(test_art.artifact_id), "1")
                self.assertEqual(test_art.artifact_name, 'model')
                self.assertEqual(test_art.group_name, 'more_path/some')
                self.assertEqual(test_art.artifact_content, open(
                    local_file, "rb").read())

    def test_log_artifacts(self):

        with TempDir() as root_dir:
            with open(root_dir.path("file_one.txt"), "w") as f:
                f.write('DB store Test One')

            os.mkdir(root_dir.path("subdir"))
            with open(root_dir.path("subdir/file_two.txt"), "w") as f:
                f.write('DB store Test Two')

            self.store.log_artifacts(root_dir._path, 'new_path/path')

            with self.store.ManagedSessionMaker() as session:
                result = session.query(initial_artifact_store_models.SqlArtifact).all()
                self.assertEqual(len(result), 2)

                test_art = session.query(initial_artifact_store_models.SqlArtifact).\
                    filter_by(artifact_name='file_one.txt').first()
                self.assertEqual(str(test_art.artifact_id), "1")
                self.assertEqual(test_art.artifact_name, 'file_one.txt')
                self.assertEqual(test_art.group_name, 'new_path/path')
                self.assertEqual(test_art.artifact_content, open(
                    root_dir.path("file_one.txt"), "rb").read())

                test_art = session.query(initial_artifact_store_models.SqlArtifact). \
                    filter_by(artifact_name='file_two.txt').first()
                self.assertEqual(str(test_art.artifact_id), "2")
                self.assertEqual(test_art.artifact_name, 'file_two.txt')
                self.assertEqual(test_art.group_name, 'new_path/path/subdir')
                self.assertEqual(test_art.artifact_content, open(
                    root_dir.path("subdir/file_two.txt"), "rb").read())