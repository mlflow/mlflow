from concurrent.futures import as_completed
from datetime import datetime
from functools import lru_cache
import logging
import math
import os
import requests
import json
from mimetypes import guess_type

import posixpath
import urllib.parse

from mlflow.entities import FileInfo
from mlflow.environment_variables import (
    MLFLOW_S3_UPLOAD_EXTRA_ARGS,
    MLFLOW_S3_ENDPOINT_URL,
    MLFLOW_S3_IGNORE_TLS,
)
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.cloud_artifact_repo import (
    CloudArtifactRepository,
    _MULTIPART_UPLOAD_CHUNK_SIZE,
)
from mlflow.utils import data_utils
from mlflow.utils.file_utils import read_chunk, relative_path_to_artifact_path
from mlflow.utils.rest_utils import augmented_raise_for_status
from mlflow.utils.request_utils import cloud_storage_http_request

from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository, _get_s3_client, MLFLOW_S3_ENDPOINT_URL

_logger = logging.getLogger(__name__)

_MAX_CACHE_SECONDS = 300


class R2ArtifactRepository(S3ArtifactRepository):
    """Stores artifacts on Cloudflare R2."""

    def __init__(
            self, artifact_uri, access_key_id=None, secret_access_key=None, session_token=None
    ):
        super().__init__(artifact_uri)
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._session_token = session_token
        self.bucket, self.bucket_path = self.parse_s3_uri(self.artifact_uri)

    def parse_s3_uri(self, uri):
        """Parse an S3 URI, returning (bucket, path)"""
        return "smurching-dev-models", "models"

    def _get_s3_client(self):
        return _get_s3_client(
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            session_token=self._session_token,
        )
