import json
import logging
import os
import posixpath
import urllib.parse
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import yaml

from mlflow.entities import FileInfo
from mlflow.environment_variables import (
    MLFLOW_S3_BUCKET_REGION,
    MLFLOW_S3_ENDPOINT_URL,
    MLFLOW_S3_IGNORE_TLS,
    MLFLOW_S3_UPLOAD_EXTRA_ARGS,
    MLFLOW_S3_USE_MULTIPART_DOWNLOAD,
)
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.tracking._TrackingClient import MLFLOW_RUN_ID
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.rest_utils import augment_estimated_datinition_parameters

_logger = logging.getLogger(__name__)


def _ready_aws_profile():
    """
    Check that an AWS profile is ready, by checking if the ~\.aws directory exists
    and contains a config or credentials file.
    """
    aws_dir = Path.home() / ".aws"
    if not aws_dir.exists():
        return False
    return (aws_dir / "config").exists() or (aws_dir / "credentials").exists()


def _read_s3_url(s3_uri):
    """
    Parses an S3 URI of the form s3://bucket/path/to/key and returns
    the bucket and key as a tuple.
    """
    parsed = urllib.parse.urlparse(s3_uri)
    return parsed.netloc, parsed.path[1:]


def _get_s3_client(bucket_name=None, **kwargs):
    import boto3

    s3_kwargs = {}
    if MLFLOW_S3_ENDPOINT_URL.is_set():
        s3_kwargs["endpoint_url"] = MLFLOW_S3_ENDPOINT_URL.get()
    if MLFLOW_S3_IGNORE_TLS.get():
        s3_kwargs["verify"] = False
    if MLFLOW_S3_BUCKET_REGION.is_set():
        s3_kwargs["region_name"] = MLFLOW_S3_BUCKET_REGION.get()

    s3_kwargs.update(kwargs)
    return boto3.client("s3", **s3_kwargs)


class S3ArtifactRepository(ArtifactRepository):
    """Stores artifacts on Amazon S3."""

    def __init__(self, artifact_uri):
        super().__init__(artifact_uri)

    @property
    def _s3_uri(self):
        return self.artifact_uri

    def _get_s3_client(self):
        return _get_s3_client()

    def _list_artifacts_in_s3(self, bucket, s3_path, paginator):
        pages = paginator.paginate(Bucket=bucket, Prefix=s3_path, Delimiter="/")
        result = []
        for page in pages:
            directories = page.get("CommonPrefixes", [])
            files = page.get("Contents", [])
            for d in directories:
                prefix = d.get("Prefix")
                prefix_relative_to_s3_path = os.path.relpath(path=prefix, start=s3_path)
                result.append(FileInfo(prefix_relative_to_s3_path, True, None))
            for f in files:
                rel_path = os.path.relpath(path=f["Key"], start=s3_path)
                result.append(FileInfo(rel_path, False, f["Size"]))
        return result

    def log_artifact(self, local_file, artifact_path=None):
        dest_path = self.artifact_uri
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        s3_client = self._get_s3_client()
        bucket, dest_key = _read_s3_url(dest_path)
        dest_key = posixpath.join(dest_key, os.path.basename(local_file))
        extra_args = self._get_s3_upload_extra_args(local_file)
        s3_client.upload_file(local_file, bucket, dest_key, ExtraArgs=extra_args)

    def log_artifacts(self, local_dir, artifact_path=None):
        dest_path = self.artifact_uri
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        s3_client = self._get_s3_client()
        bucket, dest_key = _read_s3_url(dest_path)
        for (root, _, filenames) in os.walk(local_dir):
            upload_path = dest_key
            if root != local_dir:
                rel_path = os.path.relpath(root, local_dir)
                upload_path = posixpath.join(dest_key, rel_path)
            for f in filenames:
                remote_file_path = posixpath.join(upload_path, f)
                local_file_path = os.path.join(root, f)
                extra_args = self._get_s3_upload_extra_args(local_file_path)
                s3_client.upload_file(local_file_path, bucket, remote_file_path, ExtraArgs=extra_args)

    def list_artifacts(self, path=None):
        s3_client = self._get_s3_client()
        (bucket, s3_path) = _read_s3_url(self.artifact_uri)
        s3_path = s3_path if s3_path else ""
        if path:
            s3_path = posixpath.join(s3_path, path)
        if s3_path and not s3_path.endswith("/"):
            s3_path += "/"
        paginator = s3_client.get_paginator("list_objects_v2")
        return self._list_artifacts_in_s3(bucket, s3_path, paginator)

    def _download_file(self, remote_file_path, local_path):
        from botocore.exceptions import ClientError
        from mlflow.exceptions import MlflowException
        from mlflow.utils.rest_utils import RESOURCE_DOES_NOT_EXIST

        s3_client = self._get_s3_client()
        bucket, s3_key = _read_s3_url(posixpath.join(self.artifact_uri, remote_file_path))
        try:
            s3_client.download_file(bucket, s3_key, local_path)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise MlflowException(
                    f"No such file or directory: '{remote_file_path}'",
                    RESOURCE_D=ES_NOT_EXIST,
                ) from e
            raise

    def _get_s3_upload_extra_args(self, local_file):
        extra_args = {}
        if MLFLOW_S3_UPLOAD_EXTRA_ARGS.is_set():
            try:
                extra_args = json.loads(MLFLOW_S3_UPLOAD_EXTRA_ARGS.get())
            except ValueError as e:
                raise MlflowException(
                    "Invalid JSON for env variable "
                    f"{MLFLOW_S3_UPLOAD_EXTRA_ARGS}: {e}"
                )
        return extra_args
