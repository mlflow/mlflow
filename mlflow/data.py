import os
import re
import urllib.parse
import logging

from mlflow.utils import process

DBFS_PREFIX = "dbfs:/"
S3_PREFIX = "s3://"
GS_PREFIX = "gs://"
DBFS_REGEX = re.compile("^%s" % re.escape(DBFS_PREFIX))
S3_REGEX = re.compile("^%s" % re.escape(S3_PREFIX))
GS_REGEX = re.compile("^%s" % re.escape(GS_PREFIX))

_logger = logging.getLogger(__name__)


class DownloadException(Exception):
    pass


def _fetch_dbfs(uri, local_path):
    _logger.info(
        "=== Downloading DBFS file %s to local path %s ===", uri, os.path.abspath(local_path)
    )
    process._exec_cmd(cmd=["databricks", "fs", "cp", "-r", uri, local_path])


def _fetch_s3(uri, local_path):
    import boto3

    _logger.info(
        "=== Downloading S3 object %s to local path %s ===", uri, os.path.abspath(local_path)
    )

    client_kwargs = {}
    endpoint_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
    ignore_tls = os.environ.get("MLFLOW_S3_IGNORE_TLS")

    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    if ignore_tls:
        client_kwargs["verify"] = ignore_tls.lower() not in ["true", "yes", "1"]

    (bucket, s3_path) = parse_s3_uri(uri)
    boto3.client("s3", **client_kwargs).download_file(bucket, s3_path, local_path)


def _fetch_gs(uri, local_path):
    from google.cloud import storage

    _logger.info(
        "=== Downloading GCS file %s to local path %s ===", uri, os.path.abspath(local_path)
    )
    (bucket, gs_path) = parse_gs_uri(uri)
    storage.Client().bucket(bucket).blob(gs_path).download_to_filename(local_path)


def parse_s3_uri(uri):
    """Parse an S3 URI, returning (bucket, path)"""
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "s3":
        raise Exception("Not an S3 URI: %s" % uri)
    path = parsed.path
    if path.startswith("/"):
        path = path[1:]
    return parsed.netloc, path


def parse_gs_uri(uri):
    """Parse an GCS URI, returning (bucket, path)"""
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "gs":
        raise Exception("Not a GCS URI: %s" % uri)
    path = parsed.path
    if path.startswith("/"):
        path = path[1:]
    return parsed.netloc, path


def is_uri(string):
    parsed_uri = urllib.parse.urlparse(string)
    return len(parsed_uri.scheme) > 0
