from __future__ import print_function

import os
import re

from six.moves import urllib

from mlflow.utils import process
from mlflow.store.artifact.hdfs_artifact_repo import HdfsArtifactRepository

DBFS_PREFIX = "dbfs:/"
S3_PREFIX = "s3://"
GS_PREFIX = "gs://"
VIEWFS_PREFIX = "viewfs://"
HDFS_PREFIX = "hdfs://"

DBFS_REGEX = re.compile("^%s" % re.escape(DBFS_PREFIX))
S3_REGEX = re.compile("^%s" % re.escape(S3_PREFIX))
GS_REGEX = re.compile("^%s" % re.escape(GS_PREFIX))
VIEWFS_REGEX = re.compile("^%s" % re.escape(VIEWFS_PREFIX))
HDFS_REGEX = re.compile("^%s" % re.escape(HDFS_PREFIX))


class DownloadException(Exception):
    pass


def parse_simple_uri(uri, scheme):
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != scheme:
        raise Exception("Not an %s URI: %s" % (scheme.upper(), uri))
    path = parsed.path
    if path.startswith('/'):
        path = path[1:]
    return parsed.netloc, path


def _fetch_dbfs(uri, local_path):
    print(
        "=== Downloading DBFS file %s to local path %s ===" %
        (uri, os.path.abspath(local_path))
    )
    process.exec_cmd(cmd=["databricks", "fs", "cp", "-r", uri, local_path])


def _fetch_s3(uri, local_path):
    import boto3
    print("=== Downloading S3 object %s to local path %s ===" % (uri, os.path.abspath(local_path)))
    (bucket, s3_path) = parse_simple_uri(uri, "s3")
    boto3.client('s3').download_file(bucket, s3_path, local_path)


def _fetch_gs(uri, local_path):
    from google.cloud import storage
    print(
        "=== Downloading GCS file %s to local path %s ===" %
        (uri, os.path.abspath(local_path))
    )
    (bucket, gs_path) = parse_simple_uri(uri, "gs")
    storage.Client().bucket(bucket).blob(gs_path).download_to_filename(local_path)


def _fetch_viewfs(uri, local_path):
    print(
        "=== Downloading VIEWFS file %s to local path %s ===" %
        (uri, os.path.abspath(local_path))
    )
    (bucket, hdfs_path) = parse_simple_uri(uri, "viewfs")

    hdfs = HdfsArtifactRepository(bucket)
    hdfs.download_artifacts(hdfs_path, local_path)


def _fetch_hdfs(uri, local_path):
    print(
        "=== Downloading HDFS file %s to local path %s ===" %
        (uri, os.path.abspath(local_path))
    )
    (bucket, hdfs_path) = parse_simple_uri(uri, "hdfs")

    hdfs = HdfsArtifactRepository(bucket)
    hdfs.download_artifacts(hdfs_path, local_path)


def is_uri(string):
    parsed_uri = urllib.parse.urlparse(string)
    return len(parsed_uri.scheme) > 0


def download_uri(uri, output_path):
    if DBFS_REGEX.match(uri):
        _fetch_dbfs(uri, output_path)
    elif S3_REGEX.match(uri):
        _fetch_s3(uri, output_path)
    elif GS_REGEX.match(uri):
        _fetch_gs(uri, output_path)
    elif VIEWFS_REGEX.match(uri):
        _fetch_viewfs(uri, output_path)
    elif HDFS_REGEX.match(uri):
        _fetch_hdfs(uri, output_path)
    else:
        raise DownloadException(
            "`uri` must be a DBFS (%s), S3 (%s), GCS (%s), VIEWFS (%s) URI, got "
            "%s" % (DBFS_PREFIX, S3_PREFIX, GS_PREFIX, VIEWFS_PREFIX, uri)
        )
