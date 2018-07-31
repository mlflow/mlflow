from __future__ import print_function

import os
import re

import click
from six.moves import urllib

from mlflow.utils import process

DBFS_PREFIX = "dbfs:/"
S3_PREFIX = "s3://"
DBFS_REGEX = re.compile("^%s" % re.escape(DBFS_PREFIX))
S3_REGEX = re.compile("^%s" % re.escape(S3_PREFIX))


class DownloadException(Exception):
    pass


def _fetch_dbfs(uri, local_path):
    print("=== Downloading DBFS file %s to local path %s ===" % (uri, os.path.abspath(local_path)))
    process.exec_cmd(cmd=["databricks", "fs", "cp", "--overwrite", uri, local_path])


def _fetch_s3(uri, local_path):
    print("=== Downloading S3 object %s to local path %s ===" % (uri, os.path.abspath(local_path)))
    process.exec_cmd(cmd=["aws", "s3", "cp", uri, local_path])


def is_uri(string):
    parsed_uri = urllib.parse.urlparse(string)
    return len(parsed_uri.scheme) > 0


def download_uri(uri, output_path):
    if DBFS_REGEX.match(uri):
        _fetch_dbfs(uri, output_path)
    elif S3_REGEX.match(uri):
        _fetch_s3(uri, output_path)
    else:
        raise DownloadException("`uri` must be a DBFS (%s) or S3 (%s) URI, got "
                                "%s" % (DBFS_PREFIX, S3_PREFIX, uri))


@click.command("download")
@click.argument("uri")
@click.option("--output-path", "-o", metavar="PATH",
              help="Output path into which to download the artifact.")
def download(uri, output_path):
    """
    Download the artifact at the specified DBFS or S3 URI into the specified local output path, or
    the current directory if no output path is specified.
    """
    if output_path is None:
        output_path = os.path.basename(uri)
    download_uri(uri, output_path)
