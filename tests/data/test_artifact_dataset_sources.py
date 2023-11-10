import json
import os
from unittest import mock

import pytest

from mlflow.data.dataset_source_registry import get_dataset_source_from_json, resolve_dataset_source
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository


@pytest.mark.parametrize(
    ("source_uri", "source_type", "source_class_name"),
    [
        ("/tmp/path/to/my/local/file.txt", "local", "LocalArtifactDatasetSource"),
        ("file:///tmp/path/to/my/local/directory", "local", "LocalArtifactDatasetSource"),
        ("s3://mybucket/path/to/my/file.txt", "s3", "S3ArtifactDatasetSource"),
        ("gs://mybucket/path/to/my/dir", "gs", "GCSArtifactDatasetSource"),
        ("wasbs://user@host.blob.core.windows.net/dir", "wasbs", "AzureBlobArtifactDatasetSource"),
        ("ftp://mysite.com/path/to/my/file.txt", "ftp", "FTPArtifactDatasetSource"),
        ("sftp://mysite.com/path/to/my/dir", "sftp", "SFTPArtifactDatasetSource"),
        ("hdfs://host_name:8020/hdfs/path/to/my/file.txt", "hdfs", "HdfsArtifactDatasetSource"),
        ("viewfs://host_name:8020/path/to/my/dir", "viewfs", "HdfsArtifactDatasetSource"),
    ],
)
def test_expected_artifact_dataset_sources_are_registered_and_resolvable(
    source_uri, source_type, source_class_name
):
    dataset_source = resolve_dataset_source(source_uri)
    assert isinstance(dataset_source, FileSystemDatasetSource)
    assert dataset_source._get_source_type() == source_type
    assert type(dataset_source).__name__ == source_class_name
    assert type(dataset_source).__qualname__ == source_class_name
    assert dataset_source.uri == source_uri


@pytest.mark.parametrize(
    ("source_uri", "source_type"),
    [
        ("/tmp/path/to/my/local/file.txt", "local"),
        ("file:///tmp/path/to/my/local/directory", "local"),
        ("s3://mybucket/path/to/my/file.txt", "s3"),
        ("gs://mybucket/path/to/my/dir", "gs"),
        ("wasbs://user@host.blob.core.windows.net/dir", "wasbs"),
        ("ftp://mysite.com/path/to/my/file.txt", "ftp"),
        ("sftp://mysite.com/path/to/my/dir", "sftp"),
        ("hdfs://host_name:8020/hdfs/path/to/my/file.txt", "hdfs"),
        ("viewfs://host_name:8020/path/to/my/dir", "viewfs"),
    ],
)
def test_to_and_from_json(source_uri, source_type):
    dataset_source = resolve_dataset_source(source_uri)
    assert dataset_source._get_source_type() == source_type
    source_json = dataset_source.to_json()

    parsed_source_json = json.loads(source_json)
    assert parsed_source_json["uri"] == source_uri

    reloaded_source = get_dataset_source_from_json(
        source_json, source_type=dataset_source._get_source_type()
    )
    assert isinstance(reloaded_source, FileSystemDatasetSource)
    assert type(dataset_source) == type(reloaded_source)
    assert reloaded_source.uri == dataset_source.uri


@pytest.mark.parametrize(
    ("source_uri", "source_type"),
    [
        ("/tmp/path/to/my/local/file.txt", "local"),
        ("file:///tmp/path/to/my/local/directory", "local"),
        ("s3://mybucket/path/to/my/file.txt", "s3"),
        ("gs://mybucket/path/to/my/dir", "gs"),
        ("wasbs://user@host.blob.core.windows.net/dir", "wasbs"),
        ("ftp://mysite.com/path/to/my/file.txt", "ftp"),
        ("sftp://mysite.com/path/to/my/dir", "sftp"),
        ("hdfs://host_name:8020/hdfs/path/to/my/file.txt", "hdfs"),
        ("viewfs://host_name:8020/path/to/my/dir", "viewfs"),
    ],
)
def test_load_makes_expected_mlflow_artifacts_download_call(source_uri, source_type, tmp_path):
    dataset_source = resolve_dataset_source(source_uri)
    assert dataset_source._get_source_type() == source_type

    with mock.patch("mlflow.data.artifact_dataset_sources.download_artifacts") as download_imp_mock:
        dataset_source.load()
        download_imp_mock.assert_called_once_with(artifact_uri=source_uri, dst_path=None)

    with mock.patch("mlflow.data.artifact_dataset_sources.download_artifacts") as download_imp_mock:
        dataset_source.load(dst_path=str(tmp_path))
        download_imp_mock.assert_called_once_with(artifact_uri=source_uri, dst_path=str(tmp_path))


@pytest.mark.parametrize("dst_path", [None, "dst"])
def test_local_load(dst_path, tmp_path):
    if dst_path is not None:
        dst_path = str(tmp_path / dst_path)

    # Test string file paths
    file_path = str(tmp_path / "myfile.txt")
    with open(file_path, "w") as f:
        f.write("text")

    file_dataset_source = resolve_dataset_source(file_path)
    assert file_dataset_source._get_source_type() == "local"
    assert file_dataset_source.load(dst_path=dst_path) == dst_path or file_path
    with open(file_path) as f:
        assert f.read() == "text"

    # Test directory paths with pathlib.Path
    dir_path = tmp_path / "mydir"
    os.makedirs(dir_path)

    dir_dataset_source = resolve_dataset_source(dir_path)
    assert file_dataset_source._get_source_type() == "local"
    assert dir_dataset_source.load() == dst_path or str(dir_path)


@pytest.mark.parametrize("dst_path", [None, "dst"])
def test_s3_load(mock_s3_bucket, dst_path, tmp_path):
    if dst_path is not None:
        dst_path = str(tmp_path / dst_path)

    file_path = str(tmp_path / "myfile.txt")
    with open(file_path, "w") as f:
        f.write("text")

    S3ArtifactRepository(f"s3://{mock_s3_bucket}").log_artifact(file_path)

    s3_source_uri = f"s3://{mock_s3_bucket}/myfile.txt"
    s3_dataset_source = resolve_dataset_source(s3_source_uri)
    assert s3_dataset_source._get_source_type() == "s3"
    downloaded_source = s3_dataset_source.load(dst_path=dst_path)
    if dst_path is not None:
        assert downloaded_source == os.path.join(dst_path, "myfile.txt")
    with open(downloaded_source) as f:
        assert f.read() == "text"
