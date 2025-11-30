import json
import os
from unittest import mock

import pandas as pd
import pytest

from mlflow.data.dataset_source_registry import get_dataset_source_from_json, resolve_dataset_source
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.utils.os import is_windows
from mlflow.utils.rest_utils import cloud_storage_http_request


def test_source_to_and_from_json():
    url = "http://mywebsite.com/path/to/my/dataset.txt"
    source = HTTPDatasetSource(url)
    assert source.to_json() == json.dumps({"url": url})

    reloaded_source = get_dataset_source_from_json(
        source.to_json(), source_type=source._get_source_type()
    )
    assert isinstance(reloaded_source, HTTPDatasetSource)
    assert type(source) == type(reloaded_source)
    assert source.url == reloaded_source.url == url


def test_http_dataset_source_is_registered_and_resolvable():
    source1 = resolve_dataset_source(
        "http://mywebsite.com/path/to/my/dataset.txt", candidate_sources=[HTTPDatasetSource]
    )
    assert isinstance(source1, HTTPDatasetSource)
    assert source1.url == "http://mywebsite.com/path/to/my/dataset.txt"

    source2 = resolve_dataset_source(
        "https://otherwebsite.net", candidate_sources=[HTTPDatasetSource]
    )
    assert isinstance(source2, HTTPDatasetSource)
    assert source2.url == "https://otherwebsite.net"

    with pytest.raises(MlflowException, match="Could not find a source information resolver"):
        resolve_dataset_source("s3://mybucket", candidate_sources=[HTTPDatasetSource])

    with pytest.raises(MlflowException, match="Could not find a source information resolver"):
        resolve_dataset_source("otherscheme://mybucket", candidate_sources=[HTTPDatasetSource])

    with pytest.raises(MlflowException, match="Could not find a source information resolver"):
        resolve_dataset_source("htp://mybucket", candidate_sources=[HTTPDatasetSource])


def test_source_load(tmp_path):
    source1 = HTTPDatasetSource(
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )

    loaded1 = source1.load()
    parsed1 = pd.read_csv(loaded1, sep=";")
    # Verify that the expected data was downloaded by checking for an expected column and asserting
    # that several rows are present
    assert "fixed acidity" in parsed1.columns
    assert len(parsed1) > 10

    loaded2 = source1.load(dst_path=tmp_path)
    assert loaded2 == str(tmp_path / "winequality-red.csv")
    parsed2 = pd.read_csv(loaded2, sep=";")
    # Verify that the expected data was downloaded by checking for an expected column and asserting
    # that several rows are present
    assert "fixed acidity" in parsed2.columns
    assert len(parsed1) > 10

    source2 = HTTPDatasetSource(
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv#foo?query=param"
    )
    loaded3 = source2.load(dst_path=tmp_path)
    assert loaded3 == str(tmp_path / "winequality-red.csv")
    parsed3 = pd.read_csv(loaded3, sep=";")
    assert "fixed acidity" in parsed3.columns
    assert len(parsed1) > 10

    source3 = HTTPDatasetSource("https://github.com/")
    loaded4 = source3.load()
    assert os.path.exists(loaded4)
    assert os.path.basename(loaded4) == "dataset_source"

    source4 = HTTPDatasetSource("https://github.com")
    loaded5 = source4.load()
    assert os.path.exists(loaded5)
    assert os.path.basename(loaded5) == "dataset_source"

    def cloud_storage_http_request_with_fast_fail(*args, **kwargs):
        kwargs["max_retries"] = 1
        kwargs["timeout"] = 5
        return cloud_storage_http_request(*args, **kwargs)

    source5 = HTTPDatasetSource("https://nonexistentwebsitebuiltbythemlflowteam112312.com")
    with (
        mock.patch(
            "mlflow.data.http_dataset_source.cloud_storage_http_request",
            side_effect=cloud_storage_http_request_with_fast_fail,
        ),
        pytest.raises(Exception, match="Max retries exceeded with url"),
    ):
        source5.load()


@pytest.mark.parametrize(
    ("attachment_filename", "expected_filename"),
    [
        ("testfile.txt", "testfile.txt"),
        ('"testfile.txt"', "testfile.txt"),
        ("'testfile.txt'", "testfile.txt"),
        (None, "winequality-red.csv"),
    ],
)
def test_source_load_with_content_disposition_header(attachment_filename, expected_filename):
    def download_with_mock_content_disposition_headers(*args, **kwargs):
        response = cloud_storage_http_request(*args, **kwargs)
        if attachment_filename is not None:
            response.headers["Content-Disposition"] = f"attachment; filename={attachment_filename}"
        else:
            response.headers["Content-Disposition"] = "attachment"
        return response

    with mock.patch(
        "mlflow.data.http_dataset_source.cloud_storage_http_request",
        side_effect=download_with_mock_content_disposition_headers,
    ):
        source = HTTPDatasetSource(
            "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
        )
        source.load()
        loaded = source.load()
        assert os.path.exists(loaded)
        assert os.path.basename(loaded) == expected_filename


@pytest.mark.parametrize(
    "filename",
    [
        "/foo/bar.txt",
        "./foo/bar.txt",
        "../foo/bar.txt",
        "foo/bar.txt",
    ],
)
def test_source_load_with_content_disposition_header_invalid_filename(filename):
    def download_with_mock_content_disposition_headers(*args, **kwargs):
        response = cloud_storage_http_request(*args, **kwargs)
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        return response

    with mock.patch(
        "mlflow.data.http_dataset_source.cloud_storage_http_request",
        side_effect=download_with_mock_content_disposition_headers,
    ):
        source = HTTPDatasetSource(
            "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
        )

        with pytest.raises(MlflowException, match="Invalid filename in Content-Disposition header"):
            source.load()


@pytest.mark.skipif(not is_windows(), reason="This test only passes on Windows")
@pytest.mark.parametrize(
    "filename",
    [
        r"..\..\poc.txt",
        r"Users\User\poc.txt",
    ],
)
def test_source_load_with_content_disposition_header_invalid_filename_windows(filename):
    def download_with_mock_content_disposition_headers(*args, **kwargs):
        response = cloud_storage_http_request(*args, **kwargs)
        response.headers = {"Content-Disposition": f"attachment; filename={filename}"}
        return response

    with mock.patch(
        "mlflow.data.http_dataset_source.cloud_storage_http_request",
        side_effect=download_with_mock_content_disposition_headers,
    ):
        source = HTTPDatasetSource(
            "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
        )

        # Expect an MlflowException for invalid filenames
        with pytest.raises(MlflowException, match="Invalid filename in Content-Disposition header"):
            source.load()
