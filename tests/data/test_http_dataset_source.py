import json
import os
import pytest

import pandas as pd

from mlflow.data.dataset_source_registry import resolve_dataset_source, get_dataset_source_from_json
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.exceptions import MlflowException


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
    assert "fixed acidity" in parsed1.columns
    assert len(parsed1) > 10

    loaded2 = source1.load(dst_path=tmp_path)
    assert loaded2 == str(tmp_path / "winequality-red.csv")
    parsed2 = pd.read_csv(loaded2, sep=";")
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
