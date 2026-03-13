import numpy as np
import pytest

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.histogram_utils import (
    HistogramData,
    compute_histogram_from_values,
    load_histogram_from_json,
    save_histogram_to_json,
)


def test_histogram_data_creation():
    bin_edges = [0.0, 1.0, 2.0, 3.0]
    counts = [10.0, 20.0, 30.0]
    histogram = HistogramData(
        name="test_hist",
        step=0,
        timestamp=1000,
        bin_edges=bin_edges,
        counts=counts,
        min_value=0.0,
        max_value=3.0,
    )

    assert histogram.name == "test_hist"
    assert histogram.step == 0
    assert histogram.timestamp == 1000
    assert histogram.bin_edges == bin_edges
    assert histogram.counts == counts
    assert histogram.min_value == 0.0
    assert histogram.max_value == 3.0


def test_histogram_data_validation():
    bin_edges = [0.0, 1.0, 2.0]
    counts = [10.0, 20.0]

    histogram = HistogramData(
        name="test",
        step=0,
        timestamp=1000,
        bin_edges=bin_edges,
        counts=counts,
    )
    assert histogram is not None

    with pytest.raises(ValueError, match="bin_edges must have length n\\+1"):
        HistogramData(
            name="test",
            step=0,
            timestamp=1000,
            bin_edges=[0.0, 1.0],
            counts=[10.0, 20.0],
        )


def test_compute_histogram_from_values():
    values = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    bin_edges, counts, min_val, max_val = compute_histogram_from_values(values, num_bins=5)

    assert len(bin_edges) == 6  # 5 bins + 1
    assert len(counts) == 5
    assert min_val == 1.0
    assert max_val == 5.0
    assert np.sum(counts) == len(values)


def test_compute_histogram_with_nan_values():
    values = np.array([1.0, 2.0, np.nan, 3.0, np.nan, 4.0])
    bin_edges, counts, min_val, max_val = compute_histogram_from_values(values, num_bins=3)

    assert len(bin_edges) == 4
    assert len(counts) == 3
    assert np.sum(counts) == 4


def test_compute_histogram_all_same_values():
    values = np.array([5.0, 5.0, 5.0, 5.0])
    bin_edges, counts, min_val, max_val = compute_histogram_from_values(values, num_bins=10)

    assert len(bin_edges) == 2
    assert len(counts) == 1
    assert counts[0] == 4.0
    assert min_val == 5.0
    assert max_val == 5.0


def test_save_and_load_histogram_json(tmp_path):
    hist = HistogramData(
        name="test",
        step=0,
        timestamp=1000,
        bin_edges=[0.0, 1.0],
        counts=[10.0],
    )
    file_path = tmp_path / "histogram.json"
    save_histogram_to_json(hist, file_path)

    loaded = load_histogram_from_json(file_path)
    assert loaded.name == "test"
    assert loaded.step == 0
    assert loaded.counts == [10.0]


def test_log_histogram_from_values():
    with mlflow.start_run():
        values = np.random.randn(1000)
        mlflow.log_histogram(values, key="test_histogram", step=0)

        run_id = mlflow.active_run().info.run_id

    client = MlflowClient()
    artifacts = client.list_artifacts(run_id, path="histograms")
    assert len(artifacts) > 0
    assert any("test_histogram" in art.path for art in artifacts)


def test_log_histogram_multiple_steps():
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id

        for step in range(5):
            values = np.random.randn(100) * (step + 1)
            mlflow.log_histogram(values, key="weights/layer1", step=step)

    client = MlflowClient()
    histograms = client.get_histogram(run_id, key="weights/layer1")
    assert len(histograms) == 5
    assert all(h.step == i for i, h in enumerate(histograms))


def test_log_histogram_with_slashes_in_name():
    with mlflow.start_run():
        values = np.random.randn(100)
        mlflow.log_histogram(values, key="weights/conv1/bias", step=0)

        run_id = mlflow.active_run().info.run_id

    client = MlflowClient()
    artifacts = client.list_artifacts(run_id, path="histograms")
    artifact_paths = [art.path for art in artifacts]
    assert any("weights_conv1_bias" in path for path in artifact_paths)


def test_log_histogram_validates_values_type():
    with mlflow.start_run():
        with pytest.raises(TypeError, match="'values' must be a numpy array or list"):
            mlflow.log_histogram("not_an_array", key="test")


def test_log_histogram_custom_num_bins():
    with mlflow.start_run():
        values = np.random.randn(1000)
        mlflow.log_histogram(values, key="custom_bins", step=0, num_bins=50)

        run_id = mlflow.active_run().info.run_id

    client = MlflowClient()
    histograms = client.get_histogram(run_id, key="custom_bins")
    assert len(histograms) == 1
    assert len(histograms[0].bin_edges) == 51
    assert len(histograms[0].counts) == 50


def test_histogram_data_to_from_dict():
    histogram = HistogramData(
        name="test",
        step=5,
        timestamp=123456789,
        bin_edges=[0.0, 1.0, 2.0],
        counts=[10.0, 20.0],
        min_value=0.0,
        max_value=2.0,
    )

    data = histogram.to_dict()
    assert data["name"] == "test"
    assert data["step"] == 5
    assert data["timestamp"] == 123456789

    histogram2 = HistogramData.from_dict(data)
    assert histogram2.name == histogram.name
    assert histogram2.step == histogram.step
    assert histogram2.bin_edges == histogram.bin_edges
    assert histogram2.counts == histogram.counts
