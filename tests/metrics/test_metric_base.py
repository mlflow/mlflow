from mlflow.metrics.base import MetricValue


def test_metric_value():
    metricValue1 = MetricValue(
        scores=[1, 2, 3],
        justifications=["foo", "bar", "baz"],
        aggregate_results={"mean": 2},
    )

    metricValue2 = MetricValue(
        scores=[1, 2, 3],
        justifications=["foo", "bar", "baz"],
    )

    metricValue3 = MetricValue(scores=["1", "2", "3"])

    metricValue4 = MetricValue(scores=[1, "2", "3"])

    assert metricValue1.scores == [1, 2, 3]
    assert metricValue1.justifications == ["foo", "bar", "baz"]
    assert metricValue1.aggregate_results == {"mean": 2}

    assert metricValue2.scores == [1, 2, 3]
    assert metricValue2.justifications == ["foo", "bar", "baz"]
    assert metricValue2.aggregate_results == {
        "mean": 2.0,
        "p90": 2.8,
        "variance": 0.6666666666666666,
    }

    assert metricValue3.scores == ["1", "2", "3"]
    assert metricValue3.justifications is None
    assert metricValue3.aggregate_results is None

    assert metricValue4.scores == [1, "2", "3"]
    assert metricValue4.justifications is None
    assert metricValue4.aggregate_results is None
