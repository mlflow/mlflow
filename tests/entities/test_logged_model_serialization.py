from unittest import mock

import pytest

from mlflow.entities import LoggedModel, Metric


@pytest.mark.parametrize("include_metrics", [None, True, False])
def test_to_proto_include_metrics(include_metrics):
    metric = Metric("accuracy", 0.9, 123, 1)
    model = LoggedModel(
        experiment_id="1",
        model_id="m-test",
        name="test",
        artifact_location="file:///synthetic",
        creation_timestamp=1,
        last_updated_timestamp=2,
        tags={"purpose": "test"},
        params={"width": "128"},
        metrics=[metric],
    )
    expected = model.to_proto()
    if include_metrics is False:
        expected.data.ClearField("metrics")
    with mock.patch.object(metric, "to_proto", wraps=metric.to_proto) as to_proto:
        actual = model.to_proto(
            **({} if include_metrics is None else {"include_metrics": include_metrics})
        )
        assert actual == expected
        if include_metrics is False:
            to_proto.assert_not_called()
        else:
            to_proto.assert_called_once()
    assert model.metrics == [metric]


@pytest.mark.parametrize("metrics", [None, []])
@pytest.mark.parametrize("include_metrics", [True, False])
def test_to_proto_without_metrics(metrics, include_metrics):
    model = LoggedModel("1", "m-test", "test", "file:///synthetic", 1, 2, metrics=metrics)
    assert not model.to_proto(include_metrics=include_metrics).data.metrics


@pytest.mark.parametrize("include_metrics", [None, True, False])
def test_to_proto_large_metrics(include_metrics):
    metrics = [Metric("accuracy", i / 1000, 123 + i, i) for i in range(1000)]
    model = LoggedModel("1", "m-test", "test", "file:///synthetic", 1, 2, metrics=metrics)
    proto = model.to_proto(
        **({} if include_metrics is None else {"include_metrics": include_metrics})
    )
    assert len(proto.data.metrics) == (0 if include_metrics is False else 1000)
    if include_metrics is not False:
        assert proto.data.metrics[-1] == metrics[-1].to_proto()
        assert [m.to_proto() for m in LoggedModel.from_proto(proto).metrics] == list(
            proto.data.metrics
        )
    assert model.metrics is metrics
