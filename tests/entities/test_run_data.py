from mlflow.entities import RunData


def _check_metrics(metric_objs, metrics_dict, expected_metrics):
    assert {m.key for m in metric_objs} == {m.key for m in expected_metrics}
    assert {m.value for m in metric_objs} == {m.value for m in expected_metrics}
    assert {m.timestamp for m in metric_objs} == {m.timestamp for m in expected_metrics}
    assert {m.step for m in metric_objs} == {m.step for m in expected_metrics}
    assert len(metrics_dict) == len(expected_metrics)
    assert metrics_dict == {m.key: m.value for m in expected_metrics}


def _check_params(params_dict, expected_params):
    assert params_dict == {p.key: p.value for p in expected_params}


def _check_tags(tags_dict, expected_tags):
    assert tags_dict == {t.key: t.value for t in expected_tags}


def _check(rd, metrics, params, tags):
    assert isinstance(rd, RunData)
    _check_metrics(rd._metric_objs, rd.metrics, metrics)
    _check_params(rd.params, params)
    _check_tags(rd.tags, tags)


def test_creation_and_hydration(run_data):
    rd, metrics, params, tags = run_data
    _check(rd, metrics, params, tags)
    as_dict = {
        "metrics": {m.key: m.value for m in metrics},
        "params": {p.key: p.value for p in params},
        "tags": {t.key: t.value for t in tags},
    }
    assert dict(rd) == as_dict
    proto = rd.to_proto()
    rd2 = RunData.from_proto(proto)
    _check(rd2, metrics, params, tags)
