from mlflow.entities import RunInput


def _check_input(run_datasets, datasets):
    for d1, d2 in zip(run_datasets, datasets):
        assert d1.dataset.digest == d2.dataset.digest
        assert d1.dataset.name == d2.dataset.name
        assert d1.dataset.source_type == d2.dataset.source_type
        assert d1.dataset.source == d2.dataset.source
        for t1, t2 in zip(d1.tags, d2.tags):
            assert t1.key == t2.key
            assert t1.value == t2.value


def _check(input, datasets):
    assert isinstance(input, RunInput)
    _check_input(input.dataset_inputs, datasets)


def test_creation_and_hydration(run_input):
    run_input, datasets = run_input
    _check(run_input, datasets)
    as_dict = {
        "dataset_inputs": datasets,
    }
    assert dict(run_input) == as_dict
    proto = run_input.to_proto()
    run_input2 = RunInput.from_proto(proto)
    _check(run_input2, datasets)
