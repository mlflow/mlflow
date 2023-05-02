from mlflow.entities import RunInputs
from mlflow.entities.dataset_input import DatasetInput


def _check_inputs(run_datasets, datasets):
    for d1, d2 in zip(run_datasets, datasets):
        assert d1.dataset.digest == d2.dataset.digest
        assert d1.dataset.name == d2.dataset.name
        assert d1.dataset.source_type == d2.dataset.source_type
        assert d1.dataset.source == d2.dataset.source
        for t1, t2 in zip(d1.tags, d2.tags):
            assert t1.key == t2.key
            assert t1.value == t2.value


def _check(inputs, datasets):
    assert isinstance(inputs, RunInputs)
    _check_inputs(inputs.dataset_inputs, datasets)


def test_creation_and_hydration(run_inputs):
    run_inputs, datasets = run_inputs
    _check(run_inputs, datasets)
    as_dict = {
        "dataset_inputs": datasets,
    }
    assert dict(run_inputs) == as_dict
    proto = run_inputs.to_proto()
    run_inputs2 = RunInputs.from_proto(proto)
    _check(run_inputs2, datasets)
    assert isinstance(run_inputs2.dataset_inputs[0], DatasetInput)
