import pytest

from mlflow.demo.base import BaseDemoGenerator, DemoFeature


def test_register_and_get(fresh_registry, stub_generator):
    fresh_registry.register(stub_generator)
    assert fresh_registry.get(DemoFeature.TRACES) is stub_generator


def test_register_duplicate_raises(fresh_registry, stub_generator):
    fresh_registry.register(stub_generator)
    with pytest.raises(ValueError, match="already registered"):
        fresh_registry.register(stub_generator)


def test_get_unknown_raises(fresh_registry):
    with pytest.raises(ValueError, match="not found"):
        fresh_registry.get("unknown")


def test_list_generators(fresh_registry, stub_generator, another_stub_generator):
    assert fresh_registry.list_generators() == []

    fresh_registry.register(stub_generator)
    assert fresh_registry.list_generators() == [DemoFeature.TRACES]

    fresh_registry.register(another_stub_generator)
    assert set(fresh_registry.list_generators()) == {DemoFeature.TRACES, DemoFeature.EVALUATION}


def test_contains(fresh_registry, stub_generator):
    assert DemoFeature.TRACES not in fresh_registry

    fresh_registry.register(stub_generator)
    assert DemoFeature.TRACES in fresh_registry


def test_register_requires_name(fresh_registry):
    class NoNameGenerator(BaseDemoGenerator):
        def generate(self):
            pass

        def _data_exists(self):
            return False

    with pytest.raises(ValueError, match="must define 'name'"):
        fresh_registry.register(NoNameGenerator)
