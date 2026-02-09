from unittest import mock

from mlflow.demo import generate_all_demos
from mlflow.demo.base import DemoResult


def test_generate_all_demos_calls_generators(stub_generator, another_stub_generator):
    with mock.patch("mlflow.demo.demo_registry") as mock_registry:
        mock_registry.list_generators.return_value = ["stub", "another"]
        mock_registry.get.side_effect = lambda name: {
            "stub": lambda: stub_generator,
            "another": lambda: another_stub_generator,
        }[name]

        results = generate_all_demos()

        assert len(results) == 2
        assert all(isinstance(r, DemoResult) for r in results)


def test_generate_all_demos_skips_existing(stub_generator):
    stub_generator.data_exists_value = True
    stub_generator.stored_version_value = stub_generator.version

    with mock.patch("mlflow.demo.demo_registry") as mock_registry:
        mock_registry.list_generators.return_value = ["stub"]
        mock_registry.get.return_value = lambda: stub_generator

        results = generate_all_demos()

        assert len(results) == 0
        assert not stub_generator.generate_called


def test_generate_all_demos_empty_registry():
    with mock.patch("mlflow.demo.demo_registry") as mock_registry:
        mock_registry.list_generators.return_value = []

        results = generate_all_demos()

        assert results == []


def test_generate_all_demos_stores_version(stub_generator):
    stub_generator.data_exists_value = False
    stub_generator.stored_version_value = None

    with mock.patch("mlflow.demo.demo_registry") as mock_registry:
        mock_registry.list_generators.return_value = ["stub"]
        mock_registry.get.return_value = lambda: stub_generator

        generate_all_demos()

        assert stub_generator.stored_version_value == stub_generator.version


def test_generate_all_demos_regenerates_on_version_mismatch(stub_generator):
    stub_generator.data_exists_value = True
    stub_generator.stored_version_value = 1
    stub_generator.version = 2

    with mock.patch("mlflow.demo.demo_registry") as mock_registry:
        mock_registry.list_generators.return_value = ["stub"]
        mock_registry.get.return_value = lambda: stub_generator

        results = generate_all_demos()

        assert len(results) == 1
        assert stub_generator.generate_called
        assert stub_generator.delete_demo_called
        assert stub_generator.stored_version_value == 2
