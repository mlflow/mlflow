import threading
from unittest import mock

from mlflow.demo import generate_all_demos
from mlflow.demo.base import BaseDemoGenerator, DemoFeature, DemoResult
from mlflow.environment_variables import MLFLOW_WORKSPACE
from mlflow.utils.workspace_context import (
    clear_server_request_workspace,
    get_request_workspace,
    set_server_request_workspace,
)


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


def test_generate_all_demos_propagates_workspace_to_child_threads():
    workspace_seen_in_thread = [None]

    class ThreadSpawningGenerator(BaseDemoGenerator):
        name = DemoFeature.TRACES

        def generate(self) -> DemoResult:
            def _worker():
                workspace_seen_in_thread[0] = get_request_workspace()

            t = threading.Thread(target=_worker)
            t.start()
            t.join()
            return DemoResult(
                feature=self.name,
                entity_ids=["t1"],
                navigation_url="/test",
            )

        def _data_exists(self) -> bool:
            return False

        def store_version(self) -> None:
            pass

    generator = ThreadSpawningGenerator()

    # Simulate server middleware: set workspace via ContextVar only (no env var)
    set_server_request_workspace("test-workspace")
    try:
        assert MLFLOW_WORKSPACE.get_raw() is None

        with mock.patch("mlflow.demo.demo_registry") as mock_registry:
            mock_registry.list_generators.return_value = ["traces"]
            mock_registry.get.return_value = lambda: generator

            generate_all_demos()

        assert workspace_seen_in_thread[0] == "test-workspace"
    finally:
        clear_server_request_workspace()
