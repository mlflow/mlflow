import threading
from unittest import mock

import pytest

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


@pytest.mark.parametrize("stored_version", [None, 1])
def test_generate_all_demos_regenerates_on_version_mismatch(stub_generator, stored_version):
    stub_generator.data_exists_value = True
    stub_generator.stored_version_value = stored_version
    stub_generator.version = 2

    with mock.patch("mlflow.demo.demo_registry") as mock_registry:
        mock_registry.list_generators.return_value = ["stub"]
        mock_registry.get.return_value = lambda: stub_generator

        results = generate_all_demos()

        assert len(results) == 1
        assert stub_generator.generate_called
        assert stub_generator.delete_demo_called
        assert stub_generator.stored_version_value == 2


@pytest.mark.parametrize("allowed", [False, True])
@pytest.mark.parametrize(
    ("refresh", "data_exists", "stored_version"),
    [(False, True, None), (False, True, 1), (True, True, 2), (True, False, None)],
)
def test_generate_all_demos_checks_refresh_permission(
    stub_generator, refresh, data_exists, stored_version, allowed
):
    stub_generator.data_exists_value = data_exists
    stub_generator.stored_version_value = stored_version
    stub_generator.version = 2
    can_refresh = mock.Mock(return_value=allowed)

    with mock.patch(
        "mlflow.demo.demo_registry",
        list_generators=mock.Mock(return_value=["stub"]),
        get=mock.Mock(return_value=lambda: stub_generator),
    ) as registry:
        results = generate_all_demos(refresh=refresh, can_refresh=can_refresh)

    registry.list_generators.assert_called_once_with()
    registry.get.assert_called_once_with("stub")
    can_refresh.assert_called_once_with(stub_generator)
    assert len(results) == int(allowed)
    assert stub_generator.delete_demo_called is allowed
    assert stub_generator.generate_called is allowed
    assert stub_generator.stored_version_value == (2 if allowed else stored_version)


def test_generate_all_demos_generates_missing_features_when_refresh_denied(
    stub_generator, another_stub_generator
):
    stub_generator.data_exists_value = True
    stub_generator.stored_version_value = 1
    stub_generator.version = 2
    generators = {"stub": stub_generator, "another": another_stub_generator}
    can_refresh = mock.Mock(return_value=False)

    with mock.patch(
        "mlflow.demo.demo_registry",
        list_generators=mock.Mock(return_value=list(generators)),
        get=mock.Mock(side_effect=lambda name: lambda: generators[name]),
    ) as registry:
        results = generate_all_demos(can_refresh=can_refresh)

    registry.list_generators.assert_called_once_with()
    assert registry.get.call_args_list == [mock.call("stub"), mock.call("another")]
    can_refresh.assert_called_once_with(stub_generator)
    assert [result.feature for result in results] == [another_stub_generator.name]
    assert stub_generator.delete_demo_called is False
    assert stub_generator.generate_called is False
    assert stub_generator.stored_version_value == 1
    assert another_stub_generator.generate_called is True
    assert another_stub_generator.delete_demo_called is False


def test_generate_all_demos_explicit_refresh_without_permission_callback(stub_generator):
    stub_generator.data_exists_value = True
    stub_generator.stored_version_value = stub_generator.version

    with mock.patch(
        "mlflow.demo.demo_registry",
        list_generators=mock.Mock(return_value=["stub"]),
        get=mock.Mock(return_value=lambda: stub_generator),
    ) as registry:
        results = generate_all_demos(refresh=True)

    registry.list_generators.assert_called_once_with()
    registry.get.assert_called_once_with("stub")
    assert len(results) == 1
    assert stub_generator.delete_demo_called is True
    assert stub_generator.generate_called is True


@pytest.mark.parametrize("fail_generation", [False, True])
def test_generate_all_demos_nested_generation_inherits_and_resets_refresh_permission(
    stub_generator, another_stub_generator, fail_generation
):
    stub_generator.data_exists_value = True
    stub_generator.stored_version_value = 1
    stub_generator.version = 2
    generators = {"stub": stub_generator, "another": another_stub_generator}
    can_refresh = mock.Mock(return_value=False)

    def generate_with_dependency():
        assert generate_all_demos(features=["stub"]) == []
        if fail_generation:
            raise RuntimeError("Generation failed")
        return DemoResult(feature=DemoFeature.EVALUATION, entity_ids=[], navigation_url="/stub")

    with (
        mock.patch(
            "mlflow.demo.demo_registry",
            list_generators=mock.Mock(return_value=list(generators)),
            get=mock.Mock(side_effect=lambda name: lambda: generators[name]),
        ) as registry,
        mock.patch.object(
            another_stub_generator, "generate", side_effect=generate_with_dependency
        ) as generate,
    ):
        if fail_generation:
            with pytest.raises(RuntimeError, match="Generation failed"):
                generate_all_demos(features=["another"], can_refresh=can_refresh)
        else:
            results = generate_all_demos(features=["another"], can_refresh=can_refresh)
            assert [result.feature for result in results] == [DemoFeature.EVALUATION]

        can_refresh.assert_called_once_with(stub_generator)
        assert stub_generator.delete_demo_called is False
        assert stub_generator.generate_called is False

        results = generate_all_demos(features=["stub"])

    assert registry.list_generators.call_count == 3
    assert registry.get.call_args_list == [
        mock.call("another"),
        mock.call("stub"),
        mock.call("stub"),
    ]
    generate.assert_called_once_with()
    can_refresh.assert_called_once_with(stub_generator)
    assert len(results) == 1
    assert stub_generator.delete_demo_called is True
    assert stub_generator.generate_called is True


def test_generate_all_demos_propagates_workspace_to_child_threads():
    workspace_seen_in_thread = [None]

    class ThreadSpawningGenerator(BaseDemoGenerator):
        name = DemoFeature.TRACES

        def generate(self) -> DemoResult:
            def _worker():
                workspace_seen_in_thread[0] = get_request_workspace()

            t = threading.Thread(name="test-generate-worker", target=_worker)
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
