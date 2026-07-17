import pytest

from mlflow.demo.base import BaseDemoGenerator, DemoFeature, DemoResult
from mlflow.demo.registry import DemoRegistry


class StubGenerator(BaseDemoGenerator):
    name = DemoFeature.TRACES

    def __init__(self, version: int = 1):
        self._version = version
        self.generate_called = False
        self.data_exists_value = False
        self.stored_version_value = None
        self.delete_demo_called = False
        super().__init__()

    @property
    def version(self) -> int:
        return self._version

    @version.setter
    def version(self, value: int) -> None:
        self._version = value

    def generate(self) -> DemoResult:
        self.generate_called = True
        return DemoResult(
            feature=self.name,
            entity_ids=["entity-1"],
            navigation_url="/stub",
        )

    def _data_exists(self) -> bool:
        return self.data_exists_value

    def _get_stored_version(self) -> int | None:
        return self.stored_version_value

    def store_version(self) -> None:
        self.stored_version_value = self.version

    def delete_demo(self) -> None:
        self.delete_demo_called = True


class AnotherStubGenerator(BaseDemoGenerator):
    name = DemoFeature.EVALUATION

    def generate(self) -> DemoResult:
        return DemoResult(
            feature=self.name,
            entity_ids=["entity-2"],
            navigation_url="/evaluation",
        )

    def _data_exists(self) -> bool:
        return False


@pytest.fixture
def stub_generator():
    return StubGenerator()


@pytest.fixture
def another_stub_generator():
    return AnotherStubGenerator()


@pytest.fixture
def fresh_registry():
    return DemoRegistry()
