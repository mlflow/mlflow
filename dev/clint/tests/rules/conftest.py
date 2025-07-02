import pytest
from clint.config import Config
from clint.index import SymbolIndex


@pytest.fixture(scope="session")
def index() -> SymbolIndex:
    return SymbolIndex.build()


@pytest.fixture(scope="session")
def config() -> Config:
    return Config()
