import pytest
from clint.index import SymbolIndex


@pytest.fixture(scope="session")
def index() -> SymbolIndex:
    return SymbolIndex.build()
