from pathlib import Path

import pytest
from clint.index import SymbolIndex


@pytest.fixture(scope="session")
def index_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_dir = tmp_path_factory.mktemp("clint_tests")
    index_file = tmp_dir / "symbol_index.pkl"
    SymbolIndex.build().save(index_file)
    return index_file
