import json
from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import lint_file
from clint.rules import MissingNotebookH1Header


def test_missing_notebook_h1_header(index: SymbolIndex, config: Config, tmp_path: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["## Some other header"],
            },
            {
                "cell_type": "code",
                "source": ["print('hello')"],
            },
        ]
    }
    tmp_file = tmp_path / "test.ipynb"
    tmp_file.write_text(json.dumps(notebook))
    results = lint_file(tmp_file, config, index)
    assert len(results) == 1
    assert isinstance(results[0].rule, MissingNotebookH1Header)


def test_missing_notebook_h1_header_positive(
    index: SymbolIndex, config: Config, tmp_path: Path
) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# This is a title"],
            },
            {
                "cell_type": "code",
                "source": ["print('hello')"],
            },
        ]
    }
    tmp_file = tmp_path / "test_positive.ipynb"
    tmp_file.write_text(json.dumps(notebook))
    results = lint_file(tmp_file, config, index)
    assert len(results) == 0
