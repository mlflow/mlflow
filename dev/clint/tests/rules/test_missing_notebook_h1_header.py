import json
from pathlib import Path

from clint.config import Config
from clint.linter import lint_file
from clint.rules import MissingNotebookH1Header


def test_missing_notebook_h1_header(index_path: Path) -> None:
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
    code = json.dumps(notebook)
    config = Config(select={MissingNotebookH1Header.name})
    results = lint_file(Path("test.ipynb"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, MissingNotebookH1Header)


def test_missing_notebook_h1_header_positive(index_path: Path) -> None:
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
    code = json.dumps(notebook)
    config = Config(select={MissingNotebookH1Header.name})
    results = lint_file(Path("test_positive.ipynb"), code, config, index_path)
    assert len(results) == 0
