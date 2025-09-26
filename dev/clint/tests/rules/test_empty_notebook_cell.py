import json
from pathlib import Path

from clint.config import Config
from clint.linter import lint_file
from clint.rules.empty_notebook_cell import EmptyNotebookCell


def test_empty_notebook_cell(index_path: Path) -> None:
    notebook_content = {
        "cells": [
            {
                "cell_type": "code",
                "source": [],  # Empty cell
                "metadata": {},
                "execution_count": None,
                "outputs": [],
            },
            {
                "cell_type": "code",
                "source": ["x = 5"],
                "metadata": {},
                "execution_count": None,
                "outputs": [],
            },
            {
                "cell_type": "code",
                "source": [],  # Another empty cell
                "metadata": {},
                "execution_count": None,
                "outputs": [],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    code = json.dumps(notebook_content)
    config = Config(select={EmptyNotebookCell.name})
    violations = lint_file(Path("test_notebook.ipynb"), code, config, index_path)
    assert len(violations) == 2
    assert all(isinstance(v.rule, EmptyNotebookCell) for v in violations)
    assert violations[0].cell == 1
    assert violations[1].cell == 3
