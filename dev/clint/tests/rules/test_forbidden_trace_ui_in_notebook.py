from pathlib import Path

from clint.config import Config
from clint.linter import lint_file
from clint.rules.forbidden_trace_ui_in_notebook import ForbiddenTraceUIInNotebook


def test_forbidden_trace_ui_in_notebook(index_path: Path) -> None:
    notebook_content = """
{
 "cells": [
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# This is a normal cell"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<iframe src='http://localhost:5000/static-files/lib/notebook-trace-renderer/index.html'></iframe>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# This cell contains trace UI output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# This is a normal cell"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
"""
    code = notebook_content
    config = Config(select={ForbiddenTraceUIInNotebook.name})
    violations = lint_file(Path("test.ipynb"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, ForbiddenTraceUIInNotebook) for v in violations)
    assert violations[0].cell == 2
