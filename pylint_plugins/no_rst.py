from __future__ import annotations

import re
from pathlib import Path

from astroid import nodes
from pylint import checkers
from pylint.interfaces import IAstroidChecker

from pylint_plugins.errors import NO_RST, to_msgs

PARAM_REGEX = re.compile(r"\s+:param\s+\w+:", re.MULTILINE)
RETURN_REGEX = re.compile(r"\s+:returns?:", re.MULTILINE)

# TODO: Remove this once all docstrings are updated
IGNORE = {
    str(Path(p).resolve())
    for p in [
        "dev/set-mlflow-spark-scala-version.py",
        "mlflow/gateway/client.py",
        "mlflow/gateway/providers/utils.py",
        "mlflow/keras/callback.py",
        "mlflow/legacy_databricks_cli/configure/provider.py",
        "mlflow/metrics/base.py",
        "mlflow/metrics/genai/base.py",
        "mlflow/mleap/__init__.py",
        "mlflow/models/evaluation/default_evaluator.py",
        "mlflow/projects/databricks.py",
        "mlflow/projects/kubernetes.py",
        "mlflow/store/_unity_catalog/registry/rest_store.py",
        "mlflow/store/artifact/azure_data_lake_artifact_repo.py",
        "mlflow/store/artifact/gcs_artifact_repo.py",
        "mlflow/store/model_registry/rest_store.py",
        "mlflow/store/tracking/rest_store.py",
        "mlflow/utils/docstring_utils.py",
        "mlflow/utils/rest_utils.py",
        "tests/conftest.py",
        "tests/evaluate/test_validation.py",
        "tests/helper_functions.py",
        "tests/projects/test_project_spec.py",
        "tests/resources/data/dataset.py",
        "tests/resources/mlflow-test-plugin/mlflow_test_plugin/dummy_dataset.py",
        "tests/sagemaker/mock/__init__.py",
        "tests/store/artifact/test_cli.py",
        "tests/tracking/fluent/test_fluent.py",
        "tests/transformers/helper.py",
        "tests/utils/test_docstring_utils.py",
    ]
}


class NoRstChecker(checkers.BaseChecker):
    __implements__ = IAstroidChecker

    name = "no-rst"
    msgs = to_msgs(NO_RST)
    priority = -1

    def visit_classdef(self, node: nodes.ClassDef) -> None:
        self._check_docstring(node)

    def visit_functiondef(self, node: nodes.FunctionDef) -> None:
        self._check_docstring(node)

    visit_asyncfunctiondef = visit_functiondef

    def _check_docstring(self, node: nodes.Module | nodes.ClassDef | nodes.FunctionDef) -> None:
        if (
            node.root().file not in IGNORE
            and node.doc_node
            and (doc := node.doc_node.value)
            and (PARAM_REGEX.search(doc) or RETURN_REGEX.search(doc))
        ):
            self.add_message(NO_RST.name, node=node.doc_node)
