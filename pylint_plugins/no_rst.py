from __future__ import annotations

import re

from astroid import nodes
from pylint import checkers
from pylint.interfaces import IAstroidChecker

from pylint_plugins.errors import NO_RST, to_msgs

PARAM_REGEX = re.compile(r"\s+:param\s+\w+:", re.MULTILINE)
RETURN_REGEX = re.compile(r"\s+:returns?:", re.MULTILINE)


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
            node.doc_node
            and (doc := node.doc_node.value)
            and (PARAM_REGEX.search(doc) or RETURN_REGEX.search(doc))
        ):
            self.add_message(NO_RST.name, node=node.doc_node)
