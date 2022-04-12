import os

import astroid
from pylint.interfaces import IAstroidChecker
from pylint.checkers import BaseChecker


def _is_assert_raises_without_msg(node: astroid.Call):
    return (
        isinstance(node.func, astroid.Attribute)
        and node.func.as_string() == "self.assertRaises"
        and "msg" not in [k.arg for k in node.keywords]
    )


IGNORE_FILES = list(
    map(
        os.path.abspath,
        [
            # Instructions
            # ============
            # 1. Select a file in the list below and remove it.
            # 2. Run pylint and confirm it fails.
            # 3. Fix the lines printed out in the previous step.
            # 4. Run pylint again and confirm it succeeds now.
            # 5. Run pytest and confirm the changed lines don't fail.
            # 6. Open a PR.
            # "tests/entities/test_run_status.py",
            "tests/store/model_registry/test_sqlalchemy_store.py",
            "tests/store/db/test_utils.py",
            "tests/store/tracking/__init__.py",
            "tests/store/tracking/test_file_store.py",
            "tests/store/tracking/test_sqlalchemy_store.py",
        ],
    )
)


def _should_ignore(path: str):
    return path in IGNORE_FILES


class AssertRaisesWithoutMsg(BaseChecker):
    __implements__ = IAstroidChecker

    name = "assert-raises"
    msgs = {
        "W0003": (
            "assertRaises must be called with `msg` argument",
            name,
            "Use `msg` argument",
        ),
    }
    priority = -1

    def visit_call(self, node: astroid.Call):
        if not _should_ignore(node.root().file) and _is_assert_raises_without_msg(node):
            self.add_message(self.name, node=node)
