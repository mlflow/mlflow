import astroid
from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker
from pylint.lint import PyLinter
from typing import Union

from pylint_plugins.errors import ILLEGAL_DIRECT_IMPORT, to_msgs


# These modules must not be imported directly, rather developers should use
# the wrapper modules defined in the utilities.
BLOCKLIST_PACKAGES_TO_WRAPPERS = {
    # Package name -> our custom wrapper module
    "psutil": "mlflow.utils.process"
}


class DirectImportBlocker(BaseChecker):
    __implements__ = IAstroidChecker

    name = "direct-import-checker"
    msgs = to_msgs(ILLEGAL_DIRECT_IMPORT)
    priority = -1

    def __init__(self, linter: PyLinter) -> None:
        super().__init__(linter)
        self.stack = []

    def visit_functiondef(self, node: astroid.FunctionDef):
        self.stack.append(node)

    def leave_functiondef(self, node: astroid.FunctionDef):  # pylint: disable=unused-argument
        self.stack.pop()

    def visit_import(self, node: astroid.Import):
        self._validate(node.names[0][0], node)

    def visit_importfrom(self, node: astroid.ImportFrom):
        self._validate(node.modname, node)

    def _validate(self, package_name: str, node: Union[astroid.Import, astroid.ImportFrom]):
        """Check if the import is direct import of a blocklisted package.

        For example, `import psutils` is not allowed, rather developers should use
        `from mlflow.utils.process import psutils`.
        """
        if self._is_blocklisted(package_name):
            wrapper_module = BLOCKLIST_PACKAGES_TO_WRAPPERS[package_name]

            if self.linter.current_name == wrapper_module:
                # We are inside the wrapper module, so direct import is allowed.
                return

            self.add_message(
                ILLEGAL_DIRECT_IMPORT.name,
                node=node,
                args=(package_name, wrapper_module),
            )

    def _is_blocklisted(self, name):
        return name.split(".", 1)[0] in BLOCKLIST_PACKAGES_TO_WRAPPERS.keys()
