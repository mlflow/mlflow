import astroid
from pylint.interfaces import IAstroidChecker
from pylint.checkers import BaseChecker


class SetChecker(BaseChecker):
    __implements__ = IAstroidChecker

    name = "set-checker"
    USE_SET_LITERAL = "use-set-literal"
    msgs = {
        "W0005": (
            "Use set literal (e.g. `{'a', 'b'}`) instead of applying `set()` on list or "
            "tuple literal (e.g. `set(['a', 'b'])`)",
            USE_SET_LITERAL,
            "Use set literal",
        ),
    }
    priority = -1

    def visit_call(self, node: astroid.Call):
        if (
            node.func.as_string() == "set"
            and node.args
            and isinstance(node.args[0], (astroid.List, astroid.Tuple))
        ):
            self.add_message(SetChecker.USE_SET_LITERAL, node=node)
