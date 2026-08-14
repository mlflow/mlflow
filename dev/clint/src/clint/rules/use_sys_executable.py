import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class UseSysExecutable(Rule):
    def _message(self) -> str:
        return (
            "Use `[sys.executable, '-m', 'mlflow', ...]` when running mlflow CLI in a subprocess."
        )

    @staticmethod
    def check(node: ast.Call, resolver: Resolver) -> bool:
        """
        Returns True if `node` looks like `subprocess.Popen(["mlflow", ...])`.
        """
        resolved = resolver.resolve(node)
        if (
            resolved
            and len(resolved) == 2
            and resolved[0] == "subprocess"
            and resolved[1] in ["Popen", "run", "check_output", "check_call"]
            and node.args
        ):
            first_arg = node.args[0]
            if isinstance(first_arg, ast.List) and first_arg.elts:
                first_elem = first_arg.elts[0]
                return (
                    isinstance(first_elem, ast.Constant)
                    and isinstance(first_elem.value, str)
                    and first_elem.value == "mlflow"
                )
        return False
