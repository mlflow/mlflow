import ast

from clint.rules.base import Rule


class UseWalrusOperator(Rule):
    def _message(self) -> str:
        return (
            "Use the walrus operator `:=` when a variable is assigned and only used "
            "within an `if` block that tests its truthiness. "
            "For example, replace `a = ...; if a: use_a(a)` with `if a := ...: use_a(a)`."
        )

    @staticmethod
    def check(
        if_node: ast.If,
        prev_stmt: ast.stmt,
        following_stmts: list[ast.stmt],
    ) -> bool:
        """
        Flags::

            a = func()
            if a:
                use(a)

        Ignores: comparisons, tuple unpacking, multi-line, used in elif/else,
        used after if, line > 100 chars
        """
        # Check if previous statement is a simple assignment (not augmented, not annotated)
        if not isinstance(prev_stmt, ast.Assign):
            return False

        # Skip if the assignment statement spans multiple lines
        if (
            prev_stmt.end_lineno is not None
            and prev_stmt.lineno is not None
            and prev_stmt.end_lineno > prev_stmt.lineno
        ):
            return False

        # Must be a single target assignment to a Name
        if len(prev_stmt.targets) != 1:
            return False

        target = prev_stmt.targets[0]
        if not isinstance(target, ast.Name):
            return False

        var_name = target.id

        # The if condition must be just the variable name (truthiness test)
        if not isinstance(if_node.test, ast.Name):
            return False

        if if_node.test.id != var_name:
            return False

        # Check that the variable is used in the if body
        if not _name_used_in_stmts(var_name, if_node.body):
            return False

        # Check that the variable is NOT used in elif/else branches
        if if_node.orelse and _name_used_in_stmts(var_name, if_node.orelse):
            return False

        # Check that the variable is NOT used after the if statement
        if following_stmts and _name_used_in_stmts(var_name, following_stmts):
            return False

        # Skip if the fixed code would exceed 100 characters
        # Original: "if var:" -> Fixed: "if var := value:"
        value = prev_stmt.value
        if (
            value.end_col_offset is None
            or value.col_offset is None
            or if_node.test.end_col_offset is None
        ):
            return False
        value_width = value.end_col_offset - value.col_offset
        fixed_line_length = (
            if_node.test.end_col_offset
            + 4  # len(" := ")
            + value_width
            + 1  # len(":")
        )
        if fixed_line_length > 100:
            return False

        return True


def _name_used_in_stmts(name: str, stmts: list[ast.stmt]) -> bool:
    """Check if a name is used (loaded) in a list of statements.

    Skips nested function/class definitions to avoid false positives from
    inner scopes that shadow or independently use the same name.
    """
    return any(_name_used_in_node(name, stmt) for stmt in stmts)


def _name_used_in_node(name: str, node: ast.AST) -> bool:
    """Recursively check if a name is used."""
    match node:
        case ast.Name(id=id, ctx=ast.Load()) if id == name:
            return True
        case _:
            return any(_name_used_in_node(name, child) for child in ast.iter_child_nodes(node))


class WalrusOperatorVisitor(ast.NodeVisitor):
    """Visits all statement blocks to check for walrus operator opportunities."""

    def __init__(self) -> None:
        self.violations: list[ast.stmt] = []

    def _check_stmts(self, stmts: list[ast.stmt]) -> None:
        for idx, stmt in enumerate(stmts[1:], start=1):
            if isinstance(stmt, ast.If):
                prev_stmt = stmts[idx - 1]
                following_stmts = stmts[idx + 1 :]
                if UseWalrusOperator.check(stmt, prev_stmt, following_stmts):
                    self.violations.append(prev_stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_stmts(node.body)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_stmts(node.body)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self._check_stmts(node.body)
        self._check_stmts(node.orelse)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._check_stmts(node.body)
        self._check_stmts(node.orelse)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._check_stmts(node.body)
        self._check_stmts(node.orelse)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._check_stmts(node.body)
        self._check_stmts(node.orelse)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._check_stmts(node.body)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._check_stmts(node.body)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._check_stmts(node.body)
        for handler in node.handlers:
            self._check_stmts(handler.body)
        self._check_stmts(node.orelse)
        self._check_stmts(node.finalbody)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        for case in node.cases:
            self._check_stmts(case.body)
        self.generic_visit(node)
