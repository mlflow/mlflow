import ast
from collections.abc import Iterator
from contextlib import contextmanager


class Resolver:
    def __init__(self) -> None:
        self.name_map: dict[str, list[str]] = {}
        self._scope_stack: list[dict[str, list[str]]] = []

    def clear(self) -> None:
        """Clear all name mappings. Useful when starting to process a new file."""
        self.name_map.clear()
        self._scope_stack.clear()

    def enter_scope(self) -> None:
        """Enter a new scope by taking a snapshot of current mappings."""
        self._scope_stack.append(self.name_map.copy())

    def exit_scope(self) -> None:
        """Exit current scope by restoring the previous snapshot."""
        if self._scope_stack:
            self.name_map = self._scope_stack.pop()

    @contextmanager
    def scope(self) -> Iterator[None]:
        """Context manager for automatic scope management."""
        self.enter_scope()
        try:
            yield
        finally:
            self.exit_scope()

    def add_import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.asname:
                self.name_map[alias.asname] = alias.name.split(".")
            else:
                toplevel = alias.name.split(".", 1)[0]
                self.name_map[toplevel] = [toplevel]

    def add_import_from(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            return

        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            module_parts = node.module.split(".")
            self.name_map[name] = module_parts + [alias.name]

    def resolve(self, node: ast.expr) -> list[str] | None:
        """
        Resolve a node to its fully qualified name parts.

        Args:
            node: AST node to resolve, typically a Call, Name, or Attribute.

        Returns:
            List of name parts (e.g., ["threading", "Thread"]) or None if unresolvable
        """
        if isinstance(node, ast.Call):
            parts = self._extract_call_parts(node.func)
        elif isinstance(node, ast.Name):
            parts = [node.id]
        elif isinstance(node, ast.Attribute):
            parts = self._extract_call_parts(node)
        else:
            return None

        return self._resolve_parts(parts) if parts else None

    def _extract_call_parts(self, node: ast.expr) -> list[str]:
        if isinstance(node, ast.Name):
            return [node.id]
        elif isinstance(node, ast.Attribute) and (
            base_parts := self._extract_call_parts(node.value)
        ):
            return base_parts + [node.attr]
        return []

    def _resolve_parts(self, parts: list[str]) -> list[str] | None:
        if not parts:
            return None

        # Check if the first part is in our name mapping
        if resolved_base := self.name_map.get(parts[0]):
            return resolved_base + parts[1:]

        return None
