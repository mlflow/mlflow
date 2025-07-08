import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class ThreadPoolExecutorWithoutThreadNamePrefix(Rule):
    def _message(self) -> str:
        return (
            "`ThreadPoolExecutor()` must be called with a `thread_name_prefix` argument to improve "
            "debugging and traceability of thread-related issues."
        )

    @staticmethod
    def check(node: ast.Call, resolver: Resolver) -> bool:
        """
        Returns True if the call is ThreadPoolExecutor() without a thread_name_prefix parameter.
        """
        return (
            (resolved := resolver.resolve(node))
            and resolved == ["concurrent", "futures", "ThreadPoolExecutor"]
            and not any(keyword.arg == "thread_name_prefix" for keyword in node.keywords)
        )
