import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class UseGhToken(Rule):
    def _message(self) -> str:
        return "Use GH_TOKEN instead of GITHUB_TOKEN for the environment variable name."

    @staticmethod
    def check(node: ast.Call, resolver: Resolver) -> bool:
        """
        Returns True if the call reads the GITHUB_TOKEN environment variable.
        Handles:
        - os.getenv("GITHUB_TOKEN")
        - os.environ.get("GITHUB_TOKEN")
        """
        match node:
            case ast.Call(args=[ast.Constant(value="GITHUB_TOKEN"), *_]):
                match resolver.resolve(node.func):
                    case ["os", "getenv"] | ["os", "environ", "get"]:
                        return True
        return False
