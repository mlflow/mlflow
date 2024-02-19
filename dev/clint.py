"""
A custom linter to enforce rules that ruff doesn't cover.
"""
from __future__ import annotations

import ast
import itertools
import re
import sys
import tokenize
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import tomli

PARAM_REGEX = re.compile(r"\s+:param\s+\w+:", re.MULTILINE)
RETURN_REGEX = re.compile(r"\s+:returns?:", re.MULTILINE)
DISABLE_COMMENT_REGEX = re.compile(r"clint:\s*disable=([a-z0-9-]+)")

# https://github.com/PyCQA/isort/blob/b818cec889657cb786beafe94a6641f8fc0f0e64/isort/stdlibs/py311.py
BUILTIN_MODULES = {
    "_ast",
    "_thread",
    "abc",
    "aifc",
    "argparse",
    "array",
    "ast",
    "asynchat",
    "asyncio",
    "asyncore",
    "atexit",
    "audioop",
    "base64",
    "bdb",
    "binascii",
    "bisect",
    "builtins",
    "bz2",
    "cProfile",
    "calendar",
    "cgi",
    "cgitb",
    "chunk",
    "cmath",
    "cmd",
    "code",
    "codecs",
    "codeop",
    "collections",
    "colorsys",
    "compileall",
    "concurrent",
    "configparser",
    "contextlib",
    "contextvars",
    "copy",
    "copyreg",
    "crypt",
    "csv",
    "ctypes",
    "curses",
    "dataclasses",
    "datetime",
    "dbm",
    "decimal",
    "difflib",
    "dis",
    "distutils",
    "doctest",
    "email",
    "encodings",
    "ensurepip",
    "enum",
    "errno",
    "faulthandler",
    "fcntl",
    "filecmp",
    "fileinput",
    "fnmatch",
    "fractions",
    "ftplib",
    "functools",
    "gc",
    "getopt",
    "getpass",
    "gettext",
    "glob",
    "graphlib",
    "grp",
    "gzip",
    "hashlib",
    "heapq",
    "hmac",
    "html",
    "http",
    "idlelib",
    "imaplib",
    "imghdr",
    "imp",
    "importlib",
    "inspect",
    "io",
    "ipaddress",
    "itertools",
    "json",
    "keyword",
    "lib2to3",
    "linecache",
    "locale",
    "logging",
    "lzma",
    "mailbox",
    "mailcap",
    "marshal",
    "math",
    "mimetypes",
    "mmap",
    "modulefinder",
    "msilib",
    "msvcrt",
    "multiprocessing",
    "netrc",
    "nis",
    "nntplib",
    "ntpath",
    "numbers",
    "operator",
    "optparse",
    "os",
    "ossaudiodev",
    "pathlib",
    "pdb",
    "pickle",
    "pickletools",
    "pipes",
    "pkgutil",
    "platform",
    "plistlib",
    "poplib",
    "posix",
    "posixpath",
    "pprint",
    "profile",
    "pstats",
    "pty",
    "pwd",
    "py_compile",
    "pyclbr",
    "pydoc",
    "queue",
    "quopri",
    "random",
    "re",
    "readline",
    "reprlib",
    "resource",
    "rlcompleter",
    "runpy",
    "sched",
    "secrets",
    "select",
    "selectors",
    "shelve",
    "shlex",
    "shutil",
    "signal",
    "site",
    "smtpd",
    "smtplib",
    "sndhdr",
    "socket",
    "socketserver",
    "spwd",
    "sqlite3",
    "sre",
    "sre_compile",
    "sre_constants",
    "sre_parse",
    "ssl",
    "stat",
    "statistics",
    "string",
    "stringprep",
    "struct",
    "subprocess",
    "sunau",
    "symtable",
    "sys",
    "sysconfig",
    "syslog",
    "tabnanny",
    "tarfile",
    "telnetlib",
    "tempfile",
    "termios",
    "test",
    "textwrap",
    "threading",
    "time",
    "timeit",
    "tkinter",
    "token",
    "tokenize",
    "tomllib",
    "trace",
    "traceback",
    "tracemalloc",
    "tty",
    "turtle",
    "turtledemo",
    "types",
    "typing",
    "unicodedata",
    "unittest",
    "urllib",
    "uu",
    "uuid",
    "venv",
    "warnings",
    "wave",
    "weakref",
    "webbrowser",
    "winreg",
    "winsound",
    "wsgiref",
    "xdrlib",
    "xml",
    "xmlrpc",
    "zipapp",
    "zipfile",
    "zipimport",
    "zlib",
    "zoneinfo",
}


def ignore_map(code: str) -> dict[str, set[int]]:
    """
    Creates a mapping of rule name to line numbers to ignore.

    {
        "<rule_name>": {<line_number>, ...},
        ...
    }
    """
    mapping: dict[str, set[int]] = {}
    lines = iter(code.splitlines(True))
    for tok in tokenize.generate_tokens(lambda: next(lines)):
        if tok.type != tokenize.COMMENT:
            continue
        comment = tok.string.strip()
        if m := DISABLE_COMMENT_REGEX.search(comment):
            mapping.setdefault(m.group(1), set()).add(tok.start[0])
    return mapping


@dataclass
class Rule:
    name: str
    message: str


@dataclass
class Violation:
    rule: Rule
    path: str
    lineno: int
    col_offset: int

    def __str__(self):
        return f"{self.path}:{self.lineno}:{self.col_offset}: {self.rule.message}"


NO_RST = Rule("no-rst", "Do not use RST style. Use Google style instead.")
LAZY_BUILTIN_IMPORT = Rule(
    "lazy-builtin-import", "Builtin modules must be imported at the top level."
)


NO_RST_IGNORE = {
    "mlflow/gateway/client.py",
    "mlflow/gateway/providers/utils.py",
    "mlflow/keras/callback.py",
    "mlflow/metrics/base.py",
    "mlflow/metrics/genai/base.py",
    "mlflow/models/utils.py",
    "mlflow/projects/databricks.py",
    "mlflow/projects/kubernetes.py",
    "mlflow/store/_unity_catalog/registry/rest_store.py",
    "mlflow/store/artifact/azure_data_lake_artifact_repo.py",
    "mlflow/store/artifact/gcs_artifact_repo.py",
    "mlflow/store/model_registry/rest_store.py",
    "mlflow/store/tracking/rest_store.py",
    "mlflow/utils/docstring_utils.py",
    "mlflow/utils/rest_utils.py",
    "tests/utils/test_docstring_utils.py",
}


class Linter(ast.NodeVisitor):
    def __init__(self, path: str, ignore: dict[str, set[int]]):
        self.stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.path = path
        self.ignore = ignore
        self.violations: list[Violation] = []

    def _check(self, node: ast.AST, rule: Rule) -> None:
        if lines := self.ignore.get(rule.name):
            if node.lineno in lines:
                return
        self.violations.append(Violation(rule, self.path, node.lineno, node.col_offset))

    def _docstring(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> ast.Constant | None:
        if (
            isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.s, str)
        ):
            return node.body[0].value
        return None

    def _no_rst(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self.path in NO_RST_IGNORE:
            return

        if (nd := self._docstring(node)) and (
            PARAM_REGEX.search(nd.s) or RETURN_REGEX.search(nd.s)
        ):
            self._check(nd, NO_RST)

    def _is_in_function(self):
        return bool(self.stack)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node)
        self._no_rst(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append(node)
        self._no_rst(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.stack.append(node)
        self._no_rst(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        if self._is_in_function():
            for alias in node.names:
                if alias.name.split(".", 1)[0] in BUILTIN_MODULES:
                    self._check(node, LAZY_BUILTIN_IMPORT)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._is_in_function() and node.module.split(".", 1)[0] in BUILTIN_MODULES:
            self._check(node, LAZY_BUILTIN_IMPORT)
        self.generic_visit(node)


def lint_file(path: str) -> list[Violation]:
    with open(path) as f:
        code = f.read()
        linter = Linter(path, ignore_map(code))
        linter.visit(ast.parse(code))
        return linter.violations


def _exclude_regex() -> re.Pattern:
    with open("pyproject.toml", "rb") as f:
        data = tomli.load(f)
        exclude = data["tool"]["clint"]["exclude"]
        return re.compile("|".join(map(re.escape, exclude)))


def main():
    EXCLUDE_REGEX = _exclude_regex()
    files = sys.argv[1:]
    with ProcessPoolExecutor() as pool:
        futures = [pool.submit(lint_file, f) for f in files if not EXCLUDE_REGEX.match(f)]
        violations_iter = itertools.chain.from_iterable(f.result() for f in as_completed(futures))
        if violations := list(violations_iter):
            sys.stderr.write("\n".join(map(str, violations)) + "\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
