import functools
import importlib
import inspect
import re
import textwrap
from pathlib import Path

from docutils.parsers.rst import directives
from sphinx.directives.code import CodeBlock


def get_obj_and_module(obj_path):
    splits = obj_path.split(".")
    for i in reversed(range(1, len(splits) + 1)):
        try:
            maybe_module = ".".join(splits[:i])
            mod = importlib.import_module(maybe_module)
        except ImportError:
            continue
        return mod, functools.reduce(getattr, splits[i:], mod)

    raise Exception("Should not reach here")


def get_code_block_line(mod_file, obj_line, lineno_in_docstring):
    with mod_file.open() as f:
        lines = f.readlines()[obj_line:]
        for offset, line in enumerate(lines):
            if line.lstrip().startswith('"""'):
                extra_offset = 0
                while re.search(r"[^\"\s]", lines[offset + extra_offset]) is None:
                    extra_offset += 1
                return obj_line + offset + extra_offset + lineno_in_docstring


# fmt: off
# This function helps understand what each variable represents in `get_code_block_line`.
def _func():           # <- obj_line
    """                  <- obj_line + offset

    Docstring            <- obj_line + offset + extra_offset

    .. code-block::      <- obj_line + offset + extra_offset + lineno_in_docstring
        :test:
        ...
    """
# fmt: on


def get_code_block_location(obj_path, lineno_in_docstring, repo_root):
    mod, obj = get_obj_and_module(obj_path)
    abs_mod_file = Path(mod.__file__)
    rel_mod_file = abs_mod_file.relative_to(repo_root)
    obj_line = inspect.getsourcelines(obj)[1]
    code_block_line = get_code_block_line(abs_mod_file, obj_line, lineno_in_docstring)
    return f"{rel_mod_file}:{code_block_line}"


class TestCodeBlockDirective(CodeBlock):
    """
    Overrides the `code-block` directive to dump code blocks marked with the `:test:` option
    to files for testing.

    ```
    .. code-block:: python
        :test:

        print("Hello, world!")
    ```
    """

    option_spec = {**CodeBlock.option_spec, "test": directives.flag}

    def _dump_code_block(self):
        docs_dir = Path.cwd()
        repo_root = docs_dir.parent
        directory = docs_dir.joinpath(".examples")
        directory.mkdir(exist_ok=True)
        source, lineno_in_docstring = self.get_source_info()
        obj_path = source.split(":docstring of ")[1]
        code_block_location = get_code_block_location(obj_path, lineno_in_docstring, repo_root)
        name = re.sub(r"[\._]+", "_", obj_path).strip("")
        filename = f"test_{name}_{lineno_in_docstring}.py"
        content = textwrap.indent("\n".join(self.content), " " * 4)
        code = "\n".join(
            [
                f"# Location: {code_block_location}",
                "import pytest",
                "",
                "",
                # Show the code block location in the test report.
                f"@pytest.mark.parametrize('_', [' {code_block_location} '])",
                "def test(_):",
                content,
                "",
                "",
                'if __name__ == "__main__":',
                "    test()",
                "",
            ]
        )
        directory.joinpath(filename).write_text(code)

    def run(self):
        if "test" in self.options:
            self._dump_code_block()
        return super().run()


def setup(app):
    app.add_directive("code-block", TestCodeBlockDirective, override=True)
    return {
        "version": "builtin",
        "parallel_read_safe": False,
        "parallel_write_safe": False,
    }
