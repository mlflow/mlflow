import hashlib
import textwrap
from pathlib import Path

from sphinx.directives.code import CodeBlock


class TestCodeBlockDirective(CodeBlock):
    def _dump_code_block(self):
        docs_dir = Path.cwd()
        directory = docs_dir.joinpath(".examples")
        directory.mkdir(exist_ok=True)
        source, lineno_in_docstring = self.get_source_info()
        rel_from_root = Path(source).relative_to(docs_dir.parent)
        key = (rel_from_root, lineno_in_docstring)
        suffix = hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:32]
        filename = "test_code_block_{}.py".format(suffix)
        content = textwrap.indent("\n".join(self.content), " " * 4)
        code = "\n".join(
            [
                f"# source: {rel_from_root}",
                f"# lineno_in_docstring: {lineno_in_docstring}",
                f"# command to test this code block: python {filename}",
                "",
                "",
                "def test_code_block():",
                content,
                "",
                "",
                'if __name__ == "__main__":',
                "    test_code_block()",
                "",
            ]
        )
        directory.joinpath(filename).write_text(code)

    def run(self):
        self._dump_code_block()
        return super().run()


def setup(app):
    app.add_directive("test-code-block", TestCodeBlockDirective)
    return {
        "version": "builtin",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
