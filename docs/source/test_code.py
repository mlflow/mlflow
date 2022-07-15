import textwrap
import importlib
import inspect
import typing as t
from pathlib import Path
from functools import reduce

from docutils import nodes
from sphinx.util.docutils import SphinxDirective


PROCESSED_OBJECTS = {}


def create_test_code(code: str, location: str, id: int) -> str:
    template = """
# Location: {location}
def test_{id}():
{code}
"""
    return (
        template.format(
            location=location,
            code=textwrap.indent(code, " " * 4),
            id=id,
        ).strip()
        + "\n"
    )


def get_directive_lineno(module: str, module_file: Path, obj_name: str, lineno: int) -> int:
    mod = importlib.import_module(module)
    obj = reduce(lambda o, n: getattr(o, n), obj_name.split("."), mod)
    obj_start_lineno = inspect.getsourcelines(obj)[1]
    lines = module_file.read_text().splitlines()
    for offset, line in enumerate(lines[obj_start_lineno - 1 :]):
        if line.endswith(":"):
            return obj_start_lineno + offset + 1 + lineno


def parse_object_name(fq_obj_name: str, repo_root: Path) -> t.Tuple[t.List[str], t.List[str]]:
    slice_end = 0
    parts = fq_obj_name.split(".")
    while slice_end < len(parts):
        slice_end += 1
        if repo_root.joinpath(*parts[:slice_end]).exists():
            continue

        (*rest, last) = parts[:slice_end]
        if repo_root.joinpath(*rest, last + ".py").exists():
            module_file = repo_root.joinpath(*rest, last + ".py")
            return parts[:slice_end], module_file, parts[slice_end:]

        if repo_root.joinpath(*rest, "__init__.py").exists():
            module_file = repo_root.joinpath(*rest, "__init__.py")
            slice_end -= 1
            return parts[:slice_end], module_file, parts[slice_end:]

        raise Exception(f"Something went wrong")


class TestCodeDirective(SphinxDirective):
    """
    Custom sphinx directive to dump example code blocks into python scripts for testing.
    """

    has_content = True

    def run(self):
        docs_dir = Path.cwd().resolve()
        repo_root = docs_dir.parent
        tests_dir = docs_dir.joinpath("tests")

        source, lineno = self.get_source_info()
        code = "\n".join(self.content)
        if ":docstring of" in source:
            source, fq_obj_name = source.split(":docstring of ")
            module_parts, module_file, obj_name_parts = parse_object_name(fq_obj_name, repo_root)
            module = ".".join(module_parts)
            obj_name = ".".join(obj_name_parts)
            directive_lineno = get_directive_lineno(module, module_file, obj_name, lineno)
            directive_loc = str(module_file) + ":" + str(directive_lineno)

            script_name = "_".join(["test", *obj_name_parts]) + ".py"
            script_dir = tests_dir.joinpath(*module_parts)
            script_dir.mkdir(exist_ok=True, parents=True)
            script_path = script_dir.joinpath(script_name)
            test_code = create_test_code(code, directive_loc, directive_lineno)

            directive_ids = PROCESSED_OBJECTS.get(fq_obj_name, [])
            if directive_ids:
                if directive_lineno not in directive_ids:
                    PROCESSED_OBJECTS[fq_obj_name].append(directive_lineno)
                    script_path.write_text(script_path.read_text() + "\n\n" + test_code)
                pass
            else:
                PROCESSED_OBJECTS[fq_obj_name] = [directive_lineno]
                script_path.write_text(f"# Object: {fq_obj_name}" + "\n\n" + test_code)
        else:
            # TODO: Support test-code directives in a .rst file
            pass

        node = nodes.literal_block(code, code)
        return [node]


def setup(app):
    app.add_directive("test-code", TestCodeDirective)
    return {"version": "0.1"}
