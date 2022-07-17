import textwrap
import importlib
import inspect
from types import ModuleType
import typing as t
from pathlib import Path
from functools import reduce

from docutils import nodes
from sphinx.util.docutils import SphinxDirective


def to_test_function(code: str, directive_location: str, name: str) -> str:
    """
    Create a test function that runs the given code.
    """
    template = """
# Location: {directive_location}
def test_{name}():
{code}
"""
    return template.format(
        directive_location=directive_location,
        code=textwrap.indent(code, " " * 4),
        name=name,
    ).lstrip()


def get_directive_lineno(
    mod: ModuleType, mod_file: str, obj_name: str, lineno_in_docstring: int
) -> int:
    """
    Return the line number in the module file where the directive is defined.
    """
    obj = reduce(lambda o, n: getattr(o, n), obj_name.split("."), mod)
    obj_start_lineno = inspect.getsourcelines(obj)[1]
    lines = Path(mod_file).read_text().splitlines()
    for offset, line in enumerate(lines[obj_start_lineno - 1 :]):
        if line.endswith(":"):
            return obj_start_lineno + offset + lineno_in_docstring


def parse_object_name(fq_obj_name: str, repo_root: Path) -> t.Tuple[t.List[str], t.List[str]]:
    """
    Parse the given fully-qualified object name into a list of module names and a list of object
    names. For example, "mlflow.sklearn.log_model" would be parsed into (["mlflow", "sklearn"], ["log_model"]).
    """
    slice_end = 0
    parts = fq_obj_name.split(".")
    while slice_end < len(parts):
        slice_end += 1
        if repo_root.joinpath(*parts[:slice_end]).exists():
            continue

        (*rest, last) = parts[:slice_end]
        if repo_root.joinpath(*rest, last + ".py").exists():
            return parts[:slice_end], parts[slice_end:]

        if repo_root.joinpath(*rest, "__init__.py").exists():
            return parts[: slice_end - 1], parts[slice_end - 1 :]

        raise Exception(f"Something went wrong. Should not reach here.")


PROCESSED_OBJECTS = {}


class TestCodeBlockDirective(SphinxDirective):
    """
    Custom Sphinx directive to store test-code blocks in python scripts that can be run with pytest.
    """

    has_content = True

    def run(self):
        docs_dir = Path.cwd().resolve()
        repo_root = docs_dir.parent
        tests_dir = docs_dir.joinpath("tests")

        source, lineno_in_docstring = self.get_source_info()
        code = "\n".join(self.content)
        if ":docstring of" in source:
            source, fq_obj_name = source.split(":docstring of ")
            module_parts, obj_name_parts = parse_object_name(fq_obj_name, repo_root)
            module = ".".join(module_parts)
            obj_name = ".".join(obj_name_parts)
            mod = importlib.import_module(module)
            mod_file = inspect.getsourcefile(mod)

            script_name = "_".join(["test", *obj_name_parts]) + ".py"
            script_dir = tests_dir.joinpath(*module_parts)
            script_dir.mkdir(exist_ok=True, parents=True)
            script_path = script_dir.joinpath(script_name)

            directive_lineno = get_directive_lineno(mod, mod_file, obj_name, lineno_in_docstring)
            directive_loc = f" {mod_file}:{directive_lineno}"
            test_function = to_test_function(code, directive_loc, name=str(directive_lineno))
            directive_ids = PROCESSED_OBJECTS.get(fq_obj_name, [])
            if directive_ids:
                if directive_lineno not in directive_ids:
                    PROCESSED_OBJECTS[fq_obj_name].append(directive_lineno)
                    script_path.write_text(script_path.read_text() + "\n\n" + test_function)
            else:
                PROCESSED_OBJECTS[fq_obj_name] = [directive_lineno]
                script_path.write_text(f"# Object: {fq_obj_name}" + "\n\n" + test_function)
        else:
            # TODO: Support test-code blocks in a .rst file
            pass

        node = nodes.literal_block(code, code)
        return [node]


def setup(app):
    app.add_directive("test-code-block", TestCodeBlockDirective)
    return {"version": "0.1"}
