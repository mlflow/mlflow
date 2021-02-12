import ast
import os
from collections import namedtuple


def iterate_python_scripts(directory):
    for root, _, files in os.walk(directory):
        yield from (os.path.join(root, f) for f in files if f.endswith(".py"))


def read_file(path):
    with open(path) as f:
        return f.read()


def is_builtin_type_alias(attr_node):
    """
    Returns True if the passed-in attribute node represents a NumPy's built-in type alias
    (e.g. np.int)
    """
    return (
        isinstance(attr_node.value, ast.Name)
        and attr_node.value.id in ["numpy", "np"]
        and attr_node.attr
        in ["bool", "complex", "float", "int", "long", "object", "str", "unicode"]
    )


class NodeWithPath(namedtuple("NodeWithPath", ["node", "path"])):
    def __str__(self):
        node, path = self
        attr = node.value.id + "." + node.attr
        loc = "{}:{}:{}".format(path, node.lineno, node.col_offset)
        return "{} | {}".format(loc, attr)


class NumpyBuiltinAliasFinder(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.found = []
        self.path = None

    def visit_Attribute(self, node):
        if is_builtin_type_alias(node):
            self.found.append(NodeWithPath(node=node, path=self.path))
        self.generic_visit(node)

    def find(self, path):
        assert path.endswith(".py")
        self.path = path
        self.visit(ast.parse(read_file(path)))

    def summary(self):
        return "\n".join(map(str, self.found))


def test_finder(tmpdir):
    src = """
a = np.object
b = numpy.int
c = np.float(1.0)

def func(x=np.bool):
    pass
"""
    tmp_path = tmpdir.join("test.py").strpath
    with open(tmp_path, "w") as f:
        f.write(src)

    finder = NumpyBuiltinAliasFinder()
    finder.find(tmp_path)

    expected = [
        ("np", "object"),
        ("numpy", "int"),
        ("np", "float"),
        ("np", "bool"),
    ]
    assert len(finder.found) == len(expected)
    for (node, path), (id_expected, attr_expected) in zip(finder.found, expected):
        assert node.value.id == id_expected
        assert node.attr == attr_expected
        assert path == path


def test_no_numpy_builtin_type_aliases():
    """
    Note that this test doesn't detect built-in type aliases in comments or docstrings
    """
    finder = NumpyBuiltinAliasFinder()
    for direcory in ["mlflow", "tests", "examples"]:
        for path in iterate_python_scripts(direcory):
            finder.find(path)

    msg = (
        "NumPy's built-in type aliases (e.g. np.int) have been deprecated. "
        "Please replace them by following this instruction: "
        "https://numpy.org/devdocs/release/1.20.0-notes.html#using-the-aliases-of-builtin-types-like-np-int-is-deprecated\n"  # noqa
    )
    assert finder.found == [], msg + finder.summary()
