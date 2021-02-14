# import ast
# import os
# from collections import namedtuple

# import pytest


# def iterate_python_scripts(directory):
#     for root, _, files in os.walk(directory):
#         yield from (os.path.join(root, f) for f in files if f.endswith(".py"))


# def read_file(path):
#     with open(path) as f:
#         return f.read()


# class NodeWithPath(namedtuple("NodeWithPath", ["node", "path"])):
#     def __str__(self):
#         node, path = self
#         attr = node.value.id + "." + node.attr
#         loc = "{}:{}:{}".format(path, node.lineno, node.col_offset)
#         return "{} | {}".format(loc, attr)


# class NumpyBuiltinAliasFinder(ast.NodeVisitor):
#     def __init__(self):
#         super().__init__()
#         self.found = []
#         self.path = None
#         self.numpy_imported_as = None

#     def visit_Import(self, node):
#         if self.numpy_imported_as is None:
#             for imp in node.names:
#                 if imp.name == "numpy":
#                     self.numpy_imported_as = imp.asname or imp.name
#                     break

#         self.generic_visit(node)

#     def visit_Attribute(self, node):
#         if self.numpy_imported_as and self.is_builtin_type_alias(node):
#             self.found.append(NodeWithPath(node=node, path=self.path))
#         self.generic_visit(node)

#     def is_builtin_type_alias(self, attr_node):
#         return (
#             isinstance(attr_node.value, ast.Name)
#             and attr_node.value.id == self.numpy_imported_as
#             and attr_node.attr
#             in ["bool", "complex", "float", "int", "long", "object", "str", "unicode"]
#         )

#     def find(self, path):
#         assert path.endswith(".py")
#         self.path = path
#         self.visit(ast.parse(read_file(path)))

#     def summary(self):
#         return "\n".join(map(str, self.found))


# @pytest.mark.parametrize("numpy_imported_as", ["numpy", "np"])
# def test_finder(numpy_imported_as, tmpdir):
#     src = """
# import numpy as {asname}

# a = {asname}.object
# b = {asname}.int
# c = {asname}.float(1.0)

# def func(d={asname}.bool):
#     e = {asname}.str
# """.format(
#         asname=numpy_imported_as
#     )
#     tmp_path = tmpdir.join("test.py").strpath
#     with open(tmp_path, "w") as f:
#         f.write(src)

#     finder = NumpyBuiltinAliasFinder()
#     finder.find(tmp_path)

#     attrs_expected = [
#         "object",
#         "int",
#         "float",
#         "bool",
#         "str",
#     ]

#     assert finder.numpy_imported_as == numpy_imported_as
#     assert len(finder.found) == len(attrs_expected)
#     for (node, path), attr_expected in zip(finder.found, attrs_expected):
#         assert node.value.id == numpy_imported_as
#         assert node.attr == attr_expected
#         assert path == path


# @pytest.mark.parametrize("numpy_imported_as", ["numpy", "np"])
# def test_finder_when_numpy_is_lazily_imported_within_function(numpy_imported_as, tmpdir):
#     src = """
# def func():
#     import numpy as {asname}

#     a = {asname}.object
# """.format(
#         asname=numpy_imported_as
#     )
#     tmp_path = tmpdir.join("test.py").strpath
#     with open(tmp_path, "w") as f:
#         f.write(src)

#     finder = NumpyBuiltinAliasFinder()
#     finder.find(tmp_path)

#     attrs_expected = ["object"]
#     assert finder.numpy_imported_as == numpy_imported_as
#     assert len(finder.found) == len(attrs_expected)
#     for (node, path), attr_expected in zip(finder.found, attrs_expected):
#         assert node.value.id == numpy_imported_as
#         assert node.attr == attr_expected
#         assert path == path


# def test_finder_when_np_is_not_numpy(tmpdir):
#     src = """
# np = NP()

# a = np.object
# """
#     tmp_path = tmpdir.join("test.py").strpath
#     with open(tmp_path, "w") as f:
#         f.write(src)

#     finder = NumpyBuiltinAliasFinder()
#     finder.find(tmp_path)

#     assert finder.numpy_imported_as is None
#     assert len(finder.found) == 0


# def test_no_numpy_builtin_type_aliases():
#     """
#     Note that this test doesn't detect built-in type aliases in comments or docstrings
#     """
#     finder = NumpyBuiltinAliasFinder()
#     for direcory in ["mlflow", "tests", "examples"]:
#         for path in iterate_python_scripts(direcory):
#             finder.find(path)

#     msg = (
#         "NumPy's built-in type aliases (e.g. np.int) have been deprecated. "
#         "Please replace them by following this instruction: "
#         "https://numpy.org/devdocs/release/1.20.0-notes.html#using-the-aliases-of-builtin-types-like-np-int-is-deprecated\n"  # noqa
#     )
#     assert finder.found == [], msg + finder.summary()
