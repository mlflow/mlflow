# Remove this test script once we drop Python 3.5 support.

import ast
import os


def search_python_scripts(dirs):
    for d in dirs:
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.endswith(".py"):
                    yield os.path.join(root, f)


def read_file(path):
    with open(path) as f:
        return f.read()


def search_f_strings(node):
    for c in ast.iter_child_nodes(node):
        if isinstance(c, ast.JoinedStr):
            yield (c.lineno, c.col_offset)
        yield from search_f_strings(c)


def test_no_f_strings():
    dirs = ['mlflow', 'tests', 'examples']
    f_strings_all = []

    for path in search_python_scripts(dirs):
        src = read_file(path)
        root = ast.parse(src)
        f_strings = list(search_f_strings(root))
        if len(f_strings) > 0:
            f_strings_all += [
                "{}:{}:{}: {}".format(path, *pos, "Remove f-string")
                for pos in f_strings
            ]

    assert len(f_strings_all) == 0, '\n' + '\n'.join(f_strings_all)
