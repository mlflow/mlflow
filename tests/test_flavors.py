import ast
import os

import mlflow


def read_file(path):
    with open(path) as f:
        return f.read()


def is_model_flavor(src):
    for node in ast.iter_child_nodes(ast.parse(src)):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "FLAVOR_NAME"
        ):
            return True
    return False


def iter_flavor_names():
    for root, _, files in os.walk("mlflow"):
        for f in files:
            if f != "__init__.py":
                continue

            path = os.path.join(root, f)
            src = read_file(path)
            if is_model_flavor(src):
                basename = os.path.basename(root)
                yield os.path.splitext(basename)[0]


def test_all_flavors_can_be_accessed_from_mlflow():
    flavor_names = list(iter_flavor_names())
    assert len(flavor_names) != 0
    for flavor_name in flavor_names:
        assert hasattr(mlflow, flavor_name)
