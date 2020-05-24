import os

import pytest


@pytest.mark.parametrize("top_dir", ["mlflow", "tests"])
def test_init_exists(top_dir):
    dirs = []
    ignores = ["mlflow/server/js"]
    for root, _, files in os.walk(top_dir):
        if any(root.startswith(d) for d in ignores):
            continue

        if any(f.endswith(".py") for f in files):
            if "__init__.py" not in files:
                dirs.append(root)

    assert len(dirs) == 0, "`__init__.py` does not exist in `{}`".format(dirs)
