import os


def test_init_exists():
    dirs = []
    ignores = ["mlflow/server/js"]

    for d in ["mlflow", "tests"]:
        for root, _, files in os.walk(d):
            if any(d.startswith(i) for i in ignores):
                continue

            if any(f.endswith(".py") for f in files) and ("__init__.py" not in files):
                dirs.append(root)

    msg_template = (
        "Please add `__init__.py` in the following directories "
        "otherwise pylint ignores them:\n{}"
    )
    assert len(dirs) == 0, msg_template.format("\n".join(dirs))
