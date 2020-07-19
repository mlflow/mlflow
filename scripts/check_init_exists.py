import os


def check_init_exists():
    dirs = []
    ignores = ["mlflow/server/js"]

    for d in ["mlflow", "tests"]:
        for root, _, files in os.walk(d):
            if any(root.startswith(i) for i in ignores):
                continue

            if any(f.endswith(".py") for f in files) and ("__init__.py" not in files):
                dirs.append(root)

    msg_template = (
        "Please add `__init__.py` in the following directories "
        "to enable lint check via pylint:\n{}"
    )
    assert len(dirs) == 0, msg_template.format("\n".join(dirs))


def main():
    check_init_exists()


if __name__ == "__main__":
    main()
