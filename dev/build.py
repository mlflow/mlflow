import argparse
import contextlib
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build MLflow package.")
    parser.add_argument(
        "--package-type",
        help="Package type to build. Default is 'dev'.",
        choices=["skinny", "release", "dev"],
        default="dev",
    )
    parser.add_argument(
        "--sha",
        help="If specified, include the SHA in the wheel name as a build tag.",
    )
    return parser.parse_args()


@contextlib.contextmanager
def restore_changes():
    try:
        yield
    finally:
        subprocess.check_call(["git", "restore", ":^dev/build.py"])


def main():
    args = parse_args()

    for path in map(Path, ["build", "dist", "mlflow.egg-info", "mlflow_skinny.egg-info"]):
        if not path.exists():
            continue
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)

    with restore_changes():
        pyproject = Path("pyproject.toml")
        if args.package_type == "skinny":
            readme = Path("README.md")
            readme_skinny = Path("README_SKINNY.md")
            readme.write_text(readme_skinny.read_text() + "\n" + readme.read_text())

            pyproject.write_text(Path("pyproject.skinny.toml").read_text())

        elif args.package_type == "release":
            pyproject.write_text(Path("pyproject.release.toml").read_text())

        subprocess.check_call([sys.executable, "-m", "build"])

    if args.sha:
        # If build succeeds, there should be one wheel in the dist directory
        wheel = next(Path("dist").glob("mlflow*.whl"))
        name, version, rest = wheel.name.split("-", 2)
        build_tag = f"0.sha.{args.sha}"  # build tag must start with a digit
        wheel.rename(wheel.with_name(f"{name}-{version}-{build_tag}-{rest}"))


if __name__ == "__main__":
    main()
