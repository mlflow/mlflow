import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-type",
        help="Package type to build. Default is 'dev'.",
        choices=["skinny", "release", "dev"],
        default="dev",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.package_type == "skinny":
            with open("README_SKINNY.rst") as f1, open("README.rst") as f2:
                readme = f1.read() + "\n" + f2.read()

            with open("README.rst", "w") as f:
                f.write(readme)

            with open("pyproject.skinny.toml") as f1, open("pyproject.toml", "w") as f2:
                f2.write(f1.read())

        elif args.package_type == "release":
            with open("pyproject.release.toml") as f1, open("pyproject.toml", "w") as f2:
                f2.write(f1.read())

        subprocess.check_call([sys.executable, "-m", "build"])
    finally:
        subprocess.check_call(["git", "restore", "."])


if __name__ == "__main__":
    main()
