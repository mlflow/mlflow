import argparse
import re


def create_pom_with_pinned_scala_version(pom_file_path: str, scala_version: str) -> str:
    """
    Creates a new pom.xml file with the given scala version.

    Args:
        pom_file_path: Path to the original pom.xml file.
        scala_version: Scala version to pin to.

    Returns:
        Path to the new pom.xml file.
    """
    assert re.match(r"^\d+\.\d+\.\d+$", scala_version), f"Invalid scala version: {scala_version}"

    scala_compat_version = ".".join(scala_version.split(".")[:2])

    with open(pom_file_path) as f:
        content = f.read()

    # Replace scala.version with the specified scala version
    content = re.sub(
        r"<scala\.version>\d+\.\d+\.\d+</scala\.version>",
        f"<scala.version>{scala_version}</scala.version>",
        content,
    )
    content = re.sub(
        r"<scala\.compat\.version>\d+\.\d+</scala\.compat\.version>",
        f"<scala.compat.version>{scala_compat_version}</scala.compat.version>",
        content,
    )
    with open(pom_file_path, "w") as f:
        f.write(content)

    return pom_file_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scala-version",
        required=True,
        help="Update scala version in mlflow/java/spark 'pom.xml' file.",
        type=str,
    )
    return parser.parse_args()


def main():
    scala_version = parse_args().scala_version
    create_pom_with_pinned_scala_version("mlflow/java/spark/pom.xml", scala_version)


if __name__ == "__main__":
    main()
