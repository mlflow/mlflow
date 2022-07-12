import argparse

from packaging.version import Version


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", help="Release version to validate, e.g., '1.2.3'", required=True
    )
    return parser.parse_args()


def main():
    args = parse_args()
    version = Version(args.version)
    msg = (
        f"Invalid release version: '{args.version}', "
        "must be in the format of <major>.<minor>.<micro>"
    )
    assert len(version.release) == 3, msg


if __name__ == "__main__":
    main()
