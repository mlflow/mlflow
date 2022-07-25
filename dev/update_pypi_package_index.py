"""
A script to update the index of known PyPI packages, which is used during dependency inference to
print warnings when unrecognized packages are inferred as dependencies. The index is located
at 'mlflow/pypi_package_index.json'.

# How to run (make sure you're in the repository root):
$ python dev/update_pypi_package_index.py
"""

import argparse
import json
import posixpath
import requests
import sys
from datetime import datetime
from html.parser import HTMLParser


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        help="Path to the PyPI package index (default: mlflow/pypi_package_index.json)",
        default="mlflow/pypi_package_index.json",
        required=False,
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    package_names = set()

    class PyPIHTMLParser(HTMLParser):  # pylint: disable=abstract-method
        def handle_starttag(self, tag, attrs):
            if tag == "a":
                for name, value in attrs:
                    if name == "href":
                        # Packages are represented in the PyPI simple index
                        # as anchors with href attributes. These are the only
                        # elements we care about
                        package_name = posixpath.basename(value.rstrip("/"))
                        package_names.add(package_name)

    index_url = "https://pypi.org/simple/"
    raw_index_html = requests.get(index_url).text

    parser = PyPIHTMLParser()
    parser.feed(raw_index_html)

    formatted_package_index = {
        "index_date": datetime.today().strftime("%Y-%m-%d"),
        "package_names": list(package_names),
    }

    with open(args.path, "w") as f:
        json.dump(formatted_package_index, f, separators=(",", ":"))


if __name__ == "__main__":
    main(sys.argv[1:])
