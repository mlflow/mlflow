import json
import os
import sys
from typing import List

import pytest
import requests


def fetch_labels() -> List[str]:
    if "GITHUB_ACTIONS" not in os.environ:
        return []

    if os.getenv("GITHUB_EVENT_NAME", "") != "pull_request":
        return []

    github_event_path = os.environ["GITHUB_EVENT_PATH"]
    with open(github_event_path) as f:
        data = json.load(f)
        pr_number = data["pull_request"]["number"]

    repo = os.environ["GITHUB_REPOSITORY"]
    resp = requests.get(f"https://api.github.com/repos/{repo}/issues/{pr_number}/labels")
    resp.raise_for_status()
    return [label["name"] for label in resp.json()]


def main() -> None:
    fail_fast = "fail-fast" in fetch_labels()
    extra_options = ["--exitfirst"] if fail_fast else []
    pytest.main([*extra_options, *sys.argv[1:]])


if __name__ == "__main__":
    main()
