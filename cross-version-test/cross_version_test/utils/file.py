import re
import typing as t

import requests
import yaml


def read_yaml(path: str) -> t.Dict[str, t.Any]:
    if re.match("^https?://", path):
        resp = requests.get(path)
        resp.raise_for_status()
        return yaml.load(resp.content, Loader=yaml.SafeLoader)
    else:
        with open(path) as f:
            return yaml.load(f, Loader=yaml.SafeLoader)
