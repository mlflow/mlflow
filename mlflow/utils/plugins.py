import importlib.metadata
import sys
from typing import List


def _get_entry_points(group: str) -> List[importlib.metadata.EntryPoint]:
    if sys.version_info >= (3, 10):
        return importlib.metadata.entry_points(group=group)

    try:
        return importlib.metadata.entry_points().get(group, [])
    except AttributeError:
        return importlib.metadata.entry_points().select(group=group)


def get_entry_points(group: str) -> List[importlib.metadata.EntryPoint]:
    return _get_entry_points(group)
