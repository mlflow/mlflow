import importlib.metadata
import sys


def _get_entry_points(group: str) -> list[importlib.metadata.EntryPoint]:
    if sys.version_info >= (3, 10):
        return importlib.metadata.entry_points(group=group)

    entrypoints = importlib.metadata.entry_points()
    try:
        return entrypoints.get(group, [])
    except AttributeError:
        return entrypoints.select(group=group)


def get_entry_points(group: str) -> list[importlib.metadata.EntryPoint]:
    return _get_entry_points(group)
