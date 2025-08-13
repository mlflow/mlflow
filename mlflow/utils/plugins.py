import importlib.metadata


def _get_entry_points(group: str) -> list[importlib.metadata.EntryPoint]:
    return importlib.metadata.entry_points(group=group)


def get_entry_points(group: str) -> list[importlib.metadata.EntryPoint]:
    return _get_entry_points(group)
