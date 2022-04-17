import re
import typing as t

_FLAVOR_NAME_MATCHER = re.compile(r"^(mlflow|tests)/(.+?)(_autolog(ging)?)?(\.py|/)")


def get_flavor_name_from_path(path: str) -> t.Optional[str]:
    match = _FLAVOR_NAME_MATCHER.match(path)
    return match.group(2) if match else None


def get_changed_flavors(changed_files: t.List[str]) -> t.Set[str]:
    return set(filter(None, map(get_flavor_name_from_path, changed_files)))
