import shutil
import typing as t


def remove_comments(s: str) -> str:
    return "\n".join(l for l in s.splitlines() if not l.strip().startswith("# "))


def get_pip_install_cmd(requirements: t.List[str]) -> str:
    # Apply `repr` to wrap each requirement with single quotes
    return "pip install " + " ".join(map(repr, requirements))


def split(s: str, *, sep: str) -> t.List[str]:
    return list(filter(None, map(str.strip, s.split(sep))))


def separator(title: str, sepchar: str = "=", length: t.Optional[int] = None) -> str:
    terminal_width = shutil.get_terminal_size(fallback=(80, 24))[0]
    length = terminal_width if length is None else length
    rest = length - len(title) - 2
    left = rest // 2 if rest % 2 else (rest + 1) // 2
    right = rest - left
    return "{} {} {}".format(sepchar * left, title, sepchar * right)
